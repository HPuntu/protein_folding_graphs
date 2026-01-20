import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import networkx as nx
from collections import defaultdict
from scipy.stats import binned_statistic
import pandas as pd


# Functionality for inferring contact flip energies from graph networks using kinetic approach 

def get_bit_flipped(val1: int, val2: int):
    '''
    Determine which contact residue pair was fipped between two integer bit representation
    contact maps with hamming distance 1.
     +1 : 0 -> 1 (contact formed)
     -1 : 1 -> 0 (contact broken)
    '''
    diff = val1 ^ val2
    if diff == 0:
        return -1, 0  # no change
    # check single-bit
    if diff & (diff - 1) != 0:
        return -1, 0  # multi-bit difference; not a single contact flip

    bit_pos = diff.bit_length() - 1
    b1 = (val1 >> bit_pos) & 1
    b2 = (val2 >> bit_pos) & 1

    if b1 == 0 and b2 == 1:
        return bit_pos, +1
    elif b1 == 1 and b2 == 0:
        return bit_pos, -1
    return -1, 0


def get_residue_pair_from_bit(bit_index: int, n_residues: int):
    '''
    Given bit index of flipped bit and number of residues we can infer the
    residue pair (i, j) by converting upper-triangle flattened bit with 
    row-major over upper triangle excluding diagonal: e.g. row i contains pairs (i, i+1)...(i, n-1)
    '''
    if bit_index < 0:
        raise ValueError("bit_index must be >= 0")

    current_idx = bit_index
    for i in range(n_residues):
        row_length = (n_residues - 1) - i
        if current_idx < row_length:
            j = i + 1 + current_idx
            return (i, j)
        current_idx -= row_length

    raise ValueError(f"Bit index {bit_index} out of bounds for n_residues={n_residues}")


def infer_contact_flip_energies_from_graph(
    G,
    ints=None,         
    node_state_attr=None,        
    Gm=None,                
    manifold_edge_attr="manifold", 
    temporal_count_attr="temp_count",
    node_pop_attr="frame_count",   
    tau_ps=200.0,
    T=300.0,
    n_residues=10,
    max_paths=2000,           
    k_ref_mode="median",        
    eps=1e-12,
    beta_prior=None,            
):
    '''
    Implements the following algorithm to infer per-contact flip relative dG
    from kinetics: 

    For every temporal edge ($u$,$v$) with observed transition counts $C_{uv}^{\text{obs}}$:
        1. Find all shortest paths between $u$ and $v$ on the manifold graph where $N_\text{paths}$ is the number of degenerate shortest paths.
        2. Assign each path a flux\* $C_{uv}^{\text{obs}}/N_\text{paths}$ 
        3. For each path iterate through its constituent manifold edges $(i,j)$ adding the flux to each edges' inferred count $C_{ij}^{\text{inferred}}$ 

    Also implements smoothing with a beta bernoulli Bayesian prior.
    '''

    # constants
    kB = 0.0019872041  # kcal/mol/K
    beta = 1.0 / (kB * T)
    tau = float(tau_ps)  # in ps; rates returned in 1/ps

    # get mapping node -> state int
    if ints is None:
        if node_state_attr is None:
            raise ValueError("Provide either `ints` (node->int list) or `node_state_attr` on nodes.")
        # assumes node labels are integer indices if using ints list;
        node_to_int = {u: int(G.nodes[u][node_state_attr]) for u in G.nodes()}
    else:
        # ints is list indexed by node ids
        node_to_int = {u: int(ints[u]) for u in G.nodes()}

    # make sure we have functional manifold graph (manifold edges only for shortest paths)
    if Gm is None:
        manifold_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get(manifold_edge_attr, False)]
        Gm = G.edge_subgraph(manifold_edges).copy()
    else:
        Gm = Gm.copy()
    Gm_und = Gm.to_undirected()

    # initialise inferred_count on manifold edges in G
    for u, v in G.edges():
        if Gm_und.has_edge(u, v):
            G[u][v]["inferred_count"] = 0.0

    # temporal edges with counts
    temporal_edges = []
    for u, v, d in G.edges(data=True):
        c = d.get(temporal_count_attr, None)
        if c is None:
            continue
        c = float(c)
        if c > 0 and u != v:
            temporal_edges.append((u, v, c))

    # ALGORITHM project temporal flux onto manifold edges via shortest paths
    for u, v, count in temporal_edges:
        if not (u in Gm_und and v in Gm_und):
            continue
        try:
            paths = []
            for idx, p in enumerate(nx.all_shortest_paths(Gm_und, u, v)):
                paths.append(p)
                if max_paths is not None and len(paths) >= max_paths:
                    break
            if not paths:
                continue

            flux_per_path = count / len(paths)

            for path in paths:
                for a, b in zip(path[:-1], path[1:]):
                    # store on whichever directed edge exists in G
                    if G.has_edge(a, b) and Gm_und.has_edge(a, b):
                        G[a][b]["inferred_count"] += flux_per_path
                    elif G.has_edge(b, a) and Gm_und.has_edge(a, b):
                        G[b][a]["inferred_count"] += flux_per_path
                    else:
                        # if neither direction exists, skip (or add edge if you want)
                        pass

        except nx.NetworkXNoPath:
            print("ERROR: No path on manifold between", u, v)
            continue

    # compute per-manifold-edge flip identity and edge-level p/k/dG
    contact_counts = defaultdict(float)   # C_inf(a,b)
    contact_opps = defaultdict(float)     # N_adj(a,b)
    contact_edges = defaultdict(list)     # list of directed edges contributing

    # ALGORITHM convert inferred counts to p/k/dG
    edge_k_list = []
    for u, v, d in G.edges(data=True):
        if not Gm_und.has_edge(u, v): # 
            continue

        # infer bit flip from node states
        bit_pos, direction = get_bit_flipped(node_to_int[u], node_to_int[v])
        if bit_pos < 0:
            # not a single contact flip edge; skip per-contact aggregation
            # this also catches bad manifold labeling
            d["bit_index"] = None
            d["residue_pair"] = None
            d["flip_dir"] = 0
            d["p_flip"] = np.nan
            d["k_flip"] = np.nan
            d["dG_rel"] = np.nan
            continue

        res_pair = get_residue_pair_from_bit(bit_pos, n_residues=n_residues)

        d["bit_index"] = int(bit_pos)
        d["residue_pair"] = res_pair
        d["flip_dir"] = int(direction)  # flip direction, +1 form, -1 break

        inferred = float(d.get("inferred_count", 0.0)) # inferred count on this edge
        N_u = float(G.nodes[u].get(node_pop_attr, 0.0)) # population at node u

        # edge p/k conditional on being at node u with population count at u
        if N_u > 0:
            p_uv = inferred / N_u
            p_uv = float(np.clip(p_uv, 0.0, 1.0 - eps))
            k_uv = -np.log(1.0 - p_uv) / tau
            d["p_flip"] = p_uv
            d["k_flip"] = k_uv
            if np.isfinite(k_uv) and k_uv > 0:
                edge_k_list.append(k_uv)
        else:
            d["p_flip"] = np.nan
            d["k_flip"] = np.nan

        # per-contact (direction-agnostic) aggregation
        contact_counts[res_pair] += inferred
        contact_opps[res_pair] += N_u
        contact_edges[res_pair].append((u, v))

    # choose reference rate (default is median k)
    edge_k_arr = np.array([k for k in edge_k_list if np.isfinite(k) and k > 0])
    if edge_k_arr.size == 0:
        k_ref = np.nan
    else:
        if isinstance(k_ref_mode, (int, float)):
            k_ref = float(k_ref_mode)
        elif k_ref_mode == "max":
            k_ref = float(np.max(edge_k_arr))
        else:
            k_ref = float(np.median(edge_k_arr))

    # edge-level relative dG
    for u, v, d in G.edges(data=True):
        if not Gm_und.has_edge(u, v):
            continue
        k_uv = d.get("k_flip", np.nan)
        if np.isfinite(k_uv) and k_uv > 0 and np.isfinite(k_ref) and k_ref > 0:
            d["dG_rel"] = -(1.0 / beta) * np.log(k_uv / k_ref)
        else:
            d["dG_rel"] = np.nan

    # per-contact p/k/dG ---
    alpha_beta = beta_prior  # (alpha, beta) or None
    contact_p = {}
    contact_k = {}
    contact_dG = {}

    for pair in contact_counts.keys():
        Cinf = float(contact_counts[pair])
        Nadj = float(contact_opps[pair])

        if Nadj <= 0:
            contact_p[pair] = np.nan
            contact_k[pair] = np.nan
            contact_dG[pair] = np.nan
            continue

        # Bayesian smoothing with Beta-Bernoulli posterior mean
        if alpha_beta is not None:
            a0, b0 = alpha_beta
            p = (Cinf + a0) / (Nadj + a0 + b0)
        else:
            p = Cinf / Nadj

        p = float(np.clip(p, 0.0, 1.0 - eps))
        k = -np.log(1.0 - p) / tau

        contact_p[pair] = p
        contact_k[pair] = k

        if np.isfinite(k) and k > 0 and np.isfinite(k_ref) and k_ref > 0:
            contact_dG[pair] = -(1.0 / beta) * np.log(k / k_ref)
        else:
            contact_dG[pair] = np.nan

    rows = []
    for pair, dG in contact_dG.items():
        i, j = pair
        rows.append({
            "residue_pair": pair,
            "res_label": f"{i}-{j}",
            "C_inf": contact_counts[pair],
            "N_adj": contact_opps[pair],
            "p_flip": contact_p[pair],
            "k_flip_1_per_ps": contact_k[pair],
            "dG_rel_kcal_per_mol": dG,
            "n_edges_contexts": len(contact_edges[pair]),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # higher dG_rel => slower/harder flip
        df = df.sort_values("dG_rel_kcal_per_mol", ascending=False).reset_index(drop=True)

    out = {
        "contact_counts": dict(contact_counts),
        "contact_opps": dict(contact_opps),
        "contact_p": dict(contact_p),
        "contact_k": dict(contact_k),
        "contact_dG_rel": dict(contact_dG),
        "k_ref": k_ref,
        "beta": beta,
        "tau_ps": tau,
    }

    return G, df, out

# G_energy, df, results = infer_contact_flip_energies_from_graph(G, ints=ints, node_state_attr=G.nodes, Gm=Gm)
# df