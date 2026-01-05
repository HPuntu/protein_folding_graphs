import numpy as np
import networkx as nx
from collections import Counter


# Graph builders

def temporal_edge_counts(map_uid, keep_self_loops=False):
    ''''''
    map_uid = np.asarray(map_uid, dtype=int)
    F = len(map_uid)
    if F < 2:
        return {}

    a = map_uid[:-1]
    b = map_uid[1:]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    if not keep_self_loops:
        mask = lo != hi
        lo = lo[mask]
        hi = hi[mask]
        idxs = np.nonzero(mask)[0]
    else:
        idxs = np.arange(F - 1, dtype=int)

    if lo.size == 0:
        return {}

    pairs = np.stack((lo, hi), axis=1)
    uniq_pairs, counts, first_idxs = np.unique(pairs, axis=0, return_counts=True, return_index=True)
    first_frames = idxs[first_idxs]

    edge_info = {}
    for (u, v), c, f in zip(uniq_pairs, counts, first_frames):
        edge_info[(int(u), int(v))] = {'count': int(c), 'first_frame': int(f)}
    return edge_info

def build_temporal_transition_graph(map_uid, unique_indices, keep_self_loops=False):
    '''
    Builds a graph where unique contact map nodes are connected if they are adjacent 
    (befor or after) in temporal sequence in the trajectory of contact maps for any of
    their instances in the trajectory.
    '''
    map_uid = np.asarray(map_uid, dtype=int)
    F = len(map_uid)
    if F == 0:
        return nx.Graph(), Counter(), np.array([], dtype=int)

    U = int(map_uid.max()) + 1
    node_counts = np.bincount(map_uid, minlength=U).astype(int)

    edge_info = temporal_edge_counts(map_uid, keep_self_loops=keep_self_loops)
    counter = Counter({k: v['count'] for k, v in edge_info.items()})

    G = nx.Graph()
    G.add_nodes_from(range(U))

    for (i, j), meta in edge_info.items():
        G.add_edge(i, j, weight=meta['count'], first_frame=meta['first_frame'])

    # node attributes
    for n in range(U):
        G.nodes[n]['frame_count'] = int(node_counts[n])
        G.nodes[n]['rep_frame'] = int(unique_indices[n]) if n < len(unique_indices) else -1

    return G, counter, node_counts

def build_graph_one_bit_neighbors(ints, Mbits):
    '''
    Builds a graph where nodes are neighbours if they differ by only one contact
    (Hamming distance of 1). Given ints as an array of contact maps in
    integer form.
    '''
    U = len(ints)
    int_to_uid = {int_val: uid for uid, int_val in enumerate(ints)} # dictionary mapping contact map integers to 0-indexed ids

    G = nx.Graph()
    G.add_nodes_from(range(U)) # one node per unique map
    masks = [1 << k for k in range(Mbits)] # all possible single bit flip masks covers the whole space of unique contact maps
    
    # build edges on graph by cycling through each contact map, flipping a bit and checking if neighbours exist from dict
    for uid, val in enumerate(ints):
        for m in masks:
            neigh_val = val ^ m # XOR flip
            other = int_to_uid.get(neigh_val)
            if other is not None and other != uid:
                G.add_edge(uid, other) # if neighbour exists add edge

    return G

def build_graph_pairwise(ints):
    '''
    Alternative to one_bit_neighbours approach above using pairwise comparison instead.
    Better for smaller U.
    '''
    U = len(ints)
    G = nx.Graph()
    G.add_nodes_from(range(U))

    # pairwise comparison cycle
    for i in range(U):
        vi = ints[i]
        for j in range(i+1, U):
            if (vi ^ ints[j]).bit_count() == 1:
                G.add_edge(i, j)
    return G

def build_contact_manifold_graph(ints, Mbits, method='auto'):
    '''  
    Wrap around of the two different perfomance-based computations of
    contact-space node neighborhoos.
    '''
    U = len(ints)

    if method == 'pairwise':
        G = build_graph_pairwise(ints)
    elif method == 'bitflip':
        G = build_graph_one_bit_neighbors(ints, Mbits)
    else:
        if U * Mbits < 5e7:
            G = build_graph_one_bit_neighbors(ints, Mbits)
        else:
            G = build_graph_pairwise(ints)

    return G, ints

def merge_manifold_and_temporal(Gm, temporal_edge_data):
    """
    Merge manifold graph Gm with temporal_edge_data:
      temporal_edge_data: dict keyed by (a,b) with {'count':int, 'first_frame':int}
    Returns a new Graph G with node attrs copied from Gm and edges annotated:
      - etype: 'manifold' / 'temporal' / 'both'
      - manifold: bool
      - temporal: bool
      - manifold_weight: original manifold edge weight (or None)
      - temp_count: temporal count (0 if absent)
      - first_frame: temporal first_frame (None if absent)
    """
    G = nx.Graph()
    # copy nodes (with attrs)
    G.add_nodes_from((n, dict(d)) for n, d in Gm.nodes(data=True))

    # start by copying manifold edges and their attrs
    for u, v, attr in Gm.edges(data=True):
        mw = attr.get('weight') if attr is not None else None
        G.add_edge(u, v,
                   etype='manifold',
                   manifold=True,
                   temporal=False,
                   manifold_weight=(int(mw) if mw is not None else None),
                   temp_count=0,
                   first_frame=None)

    # incorporate temporal edges (add or update)
    for (a, b), info in temporal_edge_data.items():
        a, b = int(a), int(b)
        cnt = int(info.get('count', 0))
        ff  = None if info.get('first_frame') is None else int(info.get('first_frame'))
        if G.has_edge(a, b):
            # update existing manifold edge to include temporal info
            G[a][b]['temporal'] = True
            G[a][b]['temp_count'] = cnt
            G[a][b]['first_frame'] = ff
            if G[a][b].get('manifold', False):
                G[a][b]['etype'] = 'both'
            else:
                G[a][b]['etype'] = 'temporal'
        else:
            # temporal-only edge
            G.add_edge(a, b,
                       etype='temporal',
                       manifold=False,
                       temporal=True,
                       manifold_weight=None,
                       temp_count=cnt,
                       first_frame=ff)
    return G