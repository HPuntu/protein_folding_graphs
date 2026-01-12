import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse
import scipy.sparse.linalg

# Code for the graph theoretical analysis of protein folding networks
# here we use a number of powerful tools from graph theory to 
# define the kinetics of protein folding, identify transition state
# ensembles and energy landscapes, the shortest path with 
# kinetic weighting and the folded state.

def pseudo_free_energy(node_counts, node_list):
    ''' 
    Calculate pseudo-Free Energy F for each node by boltzmann
    inversion
    '''
    # Calculate pseudo-Free Energy F for each node
    F = {}
    max_count = node_counts.max()
    for n in node_list:
        c = node_counts.get(n, 1)
        F[n] = -np.log(c / max_count) # F=0 for most populated state
    return F

def transition_probabilities(G, counts_matrix=None):
    '''
    Function to compute the transition probabilities between
    states (nodes) given counts (in this case the "temporal counts")
    we assigned to edges when building the manifold temporal graph.

    Optionally can provide independent count matrix instead
    of obtaining it from graph edge weights.
    '''

    if counts_matrix: # option to provide seperate matrix instead
        C = counts_matrix
    else:
        C = nx.to_numpy_array(G, weight='temp_count', nodelist=G.nodes())
    row_sums = C.sum(axis=1)

    # handle division by 0 cases
    row_sums[row_sums == 0] = 1 
    P = C / row_sums[:, np.newaxis] # probabilities

    return P

def thermodynamic_graph(G, P, frame_to_uid):
    ''' 
    Thermodynamic version of the contact state graph by taking
    the energy functional F ~ -ln(population) known
    as Boltzmann Inversion. We can approximate this from 
    the graph with node state counts.
    '''

    # Thermodynamic weight by boltzmann inversion
    node_counts = pd.Series(frame_to_uid).value_counts()
    node_list = list(G.nodes())

    F = pseudo_free_energy(node_counts, node_list)

    # Define Edge Weight based on Delta F
    # Weight = max(0, F_dest - F_source) (Uphill requires work, Downhill is free)
    W_therm = np.zeros_like(P)
    for i, u in enumerate(node_list):
        for j, v in enumerate(node_list):
            if G.has_edge(u, v):
                dF = F[v] - F[u]
                # Standard Metropolis-like barrier
                W_therm[i, j] = max(0, dF) + 0.1 # 0.1 is a base diffusion cost

    G_therm = G.copy()
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    for u, v in G.edges():
        i, j = node_idx[u], node_idx[v]
        
        # Thermodynamic Weight
        w_t = W_therm[i, j]
        G_therm[u][v]['weight'] = w_t

    return G_therm

def kinetic_graph(G, P):
    ''' 
    Build a kinetic verison of the contact state graph by 
    weighting manifold edges by transition probabilities.

    With this we can compute shortest paths whilst considering
    the kinetic landscape.
    '''

    # Define kinetic weighting as -ln(P_ij)
    P_safe = np.where(P > 0, P, 1e-10)
    W_kin = -np.log(P_safe)

    G_kin = G.copy()
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    for u, v in G.edges():
        i, j = node_idx[u], node_idx[v]
        
        # Kinetic Weight
        w_k = W_kin[i, j]
        # Check for infinite weight (disconnected in probability)
        if w_k > 100: w_k = 10000 
        G_kin[u][v]['weight'] = w_k

    return G_kin

def markov_state_model(G, T=300):
    '''
    
    '''
    nodes = list(G.nodes())
    n = len(nodes)
    
    # 1. Build Transition Matrix P
    # Weight by temporal counts (flux)
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight='temp_count', format='csr')
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1e-10
    
    # P = D^-1 * A
    D_inv = scipy.sparse.diags(1.0 / row_sums)
    P = D_inv @ A
    
    # 2. Compute Stationary Distribution (Eigenvector for lambda=1)
    # We look for the Left Eigenvector: u P = u  => P.T u.T = u.T
    print(f"Solving Eigenvalues for {n} microstates...")
    
    # 'LM' = Largest Magnitude. We want the largest real eigenvalue (which is 1.0)
    try:
        vals, vecs = scipy.sparse.linalg.eigs(P.T, k=3, which='LM')
    except:
        # Fallback for dense/small matrices if sparse solver gets fussy
        print("Switching to dense solver...")
        vals, vecs = np.linalg.eig(P.T.toarray())
        
    # Find the index of the eigenvalue closest to 1.0
    idx = np.argmin(np.abs(vals - 1.0))
    pi = np.real(vecs[:, idx])
    
    # Normalize pi so sum(pi) = 1
    pi = pi / np.sum(pi)
    pi = np.maximum(pi, 1e-10)
    
    # 3. Calculate Free Energy Landscape
    # F = -kT ln(pi)
    kB = 0.001987 # kcal/mol/K
    F = -kB * T * np.log(pi)
    
    # Shift so the global minimum is at 0 kcal/mol
    F = F - np.min(F)
    
    # 4. Identify States
    # The Folded Node is the Global Minimum of Free Energy (most stable)
    min_idx = np.argmin(F)
    folded_node_id = nodes[min_idx]
    
    # The Unfolded State is often the Start Node, or the most stable state 
    # in the 'other' basin (found via 2nd eigenvector), but for now 
    # let's assume Unfolded is the highest entropy state near the start.
    
    results = {
        'stationary_dist': {n: p for n, p in zip(nodes, pi)},
        'free_energy': {n: f for n, f in zip(nodes, F)},
        'folded_node': folded_node_id,
        'min_energy': F[min_idx]
    }
    
    return results


def physics_shortest_path(G_phys, start_node, folded_node, physics="kinetic"):
    ''' 
    Given a kinetic or thermodynamic based physics contact space
    graph obtain the shortest path.

    IF WE RESTRICT THE SHORTEST PATH CALCULATION HERE to real
    manifold edges only it cannot utilise temporal edges which
    teleport across multiple contact changes and so we define
    the shortest continuous path over contact spaces weighted
    by the respective physics - thus smoothing over MD noise!
    '''

    if physics is "kinetic": # Most Probable Path (Kinetic)
        try:
            path_mpp = nx.shortest_path(G_phys, source=start_node, target=folded_node, weight='weight')
            print(f"Most Probable Path (MPP): {len(path_mpp)} steps")
        except:
            print("MPP not found.")
            path_mpp = []
    
    elif physics is "thermodynamic": # Min Free Energy Barrier Path (Thermodynamic)
        try:
            path_mfep = nx.shortest_path(G_phys, source=start_node, target=folded_node, weight='weight')
            print(f"Min Free Energy Path: {len(path_mfep)} steps")
        except:
            print("MFEP not found.")
            path_mfep = []
            
def get_path_energy_profile(path, F):
    energies = [F.get(n, 0) for n in path]
    return energies

def build_physics_graph(G, physics_mode="kinetic", restrict_to_manifold=True, 
                        frame_to_uid=None, unique_maps=None):
    """
    Constructs a physics-weighted graph (Kinetic or Thermodynamic).
    
    Args:
        G: The base graph (nodes=microstates).
        physics_mode: 'kinetic', 'thermodynamic', or 'enthalpic' (WSME-like).
        restrict_to_manifold: If True, removes 'jump' edges that violate structural continuity.
        frame_to_uid: List of state IDs per frame (needed for population/thermo).
        unique_maps: (Optional) List of contact maps for 'enthalpic' mode.
    """
    
    # 1. Filter Edges (The "Manifold Constraint")
    if restrict_to_manifold:
        # We assume edges have an attribute 'edge_type' (set during graph construction).
        # If not, we assume all current edges are valid, or you can filter by weight/distance here.
        valid_edges = [
            (u, v) for u, v, d in G.edges(data=True) 
            if d.get('edge_type', 'manifold') == 'manifold' 
            or d.get('hamming', 0) <= 1 # Fallback: keep only close structural neighbors
        ]
        G_phys = G.edge_subgraph(valid_edges).copy()
        print(f"  - Constrained to Manifold: {G_phys.number_of_edges()} edges (removed jumps)")
    else:
        G_phys = G.copy()

    # 2. Get Transition Probabilities (Flux)
    # We use 'temp_count' (observed transitions).
    # CRITICAL: For unvisited manifold edges, we add a 'diffusion background'.
    
    # Initialize weights
    for u, v in G_phys.edges():
        raw_count = G_phys[u][v].get('temp_count', 0)
        # Add epsilon counts (diffusion) so unvisited structural edges are possible but expensive
        G_phys[u][v]['flux'] = raw_count + 1e-5 

    # Normalize to Probabilities P_ij
    for u in G_phys.nodes():
        total_flux = sum([G_phys[u][v]['flux'] for v in G_phys.neighbors(u)])
        for v in G_phys.neighbors(u):
            G_phys[u][v]['prob'] = G_phys[u][v]['flux'] / total_flux

    # 3. Apply Weighting Schemes
    
    # --- SCHEME A: KINETIC (Most Probable Path) ---
    if physics_mode == "kinetic":
        for u, v in G_phys.edges():
            p = G_phys[u][v]['prob']
            # Weight = -ln(P). Higher prob = Lower weight (Shortest path)
            G_phys[u][v]['weight'] = -np.log(p)

    # --- SCHEME B: THERMODYNAMIC (Free Energy / Boltzmann) ---
    elif physics_mode == "thermodynamic":
        # F = -ln(Population)
        node_counts = pd.Series(frame_to_uid).value_counts()
        max_count = node_counts.max()
        
        # Calculate F for every node
        F = {}
        for n in G_phys.nodes():
            c = node_counts.get(n, 1) # avoid log(0)
            F[n] = -np.log(c / max_count)
            
        # Edge Weight = Metropolis Barrier
        # If going u->v lowers energy (dF < 0), it's "free" (or diffusion cost).
        # If going u->v raises energy (dF > 0), cost is dF.
        for u, v in G_phys.edges():
            dF = F[v] - F[u]
            G_phys[u][v]['weight'] = max(0, dF) + 0.1 # 0.1 = base diffusion cost

    # --- SCHEME C: ENTHALPIC (WSME-like Structural Energy) ---
    elif physics_mode == "enthalpic":
        if unique_maps is None:
            raise ValueError("Need 'unique_maps' (contact maps) for Enthalpic mode.")
            
        print("  - Computing Contact Enthalpy (Q)...")
        # Calculate Q (Fraction of Native Contacts) for every node
        # We assume the 'Folded State' is the one with MOST contacts.
        num_contacts = [np.sum(m) for m in unique_maps]
        max_contacts = max(num_contacts)
        
        # Enthalpy H ~ -Q (More contacts = Lower Energy)
        # We normalize Q from 0 to 1.
        # H = (1 - Q) * Scale (so Native=0, Unfolded=High)
        H = {}
        nodes = list(G.nodes()) # Ensure alignment with unique_maps indices if they match
        for n in nodes:
            # Assuming node ID 'n' maps to index 'n' in unique_maps
            q = np.sum(unique_maps[n]) / max_contacts
            H[n] = 10.0 * (1.0 - q) # Scale factor 10 arbitrary
            
        # Edge Weight = Change in Enthalpy
        for u, v in G_phys.edges():
            dH = H[v] - H[u]
            G_phys[u][v]['weight'] = max(0, dH) + 0.1

    return G_phys

def enforce_detailed_balance(G, temp_k=300):
    """
    Enforces Detailed Balance while preventing invalid probabilities > 1.0
    which cause negative edge weights in Dijkstra.
    """
    kB = 0.001987
    beta = 1.0 / (kB * temp_k)
    G_db = G.copy()
    
    # 1. Boltzmann Inversion
    total_frames = sum(nx.get_node_attributes(G_db, 'frame_count').values())
    if total_frames == 0: raise ValueError("Nodes need 'frame_count'")

    pi_dict = {}
    for n in G_db.nodes():
        count = G_db.nodes[n].get('frame_count', 0)
        prob = (count + 1e-9) / total_frames
        pi_dict[n] = prob
        G_db.nodes[n]['F_boltzmann'] = -kB * temp_k * np.log(prob)

    # 2. Symmetrize Flux
    undirected_edges = set(tuple(sorted((u, v))) for u, v in G_db.edges())
    
    for u, v in undirected_edges:
        P_ij_raw = G_db.edges[u, v].get('prob', 0.0) if G_db.has_edge(u, v) else 0.0
        P_ji_raw = G_db.edges[v, u].get('prob', 0.0) if G_db.has_edge(v, u) else 0.0
        
        flux_ij = pi_dict[u] * P_ij_raw
        flux_ji = pi_dict[v] * P_ji_raw
        flux_eq = (flux_ij + flux_ji) / 2.0
        
        # Calculate & CLAMP Probabilities
        # We ensure P never exceeds 1.0
        if pi_dict[u] > 0:
            P_ij_new = min(1.0, flux_eq / pi_dict[u])
        else:
            P_ij_new = 0.0
            
        if pi_dict[v] > 0:
            P_ji_new = min(1.0, flux_eq / pi_dict[v])
        else:
            P_ji_new = 0.0
            
        # Update Edges with Positive Weights
        if P_ij_new > 0:
            if not G_db.has_edge(u, v): G_db.add_edge(u, v)
            G_db[u][v]['prob'] = P_ij_new
            
            # Safe Logarithm: -ln(1.0) = 0.0, -ln(small) = positive
            if P_ij_new > 1e-12:
                w = -np.log(P_ij_new)
                G_db[u][v]['weight'] = max(0.0, w) # Force non-negative
            else:
                G_db[u][v]['weight'] = 100.0 # High cost for unlikely paths
            
        if P_ji_new > 0:
            if not G_db.has_edge(v, u): G_db.add_edge(v, u)
            G_db[v][u]['prob'] = P_ji_new
            
            if P_ji_new > 1e-12:
                w = -np.log(P_ji_new)
                G_db[v][u]['weight'] = max(0.0, w)
            else:
                G_db[v][u]['weight'] = 100.0

    return G_db

def project_kinetics_onto_manifold(G_manifold, G_kinetic, default_penalty=10.0):
    """
    Creates a graph that has the STRICT topology of G_manifold,
    but inherits thermodynamic weights from G_kinetic where available.
    
    Args:
        G_manifold: The k-NN / geometric graph (defines valid structural steps).
        G_kinetic: The detailed-balance graph (defines probabilities).
        default_penalty: Weight for manifold edges that were never sampled 
                         (Kinetic 'dark matter').
    """
    # 1. Start with the structure of the Manifold
    G_projected = G_manifold.copy()
    
    # 2. Iterate over strict manifold edges
    count_inherited = 0
    count_fallback = 0
    
    for u, v in G_projected.edges():
        # Check if this structural link was observed kinetically
        if G_kinetic.has_edge(u, v):
            # INHERIT: Use the "Detailed Balance" weight (-ln P)
            # We trust the simulation here.
            w = G_kinetic[u][v].get('weight', default_penalty)
            G_projected[u][v]['weight'] = w
            G_projected[u][v]['source'] = 'kinetic'
            count_inherited += 1
        else:
            # FALLBACK: Structurally close, but never transitioned.
            # We must assign a weight derived from pure Thermodynamics (Delta F).
            
            # Get Boltzmann F if available, else 0
            F_u = G_kinetic.nodes[u].get('F_boltzmann', 0)
            F_v = G_kinetic.nodes[v].get('F_boltzmann', 0)
            dF = F_v - F_u
            
            # Metropolis-style weight for unsampled edges
            # If dF < 0 (Downhill), weight is small (diffusion limited)
            # If dF > 0 (Uphill), weight is penalty
            
            # Base diffusion cost + thermodynamic barrier
            base_cost = G_projected[u][v].get('weight', 1.0) # geometric distance
            thermo_term = max(0, dF) 
            
            # We add a 'penalty' because if it was easy, we probably would have seen it.
            # This is the "Unsampled Friction"
            w = base_cost + thermo_term + (default_penalty * 0.5)
            
            G_projected[u][v]['weight'] = w
            G_projected[u][v]['source'] = 'theoretical'
            count_fallback += 1

    print(f"Projection Complete.")
    print(f"Edges with Observed Kinetics: {count_inherited}")
    print(f"Edges with Theoretical Inference: {count_fallback}")
    
    return G_projected

# Now the shortest path on this physics graph is weighted by the physic
# and real and continuous on the contact manifold space
#path_kin = nx.shortest_path(G_kin, source=start_node, target=folded_node, weight='weight')
#path_therm = nx.shortest_path(G_therm, source=start_node, target=folded_node, weight='weight')


# These next functions exploit a property of graphs called
# committor probabilities. This arises from the connectivity
# of graphs being able to "see the future" and thus at node 
# transitions we can ask not what the probability of taking 
# particular edge is but what the probability is that taking 
# that edge will lead to a folded state quicker than unfolding.

def compute_committor(G, start_node, folded_node, use_direct_solver=True):
    ''' 
    Gives every node a folding process score.
    '''
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Validate inputs
    if start_node not in node_to_idx:
        raise ValueError(f"Start node {start_node} not found in graph.")
    if folded_node not in node_to_idx:
        raise ValueError(f"Folded node {folded_node} not found in graph.")

    # 1. Build Transition Matrix P
    # We use 'temp_count' as the flux weight.
    # Format 'csr' is efficient for matrix multiplication.
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight='temp_count', format='csr')

    # Row normalize to get P: P_ij = A_ij / sum_k(A_ik)
    row_sums = np.array(A.sum(axis=1)).flatten()
    
    # Handle unconnected nodes (dead ends with no outgoing flux) to prevent div/0
    row_sums[row_sums == 0] = 1e-10 
    
    # P = D^-1 * A
    D_inv = scipy.sparse.diags(1.0 / row_sums)
    P = D_inv @ A

    # 2. Build Laplacian-like Operator L = (I - P)
    # The equation for committor probability is L q = 0
    I = scipy.sparse.eye(n, format='csr')
    L = (I - P).tolil() # Convert to LIL format which is fast for row replacement
    b = np.zeros(n)

    # 3. Apply Dirichlet Boundary Conditions
    s_idx = node_to_idx[start_node]
    f_idx = node_to_idx[folded_node]

    # Boundary: q(start) = 0
    L[s_idx, :] = 0.0
    L[s_idx, s_idx] = 1.0
    b[s_idx] = 0.0

    # Boundary: q(folded) = 1
    L[f_idx, :] = 0.0
    L[f_idx, f_idx] = 1.0
    b[f_idx] = 1.0

    # 4. Solve Linear System
    L_csr = L.tocsr() # Convert back to CSR for the solver
    
    if use_direct_solver:
        print(f"Solving Exact Committor System for {n} nodes (Direct Solver)...")
        try:
            # spsolve performs a direct LU decomposition. 
            # It is non-iterative and exact (within machine precision).
            q = scipy.sparse.linalg.spsolve(L_csr, b)
            print("  -> Solver success (Exact).")
        except RuntimeError as e:
            print(f"  -> Solver Failed: {e}")
            print("     Hint: Graph might be disconnected or singular.")
            return {}
    else:
        print(f"Solving Iterative Committor System for {n} nodes (BiCGSTAB)...")
        # BiCGSTAB is an iterative method for non-symmetric sparse systems.
        q, info = scipy.sparse.linalg.bicgstab(L_csr, b, rtol=1e-10, atol=1e-10)
        
        if info == 0:
            print("  -> Solver success (Converged).")
        elif info > 0:
            print(f"  -> Warning: Solver did not converge to tolerance (Info: {info})")
        else:
            print(f"  -> Error: Solver breakdown/illegal input (Info: {info})")
            return {}

    # 5. Post-process
    # Clip numerical noise (e.g. -1e-15 becomes 0.0)
    q = np.clip(q, 0.0, 1.0)

    return {node: val for node, val in zip(nodes, q)}

#q_values = compute_committor(G, start_node, folded_node)
# ANALYZE TSE (Transition State Ensemble)
# TSE is defined as states with q approx 0.5 (0.4 to 0.6)
#tse_nodes = [n for n, q in q_values.items() if 0.4 <= q <= 0.6]

