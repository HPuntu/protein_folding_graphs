import numpy as np
import networkx as nx

def detect_transition_edges(G, pos, shortest_path_nodes, threshold_sigma=2.0):
    '''
    Identifies 'Transition Edges' defined as 1-bit steps that are 
    visually stretched significantly more than average.
    '''
    # 1. Collect statistics for ALL 1-bit edges in the path
    # (or you could sample the whole graph for a baseline)
    strains = []
    edges_info = []

    for i in range(len(shortest_path_nodes) - 1):
        u = shortest_path_nodes[i]
        v = shortest_path_nodes[i+1]
        
        # Physical Distance (Hamming) - implicitly 1 for manifold edges
        # If your graph has weights, check them. Assuming unweighted or weight=1.
        physical_dist = 1.0 
        
        # Visual Distance (Euclidean)
        pos_u = np.array(pos[u])
        pos_v = np.array(pos[v])
        visual_dist = np.linalg.norm(pos_u - pos_v)
        
        # Calculate Strain
        strain = visual_dist / physical_dist
        
        strains.append(strain)
        edges_info.append((u, v, strain))

    strains = np.array(strains)
    
    # 2. Define the Baseline (Average step size)
    mean_strain = np.mean(strains)
    std_strain = np.std(strains)
    
    # 3. Identify Outliers
    print(f"Average Edge Length: {mean_strain:.3f} +/- {std_strain:.3f}")
    print("-" * 40)
    
    transition_candidates = []
    
    for u, v, strain in edges_info:
        # Check if this edge is an anomaly
        z_score = (strain - mean_strain) / std_strain if std_strain > 0 else 0
        
        if z_score > threshold_sigma:
            print(f"*** TRANSITION DETECTED ***")
            print(f"Edge {u} -> {v}")
            print(f"Visual Length: {strain:.3f} (Z-score: {z_score:.1f})")
            transition_candidates.append((u, v))
        elif strain > mean_strain:
            # Just print high-ish ones
            print(f"Edge {u} -> {v}: {strain:.3f}")
            
    return transition_candidates

# --- Usage ---
# Assuming 'G' is your merged graph and 'pos' is your 2D layout
#path = nx.shortest_path(G, source=start_node, target=folded_node)
#transitions = detect_transition_edges(G, pos, path)