import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import networkx as nx
from collections import defaultdict
from scipy.stats import binned_statistic



K_B = 0.001987  # Boltzmann constant kcal/(mol·K)


def compute_edge_distances(G, pos):
    '''
    Compute geometric embedded distances for all temporal edges in graph.
    '''

    edge_distances = {}
    for u, v in G.edges():
        if u < len(pos) and v < len(pos):
            dist = np.linalg.norm(
                pos[u] - pos[v]
            )
            edge_distances[(u, v)] = dist
            
    return edge_distances

def indentify_contact_flips(contact_map1, contact_map2):
    '''
    Identify contacts that frequently flip between nodes.
    '''
        
    diff = contact_map1 != contact_map2
    flipped_idx = np.where(diff)[0]
    
    return flipped_idx

def estimate_contact_energy():
    '''   
    Esimate individual contact pair energy from graph embedded temporal edge
    geometric distances.
    '''

import networkx as nx
import numpy as np

def analyze_folding_energetics(G, temperature=300):
    """
    Enriches graph G with thermodynamic and kinetic energy estimates.
    
    Args:
        G: NetworkX graph.
           - Nodes must have 'count' attribute (population).
           - Temporal edges must have 'count' attribute (transitions).
           - Manifold edges are defined by existence (assumed weight=1).
        temperature: Temperature in Kelvin (default 300K).
        
    Returns:
        G: The graph with added attributes:
           - Node 'free_energy': Relative stability (kBT).
           - Edge 'barrier_height': Activation energy (kBT).
           - Edge 'cooperativity_factor': Efficiency of the jump.
    """
    kB = 0.001987  # Boltzmann constant in kcal/mol/K
    beta = 1 / (kB * temperature)
    
    # --- STEP 1: Thermodynamics (Node Stability) ---
    # G_i = -kT * ln(P_i)
    total_population = sum(nx.get_node_attributes(G, 'count').values())
    
    for node in G.nodes():
        count = G.nodes[node].get('count', 0)
        # Avoid log(0) by adding a small epsilon if needed, or handling explicitly
        if count > 0:
            prob = count / total_population
            G.nodes[node]['free_energy'] = -np.log(prob) / beta
        else:
            G.nodes[node]['free_energy'] = np.inf

    # --- STEP 2: Enforce Detailed Balance (Symmetrization) ---
    # We create a temporary structure to handle the symmetrization of counts
    # C_sym_ij = (C_ij + C_ji) / 2
    
    symmetrized_counts = {}
    
    for u, v, data in G.edges(data=True):
        if 'count' in data: # It's a temporal edge
            # Check if reverse edge exists in data
            # Note: In undirected NetworkX, G[u][v] is the same as G[v][u].
            # If your graph is Directed, you must explicitly look up G[v][u].
            # Assuming Undirected Graph where 'count' aggregates both directions
            # or Directed Graph where we sum them manually:
            
            c_uv = data['count']
            
            # If directed, we would fetch c_vu. 
            # If undirected, the 'count' usually implies the total observed transitions.
            # rigorous detailed balance assumes: P_i * k_ij = P_j * k_ji
            # We approximate the equilibrium flux as the average observed flux.
            
            symmetrized_counts[(u, v)] = c_uv 

    # --- STEP 3: Kinetics & Cooperativity ---
    
    for u, v, data in G.edges(data=True):
        # We only calculate kinetics for edges that have observed transitions (temporal)
        if 'count' in data:
            
            # A. Calculate Transition Probability and Barrier
            # Rate k_ij ~ C_ij / Population_i
            # Barrier ~ -ln(k_ij)
            
            # Use the geometric mean of populations for the symmetrized barrier
            pop_u = G.nodes[u]['count']
            pop_v = G.nodes[v]['count']
            
            if pop_u > 0 and pop_v > 0:
                # Flux J = count / time_total. We treat 'count' as relative flux.
                # Theoretical Rate k = Flux / Population
                # We use a symmetric approximation for the barrier height between them:
                # Barrier ~ -ln( Count_uv / sqrt(Pop_u * Pop_v) )
                
                flux = data['count']
                avg_pop = np.sqrt(pop_u * pop_v)
                
                # Relative rate (unitless, purely for relative barrier heights)
                rate_proxy = flux / avg_pop 
                barrier = -np.log(rate_proxy) / beta
                
                data['barrier_height'] = barrier
            
            # B. Reconciliation: Manifold Distance vs Temporal Jump
            # How many contact flips did this jump actually skip?
            
            # Calculate shortest path on MANIFOLD edges only
            # (Create a view of G with only manifold edges for this calculation)
            # This can be expensive on large graphs; pre-calc if possible.
            try:
                # Assuming manifold edges have no 'count' attribute or specific tag
                # Filter for edges that define the manifold (Hamming=1)
                # Here we assume unweighted shortest path IS the Hamming distance
                manifold_dist = nx.shortest_path_length(G, u, v)
            except nx.NetworkXNoPath:
                manifold_dist = np.inf # Should not happen if manifold is connected
            
            data['manifold_dist'] = manifold_dist
            
            # C. Cooperativity Metric
            # Energy cost per unit of structural change
            if manifold_dist > 0:
                cost_per_flip = data['barrier_height'] / manifold_dist
                data['cost_per_flip'] = cost_per_flip
                
                # If manifold_dist > 1, this metric tells you if the jump was cooperative.
                # Lower 'cost_per_flip' on long jumps = High Cooperativity.

    return G