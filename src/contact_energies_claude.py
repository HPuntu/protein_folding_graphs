"""
Contact Energy Estimation from Graph Structure
===============================================

Estimates individual contact formation energies (ε_ij) from temporal edge lengths
in the contact-map graph. This provides data-driven energy parameters compared to
heuristic values used in models like WSME.

Usage:
------
from src.contact_energies import ContactEnergyEstimator

estimator = ContactEnergyEstimator(
    contact_maps=contact_maps,
    temporal_graph=temporal_graph,
    manifold_graph=manifold_graph,
    mds_coords=mds_embedding
)

energies = estimator.estimate_contact_energies()
estimator.plot_energy_comparison()
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import networkx as nx


class ContactEnergyEstimator:
    """Estimate individual contact formation energies from graph structure."""
    
    def __init__(
        self,
        contact_maps: np.ndarray,
        temporal_graph: nx.Graph,
        manifold_graph: nx.Graph,
        mds_coords: np.ndarray,
        temperature: float = 300.0,  # Kelvin
        n_residues: int = None
    ):
        """
        Parameters
        ----------
        contact_maps : np.ndarray, shape (n_frames, n_pairs)
            Binary contact maps for each trajectory frame
        temporal_graph : nx.Graph
            Graph with temporal edges (actual transitions)
        manifold_graph : nx.Graph
            Graph with manifold edges (single contact flips)
        mds_coords : np.ndarray, shape (n_nodes, n_dims)
            MDS embedding coordinates for graph nodes
        temperature : float
            Temperature in Kelvin for Boltzmann factor
        n_residues : int
            Number of residues in protein
        """
        self.contact_maps = contact_maps
        self.temporal_graph = temporal_graph
        self.manifold_graph = manifold_graph
        self.mds_coords = mds_coords
        self.temperature = temperature
        self.kB = 0.001987  # kcal/(mol·K) - Boltzmann constant
        
        if n_residues is None:
            # Infer from contact map size
            n_pairs = contact_maps.shape[1]
            n_residues = int((1 + np.sqrt(1 + 8 * n_pairs)) / 2)
        self.n_residues = n_residues
        
        # Create contact pair indices
        self.contact_pairs = []
        for i in range(n_residues):
            for j in range(i+1, n_residues):
                self.contact_pairs.append((i, j))
        
        self.contact_energies = {}
        self.cooperative_groups = []
        
    def compute_edge_distances(self) -> Dict[Tuple[int, int], float]:
        """Compute MDS distances for all edges in temporal graph."""
        edge_distances = {}
        
        for u, v in self.temporal_graph.edges():
            if u < len(self.mds_coords) and v < len(self.mds_coords):
                dist = np.linalg.norm(
                    self.mds_coords[u] - self.mds_coords[v]
                )
                edge_distances[(u, v)] = dist
                
        return edge_distances
    
    def identify_contact_flips(
        self,
        node_i: int,
        node_j: int
    ) -> List[Tuple[int, int]]:
        """
        Identify which contacts flip between two nodes.
        
        Returns
        -------
        flipped_contacts : List[Tuple[int, int]]
            List of (residue_i, residue_j) pairs that changed
        """
        if node_i >= len(self.contact_maps) or node_j >= len(self.contact_maps):
            return []
            
        cmap_i = self.contact_maps[node_i]
        cmap_j = self.contact_maps[node_j]
        
        diff = cmap_i != cmap_j
        flipped_idx = np.where(diff)[0]
        
        return [self.contact_pairs[idx] for idx in flipped_idx]
    
    def estimate_contact_energies(self) -> Dict[Tuple[int, int], Dict]:
        """
        Estimate energy for each contact formation.
        
        Key insight: Longer temporal edges (in MDS space) correspond to
        higher energy barriers because cooperative contact formation reduces
        the effective barrier.
        
        Returns
        -------
        energies : Dict[Tuple[int, int], Dict]
            For each contact pair (i,j), returns:
            - 'energy': estimated formation energy (kcal/mol)
            - 'n_observations': number of flip events observed
            - 'distances': list of MDS distances for this flip
            - 'cooperative_fraction': fraction involving multiple flips
        """
        edge_distances = self.compute_edge_distances()
        
        # Track flips for each contact
        contact_flip_data = defaultdict(lambda: {
            'distances': [],
            'n_contacts_flipped': [],
            'is_manifold_edge': []
        })
        
        # Analyze temporal edges
        for (u, v), dist in edge_distances.items():
            flipped = self.identify_contact_flips(u, v)
            
            if len(flipped) == 0:
                continue
                
            # Check if this is a manifold edge (single contact flip)
            is_manifold = (u, v) in self.manifold_graph.edges() or \
                         (v, u) in self.manifold_graph.edges()
            
            for contact in flipped:
                contact_flip_data[contact]['distances'].append(dist)
                contact_flip_data[contact]['n_contacts_flipped'].append(len(flipped))
                contact_flip_data[contact]['is_manifold_edge'].append(is_manifold)
        
        # Estimate energies
        for contact, data in contact_flip_data.items():
            if len(data['distances']) == 0:
                continue
                
            distances = np.array(data['distances'])
            n_flips = np.array(data['n_contacts_flipped'])
            
            # Energy proportional to mean distance
            # Normalize by number of contacts flipped (cooperative events)
            # Longer edges = harder transition = higher barrier
            mean_dist = np.mean(distances)
            
            # Cooperative events (multiple contacts) have lower per-contact cost
            cooperative_mask = n_flips > 1
            cooperative_frac = np.mean(cooperative_mask) if len(cooperative_mask) > 0 else 0
            
            # Estimate energy (arbitrary units based on distance)
            # This is a simple linear model - could be improved
            energy = mean_dist * self.kB * self.temperature
            
            # Correction for cooperative effects
            if cooperative_frac > 0.5:
                # If contact usually flips cooperatively, reduce energy
                energy *= (1 - 0.3 * cooperative_frac)
            
            self.contact_energies[contact] = {
                'energy': energy,
                'n_observations': len(distances),
                'mean_distance': mean_dist,
                'std_distance': np.std(distances),
                'cooperative_fraction': cooperative_frac,
                'distances': distances.tolist()
            }
        
        return self.contact_energies
    
    def identify_cooperative_groups(
        self,
        min_cooccurrence: int = 3
    ) -> List[List[Tuple[int, int]]]:
        """
        Identify groups of contacts that frequently flip together.
        
        Parameters
        ----------
        min_cooccurrence : int
            Minimum number of times contacts must flip together
            
        Returns
        -------
        groups : List[List[Tuple[int, int]]]
            Groups of contacts that form cooperatively
        """
        # Build cooccurrence matrix
        contact_list = list(self.contact_energies.keys())
        n_contacts = len(contact_list)
        cooccurrence = np.zeros((n_contacts, n_contacts))
        
        for u, v in self.temporal_graph.edges():
            flipped = self.identify_contact_flips(u, v)
            
            if len(flipped) <= 1:
                continue
                
            # All pairs of contacts that flipped together
            for i, c1 in enumerate(flipped):
                if c1 not in contact_list:
                    continue
                idx1 = contact_list.index(c1)
                
                for c2 in flipped[i+1:]:
                    if c2 not in contact_list:
                        continue
                    idx2 = contact_list.index(c2)
                    
                    cooccurrence[idx1, idx2] += 1
                    cooccurrence[idx2, idx1] += 1
        
        # Find groups using connected components
        # Threshold by minimum cooccurrence
        adj_matrix = cooccurrence >= min_cooccurrence
        
        G = nx.Graph()
        for i in range(n_contacts):
            for j in range(i+1, n_contacts):
                if adj_matrix[i, j]:
                    G.add_edge(contact_list[i], contact_list[j],
                              weight=cooccurrence[i, j])
        
        # Find connected components
        self.cooperative_groups = [
            list(component)
            for component in nx.connected_components(G)
            if len(component) > 1
        ]
        
        return self.cooperative_groups
    
    def compare_to_wsme(
        self,
        wsme_style: str = 'uniform'
    ) -> Dict[Tuple[int, int], float]:
        """
        Compare estimated energies to WSME-style heuristics.
        
        Parameters
        ----------
        wsme_style : str
            'uniform' : All ε_ij = -1
            'distance' : ε_ij proportional to sequence separation |i-j|
            
        Returns
        -------
        wsme_energies : Dict[Tuple[int, int], float]
        """
        wsme_energies = {}
        
        for (i, j) in self.contact_energies.keys():
            if wsme_style == 'uniform':
                wsme_energies[(i, j)] = -1.0
            elif wsme_style == 'distance':
                # Longer range contacts get more favorable energy
                wsme_energies[(i, j)] = -np.abs(j - i) / self.n_residues
        
        return wsme_energies
    
    def plot_energy_comparison(
        self,
        wsme_style: str = 'uniform',
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Visualize estimated contact energies vs WSME heuristics.
        """
        if not self.contact_energies:
            self.estimate_contact_energies()
        
        wsme = self.compare_to_wsme(wsme_style)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Contact energy matrix (heatmap)
        ax = axes[0, 0]
        energy_matrix = np.zeros((self.n_residues, self.n_residues))
        
        for (i, j), data in self.contact_energies.items():
            energy_matrix[i, j] = data['energy']
            energy_matrix[j, i] = data['energy']
        
        im = ax.imshow(energy_matrix, cmap='RdYlBu_r', aspect='auto')
        ax.set_xlabel('Residue j', fontsize=12)
        ax.set_ylabel('Residue i', fontsize=12)
        ax.set_title('Estimated Contact Formation Energies', fontsize=14)
        plt.colorbar(im, ax=ax, label='Energy (kcal/mol)')
        
        # Plot 2: Energy vs sequence separation
        ax = axes[0, 1]
        separations = []
        energies = []
        
        for (i, j), data in self.contact_energies.items():
            separations.append(abs(j - i))
            energies.append(data['energy'])
        
        ax.scatter(separations, energies, alpha=0.6, s=50)
        ax.set_xlabel('Sequence separation |j-i|', fontsize=12)
        ax.set_ylabel('Estimated energy (kcal/mol)', fontsize=12)
        ax.set_title('Energy vs Sequence Separation', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Cooperative fraction
        ax = axes[1, 0]
        coop_fracs = [data['cooperative_fraction'] 
                      for data in self.contact_energies.values()]
        energies_sorted = [data['energy'] 
                          for data in self.contact_energies.values()]
        
        scatter = ax.scatter(energies_sorted, coop_fracs, 
                           c=coop_fracs, cmap='viridis', 
                           alpha=0.6, s=50)
        ax.set_xlabel('Estimated energy (kcal/mol)', fontsize=12)
        ax.set_ylabel('Cooperative fraction', fontsize=12)
        ax.set_title('Cooperative Contact Formation', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Coop. fraction')
        
        # Plot 4: Our energies vs WSME
        ax = axes[1, 1]
        our_energies = []
        wsme_energies = []
        
        for contact in self.contact_energies.keys():
            our_energies.append(self.contact_energies[contact]['energy'])
            wsme_energies.append(wsme.get(contact, 0))
        
        ax.scatter(wsme_energies, our_energies, alpha=0.6, s=50)
        ax.plot([min(wsme_energies), max(wsme_energies)],
               [min(wsme_energies), max(wsme_energies)],
               'k--', alpha=0.5, label='y=x')
        ax.set_xlabel(f'WSME ({wsme_style}) energy', fontsize=12)
        ax.set_ylabel('Our estimated energy (kcal/mol)', fontsize=12)
        ax.set_title('Comparison to WSME Model', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_cooperative_groups(
        self,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """Visualize cooperative contact formation groups."""
        if not self.cooperative_groups:
            self.identify_cooperative_groups()
        
        if len(self.cooperative_groups) == 0:
            print("No cooperative groups found.")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw protein chain
        for i in range(self.n_residues - 1):
            ax.plot([i, i+1], [0, 0], 'k-', linewidth=2, alpha=0.3)
        
        # Draw contacts colored by group
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.cooperative_groups)))
        
        for group_idx, group in enumerate(self.cooperative_groups):
            for (i, j) in group:
                ax.plot([i, j], [0, 0], 'o-', 
                       color=colors[group_idx],
                       linewidth=3, markersize=8, alpha=0.7,
                       label=f'Group {group_idx+1}' if (i,j) == group[0] else None)
        
        ax.set_xlabel('Residue index', fontsize=14)
        ax.set_title('Cooperative Contact Formation Groups', fontsize=16)
        ax.set_ylim(-0.5, 0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of contact energies."""
        if not self.contact_energies:
            self.estimate_contact_energies()
        
        energies = [d['energy'] for d in self.contact_energies.values()]
        coop_fracs = [d['cooperative_fraction'] 
                     for d in self.contact_energies.values()]
        
        return {
            'n_contacts_observed': len(self.contact_energies),
            'mean_energy': np.mean(energies),
            'std_energy': np.std(energies),
            'min_energy': np.min(energies),
            'max_energy': np.max(energies),
            'mean_cooperative_fraction': np.mean(coop_fracs),
            'n_cooperative_groups': len(self.cooperative_groups),
            'highly_cooperative_contacts': sum(1 for f in coop_fracs if f > 0.7)
        }