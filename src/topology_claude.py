"""
Folding Funnel Characterization and Visualization
==================================================

Analyzes graph topology to characterize the folding funnel structure.
Computes metrics like betweenness centrality, branching factors, and
identifies bottleneck states.

Usage:
------
from src.folding_funnel import FoldingFunnelAnalyzer

analyzer = FoldingFunnelAnalyzer(
    graph=temporal_graph,
    committors=committor_values,
    populations=state_populations,
    folded_node=folded_state_idx
)

analyzer.compute_funnel_metrics()
analyzer.plot_funnel_structure()
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import networkx as nx
from collections import defaultdict
from scipy.stats import binned_statistic


class FoldingFunnelAnalyzer:
    """Analyze and visualize folding funnel topology from contact-map graph."""
    
    def __init__(
        self,
        graph: nx.Graph,
        committors: Optional[np.ndarray] = None,
        populations: Optional[np.ndarray] = None,
        free_energies: Optional[np.ndarray] = None,
        folded_node: Optional[int] = None,
        unfolded_node: Optional[int] = None,
        mds_coords: Optional[np.ndarray] = None
    ):
        """
        Parameters
        ----------
        graph : nx.Graph
            Contact-map graph (typically temporal graph)
        committors : np.ndarray, optional
            Committor probabilities for each node
        populations : np.ndarray, optional
            Population/visit frequency for each node
        free_energies : np.ndarray, optional
            Free energy for each node
        folded_node : int, optional
            Index of folded state node
        unfolded_node : int, optional
            Index of unfolded state node
        mds_coords : np.ndarray, optional
            MDS coordinates for visualization
        """
        self.graph = graph
        self.committors = committors
        self.populations = populations
        self.free_energies = free_energies
        self.folded_node = folded_node
        self.unfolded_node = unfolded_node
        self.mds_coords = mds_coords
        
        self.n_nodes = graph.number_of_nodes()
        self.metrics = {}
        
    def compute_betweenness_centrality(self) -> np.ndarray:
        """
        Compute betweenness centrality for all nodes.
        High betweenness = bottleneck state.
        
        Returns
        -------
        betweenness : np.ndarray
            Betweenness centrality for each node
        """
        betweenness_dict = nx.betweenness_centrality(
            self.graph,
            weight=None,
            normalized=True
        )
        
        betweenness = np.array([
            betweenness_dict.get(i, 0) for i in range(self.n_nodes)
        ])
        
        self.metrics['betweenness'] = betweenness
        return betweenness
    
    def compute_branching_factor_by_layer(
        self,
        n_bins: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute branching factor as function of reaction coordinate (committor).
        
        Funnel structure: high branching early (many paths), 
        low branching late (converging to folded state).
        
        Parameters
        ----------
        n_bins : int
            Number of bins for committor values
            
        Returns
        -------
        q_bins : np.ndarray
            Committor bin centers
        branching_factors : np.ndarray
            Mean out-degree in each bin
        branching_std : np.ndarray
            Std of out-degree in each bin
        """
        if self.committors is None:
            raise ValueError("Need committors to compute branching by layer")
        
        # Compute out-degree for each node
        out_degrees = np.array([
            self.graph.out_degree(i) if self.graph.is_directed() 
            else self.graph.degree(i)
            for i in range(self.n_nodes)
        ])
        
        # Bin by committor
        bins = np.linspace(0, 1, n_bins + 1)
        q_bins = (bins[:-1] + bins[1:]) / 2
        
        branching_factors = []
        branching_std = []
        
        for i in range(n_bins):
            mask = (self.committors >= bins[i]) & (self.committors < bins[i+1])
            if mask.sum() > 0:
                branching_factors.append(out_degrees[mask].mean())
                branching_std.append(out_degrees[mask].std())
            else:
                branching_factors.append(np.nan)
                branching_std.append(np.nan)
        
        branching_factors = np.array(branching_factors)
        branching_std = np.array(branching_std)
        
        self.metrics['branching_factors'] = branching_factors
        self.metrics['branching_std'] = branching_std
        self.metrics['q_bins'] = q_bins
        
        return q_bins, branching_factors, branching_std
    
    def compute_path_degeneracy(
        self,
        source: Optional[int] = None,
        target: Optional[int] = None,
        k_paths: int = 10
    ) -> Dict:
        """
        Compute degeneracy of folding pathways.
        How many distinct paths exist from unfolded to folded?
        
        Parameters
        ----------
        source : int, optional
            Source node (defaults to unfolded_node)
        target : int, optional
            Target node (defaults to folded_node)
        k_paths : int
            Number of shortest paths to find
            
        Returns
        -------
        path_info : Dict
            Information about pathway degeneracy
        """
        if source is None:
            source = self.unfolded_node
        if target is None:
            target = self.folded_node
            
        if source is None or target is None:
            print("Need source and target nodes")
            return {}
        
        try:
            # Find k shortest simple paths
            paths = list(nx.shortest_simple_paths(
                self.graph, source, target
            ))[:k_paths]
            
            path_lengths = [len(p) for p in paths]
            
            # Compute path overlap (shared nodes)
            path_overlap = np.zeros((len(paths), len(paths)))
            for i, p1 in enumerate(paths):
                for j, p2 in enumerate(paths):
                    if i != j:
                        overlap = len(set(p1) & set(p2))
                        path_overlap[i, j] = overlap / len(p1)
            
            path_info = {
                'n_paths_found': len(paths),
                'shortest_path_length': min(path_lengths),
                'mean_path_length': np.mean(path_lengths),
                'path_length_std': np.std(path_lengths),
                'mean_path_overlap': np.mean(path_overlap[np.triu_indices_from(path_overlap, k=1)]),
                'paths': paths
            }
            
            self.metrics['path_degeneracy'] = path_info
            return path_info
            
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            print(f"Could not find paths: {e}")
            return {}
    
    def identify_bottleneck_states(
        self,
        method: str = 'betweenness',
        top_k: int = 10
    ) -> np.ndarray:
        """
        Identify bottleneck states using various criteria.
        
        Parameters
        ----------
        method : str
            'betweenness' : High betweenness centrality
            'committor' : States near q=0.5
            'combined' : Weighted combination
        top_k : int
            Number of top bottleneck states to return
            
        Returns
        -------
        bottleneck_nodes : np.ndarray
            Indices of bottleneck states
        """
        if method == 'betweenness':
            if 'betweenness' not in self.metrics:
                self.compute_betweenness_centrality()
            scores = self.metrics['betweenness']
            
        elif method == 'committor':
            if self.committors is None:
                raise ValueError("Need committors for this method")
            # States near q=0.5 are bottlenecks
            scores = -np.abs(self.committors - 0.5)
            
        elif method == 'combined':
            if 'betweenness' not in self.metrics:
                self.compute_betweenness_centrality()
            if self.committors is None:
                raise ValueError("Need committors for combined method")
                
            # Normalize both metrics
            betweenness_norm = self.metrics['betweenness'] / self.metrics['betweenness'].max()
            committor_score_norm = (1 - np.abs(self.committors - 0.5) / 0.5)
            
            # Weighted combination
            scores = 0.6 * betweenness_norm + 0.4 * committor_score_norm
        
        # Get top k
        bottleneck_nodes = np.argsort(scores)[-top_k:][::-1]
        
        self.metrics[f'bottleneck_states_{method}'] = bottleneck_nodes
        return bottleneck_nodes
    
    def compute_funnel_metrics(self) -> Dict:
        """
        Compute all funnel characterization metrics.
        
        Returns
        -------
        metrics : Dict
            All computed metrics
        """
        print("Computing funnel metrics...")
        
        # Basic graph properties
        self.metrics['n_nodes'] = self.n_nodes
        self.metrics['n_edges'] = self.graph.number_of_edges()
        self.metrics['density'] = nx.density(self.graph)
        
        # Centrality measures
        print("  - Betweenness centrality...")
        self.compute_betweenness_centrality()
        
        # Clustering coefficient (local connectivity)
        print("  - Clustering coefficient...")
        clustering = nx.clustering(self.graph.to_undirected())
        self.metrics['clustering'] = np.array([
            clustering.get(i, 0) for i in range(self.n_nodes)
        ])
        
        # Branching factor analysis
        if self.committors is not None:
            print("  - Branching factors...")
            self.compute_branching_factor_by_layer()
        
        # Path analysis
        if self.folded_node is not None and self.unfolded_node is not None:
            print("  - Path degeneracy...")
            self.compute_path_degeneracy()
            
            # Graph diameter and average path length
            try:
                if nx.is_connected(self.graph.to_undirected()):
                    self.metrics['diameter'] = nx.diameter(self.graph.to_undirected())
                    self.metrics['avg_shortest_path'] = nx.average_shortest_path_length(
                        self.graph.to_undirected()
                    )
            except:
                pass
        
        # Bottleneck identification
        if self.committors is not None:
            print("  - Bottleneck states...")
            self.identify_bottleneck_states(method='combined')
        
        print("Done!")
        return self.metrics
    
    def plot_funnel_structure(
        self,
        figsize: Tuple[int, int] = (16, 12)
    ):
        """
        Comprehensive visualization of folding funnel structure.
        """
        if not self.metrics:
            self.compute_funnel_metrics()
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Funnel visualization (2D projection if MDS available)
        ax = fig.add_subplot(gs[0:2, 0:2])
        self._plot_funnel_2d(ax)
        
        # Plot 2: Branching factor vs committor
        ax = fig.add_subplot(gs[0, 2])
        self._plot_branching_factors(ax)
        
        # Plot 3: Betweenness distribution
        ax = fig.add_subplot(gs[1, 2])
        self._plot_betweenness_distribution(ax)
        
        # Plot 4: Free energy profile
        ax = fig.add_subplot(gs[2, 0])
        self._plot_free_energy_profile(ax)
        
        # Plot 5: Pathway degeneracy
        ax = fig.add_subplot(gs[2, 1])
        self._plot_pathway_degeneracy(ax)
        
        # Plot 6: Bottleneck states
        ax = fig.add_subplot(gs[2, 2])
        self._plot_bottleneck_importance(ax)
        
        plt.suptitle('Folding Funnel Characterization', fontsize=18, y=0.995)
        
        return fig
    
    def _plot_funnel_2d(self, ax):
        """Plot 2D funnel structure."""
        if self.mds_coords is None or self.committors is None:
            ax.text(0.5, 0.5, 'No MDS coordinates available',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Size by population
        if self.populations is not None:
            sizes = self.populations * 3000
        else:
            sizes = 50
        
        # Color by committor
        scatter = ax.scatter(
            self.mds_coords[:, 0],
            self.mds_coords[:, 1],
            c=self.committors,
            s=sizes,
            cmap='RdYlBu_r',
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Draw edges
        for u, v in self.graph.edges():
            if u < len(self.mds_coords) and v < len(self.mds_coords):
                ax.plot(
                    [self.mds_coords[u, 0], self.mds_coords[v, 0]],
                    [self.mds_coords[u, 1], self.mds_coords[v, 1]],
                    'gray', alpha=0.1, linewidth=0.5, zorder=0
                )
        
        # Highlight folded/unfolded
        if self.folded_node is not None and self.folded_node < len(self.mds_coords):
            ax.scatter(
                self.mds_coords[self.folded_node, 0],
                self.mds_coords[self.folded_node, 1],
                s=300, c='gold', marker='*',
                edgecolors='black', linewidths=2,
                label='Folded', zorder=10
            )
        
        if self.unfolded_node is not None and self.unfolded_node < len(self.mds_coords):
            ax.scatter(
                self.mds_coords[self.unfolded_node, 0],
                self.mds_coords[self.unfolded_node, 1],
                s=300, c='red', marker='s',
                edgecolors='black', linewidths=2,
                label='Unfolded', zorder=10
            )
        
        plt.colorbar(scatter, ax=ax, label='Committor q')
        ax.set_xlabel('MDS Dimension 1', fontsize=12)
        ax.set_ylabel('MDS Dimension 2', fontsize=12)
        ax.set_title('Folding Funnel Structure', fontsize=14)
        ax.legend()
    
    def _plot_branching_factors(self, ax):
        """Plot branching factor vs committor."""
        if 'branching_factors' not in self.metrics:
            ax.text(0.5, 0.5, 'No branching data',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        q_bins = self.metrics['q_bins']
        bf = self.metrics['branching_factors']
        bf_std = self.metrics['branching_std']
        
        valid = ~np.isnan(bf)
        
        ax.errorbar(
            q_bins[valid], bf[valid], yerr=bf_std[valid],
            fmt='o-', linewidth=2, markersize=6,
            capsize=3, alpha=0.7
        )
        
        ax.set_xlabel('Committor q', fontsize=11)
        ax.set_ylabel('Mean out-degree', fontsize=11)
        ax.set_title('Funnel Convergence', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add interpretation text
        ax.text(0.05, 0.95, 
               'High→Low = Funnel',
               transform=ax.transAxes,
               fontsize=9, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_betweenness_distribution(self, ax):
        """Plot distribution of betweenness centrality."""
        if 'betweenness' not in self.metrics:
            return
        
        betweenness = self.metrics['betweenness']
        
        ax.hist(betweenness, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Betweenness centrality', fontsize=11)
        ax.set_ylabel('Number of states', fontsize=11)
        ax.set_title('Bottleneck Distribution', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Mark high betweenness states
        threshold = np.percentile(betweenness, 95)
        ax.axvline(threshold, color='red', linestyle='--', 
                  linewidth=2, label=f'95th %ile')
        ax.legend()
    
    def _plot_free_energy_profile(self, ax):
        """Plot free energy vs committor."""
        if self.committors is None or self.free_energies is None:
            ax.text(0.5, 0.5, 'No free energy data',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Bin by committor
        bins = np.linspace(0, 1, 21)
        fe_mean, _, _ = binned_statistic(
            self.committors, self.free_energies,
            statistic='mean', bins=bins
        )
        fe_std, _, _ = binned_statistic(
            self.committors, self.free_energies,
            statistic='std', bins=bins
        )
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        valid = ~np.isnan(fe_mean)
        
        ax.errorbar(
            bin_centers[valid], fe_mean[valid], yerr=fe_std[valid],
            fmt='o-', linewidth=2, markersize=6,
            capsize=3, alpha=0.7, color='steelblue'
        )
        
        # Scatter individual states
        ax.scatter(self.committors, self.free_energies,
                  alpha=0.2, s=20, color='lightblue')
        
        ax.set_xlabel('Committor q', fontsize=11)
        ax.set_ylabel('Free energy (kT)', fontsize=11)
        ax.set_title('Free Energy Profile', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    def _plot_pathway_degeneracy(self, ax):
        """Visualize pathway degeneracy."""
        if 'path_degeneracy' not in self.metrics:
            ax.text(0.5, 0.5, 'No pathway data',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        path_info = self.metrics['path_degeneracy']
        
        if 'paths' in path_info:
            paths = path_info['paths']
            path_lengths = [len(p) for p in paths]
            
            ax.bar(range(len(path_lengths)), path_lengths,
                  alpha=0.7, edgecolor='black')
            ax.set_xlabel('Pathway rank', fontsize=11)
            ax.set_ylabel('Path length (steps)', fontsize=11)
            ax.set_title(f"{len(paths)} Shortest Paths", fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_bottleneck_importance(self, ax):
        """Plot importance of bottleneck states."""
        if 'betweenness' not in self.metrics or self.committors is None:
            ax.text(0.5, 0.5, 'No bottleneck data',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        betweenness = self.metrics['betweenness']
        committor_score = 1 - np.abs(self.committors - 0.5) / 0.5
        
        scatter = ax.scatter(
            committor_score, betweenness,
            c=self.committors,
            s=100, alpha=0.6,
            cmap='RdYlBu_r',
            edgecolors='black',
            linewidth=0.5
        )
        
        ax.set_xlabel('TSE score (1 - |q-0.5|/0.5)', fontsize=11)
        ax.set_ylabel('Betweenness centrality', fontsize=11)
        ax.set_title('Bottleneck Identification', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Highlight top bottlenecks
        if 'bottleneck_states_combined' in self.metrics:
            bottlenecks = self.metrics['bottleneck_states_combined'][:5]
            ax.scatter(
                committor_score[bottlenecks],
                betweenness[bottlenecks],
                s=200, facecolors='none',
                edgecolors='red', linewidths=2,
                label='Top 5 bottlenecks'
            )
            ax.legend()
    
    def print_summary(self):
        """Print summary of funnel analysis."""
        if not self.metrics:
            self.compute_funnel_metrics()
        
        print("\n" + "="*70)
        print("FOLDING FUNNEL ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\nGraph topology:")
        print(f"  Nodes:                    {self.metrics.get('n_nodes', 'N/A')}")
        print(f"  Edges:                    {self.metrics.get('n_edges', 'N/A')}")
        print(f"  Density:                  {self.metrics.get('density', 0):.4f}")
        
        if 'diameter' in self.metrics:
            print(f"  Diameter:                 {self.metrics['diameter']}")
            print(f"  Avg shortest path:        {self.metrics['avg_shortest_path']:.2f}")
        
        if 'betweenness' in self.metrics:
            print(f"\nBottleneck analysis:")
            print(f"  Max betweenness:          {self.metrics['betweenness'].max():.4f}")
            print(f"  Mean betweenness:         {self.metrics['betweenness'].mean():.4f}")
        
        if 'branching_factors' in self.metrics:
            bf = self.metrics['branching_factors']
            valid = ~np.isnan(bf)
            if valid.sum() > 0:
                print(f"\nFunnel structure:")
                print(f"  Early branching (q<0.3):  {bf[self.metrics['q_bins'] < 0.3][~np.isnan(bf[self.metrics['q_bins'] < 0.3])].mean():.2f}")
                print(f"  Late branching (q>0.7):   {bf[self.metrics['q_bins'] > 0.7][~np.isnan(bf[self.metrics['q_bins'] > 0.7])].mean():.2f}")
        
        if 'path_degeneracy' in self.metrics:
            pd = self.metrics['path_degeneracy']
            print(f"\nPathway degeneracy:")
            print(f"  Shortest path length:     {pd.get('shortest_path_length', 'N/A')}")
            print(f"  Mean path length:         {pd.get('mean_path_length', 0):.2f}")
            print(f"  Path overlap:             {pd.get('mean_path_overlap', 0):.2%}")
        
        print("="*70)