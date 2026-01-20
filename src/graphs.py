import numpy as np
import networkx as nx
from collections import Counter


# Graph builders

def temporal_edge_counts(map_uid, keep_self_loops=False):
    '''
    Given a trajectory (list) of contact maps in their unique
    integer form get temporal adjacency counts for every 
    visited pair of map integers.
    '''
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
    uniq_pairs, first_idxs, counts = np.unique(pairs, axis=0, return_counts=True, return_index=True)
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

    map_uids is a list of unique contact map node ids
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

    # add as node attributes
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
    contact-space node neighborhoods.

    ints is a list of unique contact map integer representations for
    each frame in the trajectory.
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

def merge_manifold_and_temporal(Gm, temporal_edge_data, node_counts):
    '''
    Given a manifold graph (edges connect nodes with hamming
    distance 1 - one contact flip), and temporal adjacency
    counts for those nodes as edge data, merge into a 
    graph by taking the union of the two sets of edges
    and adding edge attributes of temporal counts for
    temporal edges.
    '''
    G = nx.Graph()
    # copy nodes (with attrs)
    G.add_nodes_from((n, dict(d)) for n, d in Gm.nodes(data=True))

    # We iterate through all nodes in the graph and assign the count
    for n in G.nodes():
        # Safety check: ensure n is within bounds of the counts array
        if n < len(node_counts):
            G.nodes[n]['frame_count'] = int(node_counts[n])
        else:
            G.nodes[n]['count'] = 0

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






### GRAPH TOPOLOGY

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
    