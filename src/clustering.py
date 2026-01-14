import numpy as np
from collections import defaultdict
import umap
import hdbscan
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS, SpectralEmbedding
import warnings


# Dimensinality Reduction and Clustering

def embed_flat_bits(flat_bits, pca_n=50, umap_n=10, umap_n_neighbours=15, umap_min_dist=0.1, use_umap=True, random_state=42):
    '''
    First reduces array of flattened triangle contact map bits with PCA then optionally
    applies UMAP to reduce dimensionality down to uma_n.
    '''
    X = np.asarray(flat_bits, dtype=float)
    U = X.shape[0]
    nc = min(pca_n, max(1, U-1))
    pca = PCA(n_components=nc, random_state=random_state)
    Xp = pca.fit_transform(X)
    if use_umap:
        reducer = umap.UMAP(n_components=min(umap_n, Xp.shape[1]), n_neighbors=umap_n_neighbours,
                            min_dist=umap_min_dist, random_state=random_state)
        Xc = reducer.fit_transform(Xp)
    else:
        Xc = Xp[:, :min(umap_n, Xp.shape[1])]
    return Xc, pca

def cluster_embedded(X, method='hdbscan', min_cluster_size=15, kmeans_k=500, random_state=42):
    '''
    Clusters unique contact maps upper triangle flat bit representations (typically
    post-embedding with PCA/UMAP) with HDBSCAN or kmeans.
    '''
    if method == 'hdbscan':
        c = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
        labels = c.fit_predict(X)
        return labels, c
    else:
        k = min(kmeans_k, max(2, X.shape[0]//2))
        mbk = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=1024)
        labels = mbk.fit_predict(X)
        return labels, mbk
    
def pairwise_hamming_matrix(ints):
    '''
    Computes hamming distance for all pairwise combinations of different possible
    contact maps in integer format (in bitstrings), returns square matrix (U,U).
    '''
    ints = list(map(int, ints))
    U = len(ints)
    if U == 0:
        return np.zeros((0, 0), dtype=int)
    D = np.empty((U, U), dtype=int)
    for i in range(U):
        vi = ints[i]
        # compute row i
        # use generator to keep inner loop C-level where possible
        row = np.fromiter(((vi ^ v).bit_count() for v in ints), dtype=int, count=U)
        D[i] = row
    return D

def compute_cluster_medoids_hamming(ints, labels):
    '''
    Given integer contact maps and their associated cluster labels this function
    determines the cluster medoid by finding that integer who's total Hamming distance
    to all other's in the cluster is the smallest.
    '''
    ints = np.asarray(list(map(int, ints)))
    labels = np.asarray(labels, dtype=int)
    U = len(ints)
    if U == 0:
        return {}

    medoids = {}
    # group members by label
    groups = defaultdict(list)
    for idx, lab in enumerate(labels):
        groups[int(lab)].append(idx)

    for lab, members in groups.items():
        members = np.asarray(members, dtype=int)
        if members.size == 1:
            medoids[int(lab)] = int(members[0])
            continue
        # compute pairwise hamming for members only (faster than full matrix when clusters small)
        m = members.size
        # build m x m distance matrix
        best_idx = members[0]
        best_cost = None
        for ii, i in enumerate(members):
            vi = ints[i]
            # compute distances to all members
            # fast inner loop using generator calling bit_count
            costs = np.fromiter(((vi ^ ints[j]).bit_count() for j in members), dtype=int, count=m)
            total = int(costs.sum())
            if best_cost is None or total < best_cost:
                best_cost = total
                best_idx = int(i)
        medoids[int(lab)] = best_idx
    return medoids

def assign_noise_to_nearest(labels, X):
    '''
    As HDBSCAN sets noisy points to -1 labels (don't belong to any clusters) this function
    simply assigns those noisy labelled points to the nearest real clusters. Optional 
    extra for plotting and node weights but loses potentially interesting outliers.
    '''
    mask = (labels == -1)
    if not mask.any():
        return labels
    non_noise = np.where(~mask)[0]
    if non_noise.size == 0:
        # fallback: KMeans into 2 clusters
        km = KMeans(n_clusters=2, random_state=0).fit(X)
        return km.labels_
    unique = np.unique(labels[non_noise])
    centroids = np.vstack([X[labels == u].mean(axis=0) for u in unique])
    nbr = NearestNeighbors(n_neighbors=1).fit(centroids)
    _, idx = nbr.kneighbors(X[mask])
    labels[mask] = unique[idx.ravel()]
    return labels

def embed_and_cluster_by_hamming(ints,
                                 n_components=10,
                                 embed_method='mds',   # 'mds' or 'spectral'
                                 cluster_method='agglomerative',  # 'agglomerative' or 'hdbscan'
                                 n_clusters=None,      # only for agglomerative
                                 random_state=42,
                                 warn_threshold=4000):
    ''' 
    Instead of embedding with PCA/UMAP we get a matrix of 
    hamming distances for every pair of integers in a list
    of integer representations for each frame in a trajectory then
    use MDS to embed this into 2D.
    '''
    ints = list(map(int, ints))
    U = len(ints)
    if U == 0:
        return np.zeros((0, n_components)), np.array([], dtype=int), np.zeros((0,0), dtype=int)

    if U > warn_threshold:
        warnings.warn(f"U={U} large: pairwise Hamming (U^2) will be expensive and memory-heavy.", UserWarning)

    # pairwise Hamming distances
    D = pairwise_hamming_matrix(ints) 

    # reduce n_components to feasible size (<= U-1)
    nc = min(n_components, max(1, U-1))
    if embed_method == 'mds':
        # classical MDS-like metric MDS on precomputed distances
        mds = MDS(n_components=nc, dissimilarity='precomputed', random_state=random_state)
        X_emb = mds.fit_transform(D.astype(float))
    elif embed_method == 'spectral': # option for spectral embedding instead
        # requires affinity: convert distances -> affinity via a radial basis function
        tri = D[np.triu_indices(U, k=1)]
        nz = tri[tri > 0]
        if nz.size:
            sigma = float(np.median(nz))
        else:
            sigma = 1.0
        # affinity = exp(-D**2 / (2*sigma^2))
        A = np.exp(-(D.astype(float)**2) / (2.0 * (sigma**2)))
        se = SpectralEmbedding(n_components=nc, affinity='precomputed', random_state=random_state)
        X_emb = se.fit_transform(A)
    else:
        raise ValueError("embed_method must be 'mds' or 'spectral'")

    # 3) clustering
    labels = None
    if cluster_method == 'agglomerative':
        if n_clusters is None:
            # sqrt(U) clusters if not provided
            n_clusters = max(2, int(np.sqrt(U)))
        # Agglomerative with precomputed distances
        agg = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
        labels = agg.fit_predict(D)
    elif cluster_method == 'hdbscan':
        # HDBSCAN with our precomputed matix
        cl = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=15)
        labels = cl.fit_predict(D)
    else:
        raise ValueError("cluster_method must be 'agglomerative' or 'hdbscan'")

    return X_emb, labels, D





# For larger proteins

# import networkx as nx 

# X_emb, labels, _ = embed_and_cluster_scalable(
#     ints,
#     n_components_pca=50,
#     min_cluster_size=500  # Adjust based on desired granularity
# )
# pos = {i: (float(X_emb[i, 0]), float(X_emb[i, 1])) for i in range(X_emb.shape[0])}

# G_clustered = build_cluster_graph_robust(G, labels)

# cluster_counts = Counter(labels)

# # --- Step B: Calculate Cluster Centroids (New Positions) ---
# # We compute the center of each cluster in the UMAP space
# cluster_pos = {}
# for lab in G_clustered.nodes():
#     # Find indices of all micro-states in this cluster
#     indices = [i for i, x in enumerate(labels) if x == lab]
#     # Average their UMAP positions
#     centroid = X_emb[indices].mean(axis=0)
#     cluster_pos[lab] = (centroid[0], centroid[1])

# cluster_trajectory = []
# for micro_node_id in frame_to_uid:
#     if micro_node_id < len(labels):
#         cluster_id = labels[micro_node_id]
#         if cluster_id != -1: # Skip noise frames
#             cluster_trajectory.append(cluster_id)
#         else:
#             # If noise, we just repeat the last valid cluster or skip
#             if cluster_trajectory: cluster_trajectory.append(cluster_trajectory[-1])
#             else: cluster_trajectory.append(0) 

# if -1 in cluster_counts: del cluster_counts[-1]
# folded_cluster = cluster_counts.most_common(1)[0][0]
# start_cluster = cluster_trajectory[0]

# shortest_cluster_path = nx.shortest_path(G_clustered, start_cluster, folded_cluster)

# custom_paths = {
#     "shortest_cluster_path": shortest_cluster_path
# }
# custom_paths_colors = [(0,0,200/255,1.0)]

# fig, ax = plotting.plot_graph_auto(
#     G=G_clustered,                 # The small 50-node graph
#     map_uid=cluster_trajectory,    # The trajectory converted to cluster IDs
#     pos=cluster_pos,               # Centroids of clusters
#     start_frame=0,
#     folded_node=folded_node,
#     post_fold_red=False,
#     node_size_range=(8, 48),     # Bigger nodes for clusters
#     title="lambda Repressor Clustered and MDS Embedded",
#     custom_paths=custom_paths,
#     custom_paths_colors=custom_paths_colors,
#     palette="viridis_r",
#     interactive=False,              # Static for publication
#     show_shortest=False,           # Disable first to ensure basic plotting works
#     expand_jumps=True              # Draws connections between temporal cluster jumps
# )

# plt.savefig("images/lambda_repressor_clustered_and_mds_embdedded.png")


# q_values = graph_analysis.compute_committor(G_clustered, start_cluster, folded_cluster, use_direct_solver=True)

# nx.set_node_attributes(G_clustered, q_values, 'committor')

# nodes = list(G_clustered.nodes())
# node_colors_q = [G_clustered.nodes[i].get('committor', 0.0) for i in range(len(nodes))]

# fig, ax = plotting.plot_graph_auto(
#     G=G_clustered,                 # The small 50-node graph
#     map_uid=cluster_trajectory,    # The trajectory converted to cluster IDs
#     pos=cluster_pos,               # Centroids of clusters
#     start_frame=0,
#     folded_node=folded_node,
#     post_fold_red=False,
#     node_size_range=(8, 48),     # Bigger nodes for clusters
#     title="lambda Repressor Clustered and MDS Embedded",
#     custom_paths=custom_paths,
#     palette="RdBu",
#     interactive=False,              # Static for publication
#     show_shortest=False,           # Disable first to ensure basic plotting works
#     expand_jumps=True,              # Draws connections between temporal cluster jumps
#     node_custom_color=node_colors_q,
#     node_custom_color_title="q",
# )

# committor_map = calculate_committor_cluster(G_clustered, start_cluster, folded_cluster)

# print(f"Committor calculated for {len(committor_map)} clusters.")
# print(f"Start q: {committor_map[start_cluster]}")
# print(f"Folded q: {committor_map[folded_cluster]}")

# from collections import Counter
# true_counts = Counter(labels)

# # 2. Inject these counts back into your Graph nodes
# # We also filter out nodes that might exist in the graph but have 0 population (ghosts)
# nodes_to_remove = []

# for node in G_clustered.nodes():
#     # Get count from the original data (default to 1 to avoid log(0) errors)
#     count = true_counts.get(node, 0)
    
#     if count == 0:
#         nodes_to_remove.append(node)
#     else:
#         # Update the graph attribute
#         G_clustered.nodes[node]['size'] = count
#         G_clustered.nodes[node]['population_fraction'] = count / len(labels)

# # Clean up ghost nodes
# G_clustered.remove_nodes_from(nodes_to_remove)

# print(f"Updated population data for {G_clustered.number_of_nodes()} clusters.")

# # 3. Check the spread
# sizes = [d['size'] for n, d in G_clustered.nodes(data=True)]

# total_frames = sum(sizes)
# free_energy_map = {}

# for i, n in enumerate(G_clustered.nodes):
#     pop = sizes[i]
#     # Avoid log(0)
#     if pop > 0:
#         free_energy_map[n] = -np.log(pop) 
#     else:
#         free_energy_map[n] = 10 # High energy for empty states

# # 2. Extract Data for Plotting
# x_q = []
# y_F = []

# for n in G_clustered.nodes():
#     x_q.append(committor_map.get(n, 0))
#     y_F.append(free_energy_map.get(n, 10))

# # 3. Plot
# plt.figure(figsize=(10, 6))
# sc = plt.scatter(x_q, y_F, s=sizes, c=x_q, cmap='RdBu_r', alpha=0.7, edgecolors='grey')
# plt.xlabel("Committor Probability (q)")
# plt.ylabel("Free Energy (-ln P)")
# plt.title("Projected Free Energy Landscape on Reaction Coordinates")
# plt.colorbar(sc, label="Reaction Progress")
# plt.grid(True, alpha=0.3)

# plt.savefig("images/lambda_repressor_reaction_coordinates_pseudo_f.png")
# plt.show()