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
    """
    Computes Hamming distance matrix from ints, embed it, and clusters on that distance.
    Uses Multidimensional Scaling (MDS) by default to embed nodes by relative 
    Hamming distance, and agglomerative clustering. Slow for large U.
    """
    ints = list(map(int, ints))
    U = len(ints)
    if U == 0:
        return np.zeros((0, n_components)), np.array([], dtype=int), np.zeros((0,0), dtype=int)

    if U > warn_threshold:
        warnings.warn(f"U={U} large: pairwise Hamming (U^2) will be expensive and memory-heavy.", UserWarning)

    # 1) exact pairwise Hamming distances
    D = pairwise_hamming_matrix(ints)   # shape (U, U), dtype=int

    # 2) embedding
    # reduce n_components to feasible size (<= U-1)
    nc = min(n_components, max(1, U-1))
    if embed_method == 'mds':
        # classical MDS-like metric MDS on precomputed distances
        mds = MDS(n_components=nc, dissimilarity='precomputed', random_state=random_state)
        X_emb = mds.fit_transform(D.astype(float))
    elif embed_method == 'spectral':
        # spectral embedding requires affinity: convert distances -> affinity via RBF with sigma
        # choose sigma = median nonzero distance (safe heuristic)
        tri = D[np.triu_indices(U, k=1)]
        nz = tri[tri > 0]
        if nz.size:
            sigma = float(np.median(nz))
        else:
            sigma = 1.0
        # affinity = exp(-D**2 / (2*sigma^2)); use float
        A = np.exp(-(D.astype(float)**2) / (2.0 * (sigma**2)))
        se = SpectralEmbedding(n_components=nc, affinity='precomputed', random_state=random_state)
        X_emb = se.fit_transform(A)
    else:
        raise ValueError("embed_method must be 'mds' or 'spectral'")

    # 3) clustering
    labels = None
    if cluster_method == 'agglomerative':
        if n_clusters is None:
            # heuristic: choose sqrt(U) clusters if not provided
            n_clusters = max(2, int(np.sqrt(U)))
        # Agglomerative with precomputed distances (average linkage works with distances)
        agg = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
        labels = agg.fit_predict(D)
    elif cluster_method == 'hdbscan':
        # HDBSCAN supports precomputed distances by passing metric='precomputed'
        # It expects a condensed distance matrix for some versions; but passing the full matrix works.
        cl = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=15)
        labels = cl.fit_predict(D)
    else:
        raise ValueError("cluster_method must be 'agglomerative' or 'hdbscan'")

    return X_emb, labels, D

