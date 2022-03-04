import numpy as np
from sklearn import metrics
import scipy.sparse as sp
from pynndescent import NNDescent

from hnne.cool_functions import cool_mean


def clust_rank(
        mat,
        initial_rank=None,
        distance='cosine',
        verbose=False,
        low_memory_nndescent=False,
        ann_threshold=30000):
    knn_index = None
    s = mat.shape[0]
    if initial_rank is not None:
        orig_dist = []
    elif s <= ann_threshold:
        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)
        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)
    else:
        print('Using PyNNDescent to compute 1st-neighbours at this step ...')
        if low_memory_nndescent:
            print('Running on low memory...')
        knn_index = NNDescent(
            mat, 
            n_neighbors=2, 
            metric=distance, 
            verbose=verbose, 
            low_memory=low_memory_nndescent,
            n_trees=16 if low_memory_nndescent else None)
        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        print('Step PyNNDescent done ...')

    sparce_adjacency_matrix = sp.csr_matrix(
        (np.ones_like(initial_rank, dtype=np.float32),
         (np.arange(0, s), initial_rank)),
        shape=(s, s))
    
    return sparce_adjacency_matrix, orig_dist, initial_rank, knn_index


def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)
    return u, num_clust


def get_merge(c, u, data):
    if len(c) != 0:
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u
    
    mat = cool_mean(data, c)
    
    return c, mat


def update_adj(adj, d):
    # Update adj, keep one merge at a time
    idx = adj.nonzero()
    v = np.argsort(d[idx])
    v = v[:2]
    x = [idx[0][v[0]], idx[0][v[1]]]
    y = [idx[1][v[0]], idx[1][v[1]]]
    a = sp.lil_matrix(adj.get_shape())
    a[x, y] = 1
    return a


def req_numclust(c, data, req_clust, distance):
    iter_ = len(np.unique(c)) - req_clust
    c_, mat = get_merge([], c, data)
    for i in range(iter_):
        adj, orig_dist, _, _ = clust_rank(mat, initial_rank=None, distance=distance)
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
    return c_


# noinspection PyPep8Naming
def FINCH(
        data,
        initial_rank=None,
        distance='cosine',
        ensure_early_exit=True,
        verbose=True,
        low_memory_nndescent=False,
        ann_threshold=30000):
    """ FINCH clustering algorithm.

    Args:
        data: Input matrix with features in rows.
        initial_rank: Nx1 first integer neighbor indices (optional).
        distance: One of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] Recommended 'cosine'.
        ensure_early_exit: [Optional flag] may help in large, high dim datasets,
            ensure purity of merges and helps early exit
        verbose: Print verbose output.
        low_memory_nndescent: Reduce the number of trees used in NNDescent to lower memory requirements
        ann_threshold: data size threshold below which nearest neighbors are approximated with ANNs
    Returns:
        c: NxP matrix where P is the partition. Cluster label for every partition.
        num_clust: Number of clusters.
        partition_clustering:

    The code implements the FINCH algorithm described in our CVPR 2019 paper
        Sarfraz et al. "Efficient Parameter-free Clustering Using First Neighbor Relations", CVPR2019
         https://arxiv.org/abs/1902.11266
    For academic purpose only. The code or its re-implementation should not be used for commercial use.
    Please contact the author below for licensing information.
    Copyright
    M. Saquib Sarfraz (saquib.sarfraz@kit.edu)
    Karlsruhe Institute of Technology (KIT)
    """
    data = data.astype(np.float32)

    min_sim = None
    
    adj, orig_dist, first_neighbors, _ = clust_rank(
        data,
        initial_rank,
        distance,
        verbose=verbose,
        low_memory_nndescent=low_memory_nndescent,
        ann_threshold=ann_threshold
    )
    initial_rank = None
    
    group, num_clust = get_clust(adj, [], min_sim)
    
    c, mat = get_merge([], group, data)
    lowest_level_centroids = mat
    
    if verbose:
        print('Partition 0: {} clusters'.format(num_clust))

    if ensure_early_exit:
        if orig_dist.shape[-1] > 2:
            min_sim = np.max(orig_dist * adj.toarray())

    exit_clust = 2
    c_ = c
    k = 1
    num_clust = [num_clust]
    partition_clustering = []
    first_knn_index = None
    while exit_clust > 1:    
        adj, orig_dist, first_neighbors, knn_index = clust_rank(
            mat,
            initial_rank,
            distance,
            verbose=verbose,
            low_memory_nndescent=low_memory_nndescent,
            ann_threshold=ann_threshold
        )
        if first_knn_index is None:
            first_knn_index = knn_index
        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)

        partition_clustering.append(u)
        
        c_, mat = get_merge(c_, u, data)
        c = np.column_stack((c, c_))
        
        num_clust.append(num_clust_curr)
        exit_clust = num_clust[-2] - num_clust_curr

        if num_clust_curr == 1 or exit_clust <= 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            break

        if verbose:
            print('Partition {}: {} clusters'.format(k, num_clust[k]))
        k += 1

    return c, num_clust, partition_clustering, lowest_level_centroids, first_knn_index
