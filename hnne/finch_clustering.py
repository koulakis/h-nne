####################################################################################################################
# Code adapted with few changes from the FINCH clustering algorithm to generate the levels of the h-NNE hierarchy. #
# FINCH repository: https://github.com/ssarfraz/FINCH-Clustering                                                   #
# Original script: https://github.com/ssarfraz/FINCH-Clustering/blob/master/python/finch.py                        #
####################################################################################################################

import numpy as np
from sklearn import metrics
import scipy.sparse as sp
from pynndescent import NNDescent

from hnne.cool_functions import cool_mean


def clust_rank(
        mat,
        initial_rank=None,
        metric='cosine',
        verbose=False,
        ann_threshold=40000):
    knn_index = None
    s = mat.shape[0]
    if initial_rank is not None:
        orig_dist = []
    elif s <= ann_threshold:
        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=metric)
        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)
    else:
        if verbose:
            print('Using PyNNDescent to compute 1st-neighbours at this step ...')
        knn_index = NNDescent(
            mat, 
            n_neighbors=2, 
            metric=metric,
            verbose=verbose)
        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        if verbose:
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
        adj, orig_dist, _, _ = clust_rank(mat, initial_rank=None, metric=distance)
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
        ann_threshold=40000):
    """FINCH clustering algorithm.

    Parameters
    ----------
        data: array, shape (n_samples, n_features)
            Input matrix with features in rows.

        initial_rank: array, shape (n_samples, 1) (optional)
            First integer neighbor indices.

        distance: str (default 'cosine')
            One of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] Recommended 'cosine'.

        ensure_early_exit: bool (default True)
            May help in large, high dim datasets, ensure purity of merges and helps early exit.

        verbose: bool (default True)
            Print verbose output.

        ann_threshold: int (default 40000)
            Data size threshold below which nearest neighbors are approximated with ANNs.

    Returns
    -------
        c: array of shape (n_samples, n_partitions)
            Matrix with labels indicating cluster participation. There is one column per partition.

        num_clust: array of shape (n_partitions)
            Number of clusters per partition.

        partition_clustering: list of arrays of shapes equal to the values of num_clust
            List of arrays with labels indicating the centroids cluster participation per level.

        lowest_level_centroids: array of shape (num_clust[0], n_features)
            The feature coordinates of the lowest level centroids.

    References
    ----------
        The code implements the FINCH algorithm described in our CVPR 2019 paper
        [1] Sarfraz et al. "Efficient Parameter-free Clustering Using First Neighbor Relations", CVPR2019
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
        ann_threshold=ann_threshold
    )
    initial_rank = None
    
    group, num_clust = get_clust(adj, [], min_sim)
    
    c, mat = get_merge([], group, data)
    lowest_level_centroids = mat
    
    if verbose:
        print('Level 0: {} clusters'.format(num_clust))

    if ensure_early_exit:
        if orig_dist.shape[-1] > 2:
            min_sim = np.max(orig_dist * adj.toarray())

    exit_clust = 2
    c_ = c
    k = 1
    num_clust = [num_clust]
    partition_clustering = []
    while exit_clust > 1:    
        adj, orig_dist, first_neighbors, knn_index = clust_rank(
            mat,
            initial_rank,
            distance,
            verbose=verbose,
            ann_threshold=ann_threshold
        )

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
            print('Level {}: {} clusters'.format(k, num_clust[k]))
        k += 1

    return c, num_clust, partition_clustering, lowest_level_centroids
