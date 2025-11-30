####################################################################################################################
# Code adapted with few changes from the FINCH clustering algorithm to generate the levels of the h-NNE hierarchy. #
# FINCH repository: https://github.com/ssarfraz/FINCH-Clustering                                                   #
# Original script: https://github.com/ssarfraz/FINCH-Clustering/blob/master/python/finch.py
####################################################################################################################
from typing import Optional

import cupy as cp
import cupyx.scipy.sparse as sp
from cupyx.scipy.sparse import csgraph
from cuml.internals import memory_utils
memory_utils.set_global_output_type('cupy')
from cuml.neighbors import NearestNeighbors
from hnne.cool_functions import cool_mean


def clust_rank(
    mat,
    initial_rank=None,
    metric="cosine",
):
    knn_index = None
    s = mat.shape[0]
    if initial_rank is not None:
        orig_dist = []
    else:
        # Always use kNN to avoid dense NxN
        nn = NearestNeighbors(
            n_neighbors=2,
            metric=metric,
            output_type='cupy',
        )
        nn.fit(mat)
        orig_dist, result = nn.kneighbors(mat)
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12

    initial_rank = cp.asarray(initial_rank)
    sparse_adjacency_matrix = sp.csr_matrix(
        (cp.ones_like(initial_rank, dtype=cp.float32), (cp.arange(0, s), initial_rank)),
        shape=(s, s),
    )

    return sparse_adjacency_matrix, orig_dist, initial_rank, knn_index


def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        # Filter edges with distance > min_sim without materializing dense matrix
        # Use adjacency indices to gather corresponding distances
        row_idx, col_idx = a.nonzero()
        edge_dists = orig_dist[row_idx, col_idx]
        mask = edge_dists <= min_sim
        # Build a masked sparse matrix with only allowed edges
        data = cp.ones_like(row_idx, dtype=cp.float32)[mask]
        row_kept = row_idx[mask]
        col_kept = col_idx[mask]
        a = sp.csr_matrix((data, (row_kept, col_kept)), shape=a.get_shape())

    num_clust, u = csgraph.connected_components(
        csgraph=a, directed=True, connection="weak", return_labels=True
    )
    return u, num_clust


def get_merge(c, u, data):
    if len(c) != 0:
        _, ig = cp.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u

    mat = cool_mean(data, c)

    return c, mat


def update_adj(adj, d):
    # Update adj, keep one merge at a time
    idx = adj.nonzero()
    v = cp.argsort(d[idx])
    v = v[:2]
    x = [idx[0][v[0]], idx[0][v[1]]]
    y = [idx[1][v[0]], idx[1][v[1]]]
    a = sp.lil_matrix(adj.get_shape())
    a[x, y] = 1
    return a


def req_numclust(c, data, req_clust, distance):
    iter_ = len(cp.unique(c)) - req_clust
    c_, mat = get_merge([], c, data)
    for i in range(iter_):
        adj, orig_dist, _, _ = clust_rank(mat, initial_rank=None, metric=distance)
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
    return c_


# noinspection PyPep8Naming
def FINCH(
    data: cp.ndarray,
    initial_rank: Optional[cp.ndarray] = None,
    distance: str = "cosine",
    ensure_early_exit: bool = True,
    verbose: bool = True,
):
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

        random_state: Optional[int] (default None)
            An optional random state for reproducibility purposes. It fixes the state of ANN.

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
        https://openaccess.thecvf.com/content_CVPR_2019/papers/Sarfraz_Efficient_Parameter-Free_Clustering_Using_First_Neighbor_Relations_CVPR_2019_paper.pdf
        Original code author:
            M. Saquib Sarfraz (saquib.sarfraz@kit.edu)
            Karlsruhe Institute of Technology (KIT)
    """
    data = data.astype(cp.float32)

    min_sim = None

    adj, orig_dist, first_neighbors, _ = clust_rank(
        data,
        initial_rank,
        distance,
    )
    initial_rank = None

    group, num_clust = get_clust(adj, [], min_sim)

    c, mat = get_merge([], group, data)
    lowest_level_centroids = mat

    if verbose:
        print("Level 0: {} clusters".format(num_clust))

    if ensure_early_exit:
        if orig_dist.shape[-1] > 2:
            min_sim = cp.max(orig_dist * adj.toarray())

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
        )

        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)

        partition_clustering.append(u)

        c_, mat = get_merge(c_, u, data)
        c = cp.column_stack((c, c_))

        num_clust.append(num_clust_curr)
        exit_clust = num_clust[-2] - num_clust_curr

        if num_clust_curr == 1 or exit_clust <= 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            break

        if verbose:
            print("Level {}: {} clusters".format(k, num_clust[k]))
        k += 1

    return c, num_clust, partition_clustering, lowest_level_centroids
