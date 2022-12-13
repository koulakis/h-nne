import numpy as np
import scipy.sparse as sp
from pynndescent import NNDescent
import torch


def fast_knns2spmat(nbrs: np.ndarray, dists: np.ndarray) -> sp.csr_matrix:
    n = len(nbrs)

    row, col = np.where(dists > 0)

    data = dists[row, col]
    col = nbrs[row, col]
    assert len(row) == len(col) == len(data)

    spmat = sp.csr_matrix((data, (row, col)), shape=(n, n))
    return spmat


def build_symmetric_adj(adj, self_loop=True):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    return adj


def row_normalize(mx):
    rowsum = np.array(mx.sum(1))
    rowsum[rowsum <= 0] = 1
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_indices_values(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    values = sparse_mx.data
    shape = np.array(sparse_mx.shape)
    return indices, values, shape


def indices_values_to_sparse_tensor(indices, values, shape):
    indices = torch.from_numpy(indices)
    values = torch.from_numpy(values)
    shape = torch.Size(shape)

    # noinspection PyUnresolvedReferences
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    indices, values, shape = sparse_mx_to_indices_values(sparse_mx)
    return indices_values_to_sparse_tensor(indices, values, shape)


def compute_knns(data, k, metric, verbose=False):
    descent = NNDescent(
        data,
        n_neighbors=k,
        metric=metric,
        verbose=verbose)
    return descent.neighbor_graph


def prepare_adj_matrix(knn_idx, knn_dist):
    adj = fast_knns2spmat(knn_idx, knn_dist)
    adj = build_symmetric_adj(adj, self_loop=True)
    adj = row_normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


def aggregate_nns(data, n_neighbors, iterations, metric='cosine'):
    knn_idx, knn_dist = compute_knns(data, n_neighbors, metric=metric, verbose=False)
    adj = prepare_adj_matrix(knn_idx, knn_dist).to(torch.float32)

    agg = data
    for _ in range(iterations):
        agg = torch.spmm(adj, torch.from_numpy(agg).to(torch.float32)).numpy()

    return agg
