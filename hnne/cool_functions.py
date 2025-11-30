import cupy as cp
import cupyx.scipy.sparse as sp


def cool_mean(data, partition):
    """Efficiently calculate the mean of all rows of a matrix M over a partition u. The number of classes in the
    partition is implicitly defined from the values of u.

    Parameters
    ----------
        data: Matrix of dimensions (n, f) with n data points of f features each.

        partition: Partition of the data points in the form of an (n, ) array with k different integer values.

    Returns
    -------
        group_mean: A (k, f) matrix with the vectors averaged over the k partition values.
    """
    s = data.shape[0]
    un, nf = cp.unique(partition, return_counts=True)
    umat = sp.csr_matrix(
        (cp.ones(s, dtype="float32"), (cp.arange(0, s), partition)), shape=(s, len(un))
    )
    return (umat.T @ data) / nf[..., cp.newaxis]


def cool_max(arr, partition):
    """Efficiently calculate the max of all elements of an array arr of **positive** reals over a partition u."""

    # Use segment max via sorting, avoid sparse->dense blowup
    # Get order by partition ids
    order = cp.argsort(partition)
    part_sorted = partition[order]
    arr_sorted = arr[order]

    # Find boundaries where partition id changes
    is_start = cp.empty_like(part_sorted, dtype=cp.bool_)
    is_start[0] = True
    is_start[1:] = part_sorted[1:] != part_sorted[:-1]

    # Compute segmented max by scanning
    # Initialize output with -inf to handle empty segments robustly
    num_segments = int(cp.max(partition)) + 1
    out = cp.full((num_segments,), -cp.inf, dtype=arr.dtype)
    # Scatter max into buckets
    # We can use cp.maximum.at which performs unbuffered in-place segment max
    cp.maximum.at(out, part_sorted, arr_sorted)

    return out


def cool_max_radius(data, partition):
    """Efficiently calculate the maximum norm of the rows of a matrix data over a partition u. The number of
    classes in the partition is implicitly defined from the values of u.

    Parameters
    ----------
        data: Matrix of dimensions (n, f) with n data points of f features each.

        partition: Partition of the data points in the form of an (n, ) array with k different integer values.

    Returns
    -------
        group_max_radius: A (k, ) array with the maximum vector norms over the k partition values.
    """
    norms = cp.linalg.norm(data, axis=1)
    norm_maxes = cool_max(norms, partition)
    return norm_maxes


def cool_std(data, means, partition, epsilon=1e-12):
    """Efficiently calculate the standard deviation of all rows of a matrix data over a partition u. The means of
    each partition class are passed with the means matrix. The number of classes in the partition is implicitly
    defined from the values of u.

    Parameters
    ----------
        data: Matrix of dimensions (n, f) with n data points of f features each.
        means: Matrix of dimensions (n, f) with the means of the data per class. This implies that each mean is
            repeated over the vectors belonging to the same class.
        partition: Partition of the data points in the form of an (n, ) array with k different integer values.
        epsilon: Small constant to ensure that the standard deviation is not 0. This is specific to this codebase.

    Returns
    -------
        group_std: A (n, f) matrix with the standard deviation vectors over the k partition values. Rows belonging to
            the same partition class contain the same values.
    """
    return cp.sqrt(cool_mean((data - means) ** 2, partition)) + epsilon


def cool_normalize(data, partition):
    """Efficiently normalize the rows of a matrix data over a partition u. The number of classes in the partition is
    implicitly defined from the values of u.

    Parameters
    ----------
        data: Matrix of dimensions (n, f) with n data points of f features each.

        partition: Partition of the data points in the form of an (n, ) array with k different integer values.

    Returns
    -------
        group_normalized: A (n, f) matrix of the original data normalized over the k partition classes.
    """
    means = cool_mean(data, partition)
    means_per_partition = means[partition]
    stds = cool_std(data, means_per_partition, partition)

    return (data - means_per_partition) / stds[partition], [means, stds]
