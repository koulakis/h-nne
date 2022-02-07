import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise
from pynndescent import NNDescent
from numba import jit

from hnne.finch_clustering import cool_mean, FINCH


def project_with_adjacency_matrix(data, adjacency_matrix, dim=2):
    truncated_svd = TruncatedSVD(n_components=dim, n_iter=10, random_state=42)
    truncated_svd.fit(adjacency_matrix)
    
    projection = truncated_svd.components_
    w = np.dot(data.T, projection.T)
    proj = np.dot(data, w)
    
    return StandardScaler().fit_transform(proj)


def project_with_pca(data, partitions, partition_sizes, dim=2, min_number_of_anchors=1000):
    pca = PCA(n_components=dim)
    large_partitions = np.where(np.array(partition_sizes) > min_number_of_anchors)[0]
    partition_idx = large_partitions.max() if any(large_partitions) else 0
    print(f'Projecting on the {partition_idx}th partition with {partition_sizes[partition_idx]} anchors.')
    selected_anchors = cool_mean(data, partitions[:, partition_idx])
    pca.fit(selected_anchors)
    return pca.transform(data), pca, partition_idx


def project_points(
        data, 
        dim=2, 
        projection_type='pca', 
        adjacency_matrices=None, 
        partition_sizes=None, 
        partitions=None):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    if len(data) < dim or projection_type == 'pca':
        projected_points, pca, partition_idx = project_with_pca(data, partitions, partition_sizes, dim=dim)
    elif projection_type == 'adjacency':
        projected_points = project_with_adjacency_matrix(data, adjacency_matrices[0], dim=dim)
    elif projection_type == 'random_projection':
        random_components = np.random.random((data.shape[1], dim))
        projected_points = np.dot(data, random_components) 
    elif projection_type == 'random':
        projected_points = np.random.random(size=(data.shape[0], dim))
    else:
        raise ValueError(f'Invalid projection type: {projection_type}')
        
    # TODO: Return an appropriate model/function for each projection
    return projected_points, pca if projection_type == 'pca' else None, partition_idx, scaler


def get_finch_anchors(projected_points, partitions=None):   
    all_projected_anchors = []    
    for i in range(partitions.shape[-1]):
        projected_anchors = cool_mean(projected_points, partitions[:, i])
        all_projected_anchors.append(projected_anchors)
    return all_projected_anchors


@jit
def cool_max(M, u):
    lth = len(M)
    c = len(np.unique(u))
    result = np.zeros(c)
    for i in range(lth):
        idx = u[i]
        if result[idx] < M[i]:
            result[idx] = M[i]
    return result


def cool_max_radius(data, partition):
    data_lth = data.shape[0]
    norms = np.linalg.norm(data, axis=1)
    norm_maxes = cool_max(norms, partition)
    return norm_maxes[partition]


def cool_std(data, means, partition, epsilon=1e-12):
    return np.sqrt(cool_mean((data - means)**2, partition)) + epsilon


def cool_normalize(data, partition):
    means = cool_mean(data, partition)[partition]
    stds = cool_std(data, means, partition)[partition]
    
    return (data - means) / stds


def norm_angle(data, theta, partition):
    rot = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    
    data = cool_normalize(data, partition)
    
    rotated_data = np.dot(data, rot)
    rotated_data = cool_normalize(rotated_data, partition)

    return np.dot(rotated_data, np.linalg.inv(rot))


def norm_angles(data, angles, partition_mapping):
    for angle in angles:
        data = norm_angle(data, angle, partition_mapping)
    return data


def norm_angle_3d(data, alpha, beta, gamma, partition):
    r_x = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ])
    r_y = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    r_z = np.array([
        [1, 0, 0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma), np.cos(gamma)]
    ])
    
    rot = np.dot(r_x, np.dot(r_y, r_z))
    
    data = cool_normalize(data, partition)
    
    rotated_data = np.dot(data, rot)
    rotated_data = cool_normalize(rotated_data, partition)

    return np.dot(rotated_data, np.linalg.inv(rot))


def norm_angles_3d(data, alphas, betas, gammas, partition_mapping):
    for alpha, beta, gamma in zip(alphas, betas, gammas):
        data = norm_angle_3d(data, alpha, beta, gamma, partition_mapping)
    return data


def move_projected_points_to_anchors(
        points, 
        anchors, 
        partition,  
        first_neighbors_list,
        radius_shrinking=.66,
        real_nn_threshold=30000
    ):
    if anchors.shape[0] <= real_nn_threshold:
        distance_matrix = pairwise.pairwise_distances(anchors, anchors, metric='euclidean')
        np.fill_diagonal(distance_matrix, 1e12)
        nearest_neighbor_idx = np.argmin(distance_matrix, axis=1).flatten()
    else:
        print('Using ann to approximate 1-nns...')
        knn_index = NNDescent(
            anchors, 
            n_neighbors=2, 
            metric='euclidean', 
            verbose=True, 
            low_memory=True)
        nns, _ = knn_index.neighbor_graph
        nearest_neighbor_idx = nns[:, 1]
        
    anchor_radii = np.linalg.norm(anchors - anchors[nearest_neighbor_idx], axis=1, keepdims=True)
    anchor_radii = anchor_radii / 2 * radius_shrinking
    
    anchors_per_point = anchors[partition]
    anchor_radii_per_point = anchor_radii[partition]
    
    points_mean_per_partition = cool_mean(points, partition)
    points_mean = points_mean_per_partition[partition]
    points_centered = points - points_mean

    points_max_radius = np.expand_dims(cool_max_radius(points_centered, partition), axis=1)
    points_max_radius = np.where(points_max_radius == 0, 1., points_max_radius)
    
    return anchors_per_point + anchor_radii_per_point * points_centered / points_max_radius, anchor_radii[:, 0].tolist(), points_mean_per_partition, points_max_radius


def project_single_cluster_with_pca(data, dim=2):
    if data.shape[0] < dim or data.shape[1] < dim:
        return np.random.random((data.shape[0], dim))
    data_centered = data - data.mean(axis=0)
    pca = PCA(n_components=dim)
    return pca.fit_transform(data_centered)


def project_with_pca_based_on_first_partition(data, partitions, partition_sizes, partition_idx, dim=2):
    results = np.zeros((data.shape[0], dim))
    if data.shape[1] > 64:
        data, _, _ = project_with_pca(data, partitions, partition_sizes, dim=64)
    
    partition = partitions[:, partition_idx]
    for i in range(partition_sizes[partition_idx]):
        cluster_idx = partition == i
        results[cluster_idx] = project_single_cluster_with_pca(data[cluster_idx])
    
    return results


def multi_step_projection(
    data, 
    partitions, 
    adjacency_matrices, 
    partition_labels, 
    first_neighbors_list,
    inflate_pointclouds=True,
    radius_shrinking=0.66,
    dim=2,
    real_nn_threshold=40000,
    partition_sizes=None,
    projection_type='pca',
    remove_partitions_above_pca_partition=False,
    project_first_partition_pca=False,
    decompress_points=False
):
    projected_points, pca, pca_partition_idx, scaler = project_points(
        data, 
        dim=dim, 
        projection_type=projection_type,
        adjacency_matrices=adjacency_matrices,
        partition_sizes=partition_sizes, 
        partitions=partitions)
    
#     if remove_partitions_above_pca_partition:
#         if pca_partition_idx == 0:
#             pca_partition_idx += 1
#         pca_partition_idx += 1
#         partition_sizes = partition_sizes[:pca_partition_idx]
#         partitions = partitions[:, :pca_partition_idx]
#         adjacency_matrices = adjacency_matrices[:pca_partition_idx]
#         partition_labels = partition_labels[:pca_partition_idx]
#         first_neighbors_list = first_neighbors_list[:pca_partition_idx]
        
    print(partition_sizes)
    reversed_partition_range = list(reversed(range(partitions.shape[1])))
    projected_anchors = get_finch_anchors(projected_points, partitions=partitions)    
    # TODO: Further develop this step to efficiently project all anchors by: projecting to 64 dims with PCA (if dim > 64) and then parallel projecting 
    # TODO: anchor clusters with PCA to 2 (or other) dimensions. If projection_dim > 64, then just project everything with the same PCA.
    if project_first_partition_pca:
        projected_points = project_with_pca_based_on_first_partition(data, partitions, partition_sizes, partition_idx=0, dim=dim)
    
    projected_anchors = [projected_points] + projected_anchors
    curr_anchors = projected_anchors[-1]
    
#     if decompress_points:
# #         weights = pd.DataFrame(partitions[:, -1]).groupby(0).size().sort_index()
# #         weights = len(weights) * weights / weights.sum()
# #         weights = 1 / weights
#         weights = np.ones(len(curr_anchors))
#         curr_anchors = force_directed_graph_decompression(curr_anchors, weights=weights)
    
    anchor_radii = []
    shrinking_radii = []#[.9, .9, .9]
    cnt = 0
    moved_anchors = [curr_anchors]
    points_means = []
    points_max_radii = []
    for i in reversed_partition_range:
        if i == 0:
            partition_mapping = partitions[:, 0]
        else:
            partition_mapping = partition_labels[i - 1]
        
        current_points = projected_anchors[i]
        if inflate_pointclouds:
            if dim == 2:
                thetas = np.linspace(0, np.pi/2, 6)
                current_points = norm_angles(
                    current_points, 
                    thetas, 
                    partition_mapping)
            if dim == 3:
                alphas, beta, gammas = 3*[np.linspace(0, np.pi/2, 6)]
                current_points = norm_angles_3d(
                    current_points, 
                    alphas, 
                    beta, 
                    gammas,
                    partition_mapping)
        
        if cnt < len(shrinking_radii):
            curr_radius_shrinking = shrinking_radii[cnt]
        else:
            curr_radius_shrinking = radius_shrinking
        cnt+=1
        print(f'Shrinking with rate: {curr_radius_shrinking}')
        curr_anchors, radii, points_mean, points_max_radius = move_projected_points_to_anchors(
            current_points,
            curr_anchors, 
            partition_mapping, 
            first_neighbors_list[i],
            radius_shrinking=curr_radius_shrinking,
            real_nn_threshold=real_nn_threshold,
        )
        anchor_radii.append(radii)
        moved_anchors.append(curr_anchors)
        points_means.append(points_mean)
        points_max_radii.append(points_max_radius)

    return curr_anchors, anchor_radii, moved_anchors, pca, scaler, points_means, points_max_radii, projected_anchors


def full_projection(
        data, 
        distance='cosine',
        large_datasets=False,
        ann_threshold=100000,
        project_first_partition_pca=False,
        radius_shrinking=0.66,
        inflate_pointclouds=True,
        projection_type='pca',
        dim=2,
        remove_partitions_above_pca_partition=False,
        stop_at_partition=None,
        decompress_points=False
):
    print(f'Extracting FINCH partitions with {distance} distance...')
    [
        partitions, 
        partition_sizes, 
        req_c, 
        adjacency_matrices, 
        partition_labels, 
        cluster_dists,
        first_neighbors_list
    ] = FINCH( 
        data, 
        ensure_early_exit=True, 
        verbose=large_datasets,
        low_memory_nndescent=large_datasets,
        distance=distance,
        ann_threshold=ann_threshold
    )
            
    if partition_sizes[-1] < 3:
        partition_sizes = partition_sizes[:-1]
        partitions = partitions[:, :-1] 
        adjacency_matrices = adjacency_matrices[:-1]
        partition_labels = partition_labels[:-1]
        first_neighbors_list = first_neighbors_list[:-1]
        
    if stop_at_partition is not None:
        partition_sizes = partition_sizes[:stop_at_partition]
        partitions = partitions[:, :stop_at_partition] 
        adjacency_matrices = adjacency_matrices[:stop_at_partition]
        partition_labels = partition_labels[:stop_at_partition]
        first_neighbors_list = first_neighbors_list[:stop_at_partition]

    print(f'Projecting to {dim} dimensions...')
    projection, projected_centroid_radii, projected_centroids, _, _, _, _, _ = multi_step_projection(
        data, 
        partitions, 
        adjacency_matrices,
        partition_labels,
        first_neighbors_list,
        inflate_pointclouds=inflate_pointclouds,
        radius_shrinking=radius_shrinking,
        dim=dim,
        partition_sizes=partition_sizes,
        real_nn_threshold=ann_threshold,
        projection_type=projection_type,
        remove_partitions_above_pca_partition=remove_partitions_above_pca_partition,
        project_first_partition_pca=project_first_partition_pca,
        decompress_points=decompress_points
    )
    
    return projection, projected_centroid_radii, projected_centroids
