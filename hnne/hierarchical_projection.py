import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise
from pynndescent import NNDescent

from hnne.cool_functions import cool_mean, cool_max_radius
from hnne.point_spreading import atlas_decompression, norm_angles, norm_angles_3d


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
        partition_sizes=None, 
        partitions=None):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    pca = None
    partition_idx = None

    if len(data) < dim or projection_type == 'pca':
        projected_points, pca, partition_idx = project_with_pca(data, partitions, partition_sizes, dim=dim)
    elif projection_type == 'random_projection':
        random_components = np.random.random((data.shape[1], dim))
        projected_points = np.dot(data, random_components) 
    elif projection_type == 'random':
        projected_points = np.random.random(size=(data.shape[0], dim))
    else:
        raise ValueError(f'Invalid projection type: {projection_type}')
        
    # TODO: Return an appropriate model/function for each projection
    return projected_points, pca, partition_idx, scaler


def get_finch_anchors(projected_points, partitions=None):   
    all_projected_anchors = []    
    for i in range(partitions.shape[-1]):
        projected_anchors = cool_mean(projected_points, partitions[:, i])
        all_projected_anchors.append(projected_anchors)
    return all_projected_anchors


def move_projected_points_to_anchors(
        points, 
        anchors, 
        partition,
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

    anchors_max_radius = cool_max_radius(points_centered, partition)
    points_max_radius = np.expand_dims(anchors_max_radius[partition], axis=1)
    points_max_radius = np.where(points_max_radius == 0, 1., points_max_radius)
    
    return (
        anchors_per_point + anchor_radii_per_point * points_centered / points_max_radius,
        anchor_radii[:, 0],
        points_mean_per_partition,
        anchors_max_radius)


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
    partition_labels,
    inflate_pointclouds=True,
    radius_shrinking=0.66,
    dim=2,
    real_nn_threshold=40000,
    partition_sizes=None,
    projection_type='pca',
    project_first_partition_pca=False,
    decompression_level=2
):
    projected_points, pca, pca_partition_idx, scaler = project_points(
        data, 
        dim=dim, 
        projection_type=projection_type,
        partition_sizes=partition_sizes, 
        partitions=partitions)

    print(partition_sizes)
    reversed_partition_range = list(reversed(range(partitions.shape[1])))
    projected_anchors = get_finch_anchors(projected_points, partitions=partitions)    
    # TODO: Further develop this step to efficiently project all anchors by: projecting to 64 dims with PCA (if dim > 64) and then parallel projecting 
    # TODO: anchor clusters with PCA to 2 (or other) dimensions. If projection_dim > 64, then just project everything with the same PCA.
    if project_first_partition_pca:
        projected_points = project_with_pca_based_on_first_partition(data, partitions, partition_sizes, partition_idx=0, dim=dim)
    
    projected_anchors = [projected_points] + projected_anchors
    curr_anchors = projected_anchors[-1]
    
    if decompression_level > 0:
        curr_anchors = atlas_decompression(curr_anchors)
    
    anchor_radii = []
    moved_anchors = [curr_anchors]
    points_means = []
    points_max_radii = []
    for cnt, i in enumerate(reversed_partition_range):
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

        curr_anchors, radii, points_mean, points_max_radius = move_projected_points_to_anchors(
            current_points,
            curr_anchors, 
            partition_mapping,
            radius_shrinking=radius_shrinking,
            real_nn_threshold=real_nn_threshold,
        )
        if cnt <= decompression_level - 2:
            curr_anchors = atlas_decompression(curr_anchors)
        anchor_radii.append(radii)
        moved_anchors.append(curr_anchors)
        points_means.append(points_mean)
        points_max_radii.append(points_max_radius)

    return curr_anchors, anchor_radii, moved_anchors, pca, scaler, points_means, points_max_radii
