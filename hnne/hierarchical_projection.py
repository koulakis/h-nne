from enum import Enum

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise
from pynndescent import NNDescent

from hnne.cool_functions import cool_mean, cool_max_radius
from hnne.point_spreading import norm_angles, norm_angles_3d


class PreliminaryEmbedding(str, Enum):
    pca = 'pca'
    pca_centroids = 'pca_centroids'
    random_linear = 'random_linear'


def project_with_pca_centroids(data, partitions, partition_sizes, dim=2, min_number_of_anchors=1000, verbose=False):
    pca = PCA(n_components=dim)
    large_partitions = np.where(np.array(partition_sizes) > min_number_of_anchors)[0]
    partition_idx = large_partitions.max() if any(large_partitions) else 0
    if verbose:
        print(f'Projecting on the {partition_idx}th partition with {partition_sizes[partition_idx]} anchors.')
    selected_anchors = cool_mean(data, partitions[:, partition_idx])
    pca.fit(selected_anchors)
    return pca.transform(data), pca


def project_with_pca(data, dim=2):
    pca = PCA(n_components=dim)
    transformed_data = pca.fit_transform(data)
    return transformed_data, pca


def project_points(
        data, 
        dim=2, 
        preliminary_embedding='pca',
        partition_sizes=None, 
        partitions=None,
        verbose=False
):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    pca = None

    if preliminary_embedding == PreliminaryEmbedding.pca:
        projected_points, pca = project_with_pca(data, dim=dim)
    elif len(data) < dim or preliminary_embedding == PreliminaryEmbedding.pca_centroids:
        projected_points, pca = project_with_pca_centroids(
            data,
            partitions,
            partition_sizes,
            dim=dim,
            verbose=verbose)
    elif preliminary_embedding == PreliminaryEmbedding.random_linear:
        random_components = np.random.random((data.shape[1], dim))
        projected_points = np.dot(data, random_components)
    else:
        raise ValueError(f'Invalid preliminary embedding: {preliminary_embedding}')
        
    # TODO: Handle the case where pca is not defined (e.g. random projection)
    # TODO: Add an option to perform full (randomized) PCA
    return projected_points, pca, scaler


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
        radius=.9,
        real_nn_threshold=30000,
        verbose=False
):
    if anchors.shape[0] <= real_nn_threshold:
        distance_matrix = pairwise.pairwise_distances(anchors, anchors, metric='euclidean')
        np.fill_diagonal(distance_matrix, 1e12)
        nearest_neighbor_idx = np.argmin(distance_matrix, axis=1).flatten()
    else:
        if verbose:
            print('Using ann to approximate 1-nns of the projected points...')
        knn_index = NNDescent(
            anchors, 
            n_neighbors=2, 
            metric='euclidean', 
            verbose=verbose)
        nns, _ = knn_index.neighbor_graph
        nearest_neighbor_idx = nns[:, 1]
        
    anchor_distances_from_nns = np.linalg.norm(anchors - anchors[nearest_neighbor_idx], axis=1, keepdims=True)
    anchor_radii = anchor_distances_from_nns * radius
    
    anchors_per_point = anchors[partition]
    anchor_radii_per_point = anchor_radii[partition]
    
    points_mean_per_partition = cool_mean(points, partition)
    points_centered = points - points_mean_per_partition[partition]

    anchors_max_radius = cool_max_radius(points_centered, partition)
    anchors_max_radius = np.where(anchors_max_radius == 0., 1., anchors_max_radius)
    points_max_radius = np.expand_dims(anchors_max_radius[partition], axis=1)

    return (
        anchors_per_point + anchor_radii_per_point * points_centered / points_max_radius,
        anchor_radii[:, 0],
        points_mean_per_partition,
        anchors_max_radius)


def multi_step_projection(
    data, 
    partitions,
    partition_labels,
    radius,
    ann_threshold,
    dim=2,
    partition_sizes=None,
    preliminary_embedding='pca',
    verbose=False
):
    projected_points, pca, scaler = project_points(
        data, 
        dim=dim, 
        preliminary_embedding=preliminary_embedding,
        partition_sizes=partition_sizes, 
        partitions=partitions,
        verbose=verbose
    )

    if verbose:
        print(partition_sizes)
    reversed_partition_range = list(reversed(range(partitions.shape[1])))
    projected_anchors = get_finch_anchors(projected_points, partitions=partitions)
    
    projected_anchors = [projected_points] + projected_anchors
    curr_anchors = projected_anchors[-1]
    
    anchor_radii = []
    moved_anchors = [curr_anchors]
    points_means = []
    points_max_radii = []
    inflation_params_list = []
    for cnt, i in enumerate(reversed_partition_range):
        if i == 0:
            partition_mapping = partitions[:, 0]
        else:
            partition_mapping = partition_labels[i - 1]
        
        current_points = projected_anchors[i]
        if dim == 2:
            thetas = np.linspace(0, np.pi/2, 6)
            current_points, inflation_params = norm_angles(
                current_points,
                thetas,
                partition_mapping)
            inflation_params_list.append(inflation_params)
        if dim == 3:
            alphas, beta, gammas = 3*[np.linspace(0, np.pi/2, 6)]
            current_points, inflation_params = norm_angles_3d(
                current_points,
                alphas,
                beta,
                gammas,
                partition_mapping)
            inflation_params_list.append(inflation_params)

        curr_anchors, radii, points_mean, points_max_radius = move_projected_points_to_anchors(
            current_points,
            curr_anchors, 
            partition_mapping,
            radius=radius,
            real_nn_threshold=ann_threshold,
            verbose=verbose
        )

        anchor_radii.append(radii)
        moved_anchors.append(curr_anchors)
        points_means.append(points_mean)
        points_max_radii.append(points_max_radius)

    return curr_anchors, anchor_radii, moved_anchors, pca, scaler, points_means, points_max_radii, inflation_params_list
