import pickle
from dataclasses import dataclass
from typing import Optional, List, Any

import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from pynndescent import NNDescent

from hnne.finch_clustering import FINCH
from hnne.hierarchical_projection import multi_step_projection


@dataclass
class ClusteringParameters:
    partitions: np.ndarray
    partition_sizes: np.ndarray
    partition_labels: np.ndarray
    lowest_level_centroids: np.ndarray
    knn_index: Optional[Any]


@dataclass
class ProjectionParameters:
    pca: Optional[PCA]
    scaler: StandardScaler
    projected_centroid_radii: List[np.ndarray]
    projected_centroids: List[np.ndarray]
    points_means: List[np.ndarray]
    points_max_radii: List[np.ndarray]
    inflation_params_list: List[Any]


class HNNE(BaseEstimator):
    def __init__(
            self,
            radius_shrinking=0.9,
            dim=2,
            real_nn_threshold=40000,
            projection_type='pca',
            metric='cosine'
    ):
        self.radius_shrinking = radius_shrinking
        self.dim = dim
        self.real_nn_threshold = real_nn_threshold
        self.projection_type = projection_type
        self.metric = metric
        self.min_size_top_level = 3
        self.clustering_parameters: Optional[ClusteringParameters] = None
        self.projection_parameters: Optional[ProjectionParameters] = None

    def fit_only_clustering(self, data, verbose=True):
        if verbose:
            print('Partitioning data with FINCH...')
        [
            partitions,
            partition_sizes,
            partition_labels,
            lowest_level_centroids,
            knn_index
        ] = FINCH(
            data,
            ensure_early_exit=False,
            verbose=verbose,
            distance=self.metric,
            ann_threshold=self.real_nn_threshold
        )

        large_enough_partitions = np.argwhere(np.array(partition_sizes) >= self.min_size_top_level)
        if len(large_enough_partitions) == 0:
            raise ValueError(
                f'The dataset has too few points resulting to a hierarchy with sizes {partition_sizes}. Please provide'
                f' a larger amount of data till there exists one partition of size {self.min_size_top_level}.')
        max_partition_idx = int(large_enough_partitions.max()) + 1
        if max_partition_idx < len(partition_sizes) and verbose:
            print(f'Removing {len(partition_sizes) - max_partition_idx} levels from the top to start with a level'
                  f'of size at least {self.min_size_top_level}.')
        partition_sizes = partition_sizes[:max_partition_idx]
        partitions = partitions[:, :max_partition_idx]
        partition_labels = partition_labels[:max_partition_idx]

        self.clustering_parameters = ClusteringParameters(
            partitions,
            partition_sizes,
            partition_labels,
            lowest_level_centroids,
            knn_index
        )

        return partitions, partition_sizes, partition_labels

    def fit(
            self,
            data,
            y=None,
            verbose=True,
            skip_clustering_if_done=True
    ):
        if self.clustering_parameters is not None and skip_clustering_if_done:
            cparams = self.clustering_parameters
            partitions, partition_sizes, partition_labels = \
                cparams.partitions, cparams.partition_sizes, cparams.partition_labels

        else:
            [
                partitions,
                partition_sizes,
                partition_labels
            ] = self.fit_only_clustering(data, verbose=verbose)

        if verbose:
            print(f'Projecting to {self.dim} dimensions...')
        [
            projection,
            projected_centroid_radii,
            projected_centroids,
            pca,
            scaler,
            points_means,
            points_max_radii,
            inflation_params_list
        ] = multi_step_projection(
            data, 
            partitions,
            partition_labels,
            radius_shrinking=self.radius_shrinking,
            dim=self.dim,
            real_nn_threshold=self.real_nn_threshold,
            partition_sizes=partition_sizes,
            projection_type=self.projection_type,
            verbose=verbose
        ) 

        self.projection_parameters = ProjectionParameters(
            pca=pca,
            scaler=scaler,
            projected_centroid_radii=projected_centroid_radii,
            projected_centroids=projected_centroids,
            points_means=points_means,
            points_max_radii=points_max_radii,
            inflation_params_list=inflation_params_list
        )
        
        return projection
        
    def transform(self, data, ann_point_combination_threshold=400e6, verbose=True):
        if self.clustering_parameters is None or self.projection_parameters is None:
            raise ValueError('Unable to project as h-nne has not been fitted on a dataset.')
        cparams = self.clustering_parameters
        pparams = self.projection_parameters

        if verbose:
            print('Finding nearest centroids to new data...')
        if len(cparams.lowest_level_centroids) * len(data) > ann_point_combination_threshold:
            if cparams.knn_index is None:
                knn_index = NNDescent(
                    cparams.lowest_level_centroids,
                    n_neighbors=2,
                    metric=self.metric,
                    verbose=verbose,
                    low_memory=True)
            else:
                knn_index = cparams.knn_index
            nearest_anchor_idxs = knn_index.query(data, k=1)[0].flatten()
        else:
            orig_dist = metrics.pairwise.pairwise_distances(data, cparams.lowest_level_centroids, metric=self.metric)
            nearest_anchor_idxs = np.argmin(orig_dist, axis=1)

        if verbose:
            print('Projecting data...')
        # Project the points with pca
        data = pparams.scaler.transform(data)
        data = pparams.pca.transform(data)

        # Apply inflation to points, if applicable
        if self.dim <= 3:
            for rot, norm1_params, norm2_params in pparams.inflation_params_list[-1]:
                m1, s1 = norm1_params
                m1, s1 = m1[nearest_anchor_idxs], s1[nearest_anchor_idxs]
                m2, s2 = norm2_params
                m2, s2 = m2[nearest_anchor_idxs], s2[nearest_anchor_idxs]
                data = (data - m1) / s1
                data = np.dot(data, rot)
                data = (data - m2) / s2
                data = np.dot(data, np.linalg.inv(rot))
                data_norms = np.linalg.norm(data, axis=-1)
                data = np.where(
                    np.expand_dims(data_norms > 1, axis=-1),
                    data / np.expand_dims(data_norms, axis=-1),
                    data)

        # Compute parameters related to the nearest anchors
        projected_nearest_anchors = pparams.projected_centroids[-2][nearest_anchor_idxs]
        max_radii = np.expand_dims(pparams.points_max_radii[-1][nearest_anchor_idxs], axis=-1)
        centroid_radii = np.expand_dims(pparams.projected_centroid_radii[-1][nearest_anchor_idxs], axis=-1)

        # Normalize relative to the maximum anchor group original point radius
        points_mean = pparams.points_means[-1][nearest_anchor_idxs]
        normalized_points = (data - points_mean) / max_radii

        # Scale based on the nearest anchor radii
        return projected_nearest_anchors + normalized_points * centroid_radii

    fit_transform = fit

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
