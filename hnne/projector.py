import pickle
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from sklearn.neighbors import NearestNeighbors
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


@dataclass
class ProjectionParameters:
    pca: Optional[PCA]
    scaler: StandardScaler
    projected_centroid_radii: List[np.ndarray]
    projected_centroids: List[np.ndarray]
    points_means: List[np.ndarray]
    points_max_radii: List[np.ndarray]


class HNNE(BaseEstimator):
    def __init__(
            self,
            inflate_pointclouds=True,
            radius_shrinking=0.66,
            dim=2,
            real_nn_threshold=20000,
            projection_type='pca',
            nn_distance='cosine',
            low_memory_nndescent=False,
            decompression_level=2,
            min_size_top_level=3
    ):
        self.inflate_pointclouds = inflate_pointclouds
        self.radius_shrinking = radius_shrinking
        self.dim = dim
        self.real_nn_threshold = real_nn_threshold
        self.projection_type = projection_type
        self.nn_distance = nn_distance
        self.low_memory_nndescent = low_memory_nndescent
        self.decompression_level = decompression_level
        self.min_size_top_level = min_size_top_level
        self.clustering_parameters: Optional[ClusteringParameters] = None
        self.projection_parameters: Optional[ProjectionParameters] = None

    def fit_only_clustering(self, data, verbose=True):
        if verbose:
            print('Partitioning data with FINCH...')
        [
            partitions,
            partition_sizes,
            partition_labels,
            lowest_level_centroids
        ] = FINCH(
            data,
            ensure_early_exit=False,
            verbose=verbose,
            low_memory_nndescent=self.low_memory_nndescent,
            distance=self.nn_distance,
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
            lowest_level_centroids
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
            points_max_radii
        ] = multi_step_projection(
            data, 
            partitions,
            partition_labels,
            inflate_pointclouds=self.inflate_pointclouds,
            radius_shrinking=self.radius_shrinking,
            dim=self.dim,
            real_nn_threshold=self.real_nn_threshold,
            partition_sizes=partition_sizes,
            projection_type=self.projection_type,
            decompression_level=self.decompression_level
        ) 

        self.projection_parameters = ProjectionParameters(
            pca=pca,
            scaler=scaler,
            projected_centroid_radii=projected_centroid_radii,
            projected_centroids=projected_centroids,
            points_means=points_means,
            points_max_radii=points_max_radii
        )
        
        return projection
        
    def transform(self, data):
        if self.clustering_parameters is None or self.projection_parameters is None:
            raise ValueError('Unable to project as h-nne has not been fitted on a dataset.')
        cparams = self.clustering_parameters
        pparams = self.projection_parameters

#         knn_index = NNDescent(
#             self.lowest_level_centroids,
#             n_neighbors=1,
#             metric='cosine',
#             verbose=True,
#             low_memory=True)
#         nearest_anchor_idxs = knn_index.query(data, k=1)[0].flatten()

        print('Creating tree...')
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(cparams.lowest_level_centroids)
        print('Finding nns...')
        _, nearest_anchor_idxs = nbrs.kneighbors(data)
        nearest_anchor_idxs = nearest_anchor_idxs.flatten()
        print('Projecting points')

        # Compute parameters related to the nearest anchors
        projected_nearest_anchors = pparams.projected_centroids[-2][nearest_anchor_idxs]
        max_radii = np.expand_dims(pparams.points_max_radii[-1][nearest_anchor_idxs], axis=-1)
        centroid_radii = np.expand_dims(pparams.projected_centroid_radii[-1][nearest_anchor_idxs], axis=-1)

        # Project the points with pca
        pca_projected_points = pparams.scaler.transform(data)
        pca_projected_points = pparams.pca.transform(pca_projected_points)

        # Normalize relative to the maximum anchor group original point radius
        points_mean = pparams.points_means[-1][nearest_anchor_idxs]
        normalized_points = (pca_projected_points - points_mean) / max_radii

        # Scale based on the nearest anchor radii
        return projected_nearest_anchors + normalized_points * centroid_radii

    def fit_transform(self, *args, **kwargs):
        return self.fit(*args, **kwargs)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
