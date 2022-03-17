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
from hnne.hierarchical_projection import multi_step_projection, PreliminaryEmbedding


@dataclass
class HierarchyParameters:
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
    inflation_params_list: List[Any]
    knn_index_transform: Optional[Any]


class HNNE(BaseEstimator):
    """Hierarchical 1-Nearest Neighbor graph based Embedding

    A fast hierarchical dimensionality reduction algorithm.

    Parameters
    ----------
    dim: int (default 2)
        The dimension of the target space of the projection.

    metric: str (default 'cosine')
        The metric used to compute the distances when forming the h-nne hierarchy levels. Its value should be supported
        by both sklearn and pynndescent. Some possible values: 'cityblock', 'cosine', 'euclidean',
        'l1', 'l2', 'manhattan'.

    radius: float (default 0.45)
        The radius used to place points around centroids as a portion of the distance between nearest neighbor anchors.
        Though the theoretical value which guarantees no overlaps between anchor points is 0.2, 0.45 is a value which
        provides in practice denser visualizations with minimal loss in performance.

    ann_threshold: int (default 40000)
        A threshold above which approximate nearest neighbors will be computed instead of real nearest neighbors when
        building the levels of h-nne.

    preliminary_embedding: str (default 'pca')
        The preliminary embedding used to initiate h-nne. In terms of performance pca > pca_centroids > random_linear
        and in terms of speed performance pca < pca_centroids < random_linear.

    Attributes
    ----------
    min_size_top_level: int (default 3)
        The minimum number of centroids existing on the top level of the hierarchy. To achieve this minimum, the top
        levels which have fewer centroids are removed.

    hierarchy_parameters: Optional[HierarchyParameters]
        An object holding the parameters which encode the h-nne hierarchy. They are saved during fitting and can be
        reused both during projecting new points or projecting again with different parameters, e.g. dim.
    """
    def __init__(
            self,
            dim: int = 2,
            metric: str = 'cosine',
            radius: float = 0.4,
            ann_threshold: int = 40000,
            preliminary_embedding: str = 'pca'
    ):
        self.dim = dim
        self.radius = radius
        self.ann_threshold = ann_threshold
        try:
            preliminary_embedding = PreliminaryEmbedding[preliminary_embedding]
        except KeyError:
            raise ValueError(
                f'Invalid preliminary embedding: {preliminary_embedding}. '
                f'Please select one from: {", ".join(PreliminaryEmbedding)}.')
        self.preliminary_embedding = preliminary_embedding
        self.metric = metric
        self.min_size_top_level: int = 3
        self.hierarchy_parameters: Optional[HierarchyParameters] = None
        self.projection_parameters: Optional[ProjectionParameters] = None

    def fit_only_hierarchy(self, X: np.ndarray, verbose: bool = False):
        if verbose:
            print('Building h-NNE hierarchy using FINCH...')
        [
            partitions,
            partition_sizes,
            partition_labels,
            lowest_level_centroids
        ] = FINCH(
            data=X,
            ensure_early_exit=False,
            verbose=verbose,
            distance=self.metric,
            ann_threshold=self.ann_threshold
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

        self.hierarchy_parameters = HierarchyParameters(
            partitions,
            partition_sizes,
            partition_labels,
            lowest_level_centroids
        )

        return partitions, partition_sizes, partition_labels

    # noinspection PyUnusedLocal
    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray = None,
            dim: int = 2,
            verbose: bool = False,
            skip_hierarchy_building_if_done: bool = True
    ):
        """
        Build an h-nne hierarchy based on X and use it to project X.

        Parameters
        ----------
        X: array, shape (n_samples, n_features)
            The data to project.

        y: array, shape (n_samples, )
            Ignored.

        dim: int (default 2)
            Argument used to overwrite the original dimension of the target space of the projection.

        verbose: bool
            If true, plot info and progress messages.

        skip_hierarchy_building_if_done:
            If true, the h-nne hierarchy will be built only on the first run of fit. Warning: if you need to project
            a new dataset with the same HNNE object, then you have to set this to false.
        """
        if self.hierarchy_parameters is not None and skip_hierarchy_building_if_done:
            if verbose:
                print('Skipping the hierarchy construction as it is already available.')
            hparams = self.hierarchy_parameters
            partitions, partition_sizes, partition_labels = \
                hparams.partitions, hparams.partition_sizes, hparams.partition_labels

        else:
            [
                partitions,
                partition_sizes,
                partition_labels
            ] = self.fit_only_hierarchy(X, verbose=verbose)

        if dim is not None and dim != self.dim:
            if verbose:
                print(f'Overwriting the dimensions {self.dim} to the new value {dim}.')
            self.dim = dim

        if verbose:
            print(f'Projecting to {dim} dimensions...')
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
            data=X,
            partitions=partitions,
            partition_labels=partition_labels,
            radius=self.radius,
            ann_threshold=self.ann_threshold,
            dim=self.dim,
            partition_sizes=partition_sizes,
            preliminary_embedding=self.preliminary_embedding,
            verbose=verbose
        ) 

        self.projection_parameters = ProjectionParameters(
            pca=pca,
            scaler=scaler,
            projected_centroid_radii=projected_centroid_radii,
            projected_centroids=projected_centroids,
            points_means=points_means,
            points_max_radii=points_max_radii,
            inflation_params_list=inflation_params_list,
            knn_index_transform=None
        )
        
        return projection
        
    def transform(self, X: np.ndarray, ann_point_combination_threshold: int = 400e6, verbose: bool = False):
        if self.hierarchy_parameters is None or self.projection_parameters is None:
            raise ValueError('Unable to project as h-nne has not been fitted on a dataset.')
        hparams = self.hierarchy_parameters
        pparams = self.projection_parameters

        if verbose:
            print('Finding nearest centroids to new data...')
        if len(hparams.lowest_level_centroids) * len(X) > ann_point_combination_threshold:
            nns = 30
            if pparams.knn_index_transform is None:
                if verbose:
                    print('Setting up once a knn index for the last level centroids...')
                knn_index = NNDescent(
                    hparams.lowest_level_centroids,
                    n_neighbors=nns,
                    metric=self.metric,
                    verbose=verbose)
                knn_index.prepare()
                pparams.knn_index_transform = knn_index
            else:
                knn_index = pparams.knn_index_transform
            nearest_anchor_idxs = knn_index.query(X, k=nns)[0][:, 0]
        else:
            orig_dist = metrics.pairwise.pairwise_distances(X, hparams.lowest_level_centroids, metric=self.metric)
            nearest_anchor_idxs = np.argmin(orig_dist, axis=1)

        if verbose:
            print('Projecting data...')
        # Project the points with pca
        X = pparams.scaler.transform(X)
        X = pparams.pca.transform(X)

        # Apply inflation to points, if applicable
        if self.dim <= 3:
            for rot, norm1_params, norm2_params in pparams.inflation_params_list[-1]:
                m1, s1 = norm1_params
                m1, s1 = m1[nearest_anchor_idxs], s1[nearest_anchor_idxs]
                m2, s2 = norm2_params
                m2, s2 = m2[nearest_anchor_idxs], s2[nearest_anchor_idxs]
                X = (X - m1) / s1
                X = np.dot(X, rot)
                X = (X - m2) / s2
                X = np.dot(X, np.linalg.inv(rot))
                data_norms = np.linalg.norm(X, axis=-1)
                X = np.where(
                    np.expand_dims(data_norms > 1, axis=-1),
                    X / np.expand_dims(data_norms, axis=-1),
                    X)

        # Compute parameters related to the nearest anchors
        projected_nearest_anchors = pparams.projected_centroids[-2][nearest_anchor_idxs]
        max_radii = np.expand_dims(pparams.points_max_radii[-1][nearest_anchor_idxs], axis=-1)
        centroid_radii = np.expand_dims(pparams.projected_centroid_radii[-1][nearest_anchor_idxs], axis=-1)

        # Normalize relative to the maximum anchor group original point radius
        points_mean = pparams.points_means[-1][nearest_anchor_idxs]
        normalized_points = (X - points_mean) / max_radii

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
