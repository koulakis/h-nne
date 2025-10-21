import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pynndescent import NNDescent
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from hnne.finch.finch_clustering import FINCH
from hnne.v1.hierarchical_projection import (
    multi_step_projection as multi_step_projection_v1,
)
from hnne.v2.hierarchical_projection import PreliminaryEmbedding
from hnne.v2.hierarchical_projection import (
    multi_step_projection as multi_step_projection_v2,
)
from hnne.v2.v2_utils import HNNEVersion, normalize_hnne_version


@dataclass
class HierarchyParameters:
    partitions: np.ndarray
    requested_partition: np.ndarray
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
    v2_layout: Optional[Dict[Tuple[int, int], Tuple[float, ...]]]
    knn_index_transform: Optional[Any]


class HNNE(BaseEstimator):
    """Hierarchical 1-Nearest Neighbor graph based Embedding

     A fast hierarchical dimensionality reduction algorithm.

     Parameters
     ----------
     n_components: int (default 2)
         The dimension of the target space of the projection.

     metric: str (default 'cosine')
         The metric used to compute the distances when forming the h-nne hierarchy levels. Its value should be supported
         by both sklearn and pynndescent. Some possible values: 'cityblock', 'cosine', 'euclidean',
         'l1', 'l2', 'manhattan'.

     radius: float (default 0.45)
         The radius used to place points around centroids as a portion of the distance between nearest neighbor anchors.
         Though the theoretical value which guarantees no overlaps between anchor points is 0.2, 0.45 is a value which
         provides in practice denser visualizations with minimal loss in performance.

     ann_threshold: int (default 20000)
         A threshold above which approximate nearest neighbors will be computed instead of real nearest neighbors when
         building the levels of h-nne.

     preliminary_embedding: str (default 'pca')
         The preliminary embedding used to initiate h-nne. In terms of performance pca > pca_centroids > random_linear
         and in terms of speed performance pca < pca_centroids < random_linear.

     random_state: Optional[int] (default None)
         An optional random state for reproducibility purposes. It fixes the state of PCA and ANN.

     preferred_num_clust: Optional[int] (default None)
         preferred clusters view. set to number of ground truth classes or clusters in your data.

    hnne_version : Literal["v1","v2","version_1","version_2","1","2","auto"], optional (default "version_2")
         Selects which h-NNE pipeline variant to run.

         - **"v2"/"version_2"/"2"**: Use the packing-aware v2 pipeline. A small set of
           coarse FINCH levels is laid out with the fast hierarchical packing layer
           (2D fast path or ND fallback) and the resulting anchor radii seed the first
           descent step. Subsequent levels proceed with the standard h-NNE update.
         - **"v1"/"version_1"/"1"**: Legacy behavior. No packing layer; points are
           mapped into 1-NN–capped basins around anchors at each level.
         - **"auto"**: Currently resolves to v2 (recommended). Kept for forward
           compatibility if we later introduce adaptive selection.

         The value is normalized internally (case-insensitive); synonyms above all map
         to one of {"v1","v2"}. When "v1" is selected, any v2-specific knobs are ignored.

     start_cluster_view : {"auto", int, None} (default "auto")
         Controls where the v2 packing starts in the FINCH hierarchy (coarse→fine).

         - **"auto"/None**: Pick a start/end band automatically from dataset size *N*
           (coarse rule-of-thumb policy). This chooses a coarse start level and how
           many child levels to include before handing off to the classic descent.
         - **int**: Desired starting number of clusters; the nearest FINCH level is
           used. The number of child levels to include is still chosen automatically
           from *N*.

     v2_size_threshold : Optional[int] (default None)
         Upper bound on the cluster count for levels to be packed by v2. If `None`,
         the automatic policy (based on *N*) selects a reasonable end level. If set,
         the finest level used by v2 is the finest level whose cluster count is
         `<= v2_size_threshold`. Only used when `hnne_version` is v2/"auto".

     Attributes
     ----------
     min_size_top_level: int (default 3)
         The minimum number of centroids existing on the top level of the hierarchy. To achieve this minimum, the top
         levels which have fewer centroids are removed.

     faiss_threshold: int | None = 10_000_000
     faiss_use_gpu: bool = False
         For very large data size > faiss_threshold, FINCH use faiss for computing 1-nn

     hierarchy_parameters: Optional[HierarchyParameters]
         An object holding the parameters which encode the h-nne hierarchy. They are saved during fitting and can be
         reused both during projecting new points or projecting again with different parameters, e.g. n_components.

     References
     ----------
         The code implements the h-NNE algorithm described in our CVPR 2022 paper:
         [1] M. Saquib Sarfraz*, Marios Koulakis*, Constantin Seibold, Rainer Stiefelhagen.
         Hierarchical Nearest Neighbor Graph Embedding for Efficient Dimensionality Reduction. CVPR 2022.
         https://openaccess.thecvf.com/content/CVPR2022/papers/Sarfraz_Hierarchical_Nearest_Neighbor_Graph_Embedding_for_Efficient_Dimensionality_Reduction_CVPR_2022_paper.pdf

         Please contact the authors below for licensing information.
         Marios Koulakis (marios.koulakis@gmail.com)
         M. Saquib Sarfraz (saquibsarfraz@gmail.com)
    """

    def __init__(
        self,
        n_components: int = 2,
        metric: str = "cosine",
        radius: float = 0.4,
        ann_threshold: int = 20_000,
        preliminary_embedding: str = "pca",
        random_state: Optional[int] = None,
        preferred_num_clust: Optional[int] = None,
        hnne_version: HNNEVersion = "version_2",
        start_cluster_view: Union[str, int, None] = "auto",
        v2_size_threshold: Optional[int] = None,
        faiss_threshold: int = 10_000_000,
        faiss_use_gpu: bool = False,
    ):
        self.n_components = n_components
        self.radius = radius
        self.ann_threshold = ann_threshold
        self.random_state = random_state
        self.preferred_num_clust = preferred_num_clust
        self.hnne_version = normalize_hnne_version(hnne_version)
        self.v2_size_threshold = v2_size_threshold
        self.start_cluster_view = start_cluster_view
        try:
            preliminary_embedding = PreliminaryEmbedding[preliminary_embedding]
        except KeyError:
            raise ValueError(
                f"Invalid preliminary embedding: {preliminary_embedding}. "
                f'Please select one from: {", ".join(PreliminaryEmbedding)}.'
            )

        self.preliminary_embedding = preliminary_embedding
        self.metric = metric
        self.min_size_top_level: int = 3

        # faiss
        self.faiss_threshold = faiss_threshold
        self.faiss_use_gpu = faiss_use_gpu
        # --
        self.hierarchy_parameters: Optional[HierarchyParameters] = None
        self.projection_parameters: Optional[ProjectionParameters] = None

    def fit_only_hierarchy(self, X: np.ndarray, verbose: bool = False):
        if verbose:
            print("Building h-NNE hierarchy using FINCH...")
        [
            partitions,
            requested_partition,
            partition_sizes,
            partition_labels,
            lowest_level_centroids,
        ] = FINCH(
            data=X,
            req_clust=self.preferred_num_clust,
            ensure_early_exit=False,
            verbose=verbose,
            distance=self.metric,
            ann_threshold=self.ann_threshold,
            random_state=self.random_state,
            faiss_threshold=self.faiss_threshold,
            faiss_use_gpu=self.faiss_use_gpu,
        )

        large_enough_partitions = np.argwhere(
            np.array(partition_sizes) >= self.min_size_top_level
        )
        if len(large_enough_partitions) == 0:
            raise ValueError(
                f"The dataset has too few points resulting to a hierarchy with sizes {partition_sizes}. Please provide"
                f" a larger amount of data till there exists one partition of size {self.min_size_top_level}."
            )
        max_partition_idx = int(large_enough_partitions.max()) + 1
        if max_partition_idx < len(partition_sizes) and verbose:
            print(
                f"Removing {len(partition_sizes) - max_partition_idx} levels from the top to start with a level"
                f"of size at least {self.min_size_top_level}."
            )
        partition_sizes = partition_sizes[:max_partition_idx]
        partitions = partitions[:, :max_partition_idx]
        partition_labels = partition_labels[:max_partition_idx]

        self.hierarchy_parameters = HierarchyParameters(
            partitions,
            requested_partition,
            partition_sizes,
            partition_labels,
            lowest_level_centroids,
        )

        return partitions, requested_partition, partition_sizes, partition_labels

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        override_n_components: Optional[int] = None,
        verbose: bool = False,
        skip_hierarchy_building_if_done: bool = True,
    ):
        """
        Build an h-nne hierarchy based on X and use it to project X.

        Parameters
        ----------
        X: array, shape (n_samples, n_features)
            The data to project.

        y: array, shape (n_samples, )
            Ignored.

        override_n_components: Optional[int] (default None)
            Argument used to overwrite the original dimension of the target space of the projection.

        verbose: bool (default False)
            If true, plot info and progress messages.

        skip_hierarchy_building_if_done: bool (default True)
            If true, the h-nne hierarchy will be built only on the first run of fit. Warning: if you need to project
            a new dataset with the same HNNE object, then you have to set this to false.
        """
        if self.hierarchy_parameters is not None and skip_hierarchy_building_if_done:
            if verbose:
                print("Skipping the hierarchy construction as it is already available.")
            hparams = self.hierarchy_parameters
            partitions, requested_partition, partition_sizes, partition_labels = (
                hparams.partitions,
                hparams.requested_partition,
                hparams.partition_sizes,
                hparams.partition_labels,
            )

        else:
            [partitions, requested_partition, partition_sizes, partition_labels] = (
                self.fit_only_hierarchy(X, verbose=verbose)
            )

        if (override_n_components is not None) and (
            override_n_components != self.n_components
        ):
            if verbose:
                print(
                    f"Overwriting the dimensions {self.n_components} to the new value {override_n_components}."
                )
            self.n_components = override_n_components

        if verbose:
            print(f"Projecting to {self.n_components} dimensions...")
        if self.hnne_version == "v1":
            [
                projection,
                projected_centroid_radii,
                projected_centroids,
                pca,
                scaler,
                points_means,
                points_max_radii,
                inflation_params_list,
            ] = multi_step_projection_v1(
                data=X,
                partitions=partitions,
                partition_labels=partition_labels,
                radius=self.radius,
                ann_threshold=self.ann_threshold,
                dim=self.n_components,
                partition_sizes=partition_sizes,
                preliminary_embedding=self.preliminary_embedding,
                random_state=self.random_state,
                verbose=verbose,
            )
            v2_layout = None
        elif self.hnne_version == "v2":
            [
                projection,
                projected_centroid_radii,
                projected_centroids,
                pca,
                scaler,
                points_means,
                points_max_radii,
                inflation_params_list,
                v2_layout,
            ] = multi_step_projection_v2(
                data=X,
                partitions=partitions,
                partition_labels=partition_labels,
                radius=self.radius,
                ann_threshold=self.ann_threshold,
                dim=self.n_components,
                partition_sizes=partition_sizes,
                preliminary_embedding=self.preliminary_embedding,
                preferred_num_clust=self.preferred_num_clust,
                requested_partition=requested_partition,
                hnne_version=self.hnne_version,
                v2_size_threshold=self.v2_size_threshold,
                start_cluster_view=self.start_cluster_view,
                random_state=self.random_state,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown h-NNE version: {self.hnne_version}.")

        self.projection_parameters = ProjectionParameters(
            pca=pca,
            scaler=scaler,
            projected_centroid_radii=projected_centroid_radii,
            projected_centroids=projected_centroids,
            points_means=points_means,
            points_max_radii=points_max_radii,
            inflation_params_list=inflation_params_list,
            v2_layout=v2_layout,
            knn_index_transform=None,
        )

        return projection

    def transform(
        self,
        X: np.ndarray,
        ann_point_combination_threshold: int = 400e6,
        verbose: bool = False,
    ):
        NNS = 30
        MAX_NORM_ALLOWED = 3.0

        if self.hierarchy_parameters is None or self.projection_parameters is None:
            raise ValueError(
                "Unable to project as h-nne has not been fitted on a dataset."
            )
        hparams = self.hierarchy_parameters
        pparams = self.projection_parameters

        if verbose:
            print("Finding nearest centroids to new data...")
        if (
            len(hparams.lowest_level_centroids) * len(X)
            > ann_point_combination_threshold
        ):
            if pparams.knn_index_transform is None:
                if verbose:
                    print("Setting up once a knn index for the last level centroids...")
                knn_index = NNDescent(
                    hparams.lowest_level_centroids,
                    n_neighbors=NNS,
                    metric=self.metric,
                    verbose=verbose,
                )
                knn_index.prepare()
                pparams.knn_index_transform = knn_index
            else:
                knn_index = pparams.knn_index_transform
            nearest_anchor_idxs = knn_index.query(X, k=NNS)[0][:, 0]
        else:
            orig_dist = metrics.pairwise.pairwise_distances(
                X, hparams.lowest_level_centroids, metric=self.metric
            )
            nearest_anchor_idxs = np.argmin(orig_dist, axis=1)

        if verbose:
            print("Projecting data...")
        # Project the points with pca
        X = pparams.scaler.transform(X)
        X = pparams.pca.transform(X)

        # Apply inflation to points, if applicable
        if self.n_components <= 3:
            for rot, norm1_params, norm2_params in pparams.inflation_params_list[-1]:
                m1, s1 = norm1_params
                m1, s1 = m1[nearest_anchor_idxs], s1[nearest_anchor_idxs]
                m2, s2 = norm2_params
                m2, s2 = m2[nearest_anchor_idxs], s2[nearest_anchor_idxs]
                X = (X - m1) / s1
                X = np.dot(X, rot)
                X = (X - m2) / s2
                X = np.dot(X, np.linalg.inv(rot))
            # Some points might be placed on the wrong centroid and get pushed too far away after applying inflation.
            # To remedy this, we allow points only to inflate up to a number of stds of each centroid partition.
            data_norms = np.linalg.norm(X, axis=-1)
            X = np.where(
                np.expand_dims(data_norms > MAX_NORM_ALLOWED, axis=-1),
                MAX_NORM_ALLOWED * X / np.expand_dims(data_norms, axis=-1),
                X,
            )

        # Compute parameters related to the nearest anchors
        projected_nearest_anchors = pparams.projected_centroids[-2][nearest_anchor_idxs]
        max_radii = np.expand_dims(
            pparams.points_max_radii[-1][nearest_anchor_idxs], axis=-1
        )
        centroid_radii = np.expand_dims(
            pparams.projected_centroid_radii[-1][nearest_anchor_idxs], axis=-1
        )

        # Normalize relative to the maximum anchor group original point radius
        points_mean = pparams.points_means[-1][nearest_anchor_idxs]
        normalized_points = (X - points_mean) / max_radii

        # Scale based on the nearest anchor radii
        return projected_nearest_anchors + normalized_points * centroid_radii

    fit_transform = fit

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
