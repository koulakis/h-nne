from tqdm import tqdm
import numpy as np
import pickle
from pynndescent import NNDescent

from hnne.finch_clustering import cool_mean, FINCH
from hnne.hierarchical_projection import multi_step_projection


class HNNE:
    def __init__(
            self,
            inflate_pointclouds=False,
            radius_shrinking=0.66,
            dim=2,
            real_nn_threshold=20000,
            projection_type='pca',
            distance='cosine',
            low_memory_nndescent=False,
            decompression_level=0
    ):
        self.inflate_pointclouds = inflate_pointclouds
        self.radius_shrinking = radius_shrinking
        self.dim = dim
        self.real_nn_threshold = real_nn_threshold
        self.projection_type = projection_type
        self.distance = distance
        self.low_memory_nndescent = low_memory_nndescent
        self.decompression_level = decompression_level
        self.clustering_output = None

    def fit_only_clustering(self, data, verbose=True):
        if verbose:
            print('Partitioning data with FINCH...')
        [
            partitions,
            partition_sizes,
            partition_labels
        ] = FINCH(
            data,
            ensure_early_exit=False,
            verbose=verbose,
            low_memory_nndescent=self.low_memory_nndescent,
            distance=self.distance,
            ann_threshold=self.real_nn_threshold
        )

        self.clustering_output = partitions, partition_sizes, partition_labels

        return partitions, partition_sizes, partition_labels

    def fit(
            self,
            data,
            y=None,
            verbose=True,
            skip_clustering_if_done=True
    ):
        if self.clustering_output is not None and skip_clustering_if_done:
            partitions, partition_sizes, partition_labels = self.clustering_output
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
            projected_anchors
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
        
        self.pca = pca
        self.scaler = scaler
        self.lowest_level_centroids = cool_mean(data, partitions[:, 0])
        self.projected_centroid_radii = projected_centroid_radii
        self.projected_centroids = projected_centroids
        self.points_means = points_means
        self.points_max_radii = points_max_radii
        self.partitions = partitions
        self.projected_anchors = projected_anchors
        
        return projection, partitions
        
    def transform(self, data):
        projections = []
#         knn_index = NNDescent(
#             self.lowest_level_centroids, 
#             n_neighbors=1, 
#             metric='cosine', 
#             verbose=True, 
#             low_memory=True)
#         nearest_anchor_idxs = knn_index.query(data, k=1)[0].flatten()

        from sklearn.neighbors import NearestNeighbors
        print('Creating tree...')
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.lowest_level_centroids)
        print('Finding nns...')
        _, nearest_anchor_idxs = nbrs.kneighbors(data)
        nearest_anchor_idxs = nearest_anchor_idxs.flatten()

#         from sklearn import metrics
#         nn_dists = metrics.pairwise.pairwise_distances(data, self.lowest_level_centroids, metric='cosine')
#         nearest_anchor_idxs = np.argmin(nn_dists, axis=1).flatten()
        
        for i, point in enumerate(tqdm(data)):
#             nearest_anchor_idx = np.argmin(np.linalg.norm(point - self.lowest_level_centroids, axis=1))
            nearest_anchor_idx = nearest_anchor_idxs[i]
            projected_nearest_anchor = self.projected_centroids[-2][nearest_anchor_idx]
            projected_nearest_anchor_radius = self.projected_centroid_radii[-1][nearest_anchor_idx]
            
            pca_projected_point = self.scaler.transform([point])[0]
            pca_projected_point = self.pca.transform([pca_projected_point])[0]

            max_radius = self.points_max_radii[-1][self.partitions[:, 0] == nearest_anchor_idx][0, 0]
            points_mean = self.points_means[-1][nearest_anchor_idx]
            normalized_point = (pca_projected_point - points_mean) / max_radius
            projected_point = normalized_point * self.projected_centroid_radii[-1][nearest_anchor_idx] + projected_nearest_anchor
            
            projections.append(projected_point)

        return np.array(projections)

    def fit_transform(self, *args, **kwargs):
        return self.fit(*args, **kwargs)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
