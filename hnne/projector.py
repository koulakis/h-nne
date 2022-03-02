from tqdm import tqdm
import numpy as np
import pickle
from pynndescent import NNDescent

from hnne.finch_clustering import cool_mean, FINCH
from hnne.hierarchical_projection import multi_step_projection


class HNNEProjector:
    def __init__(
            self,
            inflate_pointclouds=False,
            radius_shrinking=0.66,
            dim=2,
            real_nn_threshold=40000,
            projection_type='pca',
            distance='cosine',
            low_memory_nndescent=False,
            decompress_points=False # TODO: Change to true after tests
    ):
        self.inflate_pointclouds = inflate_pointclouds
        self.radius_shrinking = radius_shrinking
        self.dim = dim
        self.real_nn_threshold = real_nn_threshold
        self.projection_type = projection_type
        self.distance = distance
        self.low_memory_nndescent = low_memory_nndescent
        self.decompress_points = decompress_points
        
    def fit(
            self,
            data,
            y=None,
            verbose=True,
            stop_at_partition=None
    ):        
        if verbose:
            print('Generating h-NNE hierarchy...')
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
        
        if stop_at_partition is not None or partition_sizes[-1] < 3:
            if verbose:
                print('Filtering last partitions')
            stop_at_partition = -1 if stop_at_partition is None else stop_at_partition
            
            partition_sizes = partition_sizes[:stop_at_partition]
            partitions = partitions[:, :stop_at_partition]
            partition_labels = partition_labels[:stop_at_partition]

        if verbose:
            print(f'Projecting to {self.dim} dimensions...')

        projection, projected_centroid_radii, projected_centroids, pca, scaler, points_means, points_max_radii, projected_anchors = multi_step_projection(
            data, 
            partitions,
            partition_labels,
            inflate_pointclouds=self.inflate_pointclouds,
            radius_shrinking=self.radius_shrinking,
            dim=self.dim,
            real_nn_threshold=self.real_nn_threshold,
            partition_sizes=partition_sizes,
            projection_type=self.projection_type,
            decompress_points=self.decompress_points
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
        
        return projection
        
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

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
