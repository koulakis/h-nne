import numpy as np
import pandas as pd
from IPython.core.display import display
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.manifold import trustworthiness
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from pynndescent import NNDescent
from scipy.stats import mode


def dim_reduction_benchmark(
        k_values, 
        data, 
        embedding, 
        labels, 
        n_splits=10, 
        shuffle=True,
        sample_train_set=300000,
        compute_trustworthiness=False,
        use_ann=False,
        seed=5536):
    """Run a benchmark on the embeddings of a dataset in a lower dimension. The benchmark
    runs for a list of different number of k nearest neighbors and returns for each setting
    the accuracy and trustworthiness of the embedding.
    
    Args:
        - k_values: a list of different number of nearest neighbors to evaluate on
        - data: a numpy array with the raw data with dimensions (n_datapoints, n_features)
        - embedding: an embedding in a numpy array of dimensions (n_datapoints, dim_of_embedding)
        - labels: a numpy array with the ground truth classes of the data of dimensions
            (n_datapoints, )
        - n_splits: number of splits to consider in the cross-validation
        - shuffle: if true, the data will be shuffled before performing the splits
        - compute_trustworthiness: flag to disable expensive computation of trustworthiness
        - seed: seed to fix the dataset shuffle
        
    Returns:
        - accuracies: a dictionary of the form {k: accuracies}, where k is a number of
            nearest neighbors and accuracies are the corresponding accuracies from the
            n_splits-fold validation
        - trustworthinesses: dictionary with the same structure as accuracies only with
            trustworthiness values this time
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    accuracies = dict()
    
    if 0.9 * data.shape[0] > sample_train_set:
        print(f'Will subsample train datasets from {0.9 * data.shape[0]} to {sample_train_set} for {k_values}-nn evaluation.')
    
    with tqdm(total=len(k_values) * kf.n_splits) as pbar:
        for i, k in enumerate(k_values):
            k_embedding = embedding
            accuracies[k] = []

            for train_index, test_index in kf.split(k_embedding, labels):
                if 0.9 * data.shape[0] > sample_train_set:
                    train_index = np.random.choice(train_index, size=sample_train_set, replace=False)
                E_train, E_test = k_embedding[train_index], k_embedding[test_index]

                y_train, y_test = labels[train_index], labels[test_index]
                
                if use_ann:
                    knn_index = NNDescent(E_train, n_trees=16, verbose=True)
                    pred_test_idx, _ = knn_index.query(E_test, k=k)
                    pred_test = mode(y_train[pred_test_idx], axis=1)[0].flatten()
                    
                else:
                    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
                    knn.fit(E_train, y_train)

                    pred_test = knn.predict(E_test)

                accuracy = accuracy_score(y_test, pred_test)
                accuracies[k].append(accuracy)
                pbar.update(1)
                
            accuracies[k] = np.array(accuracies[k])
                
        if compute_trustworthiness:
            if data.shape[0] < 20000:
                tw = trustworthiness(data, embedding, n_neighbors=5)
            else:                
                data_index = np.random.choice(data.shape[0], size=20000, replace=False)
                tw = trustworthiness(data[data_index], k_embedding[data_index], n_neighbors=5)
        else:
            tw = None
                
    return accuracies, tw


def format_metric(metric_values, digit_precision=3):
    """Format an output metric of the benchmark function.
    
    Args:
        metric_values: a dictionary of the form {k: value}, where k is a number
            of nearest neighbors and value is the corresponding value of the metric
        digit_precision: the precision used to print the metric value and its std
    """
    return {
        k: f'{value.mean():.{digit_precision}f} (\u00B1{value.std():.{digit_precision}f})' 
        for k, value in metric_values.items()}


def validate_finch_projection(
        data,
        targets,
        projection,
        projected_centroid_radii,
        projected_centroids,
        figsize=(15, 15),
        compute_trustworthiness=False,
        plot_projection=True,
        k_values=None,
        plot_without_cells=True,
        xlim=None,
        ylim=None
):
    if k_values is None:
        k_values = [10, 20]

    proj_knn_acc, tw = dim_reduction_benchmark(k_values, data, projection, targets, compute_trustworthiness=compute_trustworthiness)
        
    display(pd.DataFrame({
        'proj_KNN_ACC': format_metric(proj_knn_acc),
        'trustworthiness': {v: tw for v in k_values}
    }))

    if plot_projection:
        _, ax = plt.subplots(figsize=figsize)

        voronoi_cells = Voronoi(projected_centroids)
        voronoi_plot_2d(voronoi_cells, show_vertices=False, show_points=False, ax=ax, line_alpha=0.3, line_width=.5)

        for point, radius in zip(projected_centroids, projected_centroid_radii):
            ax.add_patch(plt.Circle(point, radius, color='orange', alpha=.1, zorder=1))

        ax.scatter(*projection.T, s=1, c=targets, zorder=2, cmap='Spectral')
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    if plot_without_cells:
        plt.figure(figsize=(10, 10))
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.scatter(*projection.T, s=1, c=targets, cmap='Spectral')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
