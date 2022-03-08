from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import typer

from hnne.benchmarking.utils import time_function_call
from hnne.benchmarking.data import dataset_loaders, DatasetGroup, dataset_validation_knn_values
from hnne.benchmarking.evaluation import format_metric, dim_reduction_benchmark
from hnne import HNNE


def main(
        experiment_name: str,
        output_directory: Path,
        data_path: Path,
        dataset_group: DatasetGroup = DatasetGroup.small,
        points_plot_limit: int = 50000,
        figsize: Tuple[int, int] = (20, 20),
        skip_done: bool = True,
        scale_data: bool = False,
        dim: int = 2,
        continue_on_error: bool = False,
        radius_factor: float = 1.3,
        metric: str = 'cosine',
        validate_only_1nn: bool = True,
        ann_threshold: int = 40000,
        compute_trustworthiness: bool = False,
        preliminary_embedding: str = 'pca',
        verbose: bool = False
):
    if dataset_group == DatasetGroup.large:
        print('Hold tight, processing large datasets...')

    experiment_directory = output_directory / experiment_name
    projections_directory = experiment_directory / 'projections'
    scores_directory = experiment_directory / 'scores'
    plots_directory = experiment_directory / 'plots'
    for directory in [
        experiment_directory, projections_directory, scores_directory, plots_directory
    ]:
        directory.mkdir(exist_ok=True)

    loaders = dataset_loaders(dataset_group=dataset_group)
    for dataset_name, loader in loaders.items():
        try:
            knn_values = [1] if validate_only_1nn else dataset_validation_knn_values[dataset_name]

            filename = f'{dataset_name}'
            out_projection_path = projections_directory / f'{filename}.npz'
            out_score_path = scores_directory / f'{filename}.csv'
            out_plot_path = plots_directory / f'{filename}.png'

            if skip_done and out_score_path.exists() and out_projection_path.exists():
                print(f'Skipping extraction of existing partitions for {dataset_name}.')
                continue

            print(f'Loading {dataset_name}...')
            data, targets = loader(data_path)
            if scale_data:
                print('Scaling data...')
                data = StandardScaler().fit_transform(data)

            hnne = HNNE(
                radius_factor=radius_factor,
                dim=dim,
                ann_threshold=ann_threshold,
                preliminary_embedding=preliminary_embedding,
                metric=metric
            )

            _, time_elapsed_finch = time_function_call(hnne.fit_only_hierarchy, data, verbose=verbose)

            projection, time_elapsed_projection = time_function_call(
                hnne.fit_transform,
                data,
                verbose=verbose
            )

            np.savez(out_projection_path, projection=projection)

            print(f'Finch time: {time_elapsed_finch}, projection time: {time_elapsed_projection}')

            print(f'Validating {dataset_name} on {knn_values} nearest neighbors...')
            proj_knn_acc, tw = dim_reduction_benchmark(
                knn_values,
                data,
                projection,
                targets,
                compute_trustworthiness=compute_trustworthiness)
            scores = pd.DataFrame({
                'proj_KNN_ACC': format_metric(proj_knn_acc),
                'trustworthiness': len(knn_values) * [tw],
                'proj_time': [time_elapsed_projection] + (len(knn_values) - 1) * [''],
                'finch_time': [time_elapsed_finch] + (len(knn_values) - 1) * ['']
            },
                index=pd.Series(knn_values, name='k_value'))

            scores.to_csv(out_score_path)

            if dim == 2:
                print(f'Exporting projection plots...')
                if projection.shape[0] > points_plot_limit:
                    idx = np.random.choice(projection.shape[0], points_plot_limit, replace=False)
                    projection = projection[idx]
                    targets = targets[idx]

                plt.figure(figsize=figsize)
                plt.scatter(*projection.T, s=1, c=targets, cmap='Spectral')
                plt.gca().set_aspect('equal', adjustable='box')
                plt.savefig(out_plot_path)

        except Exception as e:
            if continue_on_error:
                print(f'Failed to run experiment on {dataset_name}.')
                print(f'Error: {e}')
            else:
                raise e


if __name__ == '__main__':
    typer.run(main)
