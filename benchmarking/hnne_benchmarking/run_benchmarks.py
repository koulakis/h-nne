import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from hnne_benchmarking.data import (
    DatasetGroup,
    dataset_loaders,
    dataset_validation_knn_values,
)
from hnne_benchmarking.evaluation import dim_reduction_benchmark, format_metric
from hnne_benchmarking.utils import (
    build_scores_overview,
    format_time,
    plot_proj_grid,
    time_function_call,
)
from sklearn.preprocessing import StandardScaler

from hnne.projector import HNNE


def run_eval(
    data_path: Path,
    dataset_group: DatasetGroup = DatasetGroup.small,
    n_components: int = 2,
    distance="cosine",
    radius: float = 0.4,
    ann_threshold=10_000,
    preliminary_embedding="pca",
    validate_only_1nn: bool = True,
    compute_trustworthiness: bool = True,
    random_state: int = 42,
    verbose: bool = False,
    # hnne v2 params
    prefered_num_clust=None,
    hnne_version="v2",
    start_cluster_view: int = "auto",
    v2_size_threshold: int = None,
    # save params
    plot_projection: bool = True,
    save_experiment=False,
    experiment_name: str = "test",
    output_directory="./eval_results",
    points_plot_limit: int = 5000_000,
    figsize: Tuple[int, int] = (10, 10),
    skip_done: bool = True,
    scale_data: bool = False,
):
    if dataset_group == DatasetGroup.large:
        print("Hold tight, processing large datasets...")

    all_projections = []

    if save_experiment:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        experiment_directory = Path(output_directory, experiment_name)
        projections_directory = Path(experiment_directory, "projections")
        scores_directory = Path(experiment_directory, "scores")
        plots_directory = Path(experiment_directory, "plots")
        for directory in [
            experiment_directory,
            projections_directory,
            scores_directory,
            plots_directory,
        ]:
            directory.mkdir(exist_ok=True)

    loaders = dataset_loaders(dataset_group=dataset_group)
    scores_by_dataset = {}
    for i, [dataset_name, loader] in enumerate(loaders.items()):
        try:
            knn_values = (
                [1]
                if validate_only_1nn
                else dataset_validation_knn_values[dataset_name]
            )
            if save_experiment:
                filename = f"{dataset_name}"
                out_projection_path = projections_directory / f"{filename}.npz"
                out_score_path = scores_directory / f"{filename}.csv"
                out_plot_path = plots_directory / f"{filename}.png"

                if (
                    skip_done
                    and out_score_path.exists()
                    and out_projection_path.exists()
                ):
                    print(
                        f"Skipping extraction of existing partitions for {dataset_name}."
                    )
                    continue

            print(f"Loading {dataset_name}...")
            data, targets = loader(data_path)
            if scale_data:
                print("Scaling data...")
                data = StandardScaler().fit_transform(data)

            hnne = HNNE(
                n_components=n_components,
                metric=distance,
                radius=radius,
                ann_threshold=ann_threshold,
                preliminary_embedding=preliminary_embedding,
                random_state=random_state,
                preferred_num_clust=prefered_num_clust,
                hnne_version=hnne_version,
                v2_size_threshold=v2_size_threshold,
                start_cluster_view=start_cluster_view,
            )

            _, time_elapsed_finch = time_function_call(
                hnne.fit_only_hierarchy, data, verbose=verbose
            )

            projection, time_elapsed_projection = time_function_call(
                hnne.fit_transform, data, verbose=verbose
            )

            print(
                f"Finch time: {time_elapsed_finch}, projection time: {time_elapsed_projection}"
            )

            print(f"Validating {dataset_name} on {knn_values} nearest neighbors...")
            proj_knn_acc, tw = dim_reduction_benchmark(
                knn_values,
                data,
                projection,
                targets,
                compute_trustworthiness=compute_trustworthiness,
            )
            scores = pd.DataFrame(
                {
                    "KNN_ACC": format_metric(proj_knn_acc),
                    "trustworthiness": [tw] + (len(knn_values) - 1) * [""],
                    "proj_time": [format_time(time_elapsed_projection)]
                    + (len(knn_values) - 1) * [""],
                    "finch_time": [format_time(time_elapsed_finch)]
                    + (len(knn_values) - 1) * [""],
                },
                index=pd.Series(knn_values, name="k_value"),
            )

            scores_by_dataset[dataset_name] = scores

            if save_experiment:
                scores.to_csv(out_score_path)
                np.savez(out_projection_path, projection=projection)

            if plot_projection and n_components == 2:
                print("Exporting projection plots...")
                proj = projection
                targ = targets
                # downsample for plotting if needed (do NOT mutate original arrays)
                if proj.shape[0] > points_plot_limit:
                    idx = np.random.choice(
                        proj.shape[0], points_plot_limit, replace=False
                    )
                    proj = proj[idx]
                    targ = targ[idx]

                # ---- save individual figure (optional) ----
                if save_experiment:
                    fig, ax = plt.subplots(figsize=figsize)
                    sc = ax.scatter(
                        proj[:, 0], proj[:, 1], s=1, c=targ, cmap="Spectral"
                    )
                    ax.set_aspect("equal", adjustable="box")
                    fig.savefig(out_plot_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)  # important to avoid figure buildup in loops

                # ---- collect for the grid to show later ----
                # choose a readable title per dataset; adapt to your variables
                ds_title = (
                    str(dataset_name) if "dataset_name" in locals() else f"Dataset {i}"
                )  # or whatever identifier you have
                all_projections.append({"proj": proj, "targ": targ, "title": ds_title})

        except Exception as e:
            print(f"Failed to run experiment on {dataset_name}.")
            print(f"Error: {e}")

    if plot_projection and len(all_projections) > 0:
        out_plotgrid_path = plots_directory / "all_plots_grid.png"
        plt_grid = plot_proj_grid(
            all_projections, figsize, save_experiment, out_plotgrid_path
        )
        plt_grid.show()

    overview_scores = build_scores_overview(scores_by_dataset)
    if save_experiment:
        overview_scores.to_csv(Path(scores_directory, "all_datasets_scores.csv"))

    return overview_scores, plt_grid


if __name__ == "__main__":
    typer.run(run_eval)
