import os
from pathlib import Path
from typing import Union

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

# DATASETS


def load_mnist_test(data_path: Union[Path, str]) -> tuple[np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    x_test_mnist, y_test_mnist = zip(*MNIST(data_path, train=False, download=True))

    x_test_mnist = np.vstack([np.array(image).flatten() for image in x_test_mnist])
    y_test_mnist = np.array(y_test_mnist)

    return x_test_mnist, y_test_mnist


def load_fmnist_test(data_path: Union[Path, str]) -> tuple[np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    x_test_fmnist, y_test_fmnist = zip(
        *FashionMNIST(data_path, train=False, download=True)
    )

    x_test_fmnist = np.vstack([np.array(image).flatten() for image in x_test_fmnist])
    y_test_fmnist = np.array(y_test_fmnist)

    return x_test_fmnist, y_test_fmnist


def load_cifar10_test(data_path: Union[Path, str]) -> tuple[np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    x_test_cifar10, y_test_cifar10 = zip(
        *CIFAR10(data_path, train=False, download=True)
    )

    x_test_cifar10 = np.vstack([np.array(image).flatten() for image in x_test_cifar10])
    y_test_cifar10 = np.array(y_test_cifar10)

    return x_test_cifar10, y_test_cifar10


def get_datasets_path() -> Path:
    data_path = os.environ.get("DATA_PATH_HNNE")

    if data_path is None:
        raise ValueError(
            "To run the small dataset tests, please provide a path where the datasets will be downloaded "
            "via the DATA_PATH_HNNE environment variable."
        )

    return Path(data_path)


def default_seed_datapoints() -> np.ndarray:
    seed_datapoints = np.array(
        [
            [0, 0],
            [4, 0],
            [12, 0.2],
            [24, 0.4],
            [-3, 3],
            [-10, 8],
            [-24, 12],
            [-3, -3],
            [-10, -8.1],
        ]
    )

    seed_datapoints = (
        seed_datapoints - seed_datapoints.mean(axis=0)
    ) / seed_datapoints.std(axis=0)

    return seed_datapoints


def generate_inception_graph_dataset(
    seed_datapoints: np.ndarray = default_seed_datapoints(),
    n_levels: int = 3,
    shrinking_factor: float = 0.05,
) -> list[np.ndarray]:
    """Using a small seed dataset which defines a simple nearest neighbor graph G, start expanding each node of G
    with a copy of itself in a scale which is so much smaller that the distances of the small copies are negligible
    compared to the distances of G. We repeat this process n_levels many times and create a graph which perfectly fits
    the hierarchical nature of HNNE. This is great for testing both the fit and transform functions as they should
    generate a tree structure very similar to the one in the generation of the dataset. Furthermore, if the seed
    dataset has the same dimension as the projection space (but may exist embedded in a subspace higher dimensional
    space) and it is constructed so that it is almost isotropic, then the projection tree should be exactly the same.

    NOTE: The

    Args:
        seed_datapoints: A small initial dataset whose structure will be repeated n_level times. The mean of the centers
            should be 0 and the std 1. If not, this will be enforced.
        n_levels: The number of times to expand the vertices of the dataset, minimum value is 1.
        shrinking_factor: A coefficient by which the distances of the seed_datapoints shrink on every new level it is
            copied.

    Returns:
        An list of arrays with the different levels of the dataset. The dataset is levels[-1] and has shape (n f).
        n is the number of points which equals to n0^n_levels (by the way it is constructed) and f is the number
        of features, where (n0 f) is the shape of the seed_datapoints.
    """
    if n_levels < 1:
        raise ValueError(f"The number of levels should be at least 1, give {n_levels}.")

    seed_datapoints = (
        seed_datapoints - seed_datapoints.mean(axis=0)
    ) / seed_datapoints.std(axis=0)

    levels = [seed_datapoints]
    coefficient = shrinking_factor
    for i in range(n_levels - 1):
        last_level_points = levels[-1]

        # To get add all seed points to each last level point, we use numpy broadcasting.
        next_level_points = (coefficient * seed_datapoints)[
            None, ...
        ] + last_level_points[:, None, :]
        shape = next_level_points.shape
        next_level_points = next_level_points.reshape((shape[0] * shape[1], shape[2]))

        levels.append(next_level_points)
        coefficient *= shrinking_factor

    return levels


# SCORING


def compute_accuracy(
    projection: np.ndarray, labels: np.ndarray, seed: int = 42
) -> float:
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    accuracies = []
    for train_index, test_index in kf.split(projection, labels):
        x_train, x_test = projection[train_index], projection[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
        knn.fit(x_train, y_train)

        pred_test = knn.predict(x_test)

        accuracy = accuracy_score(y_test, pred_test)
        accuracies.append(accuracy)

    return float(np.mean(accuracies))
