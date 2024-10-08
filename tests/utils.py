from typing import Union
from pathlib import Path
import os

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10


def load_mnist_test(data_path: Union[Path, str]) -> tuple[np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    x_test_mnist, y_test_mnist = zip(*MNIST(data_path, train=False, download=True))

    x_test_mnist = np.vstack([np.array(image).flatten() for image in x_test_mnist])
    y_test_mnist = np.array(y_test_mnist)

    return x_test_mnist, y_test_mnist


def load_fmnist_test(data_path: Union[Path, str]) -> tuple[np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    x_test_fmnist, y_test_fmnist = zip(*FashionMNIST(data_path, train=False, download=True))

    x_test_fmnist = np.vstack([np.array(image).flatten() for image in x_test_fmnist])
    y_test_fmnist = np.array(y_test_fmnist)

    return x_test_fmnist, y_test_fmnist


def load_cifar10_test(data_path: Union[Path, str]) -> tuple[np.ndarray, np.ndarray]:
    data_path = Path(data_path)
    x_test_cifar10, y_test_cifar10 = zip(*CIFAR10(data_path, train=False, download=True))

    x_test_cifar10 = np.vstack([np.array(image).flatten() for image in x_test_cifar10])
    y_test_cifar10 = np.array(y_test_cifar10)

    return x_test_cifar10, y_test_cifar10


def get_datasets_path() -> Path:
    data_path = os.environ.get("DATA_PATH_HNNE")

    if data_path is None:
        raise ValueError(
            "To run the small dataset tests, please provide a path where the datasets will be downloaded "
            "via the DATA_PATH_HNNE environment variable.")

    return Path(data_path)


def compute_accuracy(projection:np.ndarray, labels:np.ndarray, seed: int = 42) -> float:
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
