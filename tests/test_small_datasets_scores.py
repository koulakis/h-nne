from pathlib import Path

import numpy as np

from utils import load_mnist_test, load_fmnist_test, load_cifar10_test, get_datasets_path, compute_accuracy
from hnne import HNNE

ASSETS_PATH = Path(__file__).parent / "assets/small_dataset_projections"


def test_mnist_scores():
    data_path = get_datasets_path()

    x_mnist, y_mnist = load_mnist_test(data_path)
    x_mnist = x_mnist.astype(np.float64)

    projection = HNNE().fit_transform(x_mnist)

    accuracy = compute_accuracy(projection, y_mnist)
    min_accuracy = 0.9338

    assert accuracy >= min_accuracy, "Accuracy on test MNIST degraded."


def test_fmnist_scores():
    data_path = get_datasets_path()

    x_fmnist, y_fmnist = load_fmnist_test(data_path)
    x_fmnist = x_fmnist.astype(np.float64)

    projection = HNNE().fit_transform(x_fmnist)

    accuracy = compute_accuracy(projection, y_fmnist)
    min_accuracy = 0.7542

    assert accuracy >= min_accuracy, "Accuracy on test FMNIST degraded."


def test_cifar10_scores():
    data_path = get_datasets_path()

    x_cifar10, y_cifar10 = load_cifar10_test(data_path)
    x_cifar10 = x_cifar10.astype(np.float64)

    projection = HNNE().fit_transform(x_cifar10)

    accuracy = compute_accuracy(projection, y_cifar10)
    min_accuracy = 0.20069

    assert accuracy >= min_accuracy, "Accuracy on test CIFAR10 degraded."
