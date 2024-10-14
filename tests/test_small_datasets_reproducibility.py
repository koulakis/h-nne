from pathlib import Path

import numpy as np
from utils import (
    get_datasets_path,
    load_cifar10_test,
    load_fmnist_test,
    load_mnist_test,
)

from hnne import HNNE

ASSETS_PATH = Path(__file__).parent / "assets/small_dataset_projections"


def test_mnist_reproducibility():
    data_path = get_datasets_path()

    x_mnist, _ = load_mnist_test(data_path)
    x_mnist = x_mnist.astype(np.float64)

    projection = HNNE().fit_transform(x_mnist)
    expected_projection = np.load(ASSETS_PATH / "mnist_expected_projection.npy")

    np.testing.assert_array_almost_equal(projection, expected_projection)


def test_fmnist_reproducibility():
    data_path = get_datasets_path()

    x_fmnist, _ = load_fmnist_test(data_path)
    x_fmnist = x_fmnist.astype(np.float64)

    projection = HNNE().fit_transform(x_fmnist)
    expected_projection = np.load(ASSETS_PATH / "fmnist_expected_projection.npy")

    np.testing.assert_array_almost_equal(projection, expected_projection)


def test_cifar10_reproducibility():
    data_path = get_datasets_path()

    x_cifar10, _ = load_cifar10_test(data_path)
    x_cifar10 = x_cifar10.astype(np.float64)

    projection = HNNE().fit_transform(x_cifar10)
    expected_projection = np.load(ASSETS_PATH / "cifar10_expected_projection.npy")

    # Unfortunately there are some small discrepancies on consecutive runs on CIFAR10.
    np.testing.assert_almost_equal(projection, expected_projection, decimal=5)
