import unittest
from pathlib import Path

import numpy as np
from utils import (
    compute_accuracy,
    get_datasets_path,
    load_cifar10_test,
    load_fmnist_test,
    load_mnist_test,
)

from hnne import HNNE


class TestOnSmallDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the path of the expected test result assets and the location to save the datasets.
        cls.assets_path = Path(__file__).parent / "assets/small_dataset_projections"
        data_path = get_datasets_path()

        # MNIST
        x_mnist, y_mnist = load_mnist_test(data_path)
        x_mnist = x_mnist.astype(np.float64)
        cls.x_mnist, cls.y_mnist = x_mnist, y_mnist

        # FMNIST
        x_fmnist, y_fmnist = load_fmnist_test(data_path)
        x_fmnist = x_fmnist.astype(np.float64)
        cls.x_fmnist, cls.y_fmnist = x_fmnist, y_fmnist

        # CIFAR10
        x_cifar10, y_cifar10 = load_cifar10_test(data_path)
        x_cifar10 = x_cifar10.astype(np.float64)
        cls.x_cifar10, cls.y_cifar10 = x_cifar10, y_cifar10

    def test_mnist_reproducibility(self):
        projection = HNNE(hnne_version="v1").fit_transform(self.x_mnist)
        expected_projection = np.load(
            self.assets_path / "mnist_expected_projection.npy"
        )

        np.testing.assert_array_almost_equal(projection, expected_projection)

    def test_fmnist_reproducibility(self):
        projection = HNNE(hnne_version="v1").fit_transform(self.x_fmnist)
        expected_projection = np.load(
            self.assets_path / "fmnist_expected_projection.npy"
        )

        np.testing.assert_array_almost_equal(projection, expected_projection)

    def test_cifar10_reproducibility(self):
        projection = HNNE(hnne_version="v1").fit_transform(self.x_cifar10)
        expected_projection = np.load(
            self.assets_path / "cifar10_expected_projection.npy"
        )

        # Unfortunately there are some small discrepancies on consecutive runs on CIFAR10.
        np.testing.assert_almost_equal(projection, expected_projection, decimal=5)

    def test_mnist_scores(self):
        projection = HNNE(hnne_version="v1").fit_transform(self.x_mnist)

        accuracy = compute_accuracy(projection, self.y_mnist)
        min_accuracy = 0.9338

        self.assertGreaterEqual(
            accuracy, min_accuracy, "Accuracy on test MNIST degraded."
        )

    def test_fmnist_scores(self):
        projection = HNNE(hnne_version="v1").fit_transform(self.x_fmnist)

        accuracy = compute_accuracy(projection, self.y_fmnist)
        min_accuracy = 0.7542

        self.assertGreaterEqual(
            accuracy, min_accuracy, "Accuracy on test FMNIST degraded."
        )

    def test_cifar10_scores(self):
        projection = HNNE(hnne_version="v1").fit_transform(self.x_cifar10)

        accuracy = compute_accuracy(projection, self.y_cifar10)
        min_accuracy = 0.20069

        self.assertGreaterEqual(
            accuracy, min_accuracy, "Accuracy on test CIFAR10 degraded."
        )
