import unittest

import numpy as np
from utils import generate_inception_graph_dataset

from hnne import HNNE


class TestHNNE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(636)
        cls.dummy_data_10 = np.random.random(size=(1_000, 10))
        cls.dummy_data_256 = np.random.random(size=(1_000, 256))
        cls.inception_dataset_levels = generate_inception_graph_dataset()

    def test_projector_smoke_test(self):
        hnne = HNNE()
        hnne.fit_transform(self.dummy_data_10, verbose=False)

    def test_projector_inception_dataset(self):
        inception_dataset = self.inception_dataset_levels[-1]

        # Pick a small enough radius for consistency and the Euclidean distance to match the
        # real 2D coordinates.
        hnne = HNNE(radius=0.2, metric="euclidean")
        _ = hnne.fit(inception_dataset, verbose=False)

        expected_partitions = np.stack(
            [
                np.repeat(np.arange(9**2), repeats=9),
                np.repeat(np.arange(9), repeats=9**2),
            ],
            axis=1,
        )

        self.assertEqual(
            len(self.inception_dataset_levels),
            len(hnne.projection_parameters.projected_centroids),
        )
        np.testing.assert_array_equal(
            hnne.hierarchy_parameters.partitions, expected_partitions
        )

    def test_projector_transform_yields_similar_results_to_fit_and_same_on_multiple_calls_on_random_dataset(
        self,
    ):
        dummy_data = self.dummy_data_10

        hnne = HNNE(radius=0.2)
        projection_original = hnne.fit(dummy_data, verbose=False)
        projection_with_transform = hnne.transform(dummy_data)
        second_projection_with_transform = hnne.transform(dummy_data)

        errors = np.linalg.norm(projection_with_transform - projection_original, axis=1)
        relative_errors = errors / np.linalg.norm(projection_original, axis=1)

        self.assertEqual(np.quantile(relative_errors, 0.85), 0)
        self.assertLessEqual(np.quantile(relative_errors, 0.90), 0.15)
        self.assertLessEqual(np.quantile(relative_errors, 0.95), 2)
        np.testing.assert_array_equal(
            projection_with_transform, second_projection_with_transform
        )

    def test_projector_transform_yields_similar_results_to_fit_and_same_on_multiple_calls_on_the_inception_dataset(
        self,
    ):
        inception_dataset = self.inception_dataset_levels[-1]

        # Pick a small enough radius for consistency and the Euclidean distance to match the
        # real 2D coordinates.
        hnne = HNNE(radius=0.2, metric="euclidean")
        projection_original = hnne.fit(inception_dataset, verbose=False)

        projection_with_transform = hnne.transform(inception_dataset)
        second_projection_with_transform = hnne.transform(inception_dataset)

        np.testing.assert_almost_equal(projection_original, projection_with_transform)
        np.testing.assert_array_equal(
            projection_with_transform, second_projection_with_transform
        )

    def test_default_n_components(self):
        hnne = HNNE()
        self.assertEqual(hnne.n_components, 2)

    def test_default_projection_shape(self):
        hnne = HNNE()
        projection = hnne.fit_transform(self.dummy_data_256)
        self.assertEqual(projection.shape, (self.dummy_data_256.shape[0], 2))

    def test_hnne_new_n_components(self):
        for n in [2, 3, 5, 10]:
            hnne = HNNE(n_components=n)
            projection = hnne.fit_transform(self.dummy_data_256)
            self.assertEqual(projection.shape, (self.dummy_data_256.shape[0], n))

    @unittest.skip("Skipping till a random seed parameter is added to HNNE.")
    def test_reprojection_to_different_dimension(self):
        hnne = HNNE(n_components=3)
        projection = hnne.fit_transform(self.dummy_data_256)
        self.assertEqual(projection.shape, (self.dummy_data_256.shape[0], 3))

        reprojection = hnne.fit_transform(
            self.dummy_data_256, override_dim=5, verbose=True
        )
        self.assertEqual(reprojection.shape, (self.dummy_data_256.shape[0], 5))

        direct_projection_5_dims = HNNE(n_components=5).fit_transform(
            self.dummy_data_256, verbose=True
        )
        self.assertEqual(
            direct_projection_5_dims.shape, (self.dummy_data_256.shape[0], 5)
        )
        np.testing.assert_almost_equal(reprojection, direct_projection_5_dims)
