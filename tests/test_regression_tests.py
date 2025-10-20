import unittest
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits

from hnne import HNNE


class TestHNNE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_digits().data
        cls.assets_path = (
            Path(__file__).parent / "assets/pendigits_different_version_projections"
        )

    def test_hnne_v1(self):
        hnne = HNNE(hnne_version="v1")
        projection = hnne.fit_transform(self.data, verbose=False)

        expected_projection = np.loadtxt(
            self.assets_path / "pendigits_projection_hnne_v1.txt", delimiter=","
        )

        np.testing.assert_array_equal(projection, expected_projection)

    def test_hnne_v2(self):
        hnne = HNNE(hnne_version="v2")
        projection = hnne.fit_transform(self.data, verbose=False)

        expected_projection = np.loadtxt(
            self.assets_path / "pendigits_projection_hnne_v2.txt", delimiter=","
        )

        np.testing.assert_array_equal(projection, expected_projection)
