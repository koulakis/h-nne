import unittest

import numpy as np
import pandas as pd

import hnne.cool_functions as cf


class TestCoolFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a small dummy dataset.
        data = np.array(
            [
                [-2.22984331, 1.02151004, 1.13834532],
                [-1.83693656, 0.61308542, 1.76204863],
                [-1.95971099, -4.48379165, 1.72042386],
                [0.26267476, 3.2879465, 0.75645858],
                [0.08794104, 0.94343875, -4.44867773],
                [-1.19680534, -3.33976063, -0.41174465],
            ]
        )

        # Manually compute the partition, mean and std which are shared along multiple tests.
        partition = np.array([2, 0, 2, 2, 1, 0])
        gt_mean = np.array(pd.DataFrame(data).assign(p=partition).groupby("p").mean())
        gt_std = (
            np.sqrt(
                np.array(
                    pd.DataFrame((data - gt_mean[partition]) ** 2)
                    .assign(p=partition)
                    .groupby("p")
                    .mean()
                )
            )
            + 1e-12
        )

        cls.data = data
        cls.partition = partition
        cls.gt_mean = gt_mean
        cls.gt_std = gt_std

    def test_cool_mean(self):
        cool = cf.cool_mean(self.data, self.partition)

        np.testing.assert_array_equal(self.gt_mean, cool, err_msg="Cool mean failed.")

    def test_cool_max(self):
        gt_max = np.array(
            pd.DataFrame(np.abs(self.data[:, 0]))
            .assign(p=self.partition)
            .groupby("p")
            .max()
        )[:, 0]
        cool = cf.cool_max(np.abs(self.data[:, 0]), self.partition)

        np.testing.assert_array_equal(gt_max, cool, err_msg="Cool max failed.")

    def test_cool_max_radius(self):
        gt_max_rad = np.squeeze(
            pd.DataFrame(np.linalg.norm(self.data, axis=1))
            .assign(p=self.partition)
            .groupby("p")
            .max()
        )[self.partition]
        cool = cf.cool_max_radius(self.data, self.partition)[self.partition]

        np.testing.assert_array_equal(
            gt_max_rad, cool, err_msg="Cool max radius failed."
        )

    def test_cool_std(self):
        cool = cf.cool_std(self.data, self.gt_mean[self.partition], self.partition)

        np.testing.assert_array_equal(self.gt_std, cool, err_msg="Cool std failed.")

    def test_cool_normalize(self):
        gt_norm = (self.data - self.gt_mean[self.partition]) / self.gt_std[
            self.partition
        ]
        cool, _ = cf.cool_normalize(self.data, self.partition)

        np.testing.assert_array_equal(gt_norm, cool, err_msg="Cool normalize failed.")
