import numpy as np
import pandas as pd

import hnne.cool_functions as cf


def test_cool_functions_simple_example():
    data = np.array([
        [-2.22984331, 1.02151004, 1.13834532],
        [-1.83693656, 0.61308542, 1.76204863],
        [-1.95971099, -4.48379165, 1.72042386],
        [0.26267476, 3.2879465, 0.75645858],
        [0.08794104, 0.94343875, -4.44867773],
        [-1.19680534, -3.33976063, -0.41174465]
    ])
    partition = np.array([2, 0, 2, 2, 1, 0])

    # Cool mean
    gt_mean = np.array(pd.DataFrame(data).assign(p=partition).groupby('p').mean())
    cool = cf.cool_mean(data, partition)

    np.testing.assert_array_equal(gt_mean, cool, err_msg='Cool mean failed.')

    # Cool max
    gt_max = np.array(pd.DataFrame(np.abs(data[:, 0])).assign(p=partition).groupby('p').max())[:, 0]
    cool = cf.cool_max(np.abs(data[:, 0]), partition)

    np.testing.assert_array_equal(gt_max, cool, err_msg='Cool max failed.')

    # Cool max radius
    gt_max_rad = (
        np.squeeze(
            pd.DataFrame(np.linalg.norm(data, axis=1))
            .assign(p=partition)
            .groupby('p')
            .max()
        )[partition])
    cool = cf.cool_max_radius(data, partition)

    np.testing.assert_array_equal(gt_max_rad, cool, err_msg='Cool max radius failed.')

    # Cool std
    gt_std = (
            np.sqrt(np.array(pd.DataFrame((data - gt_mean[partition])**2).assign(p=partition).groupby('p').mean()))
            + 1e-12)
    cool = cf.cool_std(data, gt_mean[partition], partition)

    np.testing.assert_array_equal(gt_std, cool, err_msg='Cool std failed.')

    # Cool normalize
    gt_norm = (data - gt_mean[partition]) / gt_std[partition]
    cool = cf.cool_normalize(data, partition)

    np.testing.assert_array_equal(gt_norm, cool, err_msg='Cool normalize failed.')



