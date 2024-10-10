import numpy as np

from hnne import HNNE
from utils import generate_inception_graph_dataset


def test_projector_smoke_test():
    dummy_data = np.random.random((1000, 10))

    hnne = HNNE()
    hnne.fit_transform(dummy_data, verbose=False)


def test_projector_inception_dataset():
    levels = generate_inception_graph_dataset()
    inception_dataset = levels[-1]

    # Pick a small enough radius for consistency and the Euclidean distance to match the
    # real 2D coordinates.
    hnne = HNNE(radius=0.2, metric="euclidean")
    _ = hnne.fit(inception_dataset)

    expected_partitions = np.stack(
        [
            np.repeat(np.arange(9**2), repeats=9),
            np.repeat(np.arange(9), repeats=9**2),
        ],
        axis=1
    )

    assert len(levels) == len(hnne.projection_parameters.projected_centroids)
    np.testing.assert_array_equal(
        hnne.hierarchy_parameters.partitions,
        expected_partitions
    )


def test_projector_transforms():
    dummy_data = np.random.uniform(size=(1000, 10))

    hnne = HNNE()
    projection_original = hnne.fit(dummy_data, verbose=False)
    projection_new = hnne.transform(dummy_data)

    assert np.min(np.linalg.norm(projection_new, axis=1)) < 2 * np.min(np.linalg.norm(projection_original, axis=1))
    assert np.max(np.linalg.norm(projection_new, axis=1)) < 2 * np.max(np.linalg.norm(projection_original, axis=1))
    assert np.quantile(np.linalg.norm(projection_new - projection_original, axis=1), .8) < 0.1
