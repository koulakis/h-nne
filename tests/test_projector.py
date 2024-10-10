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
    _ = hnne.fit(inception_dataset, verbose=False)

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


def test_projector_transform_yields_similar_results_to_fit_and_same_on_multiple_calls_on_random_dataset():
    np.random.seed(42)
    dummy_data = np.random.uniform(size=(1000, 10))

    hnne = HNNE(radius=0.2)
    projection_original = hnne.fit(dummy_data, verbose=False)
    projection_with_transform = hnne.transform(dummy_data)
    second_projection_with_transform = hnne.transform(dummy_data)

    np.testing.assert_almost_equal(projection_original, projection_with_transform)
    np.testing.assert_array_equal(projection_with_transform, second_projection_with_transform)


def test_projector_transform_yields_similar_results_to_fit_and_same_on_multiple_calls_on_the_inception_dataset():
    levels = generate_inception_graph_dataset()
    inception_dataset = levels[-1]

    # Pick a small enough radius for consistency and the Euclidean distance to match the
    # real 2D coordinates.
    hnne = HNNE(radius=0.2, metric="euclidean")
    projection_original = hnne.fit(inception_dataset, verbose=False)

    projection_with_transform = hnne.transform(inception_dataset)
    second_projection_with_transform = hnne.transform(inception_dataset)

    np.testing.assert_almost_equal(projection_original, projection_with_transform)
    np.testing.assert_array_equal(projection_with_transform, second_projection_with_transform)
