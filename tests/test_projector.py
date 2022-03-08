import numpy as np

from hnne import HNNE


def test_projector_smoke_test():
    dummy_data = np.random.random((1000, 10))

    hnne = HNNE()
    hnne.fit_transform(dummy_data, verbose=False)


def test_projector_transforms():
    dummy_data = np.random.uniform(size=(1000, 10))

    hnne = HNNE()
    projection_original = hnne.fit(dummy_data, verbose=False)
    projection_new = hnne.transform(dummy_data)

    assert np.min(np.linalg.norm(projection_new, axis=1)) < 2 * np.min(np.linalg.norm(projection_original, axis=1))
    assert np.max(np.linalg.norm(projection_new, axis=1)) < 2 * np.max(np.linalg.norm(projection_original, axis=1))
    assert np.quantile(np.linalg.norm(projection_new - projection_original, axis=1), .8) < 0.1
