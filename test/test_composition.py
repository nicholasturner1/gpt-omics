import numpy as np

from gptomics import composition as comp


def test_removemean():
    assert np.all(
        np.isclose(
            comp.removemean(np.ones((10, 10))),
            0,
        )
    )
    assert np.all(
        np.isclose(
            comp.removemean(np.ones((10, 10)), method="matrix multiply"),
            0,
        )
    )
