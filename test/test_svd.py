from __future__ import annotations

import random
from typing import Optional

import numpy as np

from gptomics.svd import SVD


def check_svd(seed: Optional[int]):
    if seed is not None:
        np.random.seed(seed)

    M1 = np.random.rand(10, 10)
    M2 = np.random.rand(10, 10)

    assert np.all(
        np.isclose(
            SVD.frommatrices(M1, M2).full(),
            M1 @ M2,
        )
    ), f"fails for seed {seed}"


def test_svd():
    check_svd(random.choice(range(1000)))
