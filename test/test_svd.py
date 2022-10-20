from __future__ import annotations

import random
from typing import Optional

import torch

from gptomics.svd import SVD


def check_svd(seed: Optional[int]):
    if seed is not None:
        torch.manual_seed(seed)

    M1 = torch.rand(10, 10)
    M2 = torch.rand(10, 10)

    assert torch.all(
        torch.isclose(SVD.frommatrices(M1, M2).full(), M1 @ M2)
    ), f"fails for seed {seed}"


def test_svd():
    check_svd(random.choice(range(1000)))
