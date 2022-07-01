"""Computing [QKV] contribution/composition terms."""
from __future__ import annotations

from typing import Union, Optional

import torch
import numpy as np
from tqdm import tqdm

from .svd import SVD
from .types import ParamMatrix


def frobnorm(M: Union[np.ndarray, SVD]) -> np.float32:
    """Frobenius norm."""
    if isinstance(M, np.ndarray):
        return np.linalg.norm(M.ravel())
    elif isinstance(M, SVD):
        return np.linalg.norm(M.S)
    else:
        raise ValueError(f"unexpected type {type(M)}")


def singularvals(M: Union[np.ndarray, SVD]) -> np.ndarray:
    """Singular values."""
    if isinstance(M, np.ndarray):
        return np.linalg.svd(M, full_matrices=False)[1]
    elif isinstance(M, SVD):
        return M.S
    else:
        raise ValueError(f"unexpected type {type(M)}")


def removemean(M: Union[np.ndarray, SVD], method="direct") -> Union[np.ndarray, SVD]:
    """Removes the mean from each column of a given matrix/vector."""
    if method == "direct":
        assert isinstance(
            M, np.ndarray
        ), "direct method only implemented for np.ndarray"
        return M - M.mean(0)

    # less direct, more fun, works with svd.SVD objects
    elif method == "matrix multiply":
        n = M.shape[0]
        E = np.eye(n, dtype=M.dtype) - 1 / n

        # transposes ensure that M's __matmul__ method is called
        # this helps to support svd.SVD objects
        return (M.T @ E).T

    else:
        raise ValueError(f"unrecognized method: {method}")


def basecomposition(
    dst_M: Union[np.ndarray, SVD],
    src_M: Union[np.ndarray, SVD],
    center: bool = True,
    denom: str = "wiki",
) -> np.float32:
    """Computes composition assuming that the matrices are properly transposed."""
    if center:
        src_M = removemean(src_M)

    numerator = frobnorm(dst_M @ src_M)
    denominator = compute_denom(dst_M, src_M, denom=denom)

    return numerator / denominator


def compute_denom(
    dst_M: ParamMatrix, src_M: ParamMatrix, denom: str = "wiki"
) -> np.float32:
    if denom == "orig":
        denominator = frobnorm(dst_M) * frobnorm(src_M)

    elif denom == "wiki":
        d1 = singularvals(dst_M)
        d2 = singularvals(src_M)

        if len(d1) > len(d2):
            d2 = zeropad(d2, len(d1))
        elif len(d2) > len(d1):
            d1 = zeropad(d1, len(d2))

        denominator = frobnorm(d1 * d2)

    elif denom == "none" or denom == "one":
        denominator = 1.0

    elif denom == "dst_orig":
        denominator = frobnorm(dst_M)

    elif denom == "dst_wiki":
        d1 = singularvals(dst_M)
        d2 = singularvals(src_M)

        if len(d1) > len(d2):
            d2 = zeropad(d2, len(d1))
        elif len(d2) > len(d1):
            d1 = zeropad(d1, len(d2))

        denominator = frobnorm(d1 * (d2 != 0))

    else:
        raise ValueError(f"unrecognized denominator: {denom}")

    return denominator


def zeropad(v: np.ndarray, newlen: int):
    assert len(v) <= newlen, f"cannot zeropad to a smaller length {len(v)}->{newlen}"

    newv = np.zeros((newlen,), dtype=v.dtype)
    newv[: len(v)] = v[:]

    return newv


def composition_singularvals(
    dst_M: Union[np.ndarray, SVD],
    src_M: Union[np.ndarray, SVD],
    center: bool = True,
    normalize: bool = True,
    wikidenom: bool = False,
) -> np.ndarray:
    """Computes the singular values from composition (but doesn't collapse them)."""
    if center:
        src_M = removemean(src_M)

    values = singularvals(dst_M @ src_M)

    if normalize and not wikidenom:
        values /= frobnorm(dst_M) * frobnorm(src_M)
    elif normalize:
        values /= np.linalg.norm(singularvals(dst_M) * singularvals(src_M))

    return values


def Qcomposition(
    dst_QK: np.ndarray, src_OV: np.ndarray, center: bool = True
) -> np.float32:
    """Computes Q composition."""
    return basecomposition(dst_QK.T, src_OV, center=center)


def Kcomposition(
    dst_QK: np.ndarray, src_OV: np.ndarray, center: bool = True
) -> np.float32:
    """Computes K composition."""
    return basecomposition(dst_QK, src_OV, center=center)


def Vcomposition(
    dst_OV: np.ndarray, src_OV: np.ndarray, center: bool = True
) -> np.float32:
    """Computes V composition."""
    return basecomposition(dst_OV, src_OV, center=center)


def MLP_in_contribution(
    dst_MLPin: np.ndarray, src_OV: np.ndarray, center: bool = True
) -> np.float32:
    """Computes contribution to the MLP hidden layer."""
    return basecomposition(dst_MLPin, src_OV, center=center)


def init_baseline_dist(
    weight: ParamMatrix,
    dest_shape: tuple[int, int],
    num_samples: int = 100,
    init_weight: bool = False,
    wikidenom: bool = True,
    sample_rank: Optional[int] = None,
) -> np.ndarray:
    """Baseline composition terms from randomly-initialized weight matrices."""

    # xavier uniform by default to match unseal
    alpha = np.sqrt(6 / (weight.shape[0] + weight.shape[1]))
    dist = torch.distributions.Uniform(-alpha, alpha)

    if init_weight:
        if sample_rank is None:
            weight = dist.rsample(weight.shape).numpy()
        else:
            weight1 = dist.rsample((weight.shape[0], sample_rank)).numpy()
            weight2 = dist.rsample((sample_rank, weight.shape[1])).numpy()

            weight = weight1 @ weight2

    terms = np.empty((num_samples,), dtype=np.float32)
    for i in tqdm(range(num_samples)):

        # sampling a random matrix
        if sample_rank is None:
            newmat = dist.rsample(dest_shape).numpy()
        else:
            newmat1 = dist.rsample((dest_shape[0], sample_rank)).numpy()
            newmat2 = dist.rsample((sample_rank, dest_shape[1])).numpy()
            newmat = newmat1 @ newmat2

        terms[i] = basecomposition(weight, newmat, center=False, wikidenom=wikidenom)

    return terms
