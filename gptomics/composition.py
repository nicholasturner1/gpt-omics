"""Computing [QKV] contribution/composition terms."""
from __future__ import annotations

import math
from typing import Union, Optional

import torch
from tqdm import tqdm

from .svd import SVD
from .types import ParamMatrix


def frobnorm(M: Union[torch.Tensor, SVD]) -> float:
    """Frobenius norm."""
    if isinstance(M, torch.Tensor):
        return torch.linalg.norm(M.ravel()).item()
    elif isinstance(M, SVD):
        return torch.linalg.norm(M.S).item()
    else:
        raise ValueError(f"unexpected type {type(M)}")


def singularvals(M: Union[torch.Tensor, SVD]) -> torch.Tensor:
    """Singular values."""
    if isinstance(M, torch.Tensor):
        return SVD.frommatrix(M).S
    elif isinstance(M, SVD):
        return M.S
    else:
        raise ValueError(f"unexpected type {type(M)}")


def removemean(
    M: Union[torch.Tensor, SVD], method="direct"
) -> Union[torch.Tensor, SVD]:
    """Removes the mean from each column of a given matrix/vector."""
    if method == "direct":
        assert isinstance(
            M, torch.Tensor
        ), "direct method only implemented for torch.Tensor"
        return M - M.mean(0)

    # less direct, more fun, works with svd.SVD objects
    elif method == "matrix multiply":
        n = M.shape[0]
        E = torch.eye(n, dtype=M.dtype, device=M.device) - 1 / n

        return E @ M

    else:
        raise ValueError(f"unrecognized method: {method}")


def basecomposition(
    dst_M: Union[torch.Tensor, SVD],
    src_M: Union[torch.Tensor, SVD],
    center: bool = True,
    denom: str = "wiki",
) -> float:
    """Computes composition assuming that the matrices are properly transposed."""
    if center:
        src_M = removemean(src_M)

    numerator = frobnorm(dst_M @ src_M)
    denominator = compute_denom(dst_M, src_M, denom=denom)

    return numerator / denominator


def compute_denom(dst_M: ParamMatrix, src_M: ParamMatrix, denom: str = "wiki") -> float:
    if denom == "orig":
        denominator = frobnorm(dst_M) * frobnorm(src_M)

    elif denom == "wiki":
        d1 = singularvals(dst_M).ravel()
        d2 = singularvals(src_M).ravel()

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


def zeropad(v: torch.Tensor, newlen: int):
    """Pads the end of a vector with zeros for elementwise multiplication."""
    assert len(v) <= newlen, f"cannot zeropad to a smaller length {len(v)}->{newlen}"

    newv = torch.zeros((newlen,), dtype=v.dtype, device=v.device)
    newv[: len(v)] = v[:]

    return newv


def composition_singularvals(
    dst_M: Union[torch.Tensor, SVD],
    src_M: Union[torch.Tensor, SVD],
    center: bool = True,
    normalize: bool = True,
    wikidenom: bool = False,
) -> torch.Tensor:
    """Computes the singular values from composition (but doesn't collapse them)."""
    if center:
        src_M = removemean(src_M)

    values = singularvals(dst_M @ src_M)

    if normalize and not wikidenom:
        values /= frobnorm(dst_M) * frobnorm(src_M)
    elif normalize:
        values /= torch.linalg.norm(singularvals(dst_M) * singularvals(src_M))

    return values


def Qcomposition(
    dst_QK: torch.Tensor, src_OV: torch.Tensor, center: bool = True
) -> float:
    """Computes Q composition."""
    return basecomposition(dst_QK.T, src_OV, center=center)


def Kcomposition(
    dst_QK: torch.Tensor, src_OV: torch.Tensor, center: bool = True
) -> float:
    """Computes K composition."""
    return basecomposition(dst_QK, src_OV, center=center)


def Vcomposition(
    dst_OV: torch.Tensor, src_OV: torch.Tensor, center: bool = True
) -> float:
    """Computes V composition."""
    return basecomposition(dst_OV, src_OV, center=center)


def MLP_in_contribution(
    dst_MLPin: torch.Tensor, src_OV: torch.Tensor, center: bool = True
) -> float:
    """Computes contribution to the MLP hidden layer."""
    return basecomposition(dst_MLPin, src_OV, center=center)


def init_baseline_dist(
    weight: ParamMatrix,
    dest_shape: tuple[int, int],
    num_samples: int = 100,
    init_weight: bool = False,
    denom: str = "wiki",
    sample_rank: Optional[int] = None,
) -> torch.Tensor:
    """Baseline composition terms from randomly-initialized weight matrices."""

    # xavier uniform by default to match unseal
    alpha = math.sqrt(6 / (weight.shape[0] + weight.shape[1]))
    dist = torch.distributions.Uniform(-alpha, alpha)

    if init_weight:
        if sample_rank is None:
            weight = dist.rsample(weight.shape).numpy()
        else:
            weight1 = dist.rsample((weight.shape[0], sample_rank)).numpy()
            weight2 = dist.rsample((sample_rank, weight.shape[1])).numpy()

            weight = weight1 @ weight2

    terms = torch.empty((num_samples,), dtype=torch.float32)
    for i in tqdm(range(num_samples)):

        # sampling a random matrix
        if sample_rank is None:
            newmat = dist.rsample(dest_shape).numpy()
        else:
            newmat1 = dist.rsample((dest_shape[0], sample_rank)).numpy()
            newmat2 = dist.rsample((sample_rank, dest_shape[1])).numpy()
            newmat = newmat1 @ newmat2

        terms[i] = basecomposition(weight, newmat, center=False, denom=denom)

    return terms
