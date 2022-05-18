"""Computing [QKV] contribution/composition terms."""
from __future__ import annotations

import numpy as np

from .svd import SVD


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


def removemean(M: np.ndarray, method="direct") -> np.ndarray:
    """Removes the mean from each column of a given matrix/vector."""
    if method == "direct":
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
    dst_M: np.ndarray, src_M: np.ndarray, center: bool = True, wikidenom: bool = False,
) -> np.float32:
    """Computes composition assuming that the matrices are properly transposed."""
    if center:
        src_M = removemean(src_M)

    numerator = frobnorm(dst_M @ src_M)

    if not wikidenom:
        denominator = frobnorm(dst_M) * frobnorm(src_M)
    else:
        denominator = np.linalg.norm(singularvals(dst_M) * singularvals(src_M))

    return numerator / denominator


def composition_singularvals(
    dst_M: np.ndarray,
    src_M: np.ndarray,
    center: bool = True,
    normalize: bool = True,
    wikidenom: bool = False,
) -> np.ndarray:
    """Computes the singular values from composition (but doesn't collapse them)."""
    if center:
        src_M = removemean(src_M)

    values = singularvals(dst_M @ src_M)

    if normalize and not wikidenom:
        values /= (frobnorm(dst_M) * frobnorm(src_M))
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
