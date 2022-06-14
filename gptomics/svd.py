"""Representing matrix multiplications as SVD factorizations."""
from __future__ import annotations

from typing import Union

import numpy as np


class SVD:
    def __init__(self, U: np.ndarray, S: np.ndarray, Vt: np.ndarray):
        self.U = U
        self.S = S
        self.Vt = Vt

    def __matmul__(self, other: Union[SVD, np.ndarray]) -> SVD:
        if isinstance(other, np.ndarray):
            other = SVD.frommatrix(other)

        # "inner core" of the product, p -> prime
        Sp = (self.S[:, np.newaxis] * self.Vt) @ (other.U * other.S)

        Up, Sp, Vtp = np.linalg.svd(Sp, full_matrices=False)

        return SVD(self.U @ Up, Sp, Vtp @ other.Vt)

    @classmethod
    def frommatrix(cls, M: np.ndarray) -> SVD:
        if len(M.shape) == 1:
            norm = np.linalg.norm(M)
            return SVD(
                (M / norm).reshape(-1, 1),
                np.array([norm], dtype=M.dtype),
                np.array([1], dtype=M.dtype).reshape(1, 1),
            )

        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        return SVD(U, S, Vt)

    @classmethod
    def frommatrices(cls, *Ms: np.ndarray) -> SVD:
        svds = [SVD.frommatrix(M) for M in Ms]

        if len(svds) == 1:
            return svds[0]

        else:  # multiple Ms
            base = svds[0]
            for svd in svds[1:]:
                base = base @ svd

            return base

    @property
    def T(self) -> SVD:
        return SVD(self.Vt.T, self.S, self.U.T)

    def full(self) -> np.ndarray:
        return (self.U * self.S) @ self.Vt

    def __repr__(self) -> str:
        return f"SVD <U: {self.U.shape}, S: {self.S.shape}, Vt: {self.Vt.shape}>"

    @property
    def shape(self) -> tuple[int, int]:
        return (self.U.shape[0], self.Vt.shape[1])

    @property
    def dtype(self) -> type:
        return self.U.dtype
