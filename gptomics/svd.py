"""Representing matrix multiplications as SVD factorizations."""
from __future__ import annotations

from typing import Union

import torch


class SVD:
    def __init__(self, U: torch.Tensor, S: torch.Tensor, Vt: torch.Tensor):
        self.U = U
        self.S = S
        self.Vt = Vt

    def __matmul__(self, other: Union[SVD, torch.Tensor]) -> SVD:
        if isinstance(other, torch.Tensor):
            other = SVD.frommatrix(other)

        # "inner core" of the product, p -> prime
        Sp = (self.S[:, None] * self.Vt) @ (other.U * other.S)

        Up, Sp, Vtp = torch.linalg.svd(Sp, full_matrices=False)

        return SVD(self.U @ Up, Sp, Vtp @ other.Vt)

    def __rmatmul__(self, other: Union[SVD, torch.Tensor]) -> SVD:  # type: ignore
        if isinstance(other, torch.Tensor):
            other = SVD.frommatrix(other)

        return other @ self

    @classmethod
    def frommatrix(cls, M: torch.Tensor) -> SVD:
        if len(M.shape) == 1:
            norm = torch.linalg.norm(M)
            return SVD(
                (M / norm).reshape(-1, 1),
                torch.tensor([norm], dtype=M.dtype, device=M.device),
                torch.tensor([1], dtype=M.dtype, device=M.device).reshape(1, 1),
            )

        U, S, Vt = torch.linalg.svd(M, full_matrices=False)

        return SVD(U, S, Vt)

    @classmethod
    def frommatrices(cls, *Ms: torch.Tensor) -> SVD:
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

    def full(self) -> torch.Tensor:
        return (self.U * self.S) @ self.Vt

    def __repr__(self) -> str:
        return f"SVD <U: {self.U.shape}, S: {self.S.shape}, Vt: {self.Vt.shape}>"

    @property
    def shape(self) -> tuple[int, int]:
        return (self.U.shape[0], self.Vt.shape[1])

    @property
    def dtype(self) -> torch.dtype:
        return self.U.dtype

    @property
    def device(self) -> torch.device:
        return self.U.device
