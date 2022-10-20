from __future__ import annotations

from typing import Union

import torch
from .svd import SVD

ParamMatrix = Union[torch.Tensor, SVD]
