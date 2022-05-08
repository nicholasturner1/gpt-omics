"""Model class

Models are designed to achieve two functions:
(1) Allow a user to easily extract parameters of interest
(2) Expose the input/output relationships within its architecture
"""
from __future__ import annotations

import json
from typing import Union
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from .svd import SVD
from . import transformersio, gptneo, torchio


GPTJ_CACHESIZE = 16


class Model:
    """ParameterBag virtual class."""

    def __init__(self):
        pass

    def QK(
        self, layer: int, head: int, factored: bool = False
    ) -> Union[SVD, np.ndarray]:
        """Extracts the QK matrix from a model."""
        raise NotImplementedError

    def OV(
        self, layer: int, head: int, factored: bool = False
    ) -> Union[SVD, np.ndarray]:
        """Extracts the OV matrix from a model."""
        raise NotImplementedError

    def Obias(self, layer: int, factored: bool = False) -> Union[SVD, np.ndarray]:
        """Extracts the output bias vector from a model (if it exists)."""
        raise NotImplementedError

    def MLPin(self, layer: int, factored: bool = False) -> Union[SVD, np.ndarray]:
        """Extracts the MLP input matrix from a model."""
        raise NotImplementedError

    def MLPout(self, layer: int, factored: bool = False) -> Union[SVD, np.ndarray]:
        """Extracts the MLP output matrix from a model."""
        raise NotImplementedError

    def MLPbias_in(self, layer: int, factored: bool = False) -> Union[SVD, np.ndarray]:
        """Extracts the MLP input bias from a model."""
        raise NotImplementedError

    def MLPbias_out(self, layer: int, factored: bool = False) -> Union[SVD, np.ndarray]:
        """Extracts the MLP output bias from a model."""
        raise NotImplementedError

    def layernorm_biases(
        self, layer: int, factored: bool = False
    ) -> tuple[Union[SVD, np.ndarray], Union[SVD, np.ndarray]]:
        """Extracts the Layer norm biases from a model."""
        raise NotImplementedError

    @property
    def num_layers(self) -> int:
        raise NotImplementedError

    @property
    def num_heads(self) -> int:
        raise NotImplementedError

    def maybe_factor(
        self, factor: bool = True, *Ms: np.ndarray
    ) -> Union[SVD, np.ndarray]:
        """Factors the matrix using an SVD if desired."""
        if factor:
            return SVD.frommatrices(*Ms)
        else:
            base = Ms[0]

            for extramat in Ms[1:]:
                base = base @ extramat

            return base

    def sends_input_to(
        self: Model,
        src_type: str,
        src_layer: int,
        dst_type: str,
        dst_layer: int
    ) -> bool:
        """Exposes input/output relationships of a model's architecture."""
        raise NotImplementedError


class GPTNeo_HF(Model):
    """GPT-Neo through HuggingFace transformers."""

    def __init__(self, modelname: str):
        super().__init__()
        self.model = transformersio.load_model(modelname)

    def QK(
        self, layer: int, head: int, factored: bool = False
    ) -> Union[SVD, np.ndarray]:
        return self.maybe_factor(factored, gptneo.QK(self.model, layer, head))

    def OV(
        self, layer: int, head: int, factored: bool = False
    ) -> Union[SVD, np.ndarray]:
        return self.maybe_factor(factored, gptneo.OV(self.model, layer, head))

    def Obias(self, layer: int, factored: bool = False) -> Union[SVD, np.ndarray]:
        return self.maybe_factor(factored, gptneo.Obias(self.model, layer))

    def MLPin(self, layer: int, factored: bool = False) -> Union[SVD, np.ndarray]:
        return self.maybe_factor(factored, gptneo.MLPin(self.model, layer))

    def MLPout(self, layer: int, factored: bool = False) -> Union[SVD, np.ndarray]:
        return self.maybe_factor(factored, gptneo.MLPout(self.model, layer))

    def MLPbias_in(self, layer: int, factored: bool = False) -> Union[SVD, np.ndarray]:
        return self.maybe_factor(factored, gptneo.MLPbias_in(self.model, layer))

    def MLPbias_out(self, layer: int, factored: bool = False) -> Union[SVD, np.ndarray]:
        return self.maybe_factor(factored, gptneo.MLPbias_out(self.model, layer))

    def layernorm_biases(
        self, layer: int, factored: bool = False
    ) -> tuple[Union[SVD, np.ndarray], Union[SVD, np.ndarray]]:
        biases = gptneo.layernorm_biases(self.model, layer)
        return (
            self.maybe_factor(factored, biases[0]),
            self.maybe_factor(factored, biases[1]),
        )

    @property
    def num_layers(self: Model) -> int:
        return self.model.config.num_layers

    @property
    def num_heads(self: Model) -> int:
        return self.model.config.num_heads

    def sends_input_to(
        self: Model,
        src_type: str,
        src_layer: int,
        dst_type: str,
        dst_layer: int
    ) -> bool:
        if src_layer > dst_layer:
            return False

        elif src_layer == dst_layer:
            # attention heads are first
            if dst_type == "att_head":
                return False

            if dst_type == "mlp_weight":
                return src_type in ["layernorm_bias1", "att_head"]

        else:  # src_layer < dst_layer
            return True


class CachedFileModel(Model):
    """A parameter bag that reads tensors from a pytorch_model.bin file.

    See torchio. Tensors are cached when reading from disk to minimize
    the number of disk accesses.
    """

    def __init__(self, config_filename: str, param_filename: str):
        super().__init__()
        self.config = self.read_config(config_filename)
        self.param_filename = param_filename
        self.tensor_names = torchio.read_tensor_names(param_filename)

    @lru_cache(maxsize=50)
    def fetch_tensor(self, tensorname: str) -> np.ndarray:
        """Fetches a tensor from disk and caches it."""
        assert tensorname in self.tensor_names, f"tensor {tensorname} not found"
        tensor = torchio.read_tensor(self.param_filename, tensorname)

        return tensor.data.numpy()

    def read_config(self: CachedFileModel, filename: str) -> SimpleNamespace:
        """Reads a config json file."""
        cfg = SimpleNamespace()

        with open(filename) as f:
            content = json.load(f)

        for k, v in content.items():
            setattr(cfg, k, v)

        return cfg


class GPTJ(CachedFileModel):
    """A CachedFileBag for interacting with GPT-J."""

    @lru_cache(maxsize=GPTJ_CACHESIZE)
    def QK(
        self, layer: int, head: int, factored: bool = True
    ) -> Union[SVD, np.ndarray]:
        assert head < self.config.n_head, f"head #{head} does not exist"
        assert layer < self.config.n_layer, f"layer #{layer} does not exist"

        # _f : the "full" set of parameters across heads
        Qf = self.fetch_tensor(f"transformer.h.{layer}.attn.q_proj.weight")
        Kf = self.fetch_tensor(f"transformer.h.{layer}.attn.k_proj.weight")

        head_dim = self.config.n_embd // self.config.n_head

        Q = Qf[head * head_dim : (head + 1) * head_dim, :]
        K = Kf[head * head_dim : (head + 1) * head_dim, :]

        return self.maybe_factor(factored, Q.T, K)

    @lru_cache(maxsize=GPTJ_CACHESIZE)
    def OV(
        self, layer: int, head: int, factored: bool = True
    ) -> Union[SVD, np.ndarray]:
        assert head < self.config.n_head, f"head #{head} does not exist"
        assert layer < self.config.n_layer, f"layer #{layer} does not exist"

        # _f : the "full" set of parameters across heads
        Of = self.fetch_tensor(f"transformer.h.{layer}.attn.out_proj.weight")
        Vf = self.fetch_tensor(f"transformer.h.{layer}.attn.v_proj.weight")

        head_dim = self.config.n_embd // self.config.n_head

        O = Of[head * head_dim : (head + 1) * head_dim, :]
        V = Vf[head * head_dim : (head + 1) * head_dim, :]

        return self.maybe_factor(factored, O, V)

    @lru_cache(maxsize=GPTJ_CACHESIZE)
    def Obias(self, layer: int) -> None:
        return None

    @lru_cache(maxsize=GPTJ_CACHESIZE)
    def MLPin(self, layer: int, factored: bool = True) -> Union[SVD, np.ndarray]:
        assert layer < self.config.n_layer, f"layer #{layer} does not exist"

        M = self.fetch_tensor(f"transformer.h.{layer}.mlp.fc_in.weight")

        return self.maybe_factor(factored, M)

    @lru_cache(maxsize=GPTJ_CACHESIZE)
    def MLPout(self, layer: int, factored: bool = True) -> Union[SVD, np.ndarray]:
        assert layer < self.config.n_layer, f"layer #{layer} does not exist"

        M = self.fetch_tensor(f"transformer.h.{layer}.mlp.fc_out.weight")

        return self.maybe_factor(factored, M)

    @lru_cache(maxsize=GPTJ_CACHESIZE)
    def MLPbias_in(self, layer: int, factored: bool = True) -> Union[SVD, np.ndarray]:
        assert layer < self.config.n_layer, f"layer #{layer} does not exist"

        M = self.fetch_tensor(f"transformer.h.{layer}.mlp.fc_in.bias")

        return self.maybe_factor(factored, M)

    @lru_cache(maxsize=GPTJ_CACHESIZE)
    def MLPbias_out(self, layer: int, factored: bool = True) -> Union[SVD, np.ndarray]:
        assert layer < self.config.n_layer, f"layer #{layer} does not exist"

        M = self.fetch_tensor(f"transformer.h.{layer}.mlp.fc_out.bias")

        return self.maybe_factor(factored, M)

    @lru_cache(maxsize=GPTJ_CACHESIZE)
    def layernorm_biases(
        self, layer: int, factored: bool = True
    ) -> Union[SVD, np.ndarray]:
        assert layer < self.config.n_layer, f"layer #{layer} does not exist"

        # only one bias for GPT-J since att and mlp applied in parallel
        bias = self.fetch_tensor(f"transformer.h.{layer}.ln_1.bias")

        return self.maybe_factor(factored, bias)

    def sends_input_to(
        self: Model,
        src_type: str,
        src_layer: int,
        dst_type: str,
        dst_layer: int
    ) -> bool:
        if src_layer > dst_layer:
            return False

        elif src_layer == dst_layer:
            return src_type == "layernorm_bias"

        else:  # src_layer < dst_layer
            return True
