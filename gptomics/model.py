"""Model class

Models are designed to achieve two functions:
(1) Allow a user to easily extract parameters of interest
(2) Expose the input/output relationships within its architecture
"""
from __future__ import annotations

import json
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from .svd import SVD
from .types import ParamMatrix
from . import transformersio, gptneo, torchio


CACHESIZE = 16


class Layer:
    """A layer Enum with an index attribute to distinguish multiples.

    The index attribute refers to LAYER index (e.g., mutliple layer norms),
    and not attention head indices.
    """

    layername = ""

    def __init__(self, index: int = 0):
        self.index = index

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.index == other.index

    def __hash__(self):
        return hash(f"{self.layername}{self.index}")


class SelfAttention(Layer):

    layername = "att_head"


class MLP(Layer):

    layername = "mlp"


class LayerNorm(Layer):

    layername = "layer norm"


class Model:
    """Model virtual class."""

    def __init__(self, gpu_svd: bool = False):
        self.gpu_svd = gpu_svd

    def qk(self, block: int, head: int, factored: bool = False) -> ParamMatrix:
        """Extracts the QK matrix from a model."""
        raise NotImplementedError

    def ov(self, block: int, head: int, factored: bool = False) -> ParamMatrix:
        """Extracts the OV matrix from a model."""
        raise NotImplementedError

    def out_bias(self, block: int, factored: bool = False) -> ParamMatrix:
        """Extracts the output bias vector from a model (if it exists)."""
        raise NotImplementedError

    def mlp_in(self, block: int, factored: bool = False) -> ParamMatrix:
        """Extracts the MLP input matrix from a model."""
        raise NotImplementedError

    def mlp_out(self, block: int, factored: bool = False) -> ParamMatrix:
        """Extracts the MLP output matrix from a model."""
        raise NotImplementedError

    def mlp_bias_in(self, block: int, factored: bool = False) -> ParamMatrix:
        """Extracts the MLP input bias from a model."""
        raise NotImplementedError

    def mlp_bias_out(self, block: int, factored: bool = False) -> ParamMatrix:
        """Extracts the MLP output bias from a model."""
        raise NotImplementedError

    def ln_weights(self, block: int, factored: bool = False) -> list[ParamMatrix]:
        """Extracts the layer norm weights from a model."""
        raise NotImplementedError

    def ln_biases(self, block: int, factored: bool = False) -> list[ParamMatrix]:
        """Extracts the layer norm biases from a model."""
        raise NotImplementedError

    @property
    def num_blocks(self) -> int:
        raise NotImplementedError

    @property
    def num_heads(self) -> int:
        raise NotImplementedError

    def maybe_factor(self, factor: bool = True, *Ms: np.ndarray) -> ParamMatrix:
        """Factors the matrix using an SVD if desired."""
        if factor:
            return SVD.frommatrices(*Ms, gpu=self.gpu_svd)
        else:
            base = Ms[0]

            for extramat in Ms[1:]:
                base = base @ extramat

            return base

    def sends_input_to(
        self, src_layer: Layer, src_block: int, dst_layer: Layer, dst_block: int
    ) -> bool:
        """Exposes input/output relationships of a model's architecture."""
        raise NotImplementedError

    def block_structure(self) -> list[Layer]:
        """Exposes the order of layers within a transformer block."""
        raise NotImplementedError

    def normalizing_layer(self, layer: Layer) -> Layer:
        """Expose which layer norm gives input to the other layers."""
        raise NotImplementedError


class GPTNeo_HF(Model):
    """GPT-Neo through HuggingFace transformers."""

    def __init__(self, modelname: str, gpu_svd: bool = False):
        super().__init__()
        self.model = transformersio.load_model(modelname)
        self.gpu_svd = gpu_svd

    def qk(self, block: int, head: int, factored: bool = False) -> ParamMatrix:
        config = self.model.config
        head_dim = config.hidden_size // config.num_heads

        assert head < config.num_heads
        assert block < config.num_layers

        attention = self.model.transformer.h[block].attn.attention
        Q = attention.q_proj.weight.data.numpy()
        K = attention.k_proj.weight.data.numpy()

        Qh = Q[head * head_dim : (head + 1) * head_dim, :]
        Kh = K[head * head_dim : (head + 1) * head_dim, :]

        return self.maybe_factor(factored, Qh.T, Kh)

    def ov(self, block: int, head: int, factored: bool = False) -> ParamMatrix:
        config = self.model.config
        head_dim = config.hidden_size // config.num_heads

        assert head < config.num_heads
        assert block < config.num_layers

        attention = self.model.transformer.h[block].attn.attention
        O = attention.out_proj.weight.data.numpy()
        V = attention.v_proj.weight.data.numpy()

        Oh = O[:, head * head_dim : (head + 1) * head_dim]
        Vh = V[head * head_dim : (head + 1) * head_dim, :]

        return self.maybe_factor(factored, Oh, Vh)

    def out_bias(self, block: int, factored: bool = False) -> ParamMatrix:
        return self.maybe_factor(factored, gptneo.out_bias(self.model, block))

    def mlp_in(self, block: int, factored: bool = False) -> ParamMatrix:
        return self.maybe_factor(factored, gptneo.mlp_in(self.model, block))

    def mlp_out(self, block: int, factored: bool = False) -> ParamMatrix:
        return self.maybe_factor(factored, gptneo.mlp_out(self.model, block))

    def mlp_bias_in(self, block: int, factored: bool = False) -> ParamMatrix:
        return self.maybe_factor(factored, gptneo.mlp_bias_in(self.model, block))

    def mlp_bias_out(self, block: int, factored: bool = False) -> ParamMatrix:
        return self.maybe_factor(factored, gptneo.mlp_bias_out(self.model, block))

    def ln_weights(self, block: int, factored: bool = False) -> list[ParamMatrix]:
        weights = gptneo.ln_weights(self.model, block)
        return [
            self.maybe_factor(factored, weights[0]),
            self.maybe_factor(factored, weights[1]),
        ]

    def ln_biases(self, block: int, factored: bool = False) -> list[ParamMatrix]:
        biases = gptneo.ln_biases(self.model, block)
        return [
            self.maybe_factor(factored, biases[0]),
            self.maybe_factor(factored, biases[1]),
        ]

    @property
    def num_blocks(self) -> int:
        return self.model.config.num_layers

    @property
    def num_heads(self) -> int:
        return self.model.config.num_heads

    def block_structure(self) -> list[Layer]:
        return [LayerNorm(0), SelfAttention(), LayerNorm(1), MLP()]

    def sends_input_to(
        self, src_layer: Layer, src_block: int, dst_layer: Layer, dst_block: int
    ) -> bool:
        if src_block > dst_block:
            return False

        elif src_block == dst_block:

            if dst_layer == LayerNorm(0):
                return False
            elif dst_layer == SelfAttention():
                return src_layer == LayerNorm(0)
            elif dst_layer == LayerNorm(1):
                return src_layer == SelfAttention()
            elif dst_layer == MLP():
                return src_layer in [SelfAttention(), LayerNorm(1)]
            else:
                raise ValueError(f"unknown dst_layer: {dst_layer}")

        else:  # src_block < dst_block
            return not isinstance(src_layer, LayerNorm)

    def normalizing_layer(self, layer: Layer) -> Layer:
        assert layer in self.block_structure(), f"unknown layer: {layer}"
        assert layer in [SelfAttention(), MLP()], f"not a normalizer layer: {layer}"

        if layer == SelfAttention():
            return LayerNorm(0)
        else:  # layer == MLP():
            return LayerNorm(1)


class CachedFileModel(Model):
    """A parameter bag that reads tensors from a pytorch_model.bin file.

    See torchio. Tensors are cached when reading from disk to minimize
    the number of disk accesses.
    """

    def __init__(
        self, config_filename: str, param_filename: str, gpu_svd: bool = False
    ):
        self.config = self.read_config(config_filename)
        self.param_filename = param_filename
        self.tensor_names = torchio.read_tensor_names(param_filename)
        self.gpu_svd = gpu_svd

    @lru_cache(maxsize=50)
    def fetch_tensor(self, tensorname: str) -> np.ndarray:
        """Fetches a tensor from disk and caches it."""
        assert tensorname in self.tensor_names, f"tensor {tensorname} not found"
        tensor = torchio.read_tensor(self.param_filename, tensorname)

        return tensor.data.numpy()

    def read_config(self, filename: str) -> SimpleNamespace:
        """Reads a config json file."""
        cfg = SimpleNamespace()

        with open(filename) as f:
            content = json.load(f)

        for k, v in content.items():
            setattr(cfg, k, v)

        return cfg


class GPTJ(CachedFileModel):
    """A CachedFileModel for interacting with GPT-J."""

    @lru_cache(maxsize=CACHESIZE)
    def qk(self, block: int, head: int, factored: bool = True) -> ParamMatrix:
        assert head < self.config.n_head, f"head #{head} does not exist"
        assert block < self.config.n_layer, f"block #{block} does not exist"

        # _f : the "full" set of parameters across heads
        Qf = self.fetch_tensor(f"transformer.h.{block}.attn.q_proj.weight")
        Kf = self.fetch_tensor(f"transformer.h.{block}.attn.k_proj.weight")

        head_dim = self.config.n_embd // self.config.n_head

        Q = Qf[head * head_dim : (head + 1) * head_dim, :]
        K = Kf[head * head_dim : (head + 1) * head_dim, :]

        return self.maybe_factor(factored, Q.T, K)

    @lru_cache(maxsize=CACHESIZE)
    def ov(self, block: int, head: int, factored: bool = True) -> ParamMatrix:
        assert head < self.config.n_head, f"head #{head} does not exist"
        assert block < self.config.n_layer, f"block #{block} does not exist"

        # _f : the "full" set of parameters across heads
        Of = self.fetch_tensor(f"transformer.h.{block}.attn.out_proj.weight")
        Vf = self.fetch_tensor(f"transformer.h.{block}.attn.v_proj.weight")

        head_dim = self.config.n_embd // self.config.n_head

        O = Of[:, head * head_dim : (head + 1) * head_dim]
        V = Vf[head * head_dim : (head + 1) * head_dim, :]

        return self.maybe_factor(factored, O, V)

    @lru_cache(maxsize=CACHESIZE)
    def out_bias(self, block: int, factored: bool = True):  # type: ignore[override]
        return None

    @lru_cache(maxsize=CACHESIZE)
    def mlp_in(self, block: int, factored: bool = True) -> ParamMatrix:
        assert block < self.config.n_layer, f"block #{block} does not exist"

        M = self.fetch_tensor(f"transformer.h.{block}.mlp.fc_in.weight")

        return self.maybe_factor(factored, M)

    @lru_cache(maxsize=CACHESIZE)
    def mlp_out(self, block: int, factored: bool = True) -> ParamMatrix:
        assert block < self.config.n_layer, f"block #{block} does not exist"

        M = self.fetch_tensor(f"transformer.h.{block}.mlp.fc_out.weight")

        return self.maybe_factor(factored, M)

    @lru_cache(maxsize=CACHESIZE)
    def mlp_bias_in(self, block: int, factored: bool = True) -> ParamMatrix:
        assert block < self.config.n_layer, f"block #{block} does not exist"

        M = self.fetch_tensor(f"transformer.h.{block}.mlp.fc_in.bias")

        return self.maybe_factor(factored, M)

    @lru_cache(maxsize=CACHESIZE)
    def mlp_bias_out(self, block: int, factored: bool = True) -> ParamMatrix:
        assert block < self.config.n_layer, f"block #{block} does not exist"

        M = self.fetch_tensor(f"transformer.h.{block}.mlp.fc_out.bias")

        return self.maybe_factor(factored, M)

    @lru_cache(maxsize=CACHESIZE)
    def ln_weights(self, block: int, factored: bool = True) -> list[ParamMatrix]:
        assert block < self.config.n_layer, f"block #{block} does not exist"

        # only one LN for GPT-J since att and mlp applied in parallel
        bias = self.fetch_tensor(f"transformer.h.{block}.ln_1.weight")

        return [self.maybe_factor(factored, bias)]

    @lru_cache(maxsize=CACHESIZE)
    def ln_biases(self, block: int, factored: bool = True) -> list[ParamMatrix]:
        assert block < self.config.n_layer, f"block #{block} does not exist"

        # only one LN for GPT-J since att and mlp applied in parallel
        bias = self.fetch_tensor(f"transformer.h.{block}.ln_1.bias")

        return [self.maybe_factor(factored, bias)]

    @property
    def num_blocks(self) -> int:
        return self.config.n_layer

    @property
    def num_heads(self) -> int:
        return self.config.n_head

    def block_structure(self) -> list[Layer]:
        return [LayerNorm(), SelfAttention(), MLP()]

    def sends_input_to(
        self, src_layer: Layer, src_block: int, dst_layer: Layer, dst_block: int
    ) -> bool:
        if src_block > dst_block:
            return False

        elif src_block == dst_block:
            if dst_layer == LayerNorm(0):
                return False

            elif dst_layer in [SelfAttention(), MLP()]:
                return src_layer == LayerNorm(0)

            else:
                raise ValueError(f"unrecognized layer: {dst_layer}")

        else:  # src_block < dst_block
            return not isinstance(src_layer, LayerNorm)

    def normalizing_layer(self, layer: Layer) -> Layer:
        assert layer in self.block_structure(), f"unknown layer: {layer}"
        assert layer in [SelfAttention(), MLP()], f"not a normalizer layer: {layer}"

        return LayerNorm(0)


class GPTNeo(CachedFileModel, GPTNeo_HF):
    """A CachedFileModel for interacting with GPT-Neo models."""

    @lru_cache(maxsize=CACHESIZE)
    def qk(self, block: int, head: int, factored: bool = True) -> ParamMatrix:
        assert head < self.num_heads, f"head #{head} does not exist"
        assert block < self.num_blocks, f"block #{block} does not exist"
        head_dim = self.config.hidden_size // self.config.num_heads

        # _f : the "full" set of parameters across heads
        Qf = self.fetch_tensor(f"transformer.h.{block}.attn.attention.q_proj.weight")
        Kf = self.fetch_tensor(f"transformer.h.{block}.attn.attention.k_proj.weight")

        Q = Qf[head * head_dim : (head + 1) * head_dim, :]
        K = Kf[head * head_dim : (head + 1) * head_dim, :]

        return self.maybe_factor(factored, Q.T, K)

    @lru_cache(maxsize=CACHESIZE)
    def ov(self, block: int, head: int, factored: bool = True) -> ParamMatrix:
        assert head < self.num_heads, f"head #{head} does not exist"
        assert block < self.num_blocks, f"block #{block} does not exist"
        head_dim = self.config.hidden_size // self.config.num_heads

        # _f : the "full" set of parameters across heads
        Of = self.fetch_tensor(f"transformer.h.{block}.attn.attention.out_proj.weight")
        Vf = self.fetch_tensor(f"transformer.h.{block}.attn.attention.v_proj.weight")

        O = Of[:, head * head_dim : (head + 1) * head_dim]
        V = Vf[head * head_dim : (head + 1) * head_dim, :]

        return self.maybe_factor(factored, O, V)

    @lru_cache(maxsize=CACHESIZE)
    def out_bias(self, block: int, factored: bool = False) -> ParamMatrix:
        out_bias = self.fetch_tensor(
            f"transformer.h.{block}.attn.attention.out_proj.bias"
        )

        return self.maybe_factor(factored, out_bias)

    @lru_cache(maxsize=CACHESIZE)
    def mlp_in(self, block: int, factored: bool = True) -> ParamMatrix:
        assert block < self.num_blocks, f"block #{block} does not exist"

        M = self.fetch_tensor(f"transformer.h.{block}.mlp.c_fc.weight")

        return self.maybe_factor(factored, M)

    @lru_cache(maxsize=CACHESIZE)
    def mlp_out(self, block: int, factored: bool = True) -> ParamMatrix:
        assert block < self.num_blocks, f"block #{block} does not exist"

        M = self.fetch_tensor(f"transformer.h.{block}.mlp.c_proj.weight")

        return self.maybe_factor(factored, M)

    @lru_cache(maxsize=CACHESIZE)
    def mlp_bias_in(self, block: int, factored: bool = True) -> ParamMatrix:
        assert block < self.num_blocks, f"block #{block} does not exist"

        M = self.fetch_tensor(f"transformer.h.{block}.mlp.c_fc.bias")

        return self.maybe_factor(factored, M)

    @lru_cache(maxsize=CACHESIZE)
    def mlp_bias_out(self, block: int, factored: bool = True) -> ParamMatrix:
        assert block < self.num_blocks, f"block #{block} does not exist"

        M = self.fetch_tensor(f"transformer.h.{block}.mlp.c_proj.bias")

        return self.maybe_factor(factored, M)

    @lru_cache(maxsize=CACHESIZE)
    def ln_weights(  # type: ignore[override]
        self, block: int, factored: bool = True
    ) -> tuple[ParamMatrix, ParamMatrix]:
        assert block < self.num_blocks, f"block #{block} does not exist"

        weights = (
            self.fetch_tensor(f"transformer.h.{block}.ln_1.weight"),
            self.fetch_tensor(f"transformer.h.{block}.ln_2.weight"),
        )

        return (
            self.maybe_factor(factored, weights[0]),
            self.maybe_factor(factored, weights[1]),
        )

    @lru_cache(maxsize=CACHESIZE)
    def ln_biases(  # type: ignore[override]
        self, block: int, factored: bool = True
    ) -> tuple[ParamMatrix, ParamMatrix]:
        assert block < self.num_blocks, f"block #{block} does not exist"

        biases = (
            self.fetch_tensor(f"transformer.h.{block}.ln_1.bias"),
            self.fetch_tensor(f"transformer.h.{block}.ln_2.bias"),
        )

        return (
            self.maybe_factor(factored, biases[0]),
            self.maybe_factor(factored, biases[1]),
        )

    @property
    def num_blocks(self) -> int:
        return self.config.num_layers

    @property
    def num_heads(self) -> int:
        return self.config.num_heads


def model_by_name(modelname: str, gpu_svd: bool = False):
    """Instantiate Models by simplified names using a default implementation."""
    known_names = ["EleutherAI/gpt-neo-125M"]
    assert modelname in known_names, f"unknown model name: {modelname}"

    if modelname == "EleutherAI/gpt-neo-125M":
        return GPTNeo_HF("EleutherAI/gpt-neo-125M", gpu_svd=gpu_svd)
