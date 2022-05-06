"""ParameterBag - an abstraction for fetching parameters from multiple model types."""
import time
import json
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from . import transformersio, gptneo, torchio
from .svd import SVD


class ParamBag:
    """ParameterBag virtual class."""

    def __init__(self):
        pass

    def QK(self, layer: int, head: int) -> np.ndarray:
        """Extracts the QK matrix from a model."""
        raise NotImplemented()

    def OV(self, layer: int, head: int) -> np.ndarray:
        """Extracts the OV matrix from a model."""
        raise NotImplemented()

    def Obias(self, layer: int) -> np.ndarray:
        """Extracts the output bias vector from a model (if it exists)."""
        raise NotImplemented()

    def MLPin(self, layer: int) -> np.ndarray:
        """Extracts the MLP input matrix from a model."""
        raise NotImplemented()

    def MLPout(self, layer: int) -> np.ndarray:
        """Extracts the MLP output matrix from a model."""
        raise NotImplemented()

    def MLPbias_in(self, layer: int) -> np.ndarray:
        """Extracts the MLP input bias from a model."""
        raise NotImplemented()

    def MLPbias_out(self, layer: int) -> np.ndarray:
        """Extracts the MLP output bias from a model."""
        raise NotImplemented()

    def layernorm_biases(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Extracts the Layer norm biases from a model."""
        raise NotImplemented()


class HuggingFaceBag(ParamBag):
    def __init__(self, modelname: str):
        self.model = transformersio.load_model(modelname)

    def QK(self, layer: int, head: int) -> np.ndarray:
        return gptneo.QK(self.model, layer, head)

    def OV(self, layer: int, head: int) -> np.ndarray:
        return gptneo.OV(self.model, layer, head)

    def Obias(self, layer: int) -> np.ndarray:
        return gptneo.Obias(self.model, layer, head)

    def MLPin(self, layer: int) -> np.ndarray:
        return gptneo.MLPin(self.model, layer)

    def MLPout(self, layer: int) -> np.ndarray:
        return gptneo.MLPout(self.model, layer)

    def MLPbias_in(self, layer: int) -> np.ndarray:
        return gptneo.MLPbias_in(self.model, layer)

    def MLPbias_out(self, layer: int) -> np.ndarray:
        return gptneo.MLPbias_out(self.model, layer)

    def layernorm_biases(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        return gptneo.layernorm_biases(self.model, layer)


class CachedFileBag(ParamBag):
    """A parameter bag that reads tensors from a pytorch_model.bin file.

    See torchio. Tensors are cached when reading from disk to minimize
    the number of disk accesses.
    """

    def __init__(self, config_filename: str, param_filename: str):
        self.config = self.read_config(config_filename)
        self.param_filename = param_filename
        self.tensor_names = torchio.read_tensor_names(param_filename)

    @lru_cache(maxsize=50)
    def fetch_tensor(self, tensorname: str) -> np.ndarray:
        """Fetches a tensor from disk and caches it."""
        assert tensorname in self.tensor_names, f"tensor {tensorname} not found"
        tensor = torchio.read_tensor(self.param_filename, tensorname)

        return tensor.data.numpy()

    def read_config(self: CachedFileBag, filename: str) -> SimpleNamespace:
        """Reads a config json file."""
        cfg = SimpleNamespace()

        with open(filename) as f:
            content = json.load(f)

        for k, v in content.items():
            setattr(cfg, k, v)

        return cfg


class GPTJBag(CachedFileBag):
    """A CachedFileBag for interacting with GPT-J."""

    def QK(self, layer: int, head: int, factored: bool = True) -> np.ndarray:
        assert head < self.config.n_head, f"head #{head} does not exist"
        assert layer < self.config.n_layer, f"layer #{layer} does not exist"

        # _f : the "full" set of parameters across heads
        Qf = self.fetch_tensor(f"transformer.h.{layer}.attn.q_proj.weight")
        Kf = self.fetch_tensor(f"transformer.h.{layer}.attn.k_proj.weight")

        head_dim = self.config.n_embd // self.config.n_head

        Q = Qf[head * head_dim : (head + 1) * head_dim, :]
        K = Kf[head * head_dim : (head + 1) * head_dim, :]
        if factored:
            return SVD.frommatrix(Q).T @ SVD.frommatrix(K)
        else:
            return Q.T @ K

    def OV(self, layer: int, head: int, factored: bool = True) -> np.ndarray:
        assert head < self.config.n_head, f"head #{head} does not exist"
        assert layer < self.config.n_layer, f"layer #{layer} does not exist"

        # _f : the "full" set of parameters across heads
        Of = self.fetch_tensor(f"transformer.h.{layer}.attn.out_proj.weight")
        Vf = self.fetch_tensor(f"transformer.h.{layer}.attn.v_proj.weight")

        head_dim = self.config.n_embd // self.config.n_head

        O = Of[head * head_dim : (head + 1) * head_dim, :]
        V = Vf[head * head_dim : (head + 1) * head_dim, :]
        if factored:
            return SVD.frommatrix(O).T @ SVD.frommatrix(V)
        else:
            return O @ V

    def Obias(self, layer: int) -> np.ndarray:
        return None

    def MLPin(self, layer: int, factored: bool = True) -> np.ndarray:
        assert layer < self.config.n_layer, f"layer #{layer} does not exist"

        M = self.fetch_tensor(f"transformer.h.{layer}.mlp.fc_in.weight")

        if factored:
            return SVD.frommatrix(M)
        else:
            return M

    def MLPout(self, layer: int, factored: bool = True) -> np.ndarray:
        assert layer < self.config.n_layer, f"layer #{layer} does not exist"

        M = self.fetch_tensor(f"transformer.h.{layer}.mlp.fc_out.weight")

        if factored:
            return SVD.frommatrix(M)
        else:
            return M

    def MLPbias_in(self, layer: int, factored: bool = True) -> np.ndarray:
        assert layer < self.config.n_layer, f"layer #{layer} does not exist"

        M = self.fetch_tensor(f"transformer.h.{layer}.mlp.fc_in.bias")

        if factored:
            return SVD.frommatrix(M)
        else:
            return M

    def MLPbias_out(self, layer: int, factored: bool = True) -> np.ndarray:
        assert layer < self.config.n_layer, f"layer #{layer} does not exist"

        M = self.fetch_tensor(f"transformer.h.{layer}.mlp.fc_out.bias")

        if factored:
            return SVD.frommatrix(M)
        else:
            return M

    def layernorm_biases(
        self, layer: int, factored: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        assert layer < self.config.n_layer, f"layer #{layer} does not exist"

        bias = self.fetch_tensor(f"transformer.h.{layer}.ln_1.bias")

        if factored:
            return SVD.frommatrix(M)
        else:
            return M
