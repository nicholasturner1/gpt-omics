"""Functions for extracting useful information from GPT-Neo HuggingFace models.

The functions to simulate the network are obv slower than running the actual model,
but they offer flexibility for analysis.
"""
from __future__ import annotations

import torch
import numpy as np
from torch import nn
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM


def qk(model: GPTNeoForCausalLM, layer: int, head: int) -> np.ndarray:
    """Extracts the QK matrix from a given head & layer from the model."""
    config = model.config
    head_dim = config.hidden_size // config.num_heads

    assert head < config.num_heads
    assert layer < config.num_layers

    return module_qk(
        model.transformer.h[layer].attn.attention, head, head_dim
    ).data.numpy()


def module_qk(module: nn.Module, head: int, head_dim: int) -> torch.Tensor:

    assert isinstance(module.q_proj, nn.Module)
    assert isinstance(module.k_proj, nn.Module)

    Q = module.q_proj.weight
    K = module.k_proj.weight

    assert isinstance(Q, torch.Tensor)
    assert isinstance(K, torch.Tensor)

    Qh = Q[head * head_dim : (head + 1) * head_dim, :]
    Kh = K[head * head_dim : (head + 1) * head_dim, :]

    return Qh.T @ Kh


def ov(
    model: GPTNeoForCausalLM, layer: int, head: int, tensor: bool = False
) -> np.ndarray:
    """Extracts the OV matrix from a given head & layer from the model."""
    config = model.config
    head_dim = config.hidden_size // config.num_heads

    assert head < config.num_heads
    assert layer < config.num_layers

    return module_ov(
        model.transformer.h[layer].attn.attention, head, head_dim
    ).data.numpy()


def module_ov(module: nn.Module, head: int, head_dim: int) -> torch.Tensor:

    assert isinstance(module.out_proj, nn.Module)
    assert isinstance(module.v_proj, nn.Module)

    O = module.out_proj.weight
    V = module.v_proj.weight

    assert isinstance(O, torch.Tensor)
    assert isinstance(V, torch.Tensor)

    Oh = O[:, head * head_dim : (head + 1) * head_dim]
    Vh = V[head * head_dim : (head + 1) * head_dim, :]

    return Oh @ Vh


def out_bias(model: GPTNeoForCausalLM, layer: int) -> np.ndarray:
    """Extracts the output bias from a given layer of the model."""
    config = model.config

    assert layer < config.num_layers

    return model.transformer.h[layer].attn.attention.out_proj.bias.data.numpy()


def mlp_in(model: GPTNeoForCausalLM, layer: int) -> np.ndarray:
    """Extracts the input weights for an MLP layer."""
    config = model.config

    assert layer < config.num_layers

    return model.transformer.h[layer].mlp.c_fc.weight.data.numpy()


def mlp_out(model: GPTNeoForCausalLM, layer: int) -> np.ndarray:
    """Extracts the output weights for an MLP layer."""
    config = model.config

    assert layer < config.num_layers

    return model.transformer.h[layer].mlp.c_proj.weight.data.numpy()


def mlp_bias_in(model: GPTNeoForCausalLM, layer: int) -> np.ndarray:
    """Extracts the bias for the input weights of an MLP layer."""
    config = model.config

    assert layer < config.num_layers

    return model.transformer.h[layer].mlp.c_fc.bias.data.numpy()


def mlp_bias_out(model: GPTNeoForCausalLM, layer: int) -> np.ndarray:
    """Extracts the bias for the output weights of an MLP layer."""
    config = model.config

    assert layer < config.num_layers

    return model.transformer.h[layer].mlp.c_proj.bias.data.numpy()


def ln_weights(model: GPTNeoForCausalLM, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Extracts the layer norm biases from a given layer of the model."""
    config = model.config

    assert layer < config.num_layers

    weight_1 = model.transformer.h[layer].ln_1.weight.data.numpy()
    weight_2 = model.transformer.h[layer].ln_2.weight.data.numpy()

    return weight_1, weight_2


def ln_biases(model: GPTNeoForCausalLM, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Extracts the layer norm biases from a given layer of the model."""
    config = model.config

    assert layer < config.num_layers

    bias_1 = model.transformer.h[layer].ln_1.bias.data.numpy()
    bias_2 = model.transformer.h[layer].ln_2.bias.data.numpy()

    return bias_1, bias_2


def attentionweights(
    inputs: np.ndarray,
    model: GPTNeoForCausalLM,
    layer: int,
    head: int,
) -> np.ndarray:
    """Computes attention weights for set of inputs to a single attention head."""
    assert len(inputs.shape) == 2

    seqlen = inputs.shape[1]
    causal_mask = np.tril(np.ones((seqlen, seqlen), dtype=np.uint8))

    raw = inputs.T @ qk(model, layer, head) @ inputs
    final = torch.nn.functional.softmax(
        # raw weights with causal mask
        torch.tensor(np.where(causal_mask == 1, raw, -1e9)),
        dim=-1,
    ).numpy()

    return final


def selfattention(
    inputs: np.ndarray,
    model: GPTNeoForCausalLM,
    layer: int,
    head: int,
) -> np.ndarray:
    """Computes the self-attention output for a set of inputs to a single head."""
    assert len(inputs.shape) == 2

    OVmat = ov(model, layer, head)

    return OVmat @ inputs @ attentionweights(inputs, model, layer, head).T
