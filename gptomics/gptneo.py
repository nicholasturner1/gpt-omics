"""Functions for extracting useful information from GPT-Neo HuggingFace models.

These functions are slower than running the actual model, but offer flexibility
for analysis.
"""
import torch
import numpy as np
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM



def QK(model: GPTNeoForCausalLM, layer: int, head: int) -> np.ndarray:
    """Extracts the QK matrix from a given head & layer from the model."""
    config = model.config
    head_dim = config.hidden_size // config.num_heads

    assert head < config.num_heads
    assert layer < config.num_layers

    attention = model.transformer.h[layer].attn.attention
    Q = attention.q_proj.weight.data.numpy()
    K = attention.k_proj.weight.data.numpy()

    Qh = Q[head * head_dim : (head + 1) * head_dim, :]
    Kh = K[head * head_dim : (head + 1) * head_dim, :]

    return Qh.T @ Kh


def OV(model: GPTNeoForCausalLM, layer: int, head: int) -> np.ndarray:
    """Extracts the OV matrix from a given head & layer from the model."""
    config = model.config
    head_dim = config.hidden_size // config.num_heads

    assert head < config.num_heads
    assert layer < config.num_layers

    attention = model.transformer.h[layer].attn.attention
    O = attention.out_proj.weight.data.numpy()
    V = attention.v_proj.weight.data.numpy()

    Oh = O[:, head * head_dim : (head + 1) * head_dim]
    Vh = V[head * head_dim : (head + 1) * head_dim, :]

    return Oh @ Vh


# I used to call this function "OVsimple" to signal that we would later account
# for the biases within the same matrix, but we later decided against this since
# the bias can't be attributed to particular heads.
OVsimple = OV


def Obias(model: GPTNeoForCausalLM, layer: int) -> np.ndarray:
    """Extracts the output bias from a given layer of the model."""
    config = model.config

    assert layer < config.num_layers

    return model.transformer.h[layer].attn.attention.out_proj.bias.data.numpy()


def MLPin(model: GPTNeoForCausalLM, layer: int) -> np.ndarray:
    """Extracts the input weights for an MLP layer."""
    config = model.config

    assert layer < config.num_layers

    return model.transformer.h[layer].mlp.c_fc.weight.data.numpy()


def MLPout(model: GPTNeoForCausalLM, layer: int) -> np.ndarray:
    """Extracts the output weights for an MLP layer."""
    config = model.config

    assert layer < config.num_layers

    return model.transformer.h[layer].mlp.c_proj.weight.data.numpy()


def MLPbias_in(model: GPTNeoForCausalLM, layer: int) -> np.ndarray:
    """Extracts the bias for the input weights of an MLP layer."""
    config = model.config

    assert layer < config.num_layers

    return model.transformer.h[layer].mlp.c_fc.bias.data.numpy()


def MLPbias_out(model: GPTNeoForCausalLM, layer: int) -> np.ndarray:
    """Extracts the bias for the output weights of an MLP layer."""
    config = model.config

    assert layer < config.num_layers

    return model.transformer.h[layer].mlp.c_proj.bias.data.numpy()


def layernorm_biases(model: GPTNeoForCausalLM, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Extracts the layer norm biases from a given layer of the model."""
    config = model.config

    assert layer < config.num_layers

    bias_1 = model.transformer.h[layer].ln_1.bias.data.numpy()
    bias_2 = model.transformer.h[layer].ln_2.bias.data.numpy()

    return bias_1, bias_2


def attentionweights(
    inputs: np.ndarray, model: GPTNeoForCausalLM, layer: int, head: int,
) -> np.ndarray:
    """Computes attention weights for set of inputs to a single attention head."""
    assert len(inputs.shape) == 2

    seqlen = inputs.shape[1]
    causal_mask = np.tril(np.ones((seqlen, seqlen), dtype=np.uint8))

    raw = inputs.T @ QK(model, layer, head) @ inputs
    final = torch.nn.functional.softmax(
        # raw weights with causal mask
        torch.tensor(np.where(causal_mask == 1, raw, -1e9)),
        dim=-1,
    ).numpy()

    return final


def selfattention(
    inputs: np.ndarray, model: GPTNeoForCausalLM, layer: int, head: int,
) -> np.ndarray:
    """Computes the self-attention output for a set of inputs to a single head."""
    assert len(inputs.shape) == 2

    OV = OV_simple(model, layer, head)

    return OV @ inputs @ attentionweights(inputs, model, layer, head).T
