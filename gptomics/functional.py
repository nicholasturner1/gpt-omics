"""Functions that actually run the network for simple tasks."""
import torch
import transformers
from torch import nn
from torch.nn import functional as F

from . import gptneo, composition as comp
from .model import Model, Layer, SelfAttention, MLP, LayerNorm


def direct_input_effect(
    model: Model,
    src_layer: Layer,
    src_block: int,
    src_type: str,
    src_index: int,
    dst_layer: Layer,
    dst_block: int,
    dst_index: int,
    term_type: str,
    denom: str = "none",
) -> torch.Tensor:
    """Runs a forward pass with only one contribution to the residual stream.

    This tests composition values for the proper input/output pair.
    """
    results = dict()
    handles = list()

    config = model.model.config

    head_dim = config.hidden_size // config.num_heads

    blocks = model.model.transformer.h
    for (i, block) in enumerate(blocks):
        for layername in ["ln_1", "attn", "ln_2", "mlp"]:
            module = getattr(block, layername)

            if is_layer_match(module, i, src_layer, src_block):
                new_handle = register_standardize_output(
                    module, src_type, src_index, head_dim
                )

            elif is_layer_match(module, i, dst_layer, dst_block):
                new_handle = register_record_input(
                    module,
                    results,
                    term_type,
                    dst_index,
                    head_dim,
                    denom=denom,
                )

            elif is_layer_norm(module):
                new_handle = register_remove_bias(module)

            else:
                new_handle = register_zero_output(module)

            handles.append(new_handle)

    hidden_states = torch.zeros((1, 1, model.model.config.hidden_size))
    # run a forward pass
    for block in blocks:
        hidden_states = block(hidden_states)[0]

    for handle in handles:
        handle.remove()

    return next(iter(results.values()))


def is_layer_norm(module: nn.Module) -> bool:
    return isinstance(module, nn.modules.normalization.LayerNorm)


def is_layer_match(
    module: nn.Module, block: int, layer_to_match: Layer, block_to_match: int
) -> bool:

    class_mapping = {
        LayerNorm(0): nn.modules.normalization.LayerNorm,
        SelfAttention(): transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoAttention,
        LayerNorm(1): nn.modules.normalization.LayerNorm,
        MLP(): transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoMLP,
    }

    return isinstance(module, class_mapping[layer_to_match]) and block == block_to_match


def register_standardize_output(
    module: nn.Module, src_type: str, src_index: int = 0, head_dim: int = 0
) -> None:
    def standardize_output_hook(
        mod: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:

        if src_type == "layernorm_bias":
            return mod.bias.view(1, 1, -1)

        elif src_type == "att_head":
            # usual matrix order is transposed here
            OV = gptneo.module_ov(mod.attention, src_index, head_dim).T

            # attention layer outputs need to be a tuple
            return (OV.view((1,) + OV.shape),)

        elif src_type == "att_bias":
            # attention layer outputs need to be a tuple
            return (mod.attention.out_proj.bias.view(1, 1, -1),)

        elif src_type == "mlp_weight":
            mlp_weight = mod.c_proj.weight

            return mlp_weight.view((1,) + mlp_weight.shape)

        elif src_type == "mlp_bias":
            return mod.c_proj.bias.view(1, 1, -1)

        elif src_type == "zero":  # sanity check
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            else:
                return torch.zeros_like(output)

        else:
            raise ValueError(f"unrecognized src_type {src_type}")

    return module.register_forward_hook(standardize_output_hook)


def register_record_input(
    module: nn.Module,
    results: dict,
    term_type: str,
    dst_index: int = 0,
    head_dim: int = 0,
    denom: str = "none",
) -> None:
    def record_input_hook(mod: nn.Module, input: torch.Tensor) -> None:
        if term_type == "Q":
            QK = gptneo.module_qk(mod.attention, dst_index, head_dim)
            result = input[0] @ QK / comp.compute_denom(input[0], QK, denom)
        elif term_type == "K":
            QK = gptneo.module_qk(mod.attention, dst_index, head_dim)
            result = input[0] @ QK.T / comp.compute_denom(input[0], QK.T, denom)
        elif term_type == "V":
            OV = gptneo.module_ov(mod.attention, dst_index, head_dim)
            result = input[0] @ OV.T / comp.compute_denom(input[0], OV.T, denom)

        elif term_type == "mlp_weight":
            result = F.linear(
                input, mod.c_fc.weight, torch.zeros_like(mod.c_fc.bias)
            ) / comp.compute_denom(input, mod.c_fc.weight)

        results[term_type] = result

    return module.register_forward_pre_hook(record_input_hook)


def subtract_bias_hook(
    module: nn.Module, input: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    return output - module.bias


def register_remove_bias(module: nn.Module) -> None:
    return module.register_forward_hook(subtract_bias_hook)


def zero_output_hook(
    module: nn.Module,
    input: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    if isinstance(output, tuple):
        output[0][:] = 0
    else:
        output[:] = 0

    return output


def register_zero_output(module: nn.Module) -> None:
    return module.register_forward_hook(zero_output_hook)
