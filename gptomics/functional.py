"""Functions that actually run the network for simple tasks."""
from __future__ import annotations

import inspect
from typing import Union, Optional

import torch
import transformers
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, GPTJForCausalLM, GPTNeoForCausalLM

from . import huggingface, transformersio, composition as comp
from .model import GPTNeo_HF, Model, Layer, SelfAttention, MLP, LayerNorm


def attention_pattern(
    modelname: str,
    prompt: str,
    cuda: bool = True,
) -> tuple[torch.Tensor, list[str]]:
    """Runs a forward pass and returns the attention patterns between token positions.

    Also returns the tokenized outputs for reference.

    Returns:
        - Attention weight values between each pair of tokens for all blocks
          and attention heads in the model
          (shape: num_blocks X num_heads X dst_token X src_token)
        - The token strings for each location in the prompt
    """
    model = transformersio.load_model(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    tokens = tokenizer.convert_ids_to_tokens(list(input_ids.squeeze()))

    return _attention_pattern(model, input_ids, cuda), tokens


def _attention_pattern(
    model, input_ids: torch.Tensor, cuda: bool = True
) -> torch.Tensor:
    if cuda:
        model = model.cuda()
        input_ids = input_ids.cuda()

    # run inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)

    # Num blocks x number of heads x tokens x tokens
    return torch.squeeze(torch.stack(outputs.attentions), 1)


def logit_attribution(
    modelname: str,
    prompt: str,
    block: Optional[int] = None,
    head: Optional[int] = None,
    cuda: bool = True,
) -> tuple[torch.Tensor, list[str]]:
    """Returns the logit attributions between tokens in the prompt for all heads.

    Returns:
        - Logit attribution values between each pair of tokens for all blocks
          and attention heads in the model
          (shape: num_blocks X num_heads X dst_token X src_token)
        - The token strings for each location in the prompt
    """
    model = transformersio.load_model(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    tokens = tokenizer.convert_ids_to_tokens(list(input_ids.squeeze()))

    return _logit_attribution(model, input_ids, block, head, cuda), tokens


def _logit_attribution(
    model,
    input_ids: torch.Tensor,
    block: Optional[int] = None,
    head: Optional[int] = None,
    cuda: bool = True,
) -> torch.Tensor:
    num_tokens = len(input_ids.ravel())
    config = model.config

    if cuda:
        model = model.cuda()
        input_ids = input_ids.cuda()

    attrs = list()

    def head_logit_attr(attn_layer, hidden_states, output) -> None:
        """nn.Module hook for extracting logit attribution values for an attention head.

        Stores the results in the externally defined "attrs" list.
        """
        assert head is not None, "head not defined"

        num_heads = num_attention_heads(attn_layer)  # need a fn to handle different
        head_dim = attn_layer.head_dim

        # dest_token X src_token
        attn_mat = output[2][0, head]

        # num_tokens X head_dim
        if "rotary" in inspect.signature(attn_layer._split_heads).parameters:
            v = attn_layer._split_heads(
                attn_layer.v_proj(hidden_states[0]),
                num_heads,
                head_dim,
                rotary=False,
            )[0, head]
        else:
            v = attn_layer._split_heads(
                attn_layer.v_proj(hidden_states[0]),
                num_heads,
                head_dim,
            )[0, head]

        # weighting each value vector by the correct attention weight
        # using unsqueeze and permute to broadcast the weights correctly
        # dest_token X src_token X head_dim
        weighted = torch.permute(
            torch.unsqueeze(v.T, 2) * torch.unsqueeze(attn_mat.T, 0), (2, 1, 0)
        )

        # Contributions to the residual stream of each weighted value vector above
        # using a standard matrix multiply
        # resid: residual_stream_sz X num_tokens^2
        Wo = attn_layer.out_proj.weight[:, head * head_dim : (head + 1) * head_dim]
        resid = torch.matmul(
            Wo,
            # permuted weighted values: head_dim X num_tokens^2
            torch.permute(weighted, (2, 0, 1)).reshape(
                head_dim, num_tokens * num_tokens
            ),
        )

        # Logit contributions to each output token in the prompt
        # num_tokens X num_tokens^2
        unembedded = torch.matmul(model.lm_head.weight[input_ids.ravel()], resid)

        # reshape recovers the dst_token X src_token matrices
        # potential_output_token X dst_token_index X src_token_index
        attrs.append(unembedded.reshape(num_tokens, num_tokens, num_tokens))

    def get_logit_attr(attn_layer, hidden_states, output) -> None:
        """nn.Module hook for extracting logit attribution values for a block.

        Stores the results in the externally defined "attrs" list.
        """
        num_heads = num_attention_heads(attn_layer)  # need a fn to handle different
        head_dim = attn_layer.head_dim

        # 1 X num_heads X dest_token X src_token
        attn_mat = output[2]  # -> need to run model with output_attentions=True

        # 1 X num_heads X num_tokens X head_dim
        v = attn_layer._split_heads(
            attn_layer.v_proj(hidden_states[0]),
            num_heads,
            head_dim,
        )

        # weighting each value vector by the correct attention weight
        # using unsqueeze and permute to broadcast the weights correctly
        # 1 X num_heads X dest_token X src_token X head_dim
        weighted = torch.permute(
            torch.unsqueeze(torch.permute(attn_mat, (0, 1, 3, 2)), 4)  # attn
            * torch.unsqueeze(v, 3),
            (0, 1, 3, 2, 4),
        )

        # Contributions to the residual stream of each weighted value vector above
        # using a batched matrix multiply
        # resid: 1 X num_heads X residual_stream_sz X num_tokens^2
        Wo = attn_layer.out_proj.weight
        resid = torch.matmul(
            # permuted Wo: 1 X num_heads X residual_stream_sz X head_dim
            torch.permute(
                torch.permute(Wo, (1, 0)).reshape(1, num_heads, head_dim, -1),
                (0, 1, 3, 2),
            ),
            # permuted weighted values: 1 X num_heads X head_dim X num_tokens^2
            torch.permute(weighted, (0, 1, 4, 2, 3)).reshape(
                1, num_heads, head_dim, num_tokens * num_tokens
            ),
        )

        # Logit contributions to each output token in the prompt
        # 1 X num_heads X num_tokens X num_tokens^2
        unembedded = torch.matmul(model.lm_head.weight[input_ids.ravel()], resid)

        # reshape recovers the dst_token X src_token matrices and removes extra 1
        # num_heads X potential_output_token X dst_token_index X src_token_index
        attrs.append(unembedded.reshape(-1, num_tokens, num_tokens, num_tokens))

    # set up hooks for all attention modules
    handles = list()

    hook = get_logit_attr if head is None else head_logit_attr

    if block is None:
        for block_ in range(config.num_layers):
            handles.append(attention_layer(model, block_).register_forward_hook(hook))

    else:
        handles.append(attention_layer(model, block).register_forward_hook(hook))

    # run the model
    with torch.no_grad():
        _ = model(input_ids=input_ids, output_attentions=True)

    # clean up handles
    for handle in handles:
        handle.remove()

    # recover attributions to each actual token in the prompt
    # [num_blocks X] num_heads X dst_token X src_token
    result = torch.stack(
        [
            attr[..., torch.arange(num_tokens)[1:], torch.arange(num_tokens)[:-1], :]
            # attr[..., torch.arange(num_tokens), torch.arange(num_tokens), :]
            for attr in attrs
        ]
    )

    return result


def attention_layer(model, block: int):
    """Extracts the attention layer module from a huggingface model."""
    if isinstance(model, GPTNeoForCausalLM):
        return model.transformer.h[block].attn.attention
    elif isinstance(model, GPTJForCausalLM):
        return model.transformer.h[block].attn
    else:
        raise ValueError(f"unrecognized model type: {type(model)}")


def num_attention_heads(attn_layer):
    """Extracts the number of attention heads from a huggingface attention layer."""
    if hasattr(attn_layer, "num_heads"):
        return attn_layer.num_heads
    elif hasattr(attn_layer, "num_attention_heads"):
        return attn_layer.num_attention_heads
    else:
        raise ValueError(f"unrecognized layer type: {type(attn_layer)}")


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
    results: dict[str, torch.Tensor] = dict()
    handles = list()

    if not isinstance(model, GPTNeo_HF):
        raise NotImplementedError("not implemented for models other than GPTNeo_HF")

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
) -> torch.utils.hooks.RemovableHandle:
    def standardize_output_hook(
        mod: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:

        if src_type == "layernorm_bias":
            assert isinstance(mod.bias, torch.Tensor)
            return mod.bias.view(1, 1, -1)

        elif src_type == "att_head":
            assert isinstance(mod.attention, nn.Module)
            # usual matrix order is transposed here
            OV = huggingface.module_ov(mod.attention, src_index, head_dim).T

            # attention layer outputs need to be a tuple
            return (OV.view((1,) + OV.shape),)

        elif src_type == "att_bias":
            # attention layer outputs need to be a tuple
            assert isinstance(mod.attention, nn.Module)
            assert isinstance(mod.attention.out_proj, nn.Module)
            assert isinstance(mod.attention.out_proj.bias, torch.Tensor)
            return (mod.attention.out_proj.bias.view(1, 1, -1),)

        elif src_type == "mlp_weight":
            assert isinstance(mod.c_proj, nn.Module)
            assert isinstance(mod.c_proj.weight, torch.Tensor)
            mlp_weight = mod.c_proj.weight

            return mlp_weight.view((1,) + mlp_weight.shape)

        elif src_type == "mlp_bias":
            assert isinstance(mod.c_proj, nn.Module)
            assert isinstance(mod.c_proj.bias, torch.Tensor)
            return mod.c_proj.bias.view(1, 1, -1)

        elif src_type == "zero":  # sanity check
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            else:
                return torch.zeros_like(output)

        else:
            raise ValueError(f"unrecognized src_type {src_type}")

    return module.register_forward_hook(standardize_output_hook)  # type: ignore


def register_record_input(
    module: nn.Module,
    results: dict,
    term_type: str,
    dst_index: int = 0,
    head_dim: int = 0,
    denom: str = "none",
) -> torch.utils.hooks.RemovableHandle:
    def record_input_hook(mod: nn.Module, input: torch.Tensor) -> None:
        if term_type == "Q":
            assert isinstance(mod.attention, nn.Module)
            QK = huggingface.module_qk(mod.attention, dst_index, head_dim)
            result = (
                input[0]
                @ QK
                / comp.compute_denom(input[0].data.numpy(), QK.data.numpy(), denom)
            )
        elif term_type == "K":
            assert isinstance(mod.attention, nn.Module)
            QK = huggingface.module_qk(mod.attention, dst_index, head_dim)
            result = (
                input[0]
                @ QK.T
                / comp.compute_denom(input[0].data.numpy(), QK.T.data.numpy(), denom)
            )
        elif term_type == "V":
            assert isinstance(mod.attention, nn.Module)
            OV = huggingface.module_ov(mod.attention, dst_index, head_dim)
            result = (
                input[0]
                @ OV.T
                / comp.compute_denom(input[0].data.numpy(), OV.T.data.numpy(), denom)
            )

        elif term_type == "mlp_weight":
            assert isinstance(mod.c_fc, nn.Module)
            assert isinstance(mod.c_fc.weight, torch.Tensor)
            assert isinstance(mod.c_fc.bias, torch.Tensor)
            result = F.linear(
                input, mod.c_fc.weight, torch.zeros_like(mod.c_fc.bias)
            ) / comp.compute_denom(input.data.numpy(), mod.c_fc.weight.data.numpy())

        results[term_type] = result

    return module.register_forward_pre_hook(record_input_hook)


def subtract_bias_hook(
    module: nn.Module, input: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    return output - module.bias


def register_remove_bias(module: nn.Module) -> torch.utils.hooks.RemovableHandle:
    return module.register_forward_hook(subtract_bias_hook)  # type: ignore


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


def register_zero_output(module: nn.Module) -> torch.utils.hooks.RemovableHandle:
    return module.register_forward_hook(zero_output_hook)  # type: ignore
