"""Functions that actually run the network for simple tasks."""
from __future__ import annotations
from tkinter import W

from typing import Union

import torch
import transformers
from torch import logit, nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from . import huggingface, transformersio, composition as comp
from .model import GPTNeo_HF, Model, Layer, SelfAttention, MLP, LayerNorm


def attention_pattern(
    modelname: str,
    prompt: str,
    cuda: bool = True,
) -> tuple[tuple[torch.Tensor, ...], list[str]]:
    """Runs a forward pass and returns the attention pattern.

    Also returns the tokenized outputs for reference.
    """

    model = transformersio.load_model(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    tokens = tokenizer.convert_ids_to_tokens(list(input_ids.squeeze()))

    if cuda:
        model = model.cuda()
        input_ids = input_ids.cuda()

    # run inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)

    # Num of block x 1 x number of heads x tokens x tokens
    return outputs.attentions, tokens

def total_attribution (
    modelname : str,
    prompt : str,
    cuda : bool = True,
) -> tuple[tuple[torch.Tensor, ...], list[str]]:
    """Runs a forward pass and returns the logit attributions for all tokens in the prompt for all heads in the model.
    """
    
    model = transformersio.load_model(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    input_ids = tokenizer(prompt, return_tensors = "pt").input_ids
    tokens = tokenizer.convert_ids_to_tokens(list(input_ids.squeeze()))
    num_tokens = len(tokens)

    config = model.config

    if cuda:
        model = model.cuda()
        input_ids = input_ids.cuda()
    # TODO: Vectorize this
    def logit_attribution(
        block_index: int,
        head_index: int,
    ) -> tuple[tuple[torch.Tensor, ...], list[str]]:
        """Runs a forward pass and returns the logit attributions for all tokens, for an attention head.
        """
        
        logits = []


        head_dim = config.hidden_size // config.num_heads
        
        def get_logit_attr(attn_layer, hidden_states, output):
            # Cols/rows of unembedding matrix correspond to tokens 
            attn_mat = output[2]
            print(type(hidden_states))
            values = attn_layer.v_proj(hidden_states[0])[0,:, head_index * head_dim:(head_index+1) * head_dim]
            print(f"values is {values.shape}")
            
            result = torch.zeros ((head_dim,num_tokens,num_tokens))
            print(f"attn_mat is {attn_mat.shape}")
            for row in range(attn_mat.shape[2]):
                result[...,row] = attn_mat[0, head_index, row] * values.T
            
            result = result.reshape(head_dim,num_tokens*num_tokens)

            out = torch.matmul(attn_layer.out_proj.weight[:, head_index * head_dim:(head_index+1) * head_dim], result)
            
            unembedded = torch.matmul(model.lm_head.weight[input_ids.ravel()], out)
            
            logits.append(unembedded.reshape(-1,num_tokens,num_tokens))
        
        handle = model.transformer.h[block_index].attn.attention.register_forward_hook(get_logit_attr)
            
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)
            
        handle.remove()
            
        return logits[0]
        
    blocks = list(range(0, config.num_layers)) 
    heads = list(range(0, config.num_heads))
    # Can we dynamically retrieve the vocab size (50257)?
    attributions = torch.zeros ((config.num_layers, config.num_heads, 50257, num_tokens, num_tokens ))
    # Need to iterate over all blocks, and over all heads in each block 
    for block in blocks:
        for head in heads: 
            attributions[block,head] = logit_attribution(block, head)

    # Iterate through tokens, find indices of tokens, grab attributions for all heads at token indices, grab column for all heads at token index in prompt (should be another blocks x heads tensor for each token in prompt)
    prompt_attributions = torch.zeros((config.num_layers, config.num_heads, num_tokens, num_tokens))    
    for prompt_index in range(num_tokens):
        # # Find token index in dictionary
        # token_attribution_index = input_ids[::, prompt_index]
        
        # # Get the logit attributions at this token_attribution_index
        # intermediary = attributions[::,::,token_attribution_index]

        # # Get the columns corresponding to the token index within the prompt
        # intermediary = torch.squeeze(intermediary)[::,::,::,prompt_index]
        # print(intermediary)
        
        # # If this still contains all the data for blocks and heads, this is the final vector
        # print(f"prompt_attributions {prompt_attributions.shape}")
        # print(f"intermediary {intermediary.shape}")
        # prompt_attributions[::,::,::,prompt_index] = intermediary
        
        prompt_attributions[:, :, :, input_ids] = torch.squeeze(attributions[:, :, input_ids])

    # blocks x heads x tokens x columns (should the last two be swapped?)
    return prompt_attributions

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
