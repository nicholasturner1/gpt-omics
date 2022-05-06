"""Some utility functions for working with HuggingFace transformers models."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM


# A helper function to load models more quickly. Taken from Eleuther discord (#gpt-j)
def no_init(loading_code):
    def dummy(self):
        return

    modules = [torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm]
    original = {}
    for mod in modules:
        original[mod] = mod.reset_parameters
        mod.reset_parameters = dummy

    result = loading_code()
    for mod in modules:
        mod.reset_parameters = original[mod]

    return result


def load_model(modelname: str) -> AutoModelForCausalLM:
    def load():
        return AutoModelForCausalLM.from_pretrained(modelname, low_cpu_mem_usage=True)

    return no_init(load)
