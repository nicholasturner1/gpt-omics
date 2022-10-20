from __future__ import annotations

import torch
import pytest

from gptomics import huggingface, transformersio


@pytest.fixture(scope="session")
def model():
    return transformersio.load_model("EleutherAI/gpt-neo-125M")


@pytest.fixture
def dummyvectors():
    torch.manual_seed(928073450)
    base_dv = torch.rand(1, 768)

    # sequence length X hidden dimension
    dvs = torch.empty((5, 768), dtype=torch.float32)
    dvs[0, :] = base_dv
    dvs[1:, :] = torch.randn(4, 768) * 0.05 + base_dv

    # Making a tensor for PyTorch code
    # batch X sequence length X hidden dimension
    return dvs.view(1, 5, 768)


def test_selfattention(model, dummyvectors):
    """Runs the self-attention test from the notebook."""
    att0 = model.transformer.h[0].attn.attention
    # we use matrices instead of 3D arrays for this
    dummyvectors_for_us = dummyvectors[0].T

    biases = att0.out_proj.bias

    torchoutput = att0(dummyvectors)[0]
    ouroutput = (
        sum(
            huggingface.selfattention(dummyvectors_for_us, model, 0, i).T
            for i in range(model.config.num_heads)
        )
        + biases
    )

    assert torch.all(torch.isclose(torchoutput, ouroutput, atol=1e-3))
