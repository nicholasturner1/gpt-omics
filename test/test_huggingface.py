import torch
import pytest
import numpy as np
from transformers import AutoModelForCausalLM

from gptomics import gptneo


@pytest.fixture(scope="session")
def model():
    return AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")


@pytest.fixture
def dummyvectors():
    np.random.seed(928073450)
    base_dv = np.random.rand(1, 768)

    # sequence length X hidden dimension
    dvs = np.empty((5, 768), dtype=np.float32)
    dvs[0, :] = base_dv
    dvs[1:, :] = np.random.randn(4, 768) * 0.05 + base_dv

    # Making a tensor for PyTorch code
    # batch X sequence length X hidden dimension
    return torch.tensor(dvs).view(1, 5, 768)


def test_selfattention(model, dummyvectors):
    """Runs the self-attention test from the notebook."""
    att0 = model.transformer.h[0].attn.attention
    dummyvectors_np = dummyvectors[0].numpy().T

    biases = att0.out_proj.bias.data.numpy()

    torchoutput = att0(dummyvectors)[0].detach().numpy()
    ouroutput = (
        sum(
            gptneo.selfattention(dummyvectors_np, model, 0, i).T
            for i in range(model.config.num_heads)
        )
        + biases
    )

    assert np.all(np.isclose(torchoutput, ouroutput, atol=1e-3))
