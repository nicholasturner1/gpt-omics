import torch

from gptomics import composition as comp


def test_frobnorm():
    for n in range(10):
        assert comp.frobnorm(torch.ones(n)) == torch.sqrt(torch.tensor(n))

    for n in range(5):
        assert comp.frobnorm(torch.ones(n * n).reshape(n, n)) == n


def test_removemean():
    # need to use double-precision here since torch's matrix multiply
    # isn't that precise (?)
    assert torch.all(
        torch.isclose(
            comp.removemean(torch.ones((10, 10), dtype=torch.float64)),
            torch.tensor(0.0, dtype=torch.float64),
        )
    )
    assert torch.all(
        torch.isclose(
            comp.removemean(
                torch.ones((10, 10), dtype=torch.float64), method="matrix multiply"
            ),
            torch.tensor(0.0, dtype=torch.float64),
        )
    )


def test_basecomposition():
    t = torch.ones(10, 10)
    assert comp.basecomposition(t, t, center=False, denom="none") == 100
    assert comp.basecomposition(t, t, center=False) == 1
