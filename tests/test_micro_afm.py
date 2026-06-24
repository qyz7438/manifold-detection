import pytest
import torch

from spectral_detection_posttrain.models.micro_afm import AFMBlock


def test_afm_identity_at_init():
    afm = AFMBlock(channels=16)
    x = torch.randn(2, 16, 32, 32)
    out = afm(x)
    assert torch.allclose(out, x, atol=1e-3)


def test_afm_output_shape():
    afm = AFMBlock(channels=16)
    x = torch.randn(2, 16, 32, 32)
    assert afm(x).shape == x.shape


@pytest.mark.parametrize("channels", [16, 64, 256])
def test_afm_gradient_flows(channels: int):
    afm = AFMBlock(channels=channels)
    x = torch.randn(1, channels, 16, 16, requires_grad=True)
    out = afm(x)
    out.mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_afm_no_nan():
    afm = AFMBlock(channels=16)
    x = torch.randn(2, 16, 32, 32) * 100.0
    out = afm(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
