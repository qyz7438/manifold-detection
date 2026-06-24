import pytest
import torch

from spectral_detection_posttrain.models.micro_afm import MultiScaleAFM


def test_multiscale_afm_preserves_shape_per_level():
    afm = MultiScaleAFM(channels=[256, 512, 1024, 1024])
    x_p2 = torch.randn(1, 256, 200, 200)
    x_p5 = torch.randn(1, 1024, 25, 25)
    out_p2 = afm(x_p2, level=0)
    out_p5 = afm(x_p5, level=3)
    assert out_p2.shape == x_p2.shape
    assert out_p5.shape == x_p5.shape


def test_multiscale_afm_gradient_flows_per_level():
    afm = MultiScaleAFM(channels=[256, 512])
    x = torch.randn(1, 256, 32, 32, requires_grad=True)
    out = afm(x, level=0)
    loss = out.mean()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_multiscale_afm_blocks_are_independent():
    afm = MultiScaleAFM(channels=[16, 32])
    x0 = torch.randn(1, 16, 8, 8, requires_grad=True)
    x1 = torch.randn(1, 32, 4, 4, requires_grad=True)
    out0 = afm(x0, level=0)
    out1 = afm(x1, level=1)
    (out0.mean() + out1.mean()).backward()
    assert x0.grad is not None and x1.grad is not None
    assert not torch.equal(x0.grad, x1.grad)
