import torch

from spectral_detection_posttrain.methods.detection.pbg import PhaseBoundaryGate


def test_pbg_identity_at_init():
    m = PhaseBoundaryGate(256)
    x = torch.randn(2, 256, 7, 7)
    y = m(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)


def test_pbg_shape_and_gradient():
    m = PhaseBoundaryGate(128, alpha_init=0.1)
    x = torch.randn(2, 128, 7, 7, requires_grad=True)
    y = m(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
