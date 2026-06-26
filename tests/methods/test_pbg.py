import torch

from spectral_detection_posttrain.methods.detection.pbg import (
    FrequencySpatialBoundaryGate,
    PhaseBoundaryGate,
)


def test_pbg_identity_at_init():
    # alpha_init=0 preserves exact identity at initialisation.
    m = FrequencySpatialBoundaryGate(256, alpha_init=0.0)
    m.eval()
    x = torch.randn(2, 256, 7, 7)
    y = m(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)


def test_pbg_alias():
    assert PhaseBoundaryGate is FrequencySpatialBoundaryGate


def test_pbg_shape_and_gradient():
    m = FrequencySpatialBoundaryGate(128, alpha_init=0.1)
    x = torch.randn(2, 128, 7, 7, requires_grad=True)
    y = m(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert any(p.grad is not None for p in m.parameters())


def test_pbg_phase_mask_modes():
    x = torch.randn(2, 64, 7, 7)
    for mode in ("none", "hard", "soft"):
        m = FrequencySpatialBoundaryGate(64, alpha_init=0.0, phase_mask=mode)
        m.eval()
        y = m(x)
        assert y.shape == x.shape
