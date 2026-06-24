import torch

from mfvpt.transforms.fourier import high_freq_perturb, low_pass_filter


def test_low_pass_filter_shape_range_and_change():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 32, 32)
    out = low_pass_filter(x, ratio=0.25)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0
    assert not torch.allclose(out, x)


def test_high_freq_perturb_shape_range_and_change():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 32, 32)
    out = high_freq_perturb(x, strength=0.10, ratio=0.25)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0
    assert not torch.allclose(out, x)


def test_invalid_ratio_raises():
    x = torch.rand(1, 3, 16, 16)
    try:
        low_pass_filter(x, ratio=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
