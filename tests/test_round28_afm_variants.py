import torch

from spectral_detection_posttrain.models.micro_afm import AFMBlock, OldAFMBlock, build_afm_block


def test_identity_afm_is_identity_at_init():
    afm = AFMBlock(channels=16, residual_mode="current")
    x = torch.randn(2, 16, 24, 24)
    assert torch.allclose(afm(x), x, atol=1e-3)


def test_old_afm_is_not_identity_at_init():
    afm = OldAFMBlock(channels=16)
    x = torch.randn(2, 16, 24, 24)
    diff = (afm(x) - x).abs().mean().item()
    assert diff > 1e-3


def test_delta_residual_is_identity_at_init():
    afm = AFMBlock(channels=16, residual_mode="delta")
    x = torch.randn(2, 16, 24, 24)
    assert torch.allclose(afm(x), x, atol=1e-3)


def test_norm_delta_residual_is_identity_at_init():
    afm = AFMBlock(channels=16, residual_mode="norm_delta")
    x = torch.randn(2, 16, 24, 24)
    assert torch.allclose(afm(x), x, atol=1e-3)


def test_afm_factory_builds_expected_variants():
    assert isinstance(build_afm_block("old", channels=16), OldAFMBlock)
    assert isinstance(build_afm_block("identity", channels=16, residual_mode="delta"), AFMBlock)
    assert build_afm_block("none", channels=16) is None


def test_delta_variants_have_gradient_flow():
    for residual_mode in ["current", "delta", "norm_delta"]:
        afm = AFMBlock(channels=16, residual_mode=residual_mode)
        x = torch.randn(1, 16, 16, 16, requires_grad=True)
        out = afm(x)
        out.mean().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
