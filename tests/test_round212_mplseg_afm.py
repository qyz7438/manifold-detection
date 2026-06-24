import torch

from spectral_detection_posttrain.models.micro_afm import MPLSegAFMBlock, build_afm_block


def test_mplseg_afm_output_shape():
    afm = MPLSegAFMBlock(in_ch=64)
    x = torch.randn(2, 64, 24, 24)
    out = afm(x)
    assert out.shape == x.shape


def test_mplseg_afm_is_not_identity_at_init():
    afm = MPLSegAFMBlock(in_ch=64)
    x = torch.randn(2, 64, 24, 24)
    out = afm(x)
    diff = (out - x).abs().mean().item()
    assert diff > 1e-3, f"MPLSeg AFM should modify features, but diff={diff}"


def test_mplseg_afm_no_nan():
    afm = MPLSegAFMBlock(in_ch=64)
    x = torch.randn(2, 64, 24, 24)
    out = afm(x)
    assert torch.isfinite(out).all()


def test_mplseg_afm_mag_gate_gets_gradient():
    afm = MPLSegAFMBlock(in_ch=16)
    x = torch.randn(2, 16, 16, 16, requires_grad=True)
    out = afm(x)
    target = torch.randn_like(out) * 0.1
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    mp_conv = afm.mp[0]
    assert mp_conv.weight.grad is not None, "mp conv grad is None"
    assert mp_conv.weight.grad.abs().sum() > 0, "mp conv grad is zero"


def test_mplseg_afm_phase_res_gets_gradient():
    afm = MPLSegAFMBlock(in_ch=16)
    x = torch.randn(2, 16, 16, 16, requires_grad=True)
    out = afm(x)
    target = torch.randn_like(out) * 0.1
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    pa_conv = afm.pa[0]
    assert pa_conv.weight.grad is not None, "pa conv grad is None"
    assert pa_conv.weight.grad.abs().sum() > 0, "pa conv grad is zero"


def test_mplseg_afm_residual_scale_gets_gradient():
    afm = MPLSegAFMBlock(in_ch=16)
    x = torch.randn(2, 16, 16, 16, requires_grad=True)
    out = afm(x)
    target = torch.randn_like(out) * 0.1
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    assert afm.residual_scale.grad is not None
    assert abs(afm.residual_scale.grad.item()) > 0


def test_mplseg_afm_factory():
    afm = build_afm_block("mplseg", channels=32)
    assert isinstance(afm, MPLSegAFMBlock)
    assert build_afm_block("none", channels=32) is None


def test_mplseg_afm_output_has_residual():
    """With residual_scale=1.0, output = x + freq_out where freq_out>=0 via ReLU."""
    afm = MPLSegAFMBlock(in_ch=16)
    x = torch.randn(2, 16, 24, 24)
    out = afm(x)
    assert (out != x).any(), "output should differ from input"
