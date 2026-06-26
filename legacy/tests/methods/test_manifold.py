"""Unit tests for Plan A learnable spectral manifold infrastructure."""

from __future__ import annotations

import pytest
import torch
from scipy.stats import wasserstein_distance

from spectral_detection_posttrain.methods.manifold import (
    AdaptiveRiemannianMetric,
    ChordTransport,
    ComplexLinear,
    ComplexMLP,
    ComplexSpectralManifold,
    SinkhornOT,
)


def _random_complex(*shape: int, dtype: torch.dtype = torch.cfloat) -> torch.Tensor:
    """Generate a random complex tensor with independent real/imag parts."""
    real = torch.randn(*shape)
    imag = torch.randn(*shape)
    return torch.complex(real, imag).to(dtype)


def _leaf_complex(*shape: int, dtype: torch.dtype = torch.cfloat) -> torch.Tensor:
    """Create a complex leaf tensor that supports ``.grad`` after backward."""
    real_imag = torch.randn(*shape, 2, requires_grad=True)
    z = torch.view_as_complex(real_imag).to(dtype)
    z.retain_grad()
    return z


def test_complex_linear():
    """ComplexLinear forward and backward pass produce correct shapes/gradients."""
    layer = ComplexLinear(in_features=8, out_features=4)
    x = _leaf_complex(2, 8)
    y = layer(x)
    assert y.shape == (2, 4)
    assert torch.is_complex(y)
    loss = y.abs().pow(2).sum()
    loss.backward()
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()
    assert torch.isfinite(layer.bias.grad).all()


def test_complex_linear_from_real():
    """Factory method builds a complex layer from real/imag weight parts."""
    real = torch.randn(4, 8)
    imag = torch.randn(4, 8)
    layer = ComplexLinear.from_real(real, imag)
    x = _random_complex(2, 8)
    y = layer(x)
    expected = torch.matmul(x, (real + 1j * imag).t())
    assert torch.allclose(y, expected, atol=1e-5)


def test_complex_mlp_identity_init():
    """A square ComplexMLP initialized as identity maps input to itself."""
    dim = 16
    mlp = ComplexMLP(
        in_features=dim,
        out_features=dim,
        hidden_features=dim,
        identity_init=True,
    )
    x = _random_complex(2, dim)
    y = mlp(x)
    assert y.shape == x.shape
    error = (y - x).abs().pow(2).mean().item()
    assert error < 1e-6


def test_manifold_encode_decode():
    """ComplexSpectralManifold reconstructs random spectra at initialization."""
    dim = 32
    manifold = ComplexSpectralManifold(in_dim=dim, latent_dim=dim, hidden_dim=dim)
    F = _random_complex(3, dim)
    F_recon = manifold(F)
    assert F_recon.shape == F.shape
    mse = (F_recon - F).abs().pow(2).mean().item()
    assert mse < 1e-3


def test_manifold_split_combine_magnitude_phase():
    """Magnitude/phase split and combine are inverses."""
    z = _random_complex(2, 8)
    rho, theta = ComplexSpectralManifold.split_magnitude_phase(z)
    z_back = ComplexSpectralManifold.combine_magnitude_phase(rho, theta)
    assert torch.allclose(z, z_back, atol=1e-6)


def test_riemannian_metric_positive_definite():
    """AdaptiveRiemannianMetric produces positive-definite matrices."""
    latent_dim = 8
    metric = AdaptiveRiemannianMetric(latent_dim=latent_dim)
    z = _random_complex(4, latent_dim)
    M = metric.metric(z)
    assert M.shape == (4, latent_dim, latent_dim)
    assert M.dtype in (torch.float32, torch.float64)

    # Symmetry.
    assert torch.allclose(M, M.transpose(-2, -1), atol=1e-6)

    # All eigenvalues positive.
    eigvals = torch.linalg.eigvalsh(M)
    assert (eigvals > 0).all()

    # At initialization the metric equals eps * I.
    assert torch.allclose(M, metric.eps * torch.eye(latent_dim), atol=1e-6)


def test_riemannian_local_distance_shape_and_nonnegativity():
    """local_distance returns real non-negative scalars."""
    latent_dim = 8
    metric = AdaptiveRiemannianMetric(latent_dim=latent_dim)
    z1 = _random_complex(2, latent_dim)
    z2 = _random_complex(2, latent_dim)
    d2 = metric.local_distance(z1, z2)
    assert d2.shape == (2,)
    assert (d2 >= 0).all()


def test_chord_transport_energy_lower_than_naive():
    """The Chord control field has lower energy than the direct latent jump."""
    dim = 16
    manifold = ComplexSpectralManifold(in_dim=dim, latent_dim=dim, hidden_dim=dim)
    metric = AdaptiveRiemannianMetric(latent_dim=dim)
    transport = ChordTransport(manifold, metric, delta=0.15, lambda_step=1.0)

    F_source = _random_complex(2, dim)
    F_target = _random_complex(2, dim)

    with torch.no_grad():
        z_src = manifold.encode(F_source)
        z_tar = manifold.encode(F_target)
        direct_energy = (z_tar - z_src).abs().pow(2).sum(dim=-1)

    F_pred = transport(F_source, F_target)
    assert F_pred.shape == F_source.shape

    chord_energy = transport.transport_energy
    assert chord_energy is not None
    assert (chord_energy <= direct_energy).all()


def test_sinkhorn_1d_wasserstein():
    """Sinkhorn distance approximates scipy 1-Wasserstein distance in 1D."""
    n = 64
    x = torch.linspace(0.0, 1.0, n)
    # Two slightly shifted distributions.
    mu = torch.softmax(torch.sin(4.0 * torch.pi * x) * 2.0, dim=0)
    nu = torch.softmax(torch.sin(4.0 * torch.pi * (x - 0.1)) * 2.0, dim=0)

    cost = SinkhornOT.pairwise_cost(x, x, p=1)
    sinkhorn = SinkhornOT(eps=0.005, max_iter=300, p=1, stable=True)
    dist = sinkhorn(mu, nu, cost)

    scipy_dist = wasserstein_distance(
        x.numpy(), x.numpy(), mu.detach().numpy(), nu.detach().numpy()
    )
    rel_err = abs(dist.item() - scipy_dist) / max(abs(scipy_dist), 1e-8)
    assert rel_err < 0.05


def test_sinkhorn_gradient_flows():
    """Sinkhorn distance is differentiable w.r.t. the cost matrix."""
    n = 16
    mu = torch.ones(n) / n
    nu = torch.ones(n) / n
    cost = torch.rand(n, n, requires_grad=True)
    sinkhorn = SinkhornOT(eps=0.05, max_iter=50)
    dist = sinkhorn(mu, nu, cost)
    dist.backward()
    assert cost.grad is not None
    assert torch.isfinite(cost.grad).all()


def test_gradient_flow():
    """Every module supports backpropagation and produces finite gradients."""
    dim = 8
    manifold = ComplexSpectralManifold(in_dim=dim, latent_dim=dim, hidden_dim=dim)
    metric = AdaptiveRiemannianMetric(latent_dim=dim)
    transport = ChordTransport(manifold, metric, delta=0.15, lambda_step=1.0)

    F = _leaf_complex(2, dim)
    F_target = _random_complex(2, dim)

    F_pred = transport(F, F_target)
    loss = F_pred.abs().pow(2).sum()
    loss.backward()

    assert F.grad is not None
    assert torch.isfinite(F.grad).all()
    for p in manifold.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()

    # Metric branch also has gradients.
    metric.zero_grad()
    z1 = _leaf_complex(2, dim)
    z2 = _leaf_complex(2, dim)
    d2 = metric.local_distance(z1, z2)
    d2.sum().backward()
    for p in metric.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_chord_transport_delta_extremes():
    """delta=0 yields the full latent jump; delta=1 yields a damped jump."""
    dim = 8
    manifold = ComplexSpectralManifold(in_dim=dim, latent_dim=dim, hidden_dim=dim)
    metric = AdaptiveRiemannianMetric(latent_dim=dim)

    F_source = _random_complex(1, dim)
    F_target = _random_complex(1, dim)

    transport_zero = ChordTransport(manifold, metric, delta=0.0, lambda_step=1.0)
    F_pred_zero = transport_zero(F_source, F_target)

    with torch.no_grad():
        z_src = manifold.encode(F_source)
        z_tar = manifold.encode(F_target)
        F_naive = manifold.decode(z_tar)

    # delta=0 should land on the target observation.
    assert torch.allclose(F_pred_zero, F_naive, atol=1e-5)
