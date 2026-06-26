"""Unit tests for Plan B adversarial defense prototype."""

from __future__ import annotations

import pytest
import torch

from spectral_detection_posttrain.methods.manifold import (
    AdaptiveRiemannianMetric,
    ChordTransport,
    ComplexSpectralManifold,
)
from spectral_detection_posttrain.methods.defense import (
    AdversarialPatchAttack,
    NaturalSpectrumModel,
    SpectralChordDefense,
    defense_success_rate,
    clean_accuracy_drop,
    robust_accuracy,
)


def _make_image(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a smooth synthetic image in [0, 1]."""
    img = torch.zeros(*shape, dtype=dtype)
    if len(shape) == 3:
        c, h, w = shape
        y = torch.linspace(-1, 1, h).view(-1, 1)
        x = torch.linspace(-1, 1, w).view(1, -1)
        pattern = (torch.sin(3.0 * x) + torch.cos(3.0 * y)) * 0.1 + 0.5
        for i in range(c):
            img[i] = pattern + i * 0.02
    return img.clamp(0, 1)


def _high_frequency_loss(image: torch.Tensor) -> torch.Tensor:
    """Mock loss that penalises high-frequency spectral energy."""
    F = torch.fft.rfft2(image)
    h, w = F.shape[-2:]
    # Focus on the upper half of frequency bins (high frequencies).
    high_freq = F[..., h // 2 :, :].abs().pow(2).mean()
    return high_freq


def test_patch_attack_changes_image():
    """The patch attack must alter the input image."""
    image = _make_image(3, 32, 32)
    loss_fn = _high_frequency_loss
    attacker = AdversarialPatchAttack(
        loss_fn,
        patch_size=(8, 8),
        location=(8, 8),
        max_iter=20,
        step_size=0.1,
    )
    patched, patch = attacker.attack(image)
    assert patched.shape == image.shape
    assert patch.shape == (3, 8, 8)
    assert not torch.allclose(patched, image, atol=1e-4)


def test_patch_attack_target_region():
    """Only the target region should differ; the rest stays intact."""
    image = _make_image(3, 32, 32)
    location = (4, 6)
    attacker = AdversarialPatchAttack(
        _high_frequency_loss,
        patch_size=(8, 8),
        location=location,
        max_iter=10,
        step_size=0.05,
    )
    patched, _ = attacker.attack(image)
    top, left = location
    ph, pw = attacker.patch_size
    # Region outside the patch should be unchanged.
    mask = torch.ones_like(image)
    mask[:, top : top + ph, left : left + pw] = 0.0
    outside = (patched - image) * mask
    assert torch.allclose(outside, torch.zeros_like(outside), atol=1e-6)
    # The patch region itself should be changed.
    patch_region = patched[:, top : top + ph, left : left + pw]
    original_region = image[:, top : top + ph, left : left + pw]
    assert not torch.allclose(patch_region, original_region, atol=1e-4)


def test_anomaly_detection_highlights_patch_frequencies():
    """Anomaly mask should respond to high-frequency perturbations."""
    # Build a smooth spectrum and a perturbed copy with injected high-frequency
    # energy.  This directly exercises detect_anomaly without relying on the
    # patch attack to reach a specific spectral signature.
    torch.manual_seed(42)
    c, h, w = 3, 32, 32
    F_clean = torch.zeros(c, h, w // 2 + 1, dtype=torch.cfloat)
    # Low-frequency content in the top-left corner with a gentle amplitude.
    F_clean[:, :4, :4] = torch.randn(c, 4, 4, dtype=torch.cfloat) * 2.0

    F_patched = F_clean.clone()
    # Inject a strong, isolated high-frequency spike.
    F_patched[:, h - 2 :, -2:] = torch.randn(c, 2, 2, dtype=torch.cfloat) * 40.0

    defense = SpectralChordDefense(
        manifold=ComplexSpectralManifold(in_dim=32 * 17, latent_dim=32, hidden_dim=32 * 17),
        transport=_mock_transport(),
        natural_model=_mock_natural_model(torch.full((c, h, w), 0.5)),
        anomaly_gate_threshold=2.0,
        window_size=5,
    )

    mask_clean = defense.detect_anomaly(F_clean)
    mask_patched = defense.detect_anomaly(F_patched)

    # The perturbed spectrum should trigger more anomalies, especially at high frequencies.
    assert mask_patched.sum() > mask_clean.sum()
    high_freq_patched = mask_patched[:, h // 2 :, :].sum()
    high_freq_clean = mask_clean[:, h // 2 :, :].sum()
    assert high_freq_patched > high_freq_clean


def test_spectral_chord_defense_output_shape():
    """Defense output must match input shape for batched and unbatched images."""
    natural = _mock_natural_model(_make_image(3, 32, 32))
    defense = SpectralChordDefense(
        manifold=ComplexSpectralManifold(in_dim=32 * 17, latent_dim=32, hidden_dim=32 * 17),
        transport=_mock_transport(),
        natural_model=natural,
    )
    # Unbatched
    x = _make_image(3, 32, 32)
    out = defense(x)
    assert out.shape == x.shape
    # Batched
    xb = _make_image(2, 3, 32, 32)
    outb = defense(xb)
    assert outb.shape == xb.shape


def test_spectral_chord_defense_preserves_clean_regions():
    """For a clean image the anomaly mask is sparse, so the output stays close."""
    clean = _make_image(3, 32, 32)
    natural = _mock_natural_model(clean)
    defense = SpectralChordDefense(
        manifold=ComplexSpectralManifold(in_dim=32 * 17, latent_dim=32, hidden_dim=32 * 17),
        transport=_mock_transport(),
        natural_model=natural,
        anomaly_gate_threshold=5.0,
    )
    defended = defense(clean)
    # Most frequencies should be untouched for a clean image; output should be close.
    relative = (defended - clean).abs().mean() / (clean.abs().mean() + 1e-8)
    assert relative < 0.05


def test_defense_reduces_adversarial_effect():
    """Defense should reduce the mock adversarial high-frequency loss."""
    clean = _make_image(3, 32, 32)
    attacker = AdversarialPatchAttack(
        _high_frequency_loss,
        patch_size=(8, 8),
        location=(8, 8),
        max_iter=30,
        step_size=0.15,
    )
    patched, _ = attacker.attack(clean)

    natural = _mock_natural_model(clean)
    defense = SpectralChordDefense(
        manifold=ComplexSpectralManifold(in_dim=32 * 17, latent_dim=32, hidden_dim=32 * 17),
        transport=_mock_transport(),
        natural_model=natural,
        anomaly_gate_threshold=2.0,
        window_size=5,
    )
    defended = defense(patched)

    loss_adv = _high_frequency_loss(patched).item()
    loss_def = _high_frequency_loss(defended).item()
    assert loss_def < loss_adv


def test_eval_defense_metrics():
    """Metric helpers should return expected scalar values."""
    clean, adv, defended = 0.9, 0.3, 0.75
    recovery = defense_success_rate(clean, adv, defended)
    expected = (defended - adv) / (clean - adv)
    assert torch.allclose(recovery, torch.tensor(expected), atol=1e-5)

    drop = clean_accuracy_drop(clean, 0.87)
    assert torch.allclose(drop, torch.tensor((clean - 0.87) / clean), atol=1e-5)

    acc = robust_accuracy(torch.tensor([0.4, 0.6, 0.9]), threshold=0.5)
    assert torch.allclose(acc, torch.tensor(2.0 / 3.0), atol=1e-5)


def test_gradient_flow():
    """Defense and natural model modules support backpropagation."""
    dim = 32 * 17
    manifold = ComplexSpectralManifold(in_dim=dim, latent_dim=32, hidden_dim=dim)
    metric = AdaptiveRiemannianMetric(latent_dim=32)
    transport = ChordTransport(manifold, metric, delta=0.15, lambda_step=1.0)
    natural = _mock_natural_model(_make_image(3, 32, 32))
    defense = SpectralChordDefense(manifold, transport, natural)

    x = _make_image(3, 32, 32).requires_grad_(True)
    out = defense(x)
    loss = out.abs().pow(2).sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for p in manifold.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def _mock_transport() -> ChordTransport:
    """Return a small ChordTransport for unit tests."""
    dim = 32 * 17
    manifold = ComplexSpectralManifold(in_dim=dim, latent_dim=32, hidden_dim=dim)
    metric = AdaptiveRiemannianMetric(latent_dim=32)
    return ChordTransport(manifold, metric, delta=0.15, lambda_step=1.0)


def _mock_natural_model(image: torch.Tensor) -> NaturalSpectrumModel:
    """Fit a natural model on several augmented versions of ``image``."""
    model = NaturalSpectrumModel(reg=1e-6)
    spectra = []
    for _ in range(5):
        noise = torch.randn_like(image) * 0.02
        spectra.append(torch.fft.rfft2((image + noise).clamp(0, 1)))
    spectra = torch.stack(spectra, dim=0)
    return model.fit(spectra)
