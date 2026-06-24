"""Tests for segmentation signal utilities."""

import torch

from spectral_detection_posttrain.methods.segmentation.signals import (
    PIXEL_CLASSIFICATION_SIGNALS,
    activation_centroid_consistency,
    aspect_ratio_plausibility,
    boundary_phase_coherence,
    boundary_reward,
    connected_component_reward,
    dice_reward,
    interior_exterior_texture_contrast,
    mask_iou_reward,
    multi_scale_saliency_consistency,
    nms_survivor_density,
    phase_correlation_score,
    radial_amplitude_profile,
    score_edge_alignment,
    signal_by_id,
    signal_ids,
)
from spectral_detection_posttrain.methods.segmentation.signals.fft import (
    compute_amplitude_profile,
    compute_lowfreq_phase_stats,
    compute_structure_similarity,
    edge_similarity_score,
    spectral_profile_similarity,
)


def _image_and_mask(size: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = torch.rand(3, size, size)
    pred = torch.zeros((size, size), dtype=torch.bool)
    target = torch.zeros((size, size), dtype=torch.bool)
    pred[8:24, 8:24] = True
    target[9:25, 9:25] = True
    return image, pred, target


def test_pixel_classification_registry_has_twenty_four_signals():
    assert len(PIXEL_CLASSIFICATION_SIGNALS) == 24


def test_signal_by_id_and_signal_ids():
    ids = signal_ids()
    assert len(ids) == len(set(ids))
    assert signal_by_id("pix.boundary.edge_alignment") is not None
    assert signal_by_id("not.a.signal") is None


def test_mask_iou_and_dice_perfect_are_one():
    mask = torch.tensor([[0, 1], [0, 1]], dtype=torch.bool)
    assert torch.isclose(mask_iou_reward(mask, mask), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(dice_reward(mask, mask), torch.tensor(1.0), atol=1e-5)


def test_boundary_reward_is_bounded():
    image, pred, target = _image_and_mask(16)
    value = boundary_reward(pred, target, tolerance=2)
    assert 0.0 <= value.item() <= 1.0


def test_connected_component_reward_penalizes_fragmentation():
    target = torch.zeros((16, 16), dtype=torch.bool)
    pred = torch.zeros((16, 16), dtype=torch.bool)
    target[2:14, 2:14] = True
    pred[2:6, 2:6] = True
    pred[10:14, 10:14] = True
    assert connected_component_reward(pred, target).item() < 1.0


def test_radial_amplitude_profile_shape():
    image, pred, _ = _image_and_mask()
    profile = radial_amplitude_profile(pred.float(), bins=8)
    assert profile.shape == (8,)
    assert torch.isfinite(profile).all()


def test_compute_amplitude_profile_shape():
    image, pred, _ = _image_and_mask()
    profile = compute_amplitude_profile(image, pred, num_bins=16)
    assert profile.shape == (16,)
    assert torch.isfinite(profile).all()


def test_compute_lowfreq_phase_stats_shape():
    image, pred, _ = _image_and_mask()
    stats = compute_lowfreq_phase_stats(image, pred)
    assert stats.shape == (4,)
    assert torch.isfinite(stats).all()


def test_spectral_profile_similarity_identical_is_one():
    image, pred, _ = _image_and_mask()
    sim = spectral_profile_similarity(image, pred, pred, num_bins=8)
    assert torch.isclose(sim, torch.tensor(1.0), atol=1e-4)


def test_phase_correlation_score_is_bounded():
    image, pred, target = _image_and_mask()
    score = phase_correlation_score(image, pred, target)
    assert 0.0 <= score.item() <= 1.0


def test_compute_structure_similarity_is_bounded():
    image, pred, target = _image_and_mask()
    score = compute_structure_similarity(image, pred, target)
    assert 0.0 <= score.item() <= 1.0


def test_edge_similarity_score_is_bounded():
    image, pred, target = _image_and_mask()
    score = edge_similarity_score(image, pred, target)
    assert 0.0 <= score.item() <= 1.0


def test_interpretable_signals_are_finite():
    image, pred, _ = _image_and_mask()
    assert torch.isfinite(boundary_phase_coherence(image, pred))
    assert torch.isfinite(interior_exterior_texture_contrast(image, pred))
    assert torch.isfinite(multi_scale_saliency_consistency(image, pred))
    assert torch.isfinite(score_edge_alignment(image, pred))
    assert torch.isfinite(activation_centroid_consistency(image, pred))
    assert torch.isfinite(aspect_ratio_plausibility(pred))
    assert torch.isfinite(nms_survivor_density(pred, [pred]))


def test_score_edge_alignment_responds_to_confidence():
    image, pred, _ = _image_and_mask()
    low_conf = score_edge_alignment(image, pred, torch.tensor(0.2))
    high_conf = score_edge_alignment(image, pred, torch.tensor(0.9))
    assert low_conf > high_conf
