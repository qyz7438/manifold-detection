from __future__ import annotations

import torch

from spectral_detection_posttrain.models.spectral_quality_head import (
    SpectralQualityHead,
    build_quality_features,
    make_quality_targets,
    normalize_r_amp,
    pairwise_ranking_loss,
    quality_input_dim,
)
from spectral_detection_posttrain.spectral.fft_features import (
    compute_amplitude_profile,
    compute_lowfreq_phase_stats,
    compute_sobel_structure_features,
)
from spectral_detection_posttrain.spectral.roi_spectral_dataset import apply_nms_to_prediction


def _sample() -> dict:
    return {
        "roi_features": torch.randn(3, 6),
        "amp_profiles": torch.randn(3, 4),
        "structure_features": torch.randn(3, 2),
        "raw_r_amp": torch.tensor([0.8, 0.6, 0.9]),
        "ious": torch.tensor([0.9, 0.2, 0.7]),
        "is_tp": torch.tensor([True, False, True]),
        "scores": torch.tensor([0.9, 0.7, 0.6]),
    }


def test_quality_head_feature_modes_and_forward() -> None:
    meta = {"roi_feature_dim": 6, "amp_bins": 4, "structure_dim": 2}
    assert quality_input_dim(meta, "roi_amp_structure") == 12
    features = build_quality_features(_sample(), "roi_amp_structure")
    head = SpectralQualityHead(input_dim=12, hidden_dim=32, dropout=0.0)
    logits = head(features)
    assert logits.shape == (3,)
    quality = head.predict_quality(features)
    assert torch.all((quality >= 0) & (quality <= 1))


def test_quality_targets_and_ranking_loss() -> None:
    sample = _sample()
    stats = {"mode": "minmax", "min": 0.5, "max": 1.0}
    targets = make_quality_targets(sample, stats)
    assert targets.tolist()[1] == 0.0
    assert targets.max() <= 1.0
    logits = torch.tensor([2.0, -1.0, 1.0])
    assert pairwise_ranking_loss(logits, sample["is_tp"], margin=0.2).item() == 0.0
    bad_logits = torch.tensor([0.0, 1.0, 0.0])
    assert pairwise_ranking_loss(bad_logits, sample["is_tp"], margin=0.2).item() > 0.0
    normalized = normalize_r_amp(torch.tensor([0.5, 0.75, 1.0]), stats)
    assert torch.allclose(normalized, torch.tensor([0.0, 0.5, 1.0]))


def test_spectral_structure_features_shapes() -> None:
    roi = torch.rand(3, 32, 32)
    assert compute_amplitude_profile(roi, num_bins=8).shape == (8,)
    assert compute_lowfreq_phase_stats(roi).shape == (4,)
    assert compute_sobel_structure_features(roi).shape == (8,)


def test_apply_nms_to_prediction_filters_scores() -> None:
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0], [20.0, 20.0, 30.0, 30.0]]),
        "labels": torch.tensor([1, 1, 1]),
        "scores": torch.tensor([0.9, 0.8, 0.01]),
    }
    filtered = apply_nms_to_prediction(prediction, iou_threshold=0.5, score_threshold=0.05)
    assert len(filtered["scores"]) == 1
    assert torch.isclose(filtered["scores"][0], torch.tensor(0.9))
