import torch

from spectral_detection_posttrain.spectral.round211_spectral_gate import radial_amplitude_profile, spectral_gate_score, shuffled_scores


def test_radial_amplitude_profile_is_finite():
    roi = torch.rand(3, 32, 32)
    profile = radial_amplitude_profile(roi, bins=8)
    assert profile.shape == (8,)
    assert torch.isfinite(profile).all()


def test_spectral_gate_score_identical_is_high():
    roi = torch.rand(3, 32, 32)
    score = spectral_gate_score(roi, roi)
    assert float(score) > 0.99


def test_shuffled_scores_preserves_shape():
    scores = torch.rand(4)
    shuffled = shuffled_scores(scores)
    assert shuffled.shape == scores.shape
