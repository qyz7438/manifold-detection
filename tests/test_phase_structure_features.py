import torch

from spectral_detection_posttrain.spectral.fft_features import (
    compute_structure_similarity,
    edge_similarity_score,
    lowfreq_phase_similarity,
    phase_correlation_score,
)


def _roi_with_square(shift_x: int = 0, shift_y: int = 0) -> torch.Tensor:
    roi = torch.zeros((3, 64, 64), dtype=torch.float32)
    y1 = 18 + shift_y
    y2 = 46 + shift_y
    x1 = 20 + shift_x
    x2 = 44 + shift_x
    roi[:, y1:y2, x1:x2] = 1.0
    return roi


def test_phase_correlation_is_high_for_identical_roi():
    roi = _roi_with_square()
    score = phase_correlation_score(roi, roi)
    assert 0.90 <= float(score.item()) <= 1.0


def test_phase_correlation_tolerates_small_translation_better_than_noise():
    roi = _roi_with_square()
    shifted = _roi_with_square(shift_x=2, shift_y=1)
    noise = torch.rand_like(roi)
    shifted_score = phase_correlation_score(roi, shifted)
    noise_score = phase_correlation_score(roi, noise)
    assert shifted_score > noise_score


def test_edge_similarity_rewards_same_structure():
    roi = _roi_with_square()
    shifted = _roi_with_square(shift_x=3, shift_y=2)
    noise = torch.rand_like(roi)
    same = edge_similarity_score(roi, roi)
    moved = edge_similarity_score(roi, shifted)
    random = edge_similarity_score(roi, noise)
    assert same > moved
    assert same > random  # identical ROI edges closer than noise
    assert 0.0 <= float(random.item()) <= 1.0


def test_lowfreq_phase_similarity_is_bounded():
    roi = _roi_with_square()
    noise = torch.rand_like(roi)
    score = lowfreq_phase_similarity(roi, noise)
    assert 0.0 <= float(score.item()) <= 1.0


def test_structure_similarity_combines_phase_and_edges():
    roi = _roi_with_square()
    shifted = _roi_with_square(shift_x=2, shift_y=2)
    noise = torch.rand_like(roi)
    same = compute_structure_similarity(roi, roi)
    moved = compute_structure_similarity(roi, shifted)
    random = compute_structure_similarity(roi, noise)
    assert same > moved
    assert moved > random
    assert 0.0 <= float(random.item()) <= 1.0
    assert 0.0 <= float(same.item()) <= 1.0
