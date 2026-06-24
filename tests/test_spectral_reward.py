import torch

from spectral_detection_posttrain.spectral.fft_features import compute_fft_amplitude
from spectral_detection_posttrain.spectral.radial_profile import radial_profile
from spectral_detection_posttrain.spectral.roi_crop import crop_and_resize_roi
from spectral_detection_posttrain.spectral.spectral_reward import auc_tp_vs_fp, spectral_reward


def test_crop_and_resize_roi_shape():
    image = torch.rand(3, 40, 50)
    roi = crop_and_resize_roi(image, torch.tensor([5.0, 6.0, 30.0, 35.0]), size=32)
    assert roi.shape == (3, 32, 32)
    assert roi.min() >= 0
    assert roi.max() <= 1


def test_fft_amplitude_and_radial_profile_shapes():
    roi = torch.rand(3, 32, 32)
    amp = compute_fft_amplitude(roi)
    profile = radial_profile(amp, num_bins=8)
    assert amp.shape == (32, 32)
    assert profile.shape == (8,)
    assert torch.isfinite(profile).all()


def test_spectral_reward_higher_for_identical_roi():
    roi = torch.rand(3, 32, 32)
    random_roi = torch.rand(3, 32, 32)
    same_reward = spectral_reward(roi, roi)
    random_reward = spectral_reward(roi, random_roi)
    assert 0.0 <= same_reward <= 1.0
    assert 0.0 <= random_reward <= 1.0
    assert same_reward >= random_reward


def test_auc_tp_vs_fp():
    assert auc_tp_vs_fp([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc_tp_vs_fp([], [0.1]) is None
