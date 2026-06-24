"""Compatibility namespace for migrated FFT signal utilities.

New code should import from `spectral_detection_posttrain.signals.fft`.
"""

from spectral_detection_posttrain.signals.fft import compute_fft_amplitude, crop_and_resize_roi, radial_profile, spectral_reward

__all__ = ["crop_and_resize_roi", "compute_fft_amplitude", "radial_profile", "spectral_reward"]
