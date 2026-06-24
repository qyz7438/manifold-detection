"""Segmentation signal extraction and verifier features.

This package mirrors ``spectral_detection_posttrain.signals`` but operates on
pixel-level masks instead of bounding boxes.  It exposes three signal families:

* ``fft`` — amplitude/phase structure features computed over masked image regions.
* ``geometry`` — spatial overlap and boundary rewards between predicted and target masks.
* ``pixel_classification`` — registry of interpretable pixel-level signal specs.

All tensor functions accept ``(C, H, W)`` images and ``(H, W)`` boolean masks so
segmentation models can consume them directly without box-level plumbing.
"""

from spectral_detection_posttrain.methods.segmentation.signals import fft, geometry, pixel_classification
from spectral_detection_posttrain.methods.segmentation.signals.fft import (
    compute_amplitude_profile,
    compute_fft_amplitude,
    compute_lowfreq_phase_stats,
    compute_structure_similarity,
    phase_correlation_score,
    radial_amplitude_profile,
)
from spectral_detection_posttrain.methods.segmentation.signals.geometry import (
    boundary_reward,
    connected_component_reward,
    dice_reward,
    mask_iou_reward,
)
from spectral_detection_posttrain.methods.segmentation.signals.interpretable import (
    activation_centroid_consistency,
    aspect_ratio_plausibility,
    boundary_phase_coherence,
    interior_exterior_texture_contrast,
    multi_scale_saliency_consistency,
    nms_survivor_density,
    score_edge_alignment,
)
from spectral_detection_posttrain.methods.segmentation.signals.pixel_classification import (
    PIXEL_CLASSIFICATION_SIGNALS,
    PixelSignalSpec,
    signal_by_id,
    signal_ids,
)

__all__ = [
    "PIXEL_CLASSIFICATION_SIGNALS",
    "PixelSignalSpec",
    "activation_centroid_consistency",
    "aspect_ratio_plausibility",
    "boundary_phase_coherence",
    "boundary_reward",
    "connected_component_reward",
    "compute_amplitude_profile",
    "compute_fft_amplitude",
    "compute_lowfreq_phase_stats",
    "compute_structure_similarity",
    "dice_reward",
    "fft",
    "geometry",
    "interior_exterior_texture_contrast",
    "mask_iou_reward",
    "multi_scale_saliency_consistency",
    "nms_survivor_density",
    "phase_correlation_score",
    "pixel_classification",
    "radial_amplitude_profile",
    "score_edge_alignment",
    "signal_by_id",
    "signal_ids",
]
