"""Canonical FFT/iFFT signal utilities."""

from .fft_features import (
    compute_amplitude_profile,
    compute_fft_amplitude,
    compute_lowfreq_phase_stats,
    compute_sobel_structure_features,
    compute_structure_similarity,
    edge_similarity_score,
    lowfreq_phase_similarity,
    phase_correlation_score,
)
from .radial_profile import radial_profile
from .raw_ifft_features import (
    crop_and_resize_boxes,
    penn_fudan_legacy_ifft_metric_bank,
    raw_ifft_feature_summary,
    sobel_edge_strength,
)
from .raw_ifft_verifier import (
    LEGACY_IFFT_FEATURE_NAMES,
    CalibratedThreshold,
    TrainEffectScorer,
    apply_selection_policy,
    calibrate_precision_threshold,
    fit_train_effect_scorer,
    parse_legacy_ifft_feature_specs,
    score_legacy_ifft_metric_bank,
    score_scene_legacy_ifft_metric_bank,
    threshold_metrics,
)
from .roi_crop import crop_and_resize_roi
from .roi_spectral_dataset import (
    RoiSpectralCandidateDataset,
    apply_nms_to_prediction,
    build_candidate_sample,
    extract_roi_box_features,
    load_candidate_cache,
    save_candidate_cache,
)
from .round211_spectral_gate import radial_amplitude_profile, shuffled_scores, spectral_gate_score
from .spectral_reward import auc_tp_vs_fp, compute_prediction_rewards, prediction_reward, spectral_reward

__all__ = [
    "LEGACY_IFFT_FEATURE_NAMES",
    "RoiSpectralCandidateDataset",
    "CalibratedThreshold",
    "TrainEffectScorer",
    "apply_selection_policy",
    "apply_nms_to_prediction",
    "auc_tp_vs_fp",
    "build_candidate_sample",
    "calibrate_precision_threshold",
    "compute_amplitude_profile",
    "compute_fft_amplitude",
    "compute_lowfreq_phase_stats",
    "compute_prediction_rewards",
    "compute_sobel_structure_features",
    "compute_structure_similarity",
    "crop_and_resize_boxes",
    "crop_and_resize_roi",
    "edge_similarity_score",
    "extract_roi_box_features",
    "fit_train_effect_scorer",
    "load_candidate_cache",
    "lowfreq_phase_similarity",
    "parse_legacy_ifft_feature_specs",
    "penn_fudan_legacy_ifft_metric_bank",
    "phase_correlation_score",
    "prediction_reward",
    "radial_amplitude_profile",
    "radial_profile",
    "raw_ifft_feature_summary",
    "save_candidate_cache",
    "score_legacy_ifft_metric_bank",
    "score_scene_legacy_ifft_metric_bank",
    "shuffled_scores",
    "sobel_edge_strength",
    "spectral_gate_score",
    "spectral_reward",
    "threshold_metrics",
]
