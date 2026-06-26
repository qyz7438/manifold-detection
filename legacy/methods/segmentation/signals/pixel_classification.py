"""Registry of interpretable pixel-level classification signals for segmentation.

This module migrates the detection-side signal families defined in
``spectral_detection_posttrain.signals.pixel_classification.registry`` and in
``scripts/diagnose_interpretable_reward_signals.py`` into a structured catalog
suitable for dense prediction.  Each spec describes what the signal measures,
whether it needs FFT, whether it requires GT statistics for calibration, and
recommended reward uses.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PixelSignalSpec:
    signal_id: str
    name: str
    description: str
    uses_fft: bool
    requires_gt_for_calibration: bool
    recommended_use: tuple[str, ...]


PIXEL_CLASSIFICATION_SIGNALS: tuple[PixelSignalSpec, ...] = (
    PixelSignalSpec(
        signal_id="pix.boundary.phase_coherence",
        name="Boundary Phase Coherence",
        description="Mask boundaries align with phase-only reconstruction edges or phase-gradient discontinuities.",
        uses_fft=True,
        requires_gt_for_calibration=False,
        recommended_use=("boundary_reward", "sample_weight"),
    ),
    PixelSignalSpec(
        signal_id="pix.boundary.edge_alignment",
        name="Edge Alignment",
        description="Predicted contours overlap strong image-gradient responses.",
        uses_fft=False,
        requires_gt_for_calibration=False,
        recommended_use=("boundary_reward", "false_boundary_penalty"),
    ),
    PixelSignalSpec(
        signal_id="pix.region.interior_exterior_contrast",
        name="Interior-Exterior Texture Contrast",
        description="Foreground texture differs from the immediate exterior ring.",
        uses_fft=True,
        requires_gt_for_calibration=False,
        recommended_use=("objectness_reward", "soft_verifier"),
    ),
    PixelSignalSpec(
        signal_id="pix.region.frequency_compactness",
        name="Frequency Compactness",
        description="Foreground mask and masked image patch have plausible frequency distribution.",
        uses_fft=True,
        requires_gt_for_calibration=False,
        recommended_use=("fragmentation_penalty", "mask_quality_prior"),
    ),
    PixelSignalSpec(
        signal_id="pix.region.spectral_residual_saliency",
        name="Spectral Residual Saliency",
        description="Predicted foreground overlaps image-level spectral saliency.",
        uses_fft=True,
        requires_gt_for_calibration=False,
        recommended_use=("weak_objectness_verifier",),
    ),
    PixelSignalSpec(
        signal_id="pix.calibration.confidence_boundary_consistency",
        name="Confidence-Boundary Consistency",
        description="Pixel confidence is high in stable interiors and lower near boundaries.",
        uses_fft=False,
        requires_gt_for_calibration=False,
        recommended_use=("calibration_regularizer",),
    ),
    PixelSignalSpec(
        signal_id="pix.consistency.multiscale_mask_stability",
        name="Multi-Scale Mask Stability",
        description="Predicted masks remain stable under image resizing.",
        uses_fft=False,
        requires_gt_for_calibration=False,
        recommended_use=("consistency_reward", "pseudo_label_filter"),
    ),
    PixelSignalSpec(
        signal_id="pix.consistency.tta_agreement",
        name="TTA Agreement",
        description="Predictions agree under reversible transforms.",
        uses_fft=False,
        requires_gt_for_calibration=False,
        recommended_use=("self_consistency_reward",),
    ),
    PixelSignalSpec(
        signal_id="pix.geometry.connected_component_plausibility",
        name="Connected Component Plausibility",
        description="Foreground components have plausible area, aspect ratio, and compactness.",
        uses_fft=False,
        requires_gt_for_calibration=True,
        recommended_use=("false_positive_penalty", "classwise_prior"),
    ),
    PixelSignalSpec(
        signal_id="pix.geometry.shape_prior",
        name="Shape Prior",
        description="Class-specific masks match calibrated shape statistics.",
        uses_fft=False,
        requires_gt_for_calibration=True,
        recommended_use=("classwise_soft_prior",),
    ),
    PixelSignalSpec(
        signal_id="pix.topology.hole_boundary_penalty",
        name="Hole/Boundary Topology Penalty",
        description="Penalizes implausible holes, broken boundaries, and fragmented topology.",
        uses_fft=False,
        requires_gt_for_calibration=True,
        recommended_use=("topology_regularizer",),
    ),
    PixelSignalSpec(
        signal_id="pix.uncertainty.entropy_edge_coupling",
        name="Entropy-Edge Coupling",
        description="Prediction entropy correlates with actual image edges instead of stable interiors.",
        uses_fft=False,
        requires_gt_for_calibration=False,
        recommended_use=("calibration_reward", "uncertainty_filter"),
    ),
    PixelSignalSpec(
        signal_id="pix.fft.amplitude_profile_similarity",
        name="Amplitude Profile Similarity",
        description="Radial amplitude profile of masked foreground matches expected target spectrum.",
        uses_fft=True,
        requires_gt_for_calibration=False,
        recommended_use=("spectral_reward", "mask_quality_reward"),
    ),
    PixelSignalSpec(
        signal_id="pix.fft.phase_correlation",
        name="Phase Correlation",
        description="Phase-only cross-correlation between predicted and target masked regions.",
        uses_fft=True,
        requires_gt_for_calibration=False,
        recommended_use=("spectral_reward", "boundary_phase_reward"),
    ),
    PixelSignalSpec(
        signal_id="pix.fft.lowfreq_phase_similarity",
        name="Low-Frequency Phase Similarity",
        description="Low-frequency phase statistics are consistent between prediction and target.",
        uses_fft=True,
        requires_gt_for_calibration=False,
        recommended_use=("spectral_reward", "structure_reward"),
    ),
    PixelSignalSpec(
        signal_id="pix.fft.structure_similarity",
        name="Structure Similarity",
        description="Combined phase, edge, and low-frequency structure similarity of masked regions.",
        uses_fft=True,
        requires_gt_for_calibration=False,
        recommended_use=("spectral_reward", "structure_reward"),
    ),
    PixelSignalSpec(
        signal_id="pix.spatial.mask_iou",
        name="Mask IoU",
        description="Pixel-level intersection-over-union between predicted and target masks.",
        uses_fft=False,
        requires_gt_for_calibration=False,
        recommended_use=("spatial_reward", "verifiable_reward"),
    ),
    PixelSignalSpec(
        signal_id="pix.spatial.dice",
        name="Dice Score",
        description="Dice overlap between predicted and target masks.",
        uses_fft=False,
        requires_gt_for_calibration=False,
        recommended_use=("spatial_reward", "verifiable_reward"),
    ),
    PixelSignalSpec(
        signal_id="pix.spatial.boundary_f1",
        name="Boundary F1",
        description="Boundary precision/recall between predicted and target masks within a tolerance band.",
        uses_fft=False,
        requires_gt_for_calibration=False,
        recommended_use=("boundary_reward", "spatial_reward"),
    ),
    PixelSignalSpec(
        signal_id="pix.spatial.connected_component",
        name="Connected Component Consistency",
        description="Foreground coverage versus extraneous fragmented predictions.",
        uses_fft=False,
        requires_gt_for_calibration=False,
        recommended_use=("fragmentation_penalty", "spatial_reward"),
    ),
    PixelSignalSpec(
        signal_id="pix.spatial.centroid_alignment",
        name="Centroid Alignment",
        description="Predicted mask centroid aligns with target mask centroid relative to mask size.",
        uses_fft=False,
        requires_gt_for_calibration=False,
        recommended_use=("localization_reward", "spatial_reward"),
    ),
    PixelSignalSpec(
        signal_id="pix.spatial.aspect_ratio_plausibility",
        name="Aspect Ratio Plausibility",
        description="Predicted mask bounding box aspect ratio is plausible for the class.",
        uses_fft=False,
        requires_gt_for_calibration=True,
        recommended_use=("shape_prior", "false_positive_penalty"),
    ),
    PixelSignalSpec(
        signal_id="pix.region.multi_scale_saliency",
        name="Multi-Scale Saliency",
        description="Predicted mask overlaps salient edges at multiple scales.",
        uses_fft=False,
        requires_gt_for_calibration=False,
        recommended_use=("objectness_reward", "weak_verifier"),
    ),
    PixelSignalSpec(
        signal_id="pix.region.activation_centroid",
        name="Activation Centroid Consistency",
        description="Saliency centroid of masked region aligns with the mask center.",
        uses_fft=False,
        requires_gt_for_calibration=False,
        recommended_use=("objectness_reward", "sample_weight"),
    ),
)

SIGNAL_COUNT: int = len(PIXEL_CLASSIFICATION_SIGNALS)


def signal_by_id(signal_id: str) -> PixelSignalSpec | None:
    for spec in PIXEL_CLASSIFICATION_SIGNALS:
        if spec.signal_id == signal_id:
            return spec
    return None


def signal_ids() -> tuple[str, ...]:
    return tuple(spec.signal_id for spec in PIXEL_CLASSIFICATION_SIGNALS)
