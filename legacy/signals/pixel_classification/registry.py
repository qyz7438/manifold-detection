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
)
