# Pixel Classification Signals

This directory records candidate verifier/reward signals for pixel-level classification tasks, including semantic segmentation and dense binary masks.

The directory is intentionally not a Python package. The name `signal` conflicts with Python's standard-library `signal` module, so code should live under `spectral_detection_posttrain/signals/`; this folder is a task-facing registry and design note.

## Signal Families

| ID | Signal | What It Checks | Uses FFT | Needs GT | Primary Use |
| --- | --- | --- | --- | --- | --- |
| pix.boundary.phase_coherence | Boundary phase coherence | Predicted mask boundaries align with phase/edge discontinuities | yes | optional | boundary reward / sample weight |
| pix.boundary.edge_alignment | Edge alignment | Mask contour overlaps Sobel/Canny-like edge response | no | optional | boundary quality reward |
| pix.region.interior_exterior_contrast | Interior-exterior texture contrast | Mask interior differs from immediate exterior ring | optional | no | objectness verifier |
| pix.region.frequency_compactness | Frequency compactness | Foreground mask has plausible low/mid/high-frequency distribution | yes | optional | false-positive suppression |
| pix.region.spectral_residual_saliency | Spectral residual saliency | Foreground overlaps image-level spectral saliency | yes | no | weak verifier / pseudo reward |
| pix.calibration.confidence_boundary_consistency | Confidence-boundary consistency | High-confidence pixels are not concentrated on uncertain boundaries | no | no | calibration regularizer |
| pix.consistency.multiscale_mask_stability | Multi-scale mask stability | Prediction remains stable under image scale changes | optional | no | consistency reward |
| pix.consistency.tta_agreement | TTA agreement | Prediction agrees under flips/crops/color jitter | no | no | self-consistency reward |
| pix.geometry.connected_component_plausibility | Component plausibility | Connected components have plausible area/aspect/compactness | no | optional | FP suppression |
| pix.geometry.shape_prior | Shape prior | Predicted regions match class-specific shape statistics | no | yes, for calibration | class-specific verifier |
| pix.topology.hole_boundary_penalty | Hole/boundary topology | Penalizes implausible holes and fragmented masks | no | optional | topology regularizer |
| pix.uncertainty.entropy_edge_coupling | Entropy-edge coupling | Uncertainty should concentrate near actual edges, not interiors | no | no | calibration reward |

## Current Priority

For short offline validation, start with:

1. `pix.boundary.phase_coherence`
2. `pix.region.interior_exterior_contrast`
3. `pix.consistency.multiscale_mask_stability`
4. `pix.uncertainty.entropy_edge_coupling`

These are interpretable, cheap to compute, and map cleanly to segmentation outputs.
