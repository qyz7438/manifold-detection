# Package Migration And Signal Layout

Date: 2026-06-21

## Canonical Layout

New code should use these namespaces:

| Area | Canonical Path | Purpose |
| --- | --- | --- |
| Core detector/matching | `spectral_detection_posttrain/core/` | Detector builders, box matching, shared low-level primitives |
| AFM | `spectral_detection_posttrain/methods/afm/` | In-network FFT/AFM model modules |
| RLVR | `spectral_detection_posttrain/methods/rlvr/` | RLVR losses, confidence rescue, verifier networks |
| DPO | `spectral_detection_posttrain/methods/dpo/` | Action verifier and DPO utilities |
| FFT signals | `spectral_detection_posttrain/signals/fft/` | FFT/iFFT features, spectral rewards, ROI spectral caches |
| Geometry signals | `spectral_detection_posttrain/signals/geometry/` | Box geometry and spatial reward signals |
| Pixel-level signals | `spectral_detection_posttrain/signals/pixel_classification/` | Code registry for segmentation/pixel-classification signals |
| Pixel signal notes | `signal/pixel_classification/` | Human-readable registry; not a Python package |
| Trainers | `spectral_detection_posttrain/trainers/` | Detection and segmentation training entry modules |

## Compatibility Shims

These old paths remain as import-compatible wrappers:

- `spectral_detection_posttrain/spectral/*` -> `spectral_detection_posttrain/signals/fft/*`
- `spectral_detection_posttrain/rlvr/*` -> `spectral_detection_posttrain/methods/rlvr/*` or DPO where appropriate
- `spectral_detection_posttrain/models/*` -> `spectral_detection_posttrain/core/models/*` or `methods/afm/*`
- `spectral_detection_posttrain/matching/*` -> `spectral_detection_posttrain/core/matching/*`
- `spectral_detection_posttrain/train/*` -> `spectral_detection_posttrain/trainers/detection/*`

Historical `scripts/round*.py` are intentionally not bulk-migrated. They are experiment artifacts and must remain reproducible.

## Signal Directory Note

The root-level `signal/` directory is intentionally a documentation/registry directory only. It must not contain `__init__.py`, because `signal` conflicts with Python's standard-library module name.

Runnable signal code belongs under `spectral_detection_posttrain/signals/`.

## Round 2221 Fusion Result

The best offline LC-HI verifier fusion is:

- `score_edge_alignment`
- `boundary_phase_coherence`
- `interior_exterior_texture_contrast`
- `reference_raw_ifft_recipe`

With balanced logistic fusion C=0.25:

- Val AUC: 0.836
- Val AP: 0.451
- Fixed-threshold val precision: 0.812
- Fixed-threshold val recall: 0.317

Adding all seven new signals or all 115 legacy iFFT features worsened transfer, so next training integration should use the narrow four-feature fusion.
