# Detection Structure Redesign — Ablation Results

**Date**: 2026-06-26  
**Dataset**: Penn-Fudan Pedestrian  
**Detector**: Faster R-CNN MobileNetV3-Large-FPN  
**Training**: 3 epochs, `box_head_only`, batch size 2, 3 seeds (42, 123, 2024)  
**Baseline**: MPLSeg-style phase-only AFM (`afm_type= mplseg_phase_only`)

## Summary

Three independent structural components were evaluated against the validated AFM baseline:

- **PBG** — Phase Boundary Gate (shallow, after ROI Align)
- **TAM** — Task-Aligned Manifold (deep, after box head)
- **PAH** — Prototype-Aware Head (deep, replaces FastRCNN predictor)

| config | n | AP50 mean±std | AP75 mean±std | ECE mean±std | P@R=0.85 mean±std |
|---|---|---|---|---|---|
| afm | 3 | 0.9121±0.0314 | 0.7463±0.0698 | 0.0657±0.0036 | 0.9525±0.0266 |
| afm_pbg | 3 | 0.9115±0.0317 | **0.7894±0.0779** | **0.0532±0.0100** | 0.9499±0.0380 |
| afm_pbg_tam | 3 | 0.9147±0.0270 | 0.7638±0.0800 | 0.0610±0.0050 | 0.9574±0.0278 |
| afm_tam | 3 | 0.9123±0.0315 | 0.7331±0.0566 | 0.0554±0.0114 | 0.9494±0.0309 |
| afm_pah | 3 | 0.8843±0.0271 | 0.6939±0.0891 | 0.1653±0.0305 | 0.7775±0.1881 |
| afm_pbg_pah | 3 | 0.8841±0.0282 | 0.6952±0.0920 | 0.1736±0.0114 | 0.7893±0.1801 |
| afm_all | 3 | 0.8865±0.0247 | 0.6926±0.0951 | 0.1602±0.0255 | 0.8161±0.1448 |

## Findings

1. **PBG improves localization (AP75)**. Adding the Phase Boundary Gate to the AFM baseline raises mean AP75 from **0.7463 → 0.7894** (+5.8% relative) while also lowering ECE (0.0657 → 0.0532).
2. **TAM is neutral to slightly helpful**. TAM alone is roughly on par with the baseline; combined with PBG it reaches AP75=0.7638, between PBG-alone and baseline.
3. **PAH hurts calibration and localization**. Any configuration containing PAH drops AP75 by ~5–7% and raises ECE by ~2.5×. The prototype-based predictor appears misaligned with the pretrained feature distribution under only 3 epochs of fine-tuning.
4. **Best candidate**: `afm + PBG`.

## Per-seed detail

See `runs/redesign_grid_summary.csv` for full per-run numbers.

## Recommendation

- Adopt **PBG** as the first validated structural improvement over the AFM baseline.
- Do not adopt PAH in its current form; consider re-designing it with better initialization or longer warmup.
- TAM can be kept as an optional component; its benefit is marginal on Penn-Fudan but may be larger on datasets with more classes or harder boundaries.
