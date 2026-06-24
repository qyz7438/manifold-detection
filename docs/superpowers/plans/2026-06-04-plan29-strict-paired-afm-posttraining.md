# Plan 2.9: Multi-Epoch AFM + RPN + Post-Training Sanity

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer two open questions from Round 2.8: (1) do AFM FFT scales ever leave zero beyond 1 epoch? (2) can unfrozen RPN learn to ignore high-frequency edge patches? Plus a minimal post-training sanity check.

**Architecture:** Penn-Fudan + Faster R-CNN MobileNetV3. 11 groups, single seed (42), clean + object_edge eval per group.

**Tech Stack:** Python, PyTorch, TorchVision, existing `spectral_detection_posttrain` package.

---

## Experiment Matrix (11 groups)

| # | Group | AFM | Trainable | Epoch | Train Data | Eval | Question |
|---|-------|-----|-----------|-------|-----------|------|---------|
| G1 | `round29_g1_baseline_full` | none | full | 1 | clean | clean+edge | shared reference |
| G2 | `round29_g2_afm_delta_1ep` | identity_delta | afm_box_head | 1 | clean | clean+edge | FFT scale baseline |
| G3 | `round29_g3_afm_delta_3ep` | identity_delta | afm_box_head | 3 | clean | clean+edge | **scale leaves 0?** |
| G4 | `round29_g4_afm_delta_5ep` | identity_delta | afm_box_head | 5 | clean | clean+edge | **scale continues?** |
| G5 | `round29_g5_boxhead_3ep` | none | box_head_only | 3 | clean | clean+edge | epoch control |
| G6 | `round29_g6_boxhead_5ep` | none | box_head_only | 5 | clean | clean+edge | epoch control |
| G7 | `round29_g7_rpn_frozen` | none | box_head_only | 1 | clean | clean+edge | RPN reference |
| G8 | `round29_g8_rpn_unfrozen_clean` | none | rpn_box_head | 1 | clean | clean+edge | unfreeze baseline |
| G9 | `round29_g9_rpn_unfrozen_mixed` | none | rpn_box_head | 1 | 50%clean+50%edge | clean+edge | **learn to ignore?** |
| B1 | `round29_b1_ckpt_eval` | none | eval_only | 0 | none | clean+edge | checkpoint ref |
| B2 | `round29_b2_posttrain` | none | box_head_only | 1 | clean | clean+edge | post-train stability |

Shared: G1's checkpoint is used by B1/B2. `rpn_box_head` = freeze backbone, train RPN objectness + box_head + box_predictor.

---

## Key Measurements

Per group:
- Standard: AP50, AP75, precision, recall, ECE, high_conf_FP, num_predictions
- Localization: matched IoU mean, center error, size error, duplicate count
- AFM (G2-G4): mag_scale, phase_scale, residual_scale per epoch
- RPN (G7-G9): RPN proposal count on edge-patch images, AP50/AP75 on edge-patch scenes

---

## Success Criteria

```text
1. All 11 groups complete without NaN.
2. G3/G4 mag_scale or phase_scale > 0 at any epoch → FFT gating is trainable beyond 1ep.
3. G9 AP50_edge > G7 AP50_edge → RPN can learn to ignore high-frequency interference.
4. B2 AP50 >= B1 AP50 - 5% → post-training from checkpoint does not collapse.
```

---

## Implementation

- G1-G6: reuse `scripts/round28_train_eval.py` with extended `--epochs` and `--trainable-mode afm_box_head`
- G7-G9: reuse `scripts/round28_train_eval.py` with new `--trainable-mode rpn_box_head` and optional `--edge-mix` flag
- B1-B2: new `scripts/round29_posttrain.py` loading G1 checkpoint
- Edge eval: reuse `scripts/round28_diagnostics.py` with `--patch-mode object_edge`
- Summarizer: reuse `scripts/round28_summarize.py` pattern
