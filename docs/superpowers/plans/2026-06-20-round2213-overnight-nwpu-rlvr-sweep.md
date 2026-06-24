# Round 2213+ Overnight NWPU RLVR Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a supervised 9-hour NWPU post-training experiment loop that improves clean AP75 beyond `round2211` while preserving FP/prediction safety.

**Architecture:** Continue from the clean `round2211` recipe and the in-progress `round2212` verifier-ranking run. Use one GPU, run one full experiment at a time, inspect results after each run, and choose the next run from a fixed priority queue.

**Tech Stack:** Python, PyTorch, TorchVision Faster R-CNN, PowerShell, `scripts/round2129_nwpu_posttrain_smoke.py`, clean eval via `evaluate_clean_detector`.

---

### Task 1: Finish Current Verifier Ranking Run

**Files:**
- Read: `runs/round2212_fft_ranking_global_clean_15ep/metrics_train.json`
- Read: `runs/round2212_fft_ranking_global_clean_15ep/eval_metrics.json`
- Read: `runs/round2212_fft_ranking_global_clean_15ep/rescue_diagnostics.json`

- [ ] Wait for PID `40732` to finish.
- [ ] Extract final AP75, best AP75, pred count, FP rate, ECE, `verifier_ranking_pair_count`, and `verifier_positive_lchi_prob_delta_mean`.
- [ ] Compare against `round2211` clean best AP75 `0.3026143912147811`.
- [ ] If AP75 improves by at least `+0.001`, prioritize verifier-ranking weight sweep next.
- [ ] If AP75 does not improve, prioritize longer `round2211` training next.

### Task 2: Verifier Ranking Weight Sweep

**Files:**
- Modify only by running: `scripts/round2129_nwpu_posttrain_smoke.py`
- Output: `runs/round2213_fft_ranking_w003_clean_15ep`
- Output: `runs/round2214_fft_ranking_w003_or_w03_clean_15ep`

- [ ] Run `verifier_ranking_loss_weight=0.003` if `round2212` is too aggressive or hurts AP75.
- [ ] Run `verifier_ranking_loss_weight=0.03` only if `round2212` improves and safety metrics stay clean.
- [ ] Keep `lr=1e-4`, `KL=1.0`, `det_loss_weight=0.5`, `policy_loss_weight=0.001`.
- [ ] Do not use `0.05` or `0.1` until `0.03` has clean evidence.

### Task 3: Longer Round2211 Continuation Recipe

**Files:**
- Output: `runs/round2215_lr1e4_30ep_clean`

- [ ] Run the `round2211` recipe for `30` epochs if `round2212` does not beat `round2211` or if epoch 15 still appears to be rising.
- [ ] Keep `KL=1.0`; do not drop to `0.1` or `0.01`.
- [ ] Treat best checkpoint as valid only if safety guard permits saving.

### Task 4: Conservative Policy Weight Sweep

**Files:**
- Output: `runs/round2216_policy003_clean_15ep`
- Output: `runs/round2217_policy005_clean_15ep`

- [ ] Run `policy_loss_weight=0.003` first.
- [ ] Run `policy_loss_weight=0.005` only if `0.003` does not increase FP/pred count.
- [ ] Keep `lr=1e-4`, `KL=1.0`, `det_loss_weight=0.5`.
- [ ] Do not run `policy_loss_weight=0.01` before seeing `0.005`.

### Task 5: Soft Verifier / Coverage Expansion

**Files:**
- Output: `runs/round2218_softgate_clean_15ep`

- [ ] Run only after at least one of Tasks 2-4 shows stable or improved AP75.
- [ ] Use `rescue_verifier_weight_mode=sigmoid`.
- [ ] Keep `rescue_verifier_gate=0.0` initially; do not lower gate and enable soft mode in the same first run.
- [ ] If soft gate helps, later test `rescue_high_iou_min=0.7` separately.

### Task 6: Reporting

**Files:**
- Create/update: `runs/round2213_overnight_summary.json`

- [ ] After each run, append clean final AP50/AP75, best AP75, baseline AP75, pred count, FP rate, ECE, key loss weights, and decision.
- [ ] Every ~1500 seconds, record current run status and current best.
- [ ] At the end of 9 hours, report best clean run and the next recommended experiment.
