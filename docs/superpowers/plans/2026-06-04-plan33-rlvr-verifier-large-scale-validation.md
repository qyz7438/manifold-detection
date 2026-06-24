# Plan 3.3 RLVR Verifier Large-Scale Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a large, controlled RLVR validation proving whether an IoU-primary, high-confidence-FP-aware verifier can causally improve detector behavior, with frequency evidence kept only as a small auxiliary/control signal.

**Architecture:** Reuse the stable KL-anchored signed ROI policy shell from `spectral_detection_posttrain.train.posttrain_rlvr`, but stop using hand-built spectral profile as the main reward. Use two policy initializations (`baseline_1ep` and `mid06_3ep`), evaluate IoU-only, IoU+high-conf-FP, real spectral auxiliary, and shuffled spectral auxiliary under a staged 288-trial maximum matrix, and judge success with fixed-recall precision, high-conf FP, AP75, reward-scale diagnostics, and TP/FP log-probability shifts.

**Tech Stack:** Python, PyTorch, TorchVision Faster R-CNN, Penn-Fudan, existing `spectral_detection_posttrain` package, pytest, PowerShell, conda env `E:\anaconda\01\envs\RLimage`.

---

## Starting Point

Plan 2.16 and Plan 2.16+ established these facts:

```text
1. Hand-built R_amp / structure profile is not a valid main RLVR reward:
   - It is numerically compressed.
   - It does not beat shuffled controls in detection post-training.

2. MPLSeg-style AFM is useful as a detector initialization / architecture probe:
   - mid06_3ep improves precision and ECE compared with mid06_1ep.
   - It is still ordinary detector training, not RLVR.

3. RLVR must return to a verifier that is both:
   - not numerically compressed;
   - causally tied to detection quality.
```

Therefore Plan 3.3 makes `IoU` and `high-conf FP penalty` the primary verifier. Spectral evidence appears only as a small auxiliary term and must beat its shuffled control before any causal claim is allowed.

---

## Scientific Claims Allowed After Plan 3.3

### RLVR Works Claim

Allowed only if a RLVR group beats its matched no-RLVR initialization under all gates:

```text
Precision@Recall=0.90 improves by >= 0.03 on clean.
High-conf FP at matched recall decreases by >= 20%.
AP50 drop is no worse than -0.01.
AP75 drop is no worse than -0.02.
num_predictions does not increase by more than 15%.
KL-to-reference remains finite and logged every epoch.
Behavior diagnostic passes:
  mean_delta_logprob(low-reward FP) <= -0.02
  mean_delta_logprob(high-reward TP) >= -0.005
  corr(reward, delta_logprob) >= 0.10
```

### Spectral Auxiliary Claim

Allowed only if real spectral auxiliary beats shuffled spectral auxiliary:

```text
real_amp_aux - shuffled_amp_aux >= 0.01 AP75 mean across clean/object_edge/object_inside/near_object
and real_amp_aux - shuffled_amp_aux >= 0.02 Precision@Recall=0.90 on clean
and real_amp_aux does not increase High-conf FP.
```

If these gates fail, the correct conclusion is:

```text
IoU-primary RLVR may be useful, but hand-built spectral auxiliary remains unproven.
```

---

## Search Budget

The full matrix is intentionally large but staged.

```text
Initializations:
  baseline_1ep
  mid06_3ep

Reward modes:
  iou_only
  iou_hconf
  iou_hconf_amp_aux
  iou_hconf_shuffled_amp_aux

Policy weights:
  0.0003
  0.001
  0.003

KL weights:
  3
  10

Unfreeze modes:
  cls
  roi

Seeds:
  42
  123
  456

Total max:
  2 * 4 * 3 * 2 * 2 * 3 = 288 RLVR trials
```

Run order:

```text
Phase A sanity gate:
  24 trials
  policy_weight=0.0003, KL=10, unfreeze=cls, all inits/rewards/seeds

Phase B policy/KL expansion:
  144 total trials
  policy_weight in {0.0003, 0.001, 0.003}, KL in {3,10}, unfreeze=cls

Phase C ROI expansion:
  288 total trials
  only if Phase B finds at least one non-collapsed RLVR group
```

Expected cost:

```text
Phase A: 3-8 GPU hours
Phase B: 12-30 GPU hours
Phase C: 24-60 GPU hours if run from scratch; less if Phase B rows are skipped
Disk: 40-120 GB depending checkpoint retention
```

Stop after Phase A if AP50/Recall collapse or behavior diagnostics fail for all RLVR groups.

---

## File Map

- Modify: `spectral_detection_posttrain/eval/detection_metrics.py`
  Add multi-target fixed-recall metrics and threshold-curve summaries.

- Create: `tests/test_round33_threshold_metrics.py`
  Unit tests for fixed-recall precision and high-conf FP at matched recall.

- Modify: `spectral_detection_posttrain/train/posttrain_rlvr.py`
  Add explicit reward weights, git/CLI metadata, reward component summaries, and per-epoch action log-probability diagnostics.

- Create: `spectral_detection_posttrain/analysis/rlvr_behavior_diagnostics.py`
  Compare baseline/current ROI log-probabilities for high-reward TP and low-reward FP actions.

- Create: `tests/test_round33_behavior_diagnostics.py`
  Unit tests for behavior diagnostic aggregation.

- Create: `scripts/round33_prepare_manifest.py`
  Create seed-specific checkpoint/config manifest for `baseline_1ep` and `mid06_3ep`.

- Create: `tests/test_round33_manifest.py`
  Verify AFM checkpoint configs match their checkpoint architecture.

- Create: `scripts/round33_run_matrix.py`
  Generate and execute the staged RLVR matrix.

- Create: `scripts/round33_eval_matrix.py`
  Evaluate every checkpoint on clean, random, object_edge, object_inside, and near_object scenes.

- Create: `scripts/round33_summarize.py`
  Aggregate metrics, behavior diagnostics, paired deltas, and allowed claims.

- Create: `runs/round33/manifest.json`
  Generated at runtime. Do not hardcode absolute paths except the Python executable.

---

## Task 1: Add Fixed-Recall Precision and Threshold Diagnostics

**Files:**
- Modify: `spectral_detection_posttrain/eval/detection_metrics.py`
- Create: `tests/test_round33_threshold_metrics.py`

- [ ] **Step 1: Write tests for multi-recall precision**

Create `tests/test_round33_threshold_metrics.py`:

```python
from spectral_detection_posttrain.eval.detection_metrics import (
    precision_at_recall,
    threshold_curve_summary,
)


def test_precision_at_recall_prefers_best_precision_after_recall_target():
    scored = [
        (0.99, True),
        (0.95, False),
        (0.90, True),
        (0.80, True),
        (0.70, False),
    ]
    assert precision_at_recall(scored, total_gt=3, target_recall=0.67) == 2 / 3


def test_threshold_curve_summary_reports_precision_at_two_recalls():
    scored = [
        (0.99, True),
        (0.96, False),
        (0.90, True),
        (0.85, True),
        (0.60, False),
    ]
    out = threshold_curve_summary(scored, total_gt=3, high_conf_threshold=0.9)
    assert out["precision_at_recall_0_85"] == 3 / 4
    assert out["precision_at_recall_0_90"] == 3 / 4
    assert out["high_conf_fp_at_recall_0_90"] == 1
    assert out["num_threshold_points"] == 5
```

- [ ] **Step 2: Run tests and verify failure**

```powershell
cd E:/CLIproject/RLimage
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round33_threshold_metrics.py -v
```

Expected: FAIL because `threshold_curve_summary` is not defined.

- [ ] **Step 3: Implement threshold summary**

Modify `spectral_detection_posttrain/eval/detection_metrics.py` by adding this function after `precision_at_recall`:

```python
def threshold_curve_summary(
    scored: list[tuple[float, bool]],
    total_gt: int,
    high_conf_threshold: float = 0.7,
    recall_targets: tuple[float, ...] = (0.85, 0.90),
) -> dict[str, float | int | None]:
    if total_gt <= 0 or not scored:
        result: dict[str, float | int | None] = {"num_threshold_points": 0}
        for target in recall_targets:
            key = f"{target:.2f}".replace(".", "_")
            result[f"precision_at_recall_{key}"] = None
            result[f"high_conf_fp_at_recall_{key}"] = None
        return result

    ordered = sorted(scored, key=lambda item: item[0], reverse=True)
    result = {"num_threshold_points": len(ordered)}
    for target in recall_targets:
        key = f"{target:.2f}".replace(".", "_")
        tp_cum = 0
        fp_cum = 0
        high_conf_fp = 0
        best_precision = None
        best_high_conf_fp = None
        for score, is_tp in ordered:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
                if score >= high_conf_threshold:
                    high_conf_fp += 1
            recall = tp_cum / max(1, total_gt)
            if recall >= target:
                precision = tp_cum / max(1, tp_cum + fp_cum)
                if best_precision is None or precision > best_precision:
                    best_precision = precision
                    best_high_conf_fp = high_conf_fp
        result[f"precision_at_recall_{key}"] = best_precision
        result[f"high_conf_fp_at_recall_{key}"] = best_high_conf_fp
    return result
```

Then update `evaluate_detection_predictions` to merge this summary:

```python
    threshold_summary = threshold_curve_summary(
        scored,
        total_gt,
        high_conf_threshold=high_conf_threshold,
        recall_targets=(0.85, 0.90),
    )

    return {
        "ap50": _compute_ap(recalls, precisions),
        "ap75": ap75,
        ...
        **threshold_summary,
    }
```

- [ ] **Step 4: Run tests and existing eval tests**

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round33_threshold_metrics.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add spectral_detection_posttrain/eval/detection_metrics.py tests/test_round33_threshold_metrics.py
git commit -m "feat: add fixed-recall threshold diagnostics"
```

---

## Task 2: Add RLVR Behavior Diagnostics

**Files:**
- Create: `spectral_detection_posttrain/analysis/rlvr_behavior_diagnostics.py`
- Create: `tests/test_round33_behavior_diagnostics.py`

- [ ] **Step 1: Write behavior diagnostic tests**

Create `tests/test_round33_behavior_diagnostics.py`:

```python
import torch

from spectral_detection_posttrain.analysis.rlvr_behavior_diagnostics import (
    summarize_logprob_shifts,
)


def test_summarize_logprob_shifts_separates_high_reward_tp_and_low_reward_fp():
    rewards = torch.tensor([1.0, 0.8, -0.5, -1.0])
    matched = torch.tensor([True, True, False, False])
    before = torch.tensor([0.2, 0.1, 0.3, 0.4])
    after = torch.tensor([0.25, 0.12, 0.1, 0.15])
    out = summarize_logprob_shifts(rewards, matched, before, after)
    assert out["high_reward_tp_count"] == 2
    assert out["low_reward_fp_count"] == 2
    assert out["high_reward_tp_delta_logprob_mean"] > 0
    assert out["low_reward_fp_delta_logprob_mean"] < 0
    assert out["reward_delta_logprob_corr"] > 0
```

- [ ] **Step 2: Run tests and verify failure**

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round33_behavior_diagnostics.py -v
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement behavior diagnostic helper**

Create `spectral_detection_posttrain/analysis/rlvr_behavior_diagnostics.py`:

```python
from __future__ import annotations

import torch


def _mean(values: torch.Tensor) -> float:
    return float(values.mean().item()) if values.numel() else 0.0


def _corr(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return 0.0
    if float(x.std(unbiased=False).item()) == 0.0:
        return 0.0
    if float(y.std(unbiased=False).item()) == 0.0:
        return 0.0
    return float(torch.corrcoef(torch.stack([x.float(), y.float()]))[0, 1].item())


def summarize_logprob_shifts(
    rewards: torch.Tensor,
    matched: torch.Tensor,
    baseline_action_logprob: torch.Tensor,
    current_action_logprob: torch.Tensor,
) -> dict[str, float | int]:
    rewards = rewards.detach().float().cpu()
    matched = matched.detach().bool().cpu()
    before = baseline_action_logprob.detach().float().cpu()
    after = current_action_logprob.detach().float().cpu()
    delta = after - before

    if rewards.numel() == 0:
        return {
            "action_count": 0,
            "high_reward_tp_count": 0,
            "low_reward_fp_count": 0,
            "high_reward_tp_delta_logprob_mean": 0.0,
            "low_reward_fp_delta_logprob_mean": 0.0,
            "reward_delta_logprob_corr": 0.0,
        }

    high_cut = torch.quantile(rewards, 0.75)
    low_cut = torch.quantile(rewards, 0.25)
    high_reward_tp = matched & (rewards >= high_cut)
    low_reward_fp = (~matched) & (rewards <= low_cut)

    return {
        "action_count": int(rewards.numel()),
        "high_reward_tp_count": int(high_reward_tp.sum().item()),
        "low_reward_fp_count": int(low_reward_fp.sum().item()),
        "delta_logprob_mean": _mean(delta),
        "high_reward_tp_delta_logprob_mean": _mean(delta[high_reward_tp]),
        "low_reward_fp_delta_logprob_mean": _mean(delta[low_reward_fp]),
        "reward_delta_logprob_corr": _corr(rewards, delta),
    }
```

- [ ] **Step 4: Run tests**

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round33_behavior_diagnostics.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add spectral_detection_posttrain/analysis/rlvr_behavior_diagnostics.py tests/test_round33_behavior_diagnostics.py
git commit -m "feat: add RLVR behavior diagnostics"
```

---

## Task 3: Improve RLVR Post-Training Logging

**Files:**
- Modify: `spectral_detection_posttrain/train/posttrain_rlvr.py`

- [ ] **Step 1: Add CLI args for explicit reward weights**

In `parse_args`, add:

```python
    parser.add_argument("--w-iou", type=float, default=1.0)
    parser.add_argument("--w-cls", type=float, default=0.2)
    parser.add_argument("--w-hconf-fp", type=float, default=None)
```

- [ ] **Step 2: Wire args into verifier config**

Replace the current `verifier_cfg = DetectionVerifierConfig(...)` block with:

```python
    verifier_cfg = DetectionVerifierConfig(
        signal=args.signal,
        temperature=args.temperature,
        w_iou=args.w_iou,
        w_cls=args.w_cls,
        w_amp=args.reward_lambda,
        w_struct=args.struct_weight,
        w_hconf_fp=float(args.alpha if args.w_hconf_fp is None else args.w_hconf_fp),
        high_conf_threshold=high_conf_threshold,
    )
```

- [ ] **Step 3: Save runtime metadata**

After `save_config(config, run_dir / "config.yaml")`, add:

```python
    import subprocess
    import sys

    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    save_json(
        {
            "git_hash": git_hash,
            "cli_args": " ".join(sys.argv),
            "seed": run_seed,
            "cudnn_benchmark": str(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": str(torch.backends.cudnn.deterministic),
        },
        run_dir / "runtime_meta.json",
    )
```

- [ ] **Step 4: Log reward component summary every epoch**

Add this import near the other imports:

```python
from spectral_detection_posttrain.analysis.rlvr_behavior_diagnostics import summarize_logprob_shifts
```

Inside the epoch loop, initialize before the batch loop:

```python
        epoch_reward_summaries = []
        epoch_behavior_summaries = []
```

After `actions = [...]`, add:

```python
            epoch_reward_summaries.append(build_reward_component_summary(actions))
```

After `loss_kl = baseline_kl_loss(class_logits, baseline_logits)`, add:

```python
            with torch.no_grad():
                action_labels_for_diag = policy_labels.to(class_logits.device)
                current_log_probs = torch.nn.functional.log_softmax(class_logits, dim=1)
                baseline_log_probs = torch.nn.functional.log_softmax(baseline_logits.to(class_logits.device), dim=1)
                arange = torch.arange(action_labels_for_diag.numel(), device=class_logits.device)
                current_selected_logprob = current_log_probs[arange, action_labels_for_diag].detach().cpu()
                baseline_selected_logprob = baseline_log_probs[arange, action_labels_for_diag].detach().cpu()
                rewards_for_diag = torch.cat([a["rewards"] for a in actions], dim=0)
                matched_for_diag = torch.cat([a["matched"] for a in actions], dim=0)
                epoch_behavior_summaries.append(
                    summarize_logprob_shifts(
                        rewards_for_diag,
                        matched_for_diag,
                        baseline_selected_logprob,
                        current_selected_logprob,
                    )
                )
```

Before building `row`, add:

```python
        def _avg_summary(key: str) -> float:
            values = [float(s.get(key, 0.0)) for s in epoch_reward_summaries]
            return sum(values) / max(1, len(values))

        def _avg_behavior(key: str) -> float:
            values = [float(s.get(key, 0.0)) for s in epoch_behavior_summaries]
            return sum(values) / max(1, len(values))
```

Extend `row`:

```python
            "reward_mean": _avg_summary("reward_mean"),
            "reward_std": _avg_summary("reward_std"),
            "amp_mean": _avg_summary("amp_mean"),
            "amp_std": _avg_summary("amp_std"),
            "structure_mean": _avg_summary("structure_mean"),
            "structure_std": _avg_summary("structure_std"),
            "matched_count_summary": _avg_summary("matched_count"),
            "candidate_count_summary": _avg_summary("candidate_count"),
            "behavior_delta_logprob_mean": _avg_behavior("delta_logprob_mean"),
            "behavior_high_reward_tp_delta_logprob_mean": _avg_behavior("high_reward_tp_delta_logprob_mean"),
            "behavior_low_reward_fp_delta_logprob_mean": _avg_behavior("low_reward_fp_delta_logprob_mean"),
            "behavior_reward_delta_logprob_corr": _avg_behavior("reward_delta_logprob_corr"),
            "behavior_high_reward_tp_count": _avg_behavior("high_reward_tp_count"),
            "behavior_low_reward_fp_count": _avg_behavior("low_reward_fp_count"),
```

- [ ] **Step 5: Add result metadata**

Extend final `result`:

```python
              "w_iou": args.w_iou,
              "w_cls": args.w_cls,
              "w_hconf_fp": float(args.alpha if args.w_hconf_fp is None else args.w_hconf_fp),
              "seed": run_seed,
```

- [ ] **Step 6: Smoke run no-update path**

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.train.posttrain_rlvr `
  --config spectral_detection_posttrain/configs/posttrain.yaml `
  --baseline runs/round216_baseline_s42/checkpoint_last.pth `
  --run-name round33_smoke_no_update `
  --signal none --unfreeze cls --optimizer adamw `
  --reward-lambda 0 --alpha 0 --beta 0 `
  --policy-loss-weight 0 --baseline-kl-weight 0 --det-loss-weight 0 `
  --epochs 1 --seed 42
```

Expected:

```text
runs/round33_smoke_no_update/runtime_meta.json exists
runs/round33_smoke_no_update/rlvr_result.json has no_update=true
```

- [ ] **Step 7: Commit**

```powershell
git add spectral_detection_posttrain/train/posttrain_rlvr.py
git commit -m "feat: log RLVR reward scale and runtime metadata"
```

---

## Task 4: Prepare Seed-Specific Manifest and Configs

**Files:**
- Create: `scripts/round33_prepare_manifest.py`
- Create: `tests/test_round33_manifest.py`

- [ ] **Step 1: Write manifest test**

Create `tests/test_round33_manifest.py`:

```python
from scripts.round33_prepare_manifest import build_config_for_init


def test_mid06_config_enables_matching_afm_architecture():
    cfg = build_config_for_init(seed=42, init_name="mid06_3ep")
    assert cfg["model"]["afm_channels"] == 256
    assert cfg["model"]["afm_type"] == "mplseg_mid"
    assert cfg["model"]["pretrained"] is False


def test_baseline_config_disables_afm():
    cfg = build_config_for_init(seed=42, init_name="baseline_1ep")
    assert cfg["model"].get("afm_channels", 0) == 0
    assert cfg["model"]["pretrained"] is False
```

- [ ] **Step 2: Implement manifest generator**

Create `scripts/round33_prepare_manifest.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from spectral_detection_posttrain.utils.config import load_config, save_config


SEEDS = [42, 123, 456]
BASE_CONFIG = Path("spectral_detection_posttrain/configs/posttrain.yaml")


def build_config_for_init(seed: int, init_name: str) -> dict:
    cfg = load_config(BASE_CONFIG)
    cfg["seed"] = int(seed)
    cfg["model"] = dict(cfg["model"])
    cfg["model"]["pretrained"] = False
    cfg["data"]["num_workers"] = 0
    cfg["eval"]["batch_size"] = 2
    cfg.setdefault("rlvr", {})
    cfg["rlvr"]["batch_size"] = 1

    if init_name == "baseline_1ep":
        cfg["model"]["afm_channels"] = 0
        cfg["model"].pop("afm_type", None)
    elif init_name == "mid06_3ep":
        cfg["model"]["afm_channels"] = 256
        cfg["model"]["afm_type"] = "mplseg_mid"
        cfg["model"]["afm_residual_mode"] = "current"
    else:
        raise ValueError(f"Unknown init_name: {init_name}")
    return cfg


def checkpoint_for_init(seed: int, init_name: str) -> str:
    if init_name == "baseline_1ep":
        return f"runs/round216_baseline_s{seed}/checkpoint_last.pth"
    if init_name == "mid06_3ep":
        return f"runs/round216p_mid06_s{seed}/checkpoint_last.pth"
    raise ValueError(f"Unknown init_name: {init_name}")


def main() -> None:
    out_dir = Path("runs/round33/configs")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for init_name in ["baseline_1ep", "mid06_3ep"]:
        for seed in SEEDS:
            cfg = build_config_for_init(seed=seed, init_name=init_name)
            cfg_path = out_dir / f"{init_name}_s{seed}.yaml"
            save_config(cfg, cfg_path)
            ckpt = checkpoint_for_init(seed=seed, init_name=init_name)
            if not Path(ckpt).exists():
                raise FileNotFoundError(ckpt)
            manifest.append({
                "init_name": init_name,
                "seed": seed,
                "config": str(cfg_path),
                "checkpoint": ckpt,
            })
    manifest_path = Path("runs/round33/manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests**

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round33_manifest.py -v
```

Expected: PASS.

- [ ] **Step 4: Generate manifest**

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/round33_prepare_manifest.py
```

Expected:

```text
runs/round33/manifest.json
```

- [ ] **Step 5: Commit**

```powershell
git add scripts/round33_prepare_manifest.py tests/test_round33_manifest.py runs/round33/manifest.json runs/round33/configs
git commit -m "feat: prepare Round 3.3 checkpoint manifest"
```

---

## Task 5: Implement Round 3.3 Matrix Runner

**Files:**
- Create: `scripts/round33_run_matrix.py`

- [ ] **Step 1: Implement staged matrix runner**

Create `scripts/round33_run_matrix.py`:

```python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PYTHON = sys.executable
REWARD_MODES = {
    "iou_only": {"signal": "none", "reward_lambda": 0.0, "w_hconf_fp": 0.0},
    "iou_hconf": {"signal": "none", "reward_lambda": 0.0, "w_hconf_fp": 0.5},
    "iou_hconf_amp_aux": {"signal": "ramp", "reward_lambda": 0.05, "w_hconf_fp": 0.5},
    "iou_hconf_shuffled_amp_aux": {"signal": "shuffled_amp", "reward_lambda": 0.05, "w_hconf_fp": 0.5},
}


def load_manifest() -> list[dict]:
    return json.loads(Path("runs/round33/manifest.json").read_text(encoding="utf-8"))


def phase_space(phase: str) -> tuple[list[float], list[float], list[str]]:
    if phase == "A":
        return [0.0003], [10.0], ["cls"]
    if phase == "B":
        return [0.0003, 0.001, 0.003], [3.0, 10.0], ["cls"]
    if phase == "C":
        return [0.0003, 0.001, 0.003], [3.0, 10.0], ["cls", "roi"]
    raise ValueError(f"Unknown phase: {phase}")


def run_trial(item: dict, reward_name: str, reward_cfg: dict, policy_weight: float, kl_weight: float, unfreeze: str, epochs: int) -> dict:
    run_name = (
        f"round33/{item['init_name']}_s{item['seed']}/"
        f"{reward_name}_pw{policy_weight:g}_kl{kl_weight:g}_{unfreeze}"
    ).replace(".", "p")
    result_path = Path("runs") / run_name / "rlvr_result.json"
    if result_path.exists():
        return {"run_name": run_name, "status": "SKIP"}
    cmd = [
        PYTHON, "-m", "spectral_detection_posttrain.train.posttrain_rlvr",
        "--config", item["config"],
        "--baseline", item["checkpoint"],
        "--run-name", run_name,
        "--signal", reward_cfg["signal"],
        "--unfreeze", unfreeze,
        "--optimizer", "adamw",
        "--reward-lambda", str(reward_cfg["reward_lambda"]),
        "--w-hconf-fp", str(reward_cfg["w_hconf_fp"]),
        "--alpha", str(reward_cfg["w_hconf_fp"]),
        "--beta", "0.0",
        "--policy-loss-weight", str(policy_weight),
        "--baseline-kl-weight", str(kl_weight),
        "--det-loss-weight", "0.0",
        "--box-loss-weight", "0.0",
        "--temperature", "1.0",
        "--policy-objective", "signed",
        "--rollout-source", "baseline",
        "--max-candidates", "40",
        "--reward-score-threshold", "0.2",
        "--epochs", str(epochs),
        "--early-stopping-patience", "3",
        "--seed", str(item["seed"]),
    ]
    r = subprocess.run(cmd, text=True, capture_output=True)
    if r.returncode != 0:
        err_path = Path("runs") / run_name / "stderr.txt"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(r.stderr[-4000:], encoding="utf-8")
        return {"run_name": run_name, "status": "CRASH", "stderr": str(err_path)}
    return {"run_name": run_name, "status": "OK"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["A", "B", "C"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    manifest = load_manifest()
    policy_weights, kl_weights, unfreeze_modes = phase_space(args.phase)
    rows = []
    for item in manifest:
        for reward_name, reward_cfg in REWARD_MODES.items():
            for policy_weight in policy_weights:
                for kl_weight in kl_weights:
                    for unfreeze in unfreeze_modes:
                        row = run_trial(item, reward_name, reward_cfg, policy_weight, kl_weight, unfreeze, args.epochs)
                        rows.append(row)
                        print(row, flush=True)
    out = Path(f"runs/round33/phase_{args.phase}_run_rows.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run by reading phase size**

```powershell
@'
from scripts.round33_run_matrix import load_manifest, phase_space, REWARD_MODES
for phase in ["A", "B", "C"]:
    pw, kl, unfreeze = phase_space(phase)
    print(phase, len(load_manifest()) * len(REWARD_MODES) * len(pw) * len(kl) * len(unfreeze))
'@ | E:\anaconda\01\envs\RLimage\python.exe -
```

Expected:

```text
A 24
B 144
C 288
```

Phase C includes the Phase B grid plus `roi`; do not run Phase C until Phase B summary passes.

- [ ] **Step 3: Commit**

```powershell
git add scripts/round33_run_matrix.py
git commit -m "feat: add Round 3.3 RLVR matrix runner"
```

---

## Task 6: Implement Stress Evaluation

**Files:**
- Create: `scripts/round33_eval_matrix.py`

- [ ] **Step 1: Implement eval runner**

Create `scripts/round33_eval_matrix.py`:

```python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PYTHON = sys.executable
SCENES = [
    ("clean", "none", "random"),
    ("random", "random", "random"),
    ("object_edge", "object_edge", "checkerboard"),
    ("object_inside", "object_inside", "checkerboard"),
    ("near_object", "near_object", "checkerboard"),
]


def iter_result_dirs(phase: str):
    rows_path = Path(f"runs/round33/phase_{phase}_run_rows.jsonl")
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("status") not in {"OK", "SKIP"}:
            continue
        run_dir = Path("runs") / row["run_name"]
        ckpt = run_dir / "checkpoint_best.pth"
        if not ckpt.exists():
            ckpt = run_dir / "checkpoint_last.pth"
        cfg = run_dir / "config.yaml"
        if ckpt.exists() and cfg.exists():
            yield row["run_name"], cfg, ckpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["A", "B", "C"], required=True)
    args = parser.parse_args()

    rows = []
    for run_name, cfg, ckpt in iter_result_dirs(args.phase):
        for scene, patch_mode, patch_type in SCENES:
            eval_run = f"{run_name}_eval_{scene}"
            metrics_path = Path("runs") / eval_run / "eval_metrics.json"
            if not metrics_path.exists():
                cmd = [
                    PYTHON, "-m", "spectral_detection_posttrain.eval.eval_detector",
                    "--config", str(cfg),
                    "--checkpoint", str(ckpt),
                    "--run-name", eval_run,
                    "--patch-mode", patch_mode,
                    "--patch-type", patch_type,
                ]
                subprocess.run(cmd, check=True)
            rows.append({"run_name": run_name, "scene": scene, "metrics_path": str(metrics_path)})
            print(rows[-1], flush=True)

    out = Path(f"runs/round33/phase_{args.phase}_eval_rows.jsonl")
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```powershell
git add scripts/round33_eval_matrix.py
git commit -m "feat: add Round 3.3 stress eval runner"
```

---

## Task 7: Implement Summary and Claim Gates

**Files:**
- Create: `scripts/round33_summarize.py`

- [ ] **Step 1: Implement summarizer**

Create `scripts/round33_summarize.py`:

```python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


KEYS = [
    "ap50", "ap75", "precision", "recall", "ece",
    "high_conf_fp_count", "num_predictions",
    "precision_at_recall_0_90", "high_conf_fp_at_recall_0_90",
]

BEHAVIOR_KEYS = [
    "reward_std",
    "behavior_high_reward_tp_delta_logprob_mean",
    "behavior_low_reward_fp_delta_logprob_mean",
    "behavior_reward_delta_logprob_corr",
]


def parse_run_name(run_name: str) -> dict[str, str]:
    parts = run_name.split("/")
    init_seed = parts[1]
    reward_tag = parts[2]
    init_name, seed = init_seed.rsplit("_s", 1)
    return {
        "init_name": init_name,
        "seed": seed,
        "tag": reward_tag,
    }


def load_eval_rows(phase: str):
    path = Path(f"runs/round33/phase_{phase}_eval_rows.jsonl")
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        metrics = json.loads(Path(row["metrics_path"]).read_text(encoding="utf-8"))
        meta = parse_run_name(row["run_name"])
        yield {**meta, "run_name": row["run_name"], "scene": row["scene"], "metrics": metrics}


def load_last_train_row(run_name: str) -> dict:
    path = Path("runs") / run_name / "metrics_train.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-1] if rows else {}


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["A", "B", "C"], required=True)
    args = parser.parse_args()

    grouped = defaultdict(list)
    for row in load_eval_rows(args.phase):
        key = (row["init_name"], row["tag"], row["scene"])
        grouped[key].append(row["metrics"])

    lines = [f"# Round 3.3 Phase {args.phase} Summary", ""]
    lines.append("| Init | Tag | Scene | AP50 | AP75 | Precision | Recall | ECE | P@R0.90 | hiFP@R0.90 | Pred |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    summary_rows = []
    for (init_name, tag, scene), metrics_list in sorted(grouped.items()):
        row = {
            "init_name": init_name,
            "tag": tag,
            "scene": scene,
        }
        for k in KEYS:
            row[k] = mean([m.get(k) for m in metrics_list])
        summary_rows.append(row)
        lines.append(
            f"| {init_name} | {tag} | {scene} | "
            f"{row['ap50']:.4f} | {row['ap75']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['ece']:.4f} | "
            f"{(row['precision_at_recall_0_90'] if row['precision_at_recall_0_90'] is not None else 0):.4f} | "
            f"{(row['high_conf_fp_at_recall_0_90'] if row['high_conf_fp_at_recall_0_90'] is not None else 0):.2f} | "
            f"{row['num_predictions']:.1f} |"
        )

    lines.append("")
    lines.append("## Behavior Diagnostics")
    lines.append("")
    lines.append("| Init | Tag | reward_std | highTP dlogp | lowFP dlogp | reward/dlogp corr |")
    lines.append("|---|---|---:|---:|---:|---:|")
    behavior_rows = []
    # Build behavior rows from eval rows so run_name is available.
    by_run = {}
    for row in load_eval_rows(args.phase):
        if row["scene"] != "clean":
            continue
        meta_key = (row["init_name"], row["tag"])
        behavior = load_last_train_row(row["run_name"])
        if behavior:
            by_run.setdefault(meta_key, []).append(behavior)
    for (init_name, tag), rows_for_group in sorted(by_run.items()):
        brow = {"init_name": init_name, "tag": tag}
        for key in BEHAVIOR_KEYS:
            brow[key] = mean([r.get(key) for r in rows_for_group])
        behavior_rows.append(brow)
        lines.append(
            f"| {init_name} | {tag} | "
            f"{(brow['reward_std'] if brow['reward_std'] is not None else 0):.4f} | "
            f"{(brow['behavior_high_reward_tp_delta_logprob_mean'] if brow['behavior_high_reward_tp_delta_logprob_mean'] is not None else 0):.4f} | "
            f"{(brow['behavior_low_reward_fp_delta_logprob_mean'] if brow['behavior_low_reward_fp_delta_logprob_mean'] is not None else 0):.4f} | "
            f"{(brow['behavior_reward_delta_logprob_corr'] if brow['behavior_reward_delta_logprob_corr'] is not None else 0):.4f} |"
        )

    out_md = Path(f"runs/round33/phase_{args.phase}_summary.md")
    out_json = Path(f"runs/round33/phase_{args.phase}_summary.json")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_json.write_text(
        json.dumps({"metrics": summary_rows, "behavior": behavior_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(out_md)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```powershell
git add scripts/round33_summarize.py
git commit -m "feat: summarize Round 3.3 RLVR matrix"
```

---

## Task 8: Execute Phase A

**Files:**
- Runtime outputs under `runs/round33/`

- [ ] **Step 1: Verify prerequisites**

```powershell
Test-Path runs/round216_baseline_s42/checkpoint_last.pth
Test-Path runs/round216p_mid06_s42/checkpoint_last.pth
```

Expected:

```text
True
True
```

- [ ] **Step 2: Generate manifest**

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/round33_prepare_manifest.py
```

- [ ] **Step 3: Run Phase A**

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/round33_run_matrix.py --phase A --epochs 5
```

- [ ] **Step 4: Evaluate Phase A**

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/round33_eval_matrix.py --phase A
```

- [ ] **Step 5: Summarize Phase A**

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/round33_summarize.py --phase A
```

- [ ] **Step 6: Inspect gates**

Open:

```text
runs/round33/phase_A_summary.md
```

Continue to Phase B only if at least one `iou_hconf` or `iou_only` group satisfies:

```text
clean AP50 >= matched initialization AP50 - 0.01
clean precision_at_recall_0_90 improves by >= 0.02
clean high_conf_fp_at_recall_0_90 decreases
object_edge AP50 does not drop by more than 0.02
```

- [ ] **Step 7: Commit Phase A summary**

```powershell
git add runs/round33/phase_A_summary.md runs/round33/phase_A_summary.json runs/round33/phase_A_run_rows.jsonl runs/round33/phase_A_eval_rows.jsonl
git commit -m "exp: add Round 3.3 Phase A RLVR results"
```

---

## Task 9: Execute Phase B and Phase C

**Files:**
- Runtime outputs under `runs/round33/`

- [ ] **Step 1: Run Phase B**

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/round33_run_matrix.py --phase B --epochs 5
E:\anaconda\01\envs\RLimage\python.exe scripts/round33_eval_matrix.py --phase B
E:\anaconda\01\envs\RLimage\python.exe scripts/round33_summarize.py --phase B
```

- [ ] **Step 2: Stop or promote**

Promote to Phase C only if Phase B identifies at least two non-collapsed groups:

```text
AP50 >= matched initialization AP50 - 0.01
Recall >= matched initialization Recall - 0.02
Precision@Recall=0.90 improves by >= 0.03
High-conf FP at recall 0.90 decreases by >= 20%
```

- [ ] **Step 3: Run Phase C if promoted**

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/round33_run_matrix.py --phase C --epochs 5
E:\anaconda\01\envs\RLimage\python.exe scripts/round33_eval_matrix.py --phase C
E:\anaconda\01\envs\RLimage\python.exe scripts/round33_summarize.py --phase C
```

- [ ] **Step 4: Commit Phase B/C summaries**

```powershell
git add runs/round33/phase_B_summary.md runs/round33/phase_B_summary.json runs/round33/phase_B_run_rows.jsonl runs/round33/phase_B_eval_rows.jsonl
git commit -m "exp: add Round 3.3 Phase B RLVR results"
```

If Phase C runs:

```powershell
git add runs/round33/phase_C_summary.md runs/round33/phase_C_summary.json runs/round33/phase_C_run_rows.jsonl runs/round33/phase_C_eval_rows.jsonl
git commit -m "exp: add Round 3.3 Phase C RLVR results"
```

---

## Task 10: Final Interpretation Report

**Files:**
- Create: `docs/round33_rlvr_large_scale_results.md`

- [ ] **Step 1: Write report with allowed claims**

Create `docs/round33_rlvr_large_scale_results.md` with this structure:

```markdown
# Round 3.3 RLVR Large-Scale Validation Results

## Setup

- Initializations:
  - baseline_1ep
  - mid06_3ep
- Reward modes:
  - iou_only
  - iou_hconf
  - iou_hconf_amp_aux
  - iou_hconf_shuffled_amp_aux
- Primary metrics:
  - AP50
  - AP75
  - Precision@Recall=0.90
  - High-conf FP at Recall=0.90
  - ECE
  - num_predictions

## Phase A Summary

Paste the table from `runs/round33/phase_A_summary.md`.

## Phase B Summary

Paste the table from `runs/round33/phase_B_summary.md` if Phase B ran.

## Phase C Summary

Paste the table from `runs/round33/phase_C_summary.md` if Phase C ran.

## Claims

State only one of:

1. IoU-primary RLVR improved detector behavior under fixed-recall precision gates.
2. RLVR shell remained stable but did not improve behavior.
3. Spectral auxiliary beat shuffled control.
4. Spectral auxiliary did not beat shuffled control.

## Next Step

If RLVR succeeds:
  move to VOC/COCO small subset with the winning verifier.

If RLVR fails:
  stop detector-side box-level RLVR and move to semantic segmentation verifier.
```

- [ ] **Step 2: Commit report**

```powershell
git add docs/round33_rlvr_large_scale_results.md
git commit -m "docs: add Round 3.3 RLVR large-scale report"
```

---

## Final Success Criteria

Plan 3.3 is complete only when:

```text
1. Phase A summary exists and includes all 24 expected trials.
2. Every result row includes git hash, CLI, seed, reward mode, init, KL, policy weight, and unfreeze mode.
3. Clean + four stress scenes are evaluated.
4. Summary reports AP50, AP75, Precision, Recall, ECE, High-conf FP, num_predictions, Precision@Recall=0.90.
5. Real spectral auxiliary is compared against shuffled spectral auxiliary.
6. The final report states allowed claims only.
```

Do not claim RLVR worked from AP50 alone. The core evidence must be fixed-recall precision, high-conf FP suppression, and policy behavior diagnostics.
