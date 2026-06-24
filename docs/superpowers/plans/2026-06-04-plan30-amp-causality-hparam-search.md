# Plan 3.0 Amp Causality And Hyperparameter Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Round 2.5 finding into a larger, controlled RLVR study: prove whether amplitude reward has causal value over shuffled amplitude, then search stable hyperparameters for AP75, patch robustness, calibration, and high-confidence error control.

**Architecture:** Keep the stable Round 2.3/2.5 training shell: baseline rollouts, signed ROI policy objective, KL to baseline, frozen detector state, and `det_loss_weight=0`. Add missing causal controls, stronger result aggregation, reward-component diagnostics, and a staged NNI search with Phase A amplitude causality, Phase B amplitude hyperparameter tuning, and Phase C structure recheck. The final report must separate verifier causality from ordinary RLVR training noise.

**Tech Stack:** Python, PyTorch, TorchVision Faster R-CNN, Penn-Fudan, existing `spectral_detection_posttrain` package, pytest, NNI GridSearch, PowerShell/Windows batch, conda env `E:\anaconda\01\envs\RLimage`.

---

## Starting Point

Round 2.5 produced the first clean positive signal:

```text
Amp-only AP75 gains vs baseline:
clean  +0.0204
edge   +0.0222
inside +0.0212
near   +0.0123
```

But it has one decisive missing control:

```text
Amp-only vs shuffled_amp_only
```

Without that, the improvement can come from either:

```text
1. Real amplitude verifier signal
2. Generic RLVR perturbation / seed / candidate-selection effect
```

Plan 3.0 exists to distinguish those two explanations.

---

## Scientific Claims Allowed After Plan 3.0

Only make these claims if the stated gates pass:

```text
Amp causal claim:
  Real amp beats shuffled amp by >= 0.005 mean AP75 across the four scenes,
  and the sign is positive in at least 3 of 4 scenes,
  and clean AP50 drop vs IoU-only is no worse than -0.005.

Amp useful but not causal:
  Real amp improves over baseline but does not beat shuffled amp.

Structure useful claim:
  Real structure beats shuffled structure by >= 0.005 AP75 on edge or inside,
  and does not lose more than 0.005 AP75 on clean and near.

Amp+Struct useful claim:
  Real amp+struct beats real amp and shuffled amp+struct under the same stability constraints.
```

If these gates fail, the correct conclusion is:

```text
The stable RLVR shell is useful, but the hand-built verifier signal is not yet proven causal.
```

---

## Search Budget

Plan 3.0 is intentionally larger than Round 2.5.

```text
Phase A: Amp causality grid
  68 trials

Phase B: Amp hyperparameter tuning
  54 trials, generated after Phase A picks the best amp strength

Phase C: Structure recheck
  24 trials

Total expected trials:
  146 trials
```

Estimated time:

```text
GPU, 3 epochs per trial: 20-40 hours
CPU-only: 3-7 days
Disk: 15-40 GB depending checkpoint retention
```

This is a long experiment. Run Phase A first and inspect the report before launching Phase B/C.

---

## File Map

- Modify: `spectral_detection_posttrain/rlvr/detection_verifier.py`
  Make `shuffled_amp` a readable alias for `shuffled_ramp`; add reward component summary helpers.

- Modify: `spectral_detection_posttrain/train/posttrain_rlvr.py`
  Add seed override, reward component diagnostics, and per-epoch amp/structure/reward statistics.

- Modify: `spectral_detection_posttrain/nni_rlvr_trial.py`
  Make all Round 3.0 trials evaluate clean, edge, inside, and near scenes; add `compute_round30_objective`; keep complete result rows.

- Create: `spectral_detection_posttrain/analysis/round30_results.py`
  Rebuilds complete result rows from run directories and eval outputs.

- Create: `spectral_detection_posttrain/analysis/build_round30_search_space.py`
  Generates Phase A/B/C JSON search spaces.

- Create: `spectral_detection_posttrain/analysis/summarize_round30_results.py`
  Aggregates all phases, computes causal pair deltas, and writes the final report.

- Create: `nni_configs/rlvr_round30_phaseA_config.yml`
  NNI config for amplitude causality grid.

- Create: `nni_configs/rlvr_round30_phaseB_config.yml`
  NNI config for amplitude hyperparameter tuning.

- Create: `nni_configs/rlvr_round30_phaseC_config.yml`
  NNI config for structure recheck.

- Create: `run_nni_rlvr_round30_phaseA.bat`
  Generates Phase A search space and launches NNI.

- Create: `run_nni_rlvr_round30_phaseB.bat`
  Generates Phase B search space from Phase A results and launches NNI.

- Create: `run_nni_rlvr_round30_phaseC.bat`
  Generates Phase C search space and launches NNI.

- Create: `tests/test_round30_results.py`
  Tests complete four-scene result rebuilding and causal delta calculations.

- Create: `tests/test_round30_search_space.py`
  Tests exact trial counts and required controls.

- Create: `tests/test_round30_training_diagnostics.py`
  Tests parse args and reward diagnostic summaries.

- Create: `docs/rlvr_plan30_amp_causality_report.md`
  Final report generated after all phases.

---

## Task 1: Add Readable Amp Signal Alias And Reward Diagnostics

**Files:**
- Modify: `spectral_detection_posttrain/rlvr/detection_verifier.py`
- Create: `tests/test_round30_training_diagnostics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_round30_training_diagnostics.py`:

```python
import pytest
import torch

from spectral_detection_posttrain.rlvr.detection_verifier import (
    DetectionVerifierConfig,
    build_reward_component_summary,
    compute_box_rewards,
    signal_uses_amp,
    signal_uses_structure,
)


def test_shuffled_amp_alias_uses_amp_not_structure():
    assert signal_uses_amp("shuffled_amp")
    assert not signal_uses_structure("shuffled_amp")


def test_amp_only_reward_ignores_structure_value():
    cfg = DetectionVerifierConfig(signal="ramp", w_iou=1.0, w_cls=0.2, w_amp=0.1, w_struct=0.9)
    ious = torch.tensor([0.7])
    class_correct = torch.tensor([1.0])
    scores = torch.tensor([0.8])
    matched = torch.tensor([True])
    s_amp = torch.tensor([0.5])
    s_struct = torch.tensor([1.0])

    reward = compute_box_rewards(cfg, ious, class_correct, scores, matched, s_amp=s_amp, s_struct=s_struct)

    assert reward.item() == pytest.approx(0.7 + 0.2 + 0.05, abs=1e-6)


def test_reward_component_summary_reports_means_and_counts():
    actions = [
        {
            "amp_values": torch.tensor([0.2, 0.8]),
            "structure_values": torch.tensor([0.1, 0.3]),
            "rewards": torch.tensor([0.5, 1.0]),
            "matched": torch.tensor([True, False]),
        },
        {
            "amp_values": torch.tensor([0.4]),
            "structure_values": torch.tensor([0.7]),
            "rewards": torch.tensor([0.2]),
            "matched": torch.tensor([True]),
        },
    ]

    summary = build_reward_component_summary(actions)

    assert summary["amp_mean"] == pytest.approx((0.2 + 0.8 + 0.4) / 3.0)
    assert summary["structure_mean"] == pytest.approx((0.1 + 0.3 + 0.7) / 3.0)
    assert summary["reward_mean"] == pytest.approx((0.5 + 1.0 + 0.2) / 3.0)
    assert summary["candidate_count"] == 3
    assert summary["matched_count"] == 2
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round30_training_diagnostics.py -v
```

Expected: fails because `shuffled_amp` is not an alias and `build_reward_component_summary` is not defined.

- [ ] **Step 3: Add signal alias**

In `spectral_detection_posttrain/rlvr/detection_verifier.py`, replace:

```python
AMP_SIGNALS = {"ramp", "shuffled_ramp", "amp_structure", "shuffled_amp_structure"}
```

with:

```python
AMP_SIGNALS = {"ramp", "shuffled_ramp", "shuffled_amp", "amp_structure", "shuffled_amp_structure"}
```

In `build_rewarded_roi_actions`, replace:

```python
    if cfg.signal == "shuffled_ramp":
        amp = shuffle_tp_values(amp, matched)
```

with:

```python
    if cfg.signal in {"shuffled_ramp", "shuffled_amp"}:
        amp = shuffle_tp_values(amp, matched)
```

- [ ] **Step 4: Add reward component summary helper**

Add to `spectral_detection_posttrain/rlvr/detection_verifier.py`:

```python
def _cat_action_tensor(actions: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    tensors = [action[key].detach().float().cpu() for action in actions if key in action and action[key].numel() > 0]
    if not tensors:
        return torch.empty((0,), dtype=torch.float32)
    return torch.cat(tensors, dim=0)


def build_reward_component_summary(actions: list[dict[str, torch.Tensor]]) -> dict[str, float | int]:
    amp = _cat_action_tensor(actions, "amp_values")
    struct = _cat_action_tensor(actions, "structure_values")
    rewards = _cat_action_tensor(actions, "rewards")
    matched = _cat_action_tensor(actions, "matched").bool()

    def _mean(values: torch.Tensor) -> float:
        return float(values.mean().item()) if values.numel() else 0.0

    def _std(values: torch.Tensor) -> float:
        return float(values.std(unbiased=False).item()) if values.numel() else 0.0

    return {
        "candidate_count": int(rewards.numel()),
        "matched_count": int(matched.sum().item()) if matched.numel() else 0,
        "amp_mean": _mean(amp),
        "amp_std": _std(amp),
        "structure_mean": _mean(struct),
        "structure_std": _std(struct),
        "reward_mean": _mean(rewards),
        "reward_std": _std(rewards),
    }
```

- [ ] **Step 5: Run tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round30_training_diagnostics.py tests/test_rlvr_verifier.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add spectral_detection_posttrain/rlvr/detection_verifier.py tests/test_round30_training_diagnostics.py
git commit -m "feat: add round30 reward diagnostics"
```

---

## Task 2: Add Seed Override And Per-Epoch Diagnostic Logging

**Files:**
- Modify: `spectral_detection_posttrain/train/posttrain_rlvr.py`
- Modify: `tests/test_round30_training_diagnostics.py`

- [ ] **Step 1: Add parse test**

Append to `tests/test_round30_training_diagnostics.py`:

```python
def test_round30_posttrain_args_accept_seed_and_shuffled_amp():
    from spectral_detection_posttrain.train.posttrain_rlvr import parse_args

    args = parse_args([
        "--config", "spectral_detection_posttrain/configs/mvp.yaml",
        "--checkpoint", "runs/baseline/checkpoint_last.pth",
        "--run-name", "round30_smoke",
        "--signal", "shuffled_amp",
        "--unfreeze", "cls",
        "--optimizer", "adamw",
        "--reward-lambda", "0.1",
        "--struct-weight", "0.0",
        "--policy-loss-weight", "0.0003",
        "--baseline-kl-weight", "10.0",
        "--det-loss-weight", "0.0",
        "--seed", "43",
        "--epochs", "1",
    ])

    assert args.signal == "shuffled_amp"
    assert args.seed == 43
```

- [ ] **Step 2: Run failing test**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round30_training_diagnostics.py::test_round30_posttrain_args_accept_seed_and_shuffled_amp -v
```

Expected: fails because `--seed` and `shuffled_amp` are not accepted by `posttrain_rlvr.py`.

- [ ] **Step 3: Update imports**

In `spectral_detection_posttrain/train/posttrain_rlvr.py`, add `build_reward_component_summary`:

```python
from spectral_detection_posttrain.rlvr.detection_verifier import (
    DetectionVerifierConfig,
    build_reward_component_summary,
    build_rewarded_roi_actions,
    signal_uses_amp,
    signal_uses_structure,
)
```

- [ ] **Step 4: Update CLI choices and seed arg**

In `parse_args`, include `shuffled_amp` in signal choices:

```python
            "shuffled_amp",
```

Add:

```python
    parser.add_argument("--seed", type=int, default=None)
```

- [ ] **Step 5: Apply seed override**

Replace:

```python
    set_seed(int(config.get("seed", 42)))
```

with:

```python
    run_seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    config["seed"] = run_seed
    set_seed(run_seed)
```

- [ ] **Step 6: Log reward component diagnostics**

After `actions = [...]`, add:

```python
            reward_summary = build_reward_component_summary(actions)
```

Replace the existing diagnostic variables:

```python
            candidate_count = sum(len(a["boxes"]) for a in actions) / max(1, len(actions))
            matched_count = sum((a["matched"]).sum().item() for a in actions) / max(1, len(actions))
            fp_count = candidate_count - matched_count
```

with:

```python
            candidate_count = reward_summary["candidate_count"] / max(1, len(actions))
            matched_count = reward_summary["matched_count"] / max(1, len(actions))
            fp_count = candidate_count - matched_count
```

Extend `progress.set_postfix(...)`:

```python
                amp=reward_summary["amp_mean"],
                struct=reward_summary["structure_mean"],
                reward=reward_summary["reward_mean"],
```

Add to the epoch `row`:

```python
            "amp_mean": reward_summary["amp_mean"],
            "amp_std": reward_summary["amp_std"],
            "structure_mean": reward_summary["structure_mean"],
            "structure_std": reward_summary["structure_std"],
            "reward_mean": reward_summary["reward_mean"],
            "reward_std": reward_summary["reward_std"],
```

Add to final `result`:

```python
              "seed": run_seed,
```

- [ ] **Step 7: Run tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round30_training_diagnostics.py tests/test_rlvr_verifier.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```powershell
git add spectral_detection_posttrain/train/posttrain_rlvr.py tests/test_round30_training_diagnostics.py
git commit -m "feat: log round30 reward components"
```

---

## Task 3: Build Complete Round 3.0 Result Rows

**Files:**
- Create: `spectral_detection_posttrain/analysis/round30_results.py`
- Create: `tests/test_round30_results.py`

- [ ] **Step 1: Write result row tests**

Create `tests/test_round30_results.py`:

```python
import pytest

from spectral_detection_posttrain.analysis.round30_results import (
    FOUR_SCENES,
    build_round30_result_row,
    compute_pair_delta,
    scene_metric_key,
)


def _metrics(ap75: float) -> dict:
    return {
        "ap50": 0.88,
        "ap75": ap75,
        "precision": 0.63,
        "recall": 0.90,
        "ece": 0.04,
        "high_conf_fp_count": 2,
        "high_conf_fp_rate": 0.03,
        "num_predictions": 130,
    }


def test_scene_metric_key_uses_short_prefixes():
    assert scene_metric_key("object_edge_checkerboard", "ap75") == "edge_ap75"
    assert scene_metric_key("object_inside_checkerboard", "ap75") == "inside_ap75"
    assert scene_metric_key("near_object_checkerboard", "ap75") == "near_ap75"


def test_build_round30_result_row_contains_all_four_scenes():
    metrics = {scene: _metrics(0.6 + idx * 0.01) for idx, scene in enumerate(FOUR_SCENES)}
    params = {
        "name": "signed_amp_l0p1_pl0p0003_kl10_seed42",
        "signal": "ramp",
        "reward_lambda": 0.1,
        "struct_weight": 0.0,
        "policy_loss_weight": 0.0003,
        "baseline_kl_weight": 10.0,
        "seed": 42,
    }

    row = build_round30_result_row(
        params=params,
        metrics=metrics,
        objective={"default": 3.0, "constraint_failed": ""},
        run_name="rlvr_x",
        checkpoint="runs/x/checkpoint_best.pth",
        eval_status="ok",
    )

    assert row["clean_ap75"] == pytest.approx(0.60)
    assert row["edge_ap75"] == pytest.approx(0.61)
    assert row["inside_ap75"] == pytest.approx(0.62)
    assert row["near_ap75"] == pytest.approx(0.63)
    assert row["seed"] == 42


def test_compute_pair_delta_averages_scene_deltas():
    real = {
        "clean_ap75": 0.66,
        "edge_ap75": 0.58,
        "inside_ap75": 0.67,
        "near_ap75": 0.61,
    }
    shuffled = {
        "clean_ap75": 0.64,
        "edge_ap75": 0.56,
        "inside_ap75": 0.65,
        "near_ap75": 0.60,
    }

    delta = compute_pair_delta(real, shuffled, metric="ap75")

    assert delta["mean_delta"] == pytest.approx((0.02 + 0.02 + 0.02 + 0.01) / 4.0)
    assert delta["positive_scene_count"] == 4
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round30_results.py -v
```

Expected: fails because `round30_results.py` does not exist.

- [ ] **Step 3: Implement result helpers**

Create `spectral_detection_posttrain/analysis/round30_results.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FOUR_SCENES = [
    "clean",
    "object_edge_checkerboard",
    "object_inside_checkerboard",
    "near_object_checkerboard",
]

SCENE_PREFIX = {
    "clean": "clean",
    "object_edge_checkerboard": "edge",
    "object_inside_checkerboard": "inside",
    "near_object_checkerboard": "near",
}

METRICS = [
    "ap50",
    "ap75",
    "precision",
    "recall",
    "ece",
    "high_conf_fp_count",
    "high_conf_fp_rate",
    "num_predictions",
]


def scene_metric_key(scene: str, metric: str) -> str:
    return f"{SCENE_PREFIX[scene]}_{metric}"


def build_round30_result_row(
    params: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    objective: dict[str, Any],
    run_name: str,
    checkpoint: str,
    eval_status: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": params.get("name", ""),
        "default": float(objective.get("default", -1.0)),
        "constraint_failed": objective.get("constraint_failed", ""),
        "run_name": run_name,
        "checkpoint": checkpoint,
        "eval_status": eval_status,
        "signal": params.get("signal", ""),
        "reward_lambda": float(params.get("reward_lambda", 0.0)),
        "struct_weight": float(params.get("struct_weight", 0.0)),
        "policy_loss_weight": float(params.get("policy_loss_weight", 0.0)),
        "baseline_kl_weight": float(params.get("baseline_kl_weight", 0.0)),
        "temperature": float(params.get("temperature", 1.0)),
        "max_candidates": int(params.get("max_candidates", 0)),
        "reward_score_threshold": float(params.get("reward_score_threshold", 0.0)),
        "seed": int(params.get("seed", 42)),
        "unfreeze": params.get("unfreeze", "cls"),
        "optimizer": params.get("optimizer", "adamw"),
        "rollout_source": params.get("rollout_source", "baseline"),
        "policy_objective": params.get("policy_objective", "signed"),
    }
    for scene in FOUR_SCENES:
        scene_metrics = metrics.get(scene, {})
        for metric in METRICS:
            row[scene_metric_key(scene, metric)] = scene_metrics.get(metric)
    return row


def compute_pair_delta(real: dict[str, Any], control: dict[str, Any], metric: str = "ap75") -> dict[str, Any]:
    deltas: dict[str, float] = {}
    values: list[float] = []
    for scene in FOUR_SCENES:
        key = scene_metric_key(scene, metric)
        if real.get(key) is None or control.get(key) is None:
            continue
        value = float(real[key]) - float(control[key])
        deltas[SCENE_PREFIX[scene]] = value
        values.append(value)
    mean_delta = sum(values) / max(1, len(values))
    return {
        "metric": metric,
        "mean_delta": mean_delta,
        "positive_scene_count": sum(1 for value in values if value > 0.0),
        "scene_deltas": deltas,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round30_results.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add spectral_detection_posttrain/analysis/round30_results.py tests/test_round30_results.py
git commit -m "feat: add round30 result schema"
```

---

## Task 4: Make NNI Trial Produce Complete Round 3.0 Rows

**Files:**
- Modify: `spectral_detection_posttrain/nni_rlvr_trial.py`
- Modify: `tests/test_round30_results.py`

- [ ] **Step 1: Add objective tests**

Append to `tests/test_round30_results.py`:

```python
def test_round30_objective_rejects_clean_ap50_drop():
    from spectral_detection_posttrain.nni_rlvr_trial import compute_round30_objective

    baseline = {scene: _metrics(0.60) for scene in FOUR_SCENES}
    metrics = {scene: _metrics(0.62) for scene in FOUR_SCENES}
    metrics["clean"]["ap50"] = baseline["clean"]["ap50"] - 0.04

    objective = compute_round30_objective(metrics, baseline)

    assert objective["default"] == -1.0
    assert objective["constraint_failed"] == "clean_ap50"


def test_round30_objective_rewards_ap75_and_ece():
    from spectral_detection_posttrain.nni_rlvr_trial import compute_round30_objective

    baseline = {scene: _metrics(0.60) for scene in FOUR_SCENES}
    metrics = {scene: _metrics(0.63) for scene in FOUR_SCENES}
    for scene in FOUR_SCENES:
        metrics[scene]["ece"] = baseline[scene]["ece"] - 0.01

    objective = compute_round30_objective(metrics, baseline)

    assert objective["constraint_failed"] == ""
    assert objective["default"] > 0
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round30_results.py -v
```

Expected: fails because `compute_round30_objective` is not defined.

- [ ] **Step 3: Import Round 3.0 helpers**

Add to `spectral_detection_posttrain/nni_rlvr_trial.py`:

```python
from spectral_detection_posttrain.analysis.round30_results import (
    FOUR_SCENES,
    build_round30_result_row,
)
```

- [ ] **Step 4: Pass seed to RLVR command**

In `_run_rlvr`, read seed:

```python
    seed = int(params.get("seed", 42))
```

Add to command:

```python
        "--seed", str(seed),
```

- [ ] **Step 5: Add Round 3.0 objective**

Add below `compute_round25_objective`:

```python
def compute_round30_objective(metrics: dict, baseline: dict) -> dict:
    for scene in FOUR_SCENES:
        if not metrics.get(scene):
            return {"default": -1.0, "constraint_failed": f"missing_{scene}"}

    checks = []
    clean = metrics["clean"]
    base_clean = baseline["clean"]
    checks.extend([
        ("clean_ap50", clean.get("ap50", 0.0) >= base_clean["ap50"] - 0.02),
        ("clean_recall", clean.get("recall", 0.0) >= base_clean["recall"] - 0.03),
        ("clean_num_predictions", clean.get("num_predictions", 10**9) <= base_clean["num_predictions"] * 1.15),
        ("clean_high_conf_fp", clean.get("high_conf_fp_count", 10**9) <= base_clean.get("high_conf_fp_count", 0) + 1),
    ])
    for scene in FOUR_SCENES[1:]:
        current = metrics[scene]
        base = baseline[scene]
        checks.extend([
            (f"{scene}_ap50", current.get("ap50", 0.0) >= base["ap50"] - 0.04),
            (f"{scene}_recall", current.get("recall", 0.0) >= base["recall"] - 0.04),
            (f"{scene}_num_predictions", current.get("num_predictions", 10**9) <= base["num_predictions"] * 1.20),
            (f"{scene}_high_conf_fp", current.get("high_conf_fp_count", 10**9) <= base.get("high_conf_fp_count", 0) + 1),
        ])
    for name, ok in checks:
        if not ok:
            return {"default": -1.0, "constraint_failed": name}

    score = 0.0
    for scene in FOUR_SCENES:
        current = metrics[scene]
        base = baseline[scene]
        ap75_gain = current.get("ap75", 0.0) - base.get("ap75", 0.0)
        ap50_gain = current.get("ap50", 0.0) - base.get("ap50", 0.0)
        ece_gain = base.get("ece", 0.0) - current.get("ece", 0.0)
        fp_gain = base.get("high_conf_fp_count", 0.0) - current.get("high_conf_fp_count", 0.0)
        score += current.get("ap50", 0.0)
        score += 1.25 * current.get("ap75", 0.0)
        score += 2.0 * ap75_gain
        score += 0.5 * ap50_gain
        score += 0.2 * ece_gain
        score += 0.02 * fp_gain
    return {"default": float(score), "constraint_failed": ""}
```

- [ ] **Step 6: Evaluate four scenes for Round 3.0**

In `main`, replace the eval scene selection block with:

```python
    eval_scenes = [
        ("clean", "none", "random"),
        ("object_edge_checkerboard", "object_edge", "checkerboard"),
        ("object_inside_checkerboard", "object_inside", "checkerboard"),
        ("near_object_checkerboard", "near_object", "checkerboard"),
    ] if "round30" in args.run_prefix or "round25" in args.run_prefix else [
        ("clean", "none", "random"),
        ("object_edge_checkerboard", "object_edge", "checkerboard"),
    ]
    for mode_key, patch_mode, patch_type in eval_scenes:
```

- [ ] **Step 7: Select Round 3.0 objective and row builder**

Replace the objective selection with:

```python
    if "round30" in args.run_prefix:
        objective = compute_round30_objective(metrics, baseline)
    elif "round25" in args.run_prefix:
        objective = compute_round25_objective(metrics, baseline)
    elif "round23" in args.run_prefix:
        objective = compute_round23_objective(metrics, baseline)
    else:
        objective = compute_round22_objective(metrics, baseline)
```

Replace `eval_status` with:

```python
    expected_eval_count = 4 if "round30" in args.run_prefix or "round25" in args.run_prefix else 2
    eval_status = "ok" if len(metrics) >= expected_eval_count else "failed"
```

Replace result row creation with:

```python
    if "round30" in args.run_prefix:
        result = build_round30_result_row(
            params=params,
            metrics=metrics,
            objective=objective,
            run_name=rlvr_result_path.parent.name,
            checkpoint=str(ckpt_path),
            eval_status=eval_status,
        )
    else:
        result = build_round23_result_row(
            params=params, metrics=metrics, objective=objective,
            run_name=rlvr_result_path.parent.name,
            checkpoint=str(ckpt_path),
            eval_status=eval_status,
        )
```

- [ ] **Step 8: Run tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round30_results.py tests/test_round23_readable_results.py -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```powershell
git add spectral_detection_posttrain/nni_rlvr_trial.py tests/test_round30_results.py
git commit -m "feat: emit complete round30 trial rows"
```

---

## Task 5: Generate Large Round 3.0 Search Spaces

**Files:**
- Create: `spectral_detection_posttrain/analysis/build_round30_search_space.py`
- Create: `tests/test_round30_search_space.py`

- [ ] **Step 1: Write search-space tests**

Create `tests/test_round30_search_space.py`:

```python
from spectral_detection_posttrain.analysis.build_round30_search_space import (
    build_phase_a_presets,
    build_phase_b_presets,
    build_phase_c_presets,
)


def test_phase_a_contains_amp_and_shuffled_amp_controls():
    presets = build_phase_a_presets()
    names = {preset["name"] for preset in presets}

    assert len(presets) == 68
    assert any("signed_amp_" in name for name in names)
    assert any("signed_shuffled_amp_" in name for name in names)
    assert any(preset["signal"] == "none" and preset["policy_loss_weight"] == 0.0003 for preset in presets)


def test_phase_b_has_expected_size_and_uses_amp_only():
    presets = build_phase_b_presets(best_reward_lambda=0.1)
    signals = {preset["signal"] for preset in presets}

    assert len(presets) == 54
    assert signals == {"ramp"}
    assert {preset["reward_lambda"] for preset in presets} == {0.1}


def test_phase_c_contains_structure_controls():
    presets = build_phase_c_presets(best_reward_lambda=0.1)
    signals = {preset["signal"] for preset in presets}

    assert len(presets) == 24
    assert signals == {"structure", "shuffled_structure", "amp_structure", "shuffled_amp_structure"}
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round30_search_space.py -v
```

Expected: fails because `build_round30_search_space.py` does not exist.

- [ ] **Step 3: Implement search-space generator**

Create `spectral_detection_posttrain/analysis/build_round30_search_space.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path


SEEDS = [42, 43]


def _base(name: str, signal: str, seed: int) -> dict:
    return {
        "name": name,
        "signal": signal,
        "reward_lambda": 0.0,
        "struct_weight": 0.0,
        "policy_loss_weight": 0.0003,
        "det_loss_weight": 0.0,
        "baseline_kl_weight": 10.0,
        "box_loss_weight": 0.0,
        "unfreeze": "cls",
        "optimizer": "adamw",
        "temperature": 1.0,
        "max_candidates": 40,
        "reward_score_threshold": 0.2,
        "rollout_source": "baseline",
        "policy_objective": "signed",
        "seed": seed,
    }


def build_phase_a_presets() -> list[dict]:
    presets: list[dict] = []
    for seed in SEEDS:
        null = _base(f"null_no_update_seed{seed}", "none", seed)
        null.update({"policy_loss_weight": 0.0, "baseline_kl_weight": 0.0})
        presets.append(null)
        presets.append(_base(f"signed_iou_seed{seed}", "none", seed))

    reward_lambdas = [0.025, 0.05, 0.1, 0.2]
    policy_weights = [0.0001, 0.0003]
    kl_weights = [5.0, 10.0]
    for seed in SEEDS:
        for signal, label in [("ramp", "amp"), ("shuffled_amp", "shuffled_amp")]:
            for reward_lambda in reward_lambdas:
                for policy_weight in policy_weights:
                    for kl_weight in kl_weights:
                        preset = _base(
                            f"signed_{label}_l{reward_lambda:g}_pl{policy_weight:g}_kl{kl_weight:g}_seed{seed}",
                            signal,
                            seed,
                        )
                        preset.update({
                            "reward_lambda": reward_lambda,
                            "policy_loss_weight": policy_weight,
                            "baseline_kl_weight": kl_weight,
                        })
                        presets.append(preset)
    return presets


def build_phase_b_presets(best_reward_lambda: float = 0.1) -> list[dict]:
    presets: list[dict] = []
    policy_weights = [0.0001, 0.0003, 0.0007]
    kl_weights = [5.0, 10.0, 20.0]
    max_candidates_values = [20, 40, 80]
    for seed in SEEDS:
        for policy_weight in policy_weights:
            for kl_weight in kl_weights:
                for max_candidates in max_candidates_values:
                    preset = _base(
                        f"amp_tune_l{best_reward_lambda:g}_pl{policy_weight:g}_kl{kl_weight:g}_mc{max_candidates}_seed{seed}",
                        "ramp",
                        seed,
                    )
                    preset.update({
                        "reward_lambda": best_reward_lambda,
                        "policy_loss_weight": policy_weight,
                        "baseline_kl_weight": kl_weight,
                        "max_candidates": max_candidates,
                    })
                    presets.append(preset)
    return presets


def build_phase_c_presets(best_reward_lambda: float = 0.1) -> list[dict]:
    presets: list[dict] = []
    struct_weights = [0.05, 0.1, 0.2]
    signals = [
        ("structure", "structure"),
        ("shuffled_structure", "shuffled_structure"),
        ("amp_structure", "amp_structure"),
        ("shuffled_amp_structure", "shuffled_amp_structure"),
    ]
    for seed in SEEDS:
        for signal, label in signals:
            for struct_weight in struct_weights:
                preset = _base(f"{label}_sw{struct_weight:g}_seed{seed}", signal, seed)
                preset.update({
                    "reward_lambda": best_reward_lambda if "amp" in signal else 0.0,
                    "struct_weight": struct_weight,
                })
                presets.append(preset)
    return presets


def write_search_space(presets: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"preset": {"_type": "choice", "_value": presets}}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["A", "B", "C"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--best-reward-lambda", type=float, default=0.1)
    args = parser.parse_args()

    if args.phase == "A":
        presets = build_phase_a_presets()
    elif args.phase == "B":
        presets = build_phase_b_presets(best_reward_lambda=args.best_reward_lambda)
    else:
        presets = build_phase_c_presets(best_reward_lambda=args.best_reward_lambda)

    write_search_space(presets, Path(args.output))
    print(f"wrote {len(presets)} presets to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round30_search_space.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Generate Phase A/B/C JSON files**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.analysis.build_round30_search_space --phase A --output nni_configs/rlvr_round30_phaseA_search_space.json
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.analysis.build_round30_search_space --phase B --best-reward-lambda 0.1 --output nni_configs/rlvr_round30_phaseB_search_space.json
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.analysis.build_round30_search_space --phase C --best-reward-lambda 0.1 --output nni_configs/rlvr_round30_phaseC_search_space.json
```

Expected:

```text
wrote 68 presets to nni_configs/rlvr_round30_phaseA_search_space.json
wrote 54 presets to nni_configs/rlvr_round30_phaseB_search_space.json
wrote 24 presets to nni_configs/rlvr_round30_phaseC_search_space.json
```

- [ ] **Step 6: Commit**

```powershell
git add spectral_detection_posttrain/analysis/build_round30_search_space.py tests/test_round30_search_space.py nni_configs/rlvr_round30_phaseA_search_space.json nni_configs/rlvr_round30_phaseB_search_space.json nni_configs/rlvr_round30_phaseC_search_space.json
git commit -m "feat: generate round30 search spaces"
```

---

## Task 6: Add NNI Configs And Run Scripts

**Files:**
- Create: `nni_configs/rlvr_round30_phaseA_config.yml`
- Create: `nni_configs/rlvr_round30_phaseB_config.yml`
- Create: `nni_configs/rlvr_round30_phaseC_config.yml`
- Create: `run_nni_rlvr_round30_phaseA.bat`
- Create: `run_nni_rlvr_round30_phaseB.bat`
- Create: `run_nni_rlvr_round30_phaseC.bat`

- [ ] **Step 1: Create Phase A config**

Create `nni_configs/rlvr_round30_phaseA_config.yml`:

```yaml
experimentName: rlvr_round30_phaseA_amp_causality
experimentWorkingDirectory: E:/CLIproject/RLimage/nni_experiments
trialCommand: E:/anaconda/01/envs/RLimage/python.exe -m spectral_detection_posttrain.nni_rlvr_trial --config spectral_detection_posttrain/configs/mvp.yaml --run-prefix nni_rlvr_round30_phaseA --rlvr-epochs 3 --early-stopping-patience 2
trialCodeDirectory: E:/CLIproject/RLimage
searchSpaceFile: rlvr_round30_phaseA_search_space.json
trialConcurrency: 1
maxTrialNumber: 68
maxExperimentDuration: 72h
tuner:
  name: GridSearch
trainingService:
  platform: local
```

- [ ] **Step 2: Create Phase B config**

Create `nni_configs/rlvr_round30_phaseB_config.yml`:

```yaml
experimentName: rlvr_round30_phaseB_amp_tuning
experimentWorkingDirectory: E:/CLIproject/RLimage/nni_experiments
trialCommand: E:/anaconda/01/envs/RLimage/python.exe -m spectral_detection_posttrain.nni_rlvr_trial --config spectral_detection_posttrain/configs/mvp.yaml --run-prefix nni_rlvr_round30_phaseB --rlvr-epochs 5 --early-stopping-patience 2
trialCodeDirectory: E:/CLIproject/RLimage
searchSpaceFile: rlvr_round30_phaseB_search_space.json
trialConcurrency: 1
maxTrialNumber: 54
maxExperimentDuration: 96h
tuner:
  name: GridSearch
trainingService:
  platform: local
```

- [ ] **Step 3: Create Phase C config**

Create `nni_configs/rlvr_round30_phaseC_config.yml`:

```yaml
experimentName: rlvr_round30_phaseC_structure_recheck
experimentWorkingDirectory: E:/CLIproject/RLimage/nni_experiments
trialCommand: E:/anaconda/01/envs/RLimage/python.exe -m spectral_detection_posttrain.nni_rlvr_trial --config spectral_detection_posttrain/configs/mvp.yaml --run-prefix nni_rlvr_round30_phaseC --rlvr-epochs 3 --early-stopping-patience 2
trialCodeDirectory: E:/CLIproject/RLimage
searchSpaceFile: rlvr_round30_phaseC_search_space.json
trialConcurrency: 1
maxTrialNumber: 24
maxExperimentDuration: 48h
tuner:
  name: GridSearch
trainingService:
  platform: local
```

- [ ] **Step 4: Create run scripts**

Create `run_nni_rlvr_round30_phaseA.bat`:

```bat
@echo off
cd /d E:\CLIproject\RLimage
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.analysis.build_round30_search_space --phase A --output nni_configs\rlvr_round30_phaseA_search_space.json
E:\anaconda\01\envs\RLimage\nni.exe experiment create --config nni_configs\rlvr_round30_phaseA_config.yml --port 8100
```

Create `run_nni_rlvr_round30_phaseB.bat`:

```bat
@echo off
cd /d E:\CLIproject\RLimage
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.analysis.build_round30_search_space --phase B --best-reward-lambda 0.1 --output nni_configs\rlvr_round30_phaseB_search_space.json
E:\anaconda\01\envs\RLimage\nni.exe experiment create --config nni_configs\rlvr_round30_phaseB_config.yml --port 8101
```

Create `run_nni_rlvr_round30_phaseC.bat`:

```bat
@echo off
cd /d E:\CLIproject\RLimage
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.analysis.build_round30_search_space --phase C --best-reward-lambda 0.1 --output nni_configs\rlvr_round30_phaseC_search_space.json
E:\anaconda\01\envs\RLimage\nni.exe experiment create --config nni_configs\rlvr_round30_phaseC_config.yml --port 8102
```

- [ ] **Step 5: Validate YAML files parse**

Run:

```powershell
@'
from pathlib import Path
import yaml
for path in Path("nni_configs").glob("rlvr_round30_phase*_config.yml"):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["trainingService"]["platform"] == "local"
    assert data["tuner"]["name"] == "GridSearch"
    print(path, data["maxTrialNumber"])
'@ | E:\anaconda\01\envs\RLimage\python.exe -
```

Expected:

```text
nni_configs\rlvr_round30_phaseA_config.yml 68
nni_configs\rlvr_round30_phaseB_config.yml 54
nni_configs\rlvr_round30_phaseC_config.yml 24
```

- [ ] **Step 6: Commit**

```powershell
git add nni_configs/rlvr_round30_phaseA_config.yml nni_configs/rlvr_round30_phaseB_config.yml nni_configs/rlvr_round30_phaseC_config.yml run_nni_rlvr_round30_phaseA.bat run_nni_rlvr_round30_phaseB.bat run_nni_rlvr_round30_phaseC.bat
git commit -m "chore: add round30 nni launch configs"
```

---

## Task 7: Add Round 3.0 Summarizer

**Files:**
- Create: `spectral_detection_posttrain/analysis/summarize_round30_results.py`
- Modify: `tests/test_round30_results.py`
- Create: `docs/rlvr_plan30_amp_causality_report.md`

- [ ] **Step 1: Add summarizer tests**

Append to `tests/test_round30_results.py`:

```python
def test_causality_gate_accepts_strong_amp_delta():
    from spectral_detection_posttrain.analysis.summarize_round30_results import causality_gate

    delta = {
        "mean_delta": 0.008,
        "positive_scene_count": 4,
        "scene_deltas": {"clean": 0.006, "edge": 0.008, "inside": 0.01, "near": 0.008},
    }

    result = causality_gate(delta, min_mean_delta=0.005)

    assert result["passed"]
    assert result["reason"] == "passed"


def test_causality_gate_rejects_mixed_sign_delta():
    from spectral_detection_posttrain.analysis.summarize_round30_results import causality_gate

    delta = {
        "mean_delta": 0.006,
        "positive_scene_count": 2,
        "scene_deltas": {"clean": 0.01, "edge": -0.002, "inside": 0.012, "near": -0.001},
    }

    result = causality_gate(delta, min_mean_delta=0.005)

    assert not result["passed"]
    assert result["reason"] == "positive_scene_count"
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round30_results.py -v
```

Expected: fails because `summarize_round30_results.py` does not exist.

- [ ] **Step 3: Implement summarizer**

Create `spectral_detection_posttrain/analysis/summarize_round30_results.py`:

```python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from spectral_detection_posttrain.analysis.round30_results import (
    FOUR_SCENES,
    SCENE_PREFIX,
    compute_pair_delta,
    load_jsonl,
    scene_metric_key,
)


def causality_gate(delta: dict[str, Any], min_mean_delta: float = 0.005) -> dict[str, Any]:
    if float(delta.get("mean_delta", 0.0)) < min_mean_delta:
        return {"passed": False, "reason": "mean_delta"}
    if int(delta.get("positive_scene_count", 0)) < 3:
        return {"passed": False, "reason": "positive_scene_count"}
    return {"passed": True, "reason": "passed"}


def _group_key(row: dict[str, Any]) -> tuple:
    return (
        row.get("signal"),
        float(row.get("reward_lambda", 0.0)),
        float(row.get("struct_weight", 0.0)),
        float(row.get("policy_loss_weight", 0.0)),
        float(row.get("baseline_kl_weight", 0.0)),
        int(row.get("max_candidates", 0)),
    )


def _metric_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / max(1, len(values))


def group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("eval_status") == "ok" and row.get("constraint_failed", "") == "":
            groups[_group_key(row)].append(row)

    summary: list[dict[str, Any]] = []
    for key, group in groups.items():
        signal, reward_lambda, struct_weight, policy_loss_weight, baseline_kl_weight, max_candidates = key
        item = {
            "signal": signal,
            "reward_lambda": reward_lambda,
            "struct_weight": struct_weight,
            "policy_loss_weight": policy_loss_weight,
            "baseline_kl_weight": baseline_kl_weight,
            "max_candidates": max_candidates,
            "count": len(group),
        }
        for scene in FOUR_SCENES:
            prefix = SCENE_PREFIX[scene]
            for metric in ["ap50", "ap75", "ece", "high_conf_fp_count", "recall"]:
                item[f"{prefix}_{metric}_mean"] = _metric_mean(group, scene_metric_key(scene, metric))
        item["mean_ap75_all_scenes"] = sum(item[f"{SCENE_PREFIX[scene]}_ap75_mean"] for scene in FOUR_SCENES) / len(FOUR_SCENES)
        summary.append(item)
    return sorted(summary, key=lambda row: row["mean_ap75_all_scenes"], reverse=True)


def _find_matching_control(rows: list[dict[str, Any]], real: dict[str, Any], control_signal: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("signal") != control_signal:
            continue
        if float(row.get("reward_lambda", 0.0)) != float(real.get("reward_lambda", 0.0)):
            continue
        if float(row.get("policy_loss_weight", 0.0)) != float(real.get("policy_loss_weight", 0.0)):
            continue
        if float(row.get("baseline_kl_weight", 0.0)) != float(real.get("baseline_kl_weight", 0.0)):
            continue
        if int(row.get("seed", 42)) != int(real.get("seed", 42)):
            continue
        return row
    return None


def compute_amp_causal_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for real in rows:
        if real.get("signal") != "ramp":
            continue
        control = _find_matching_control(rows, real, "shuffled_amp")
        if control is None:
            continue
        delta = compute_pair_delta(real, control, metric="ap75")
        gate = causality_gate(delta)
        pairs.append({
            "name": real.get("name"),
            "control": control.get("name"),
            "reward_lambda": real.get("reward_lambda"),
            "policy_loss_weight": real.get("policy_loss_weight"),
            "baseline_kl_weight": real.get("baseline_kl_weight"),
            "seed": real.get("seed"),
            "mean_ap75_delta": delta["mean_delta"],
            "positive_scene_count": delta["positive_scene_count"],
            "gate_passed": gate["passed"],
            "gate_reason": gate["reason"],
            "scene_deltas": delta["scene_deltas"],
        })
    return sorted(pairs, key=lambda row: row["mean_ap75_delta"], reverse=True)


def render_report(rows: list[dict[str, Any]]) -> str:
    grouped = group_rows(rows)
    amp_pairs = compute_amp_causal_pairs(rows)
    lines = ["# Plan 3.0 Amp Causality Report", ""]
    lines.append("## Top Hyperparameter Groups")
    lines.append("")
    lines.append("| signal | lambda | policy | KL | max candidates | count | mean AP75 | clean AP50 | edge AP75 | inside AP75 | near AP75 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in grouped[:20]:
        lines.append(
            f"| {row['signal']} | {row['reward_lambda']:.3g} | {row['policy_loss_weight']:.3g} | "
            f"{row['baseline_kl_weight']:.3g} | {row['max_candidates']} | {row['count']} | "
            f"{row['mean_ap75_all_scenes']:.6f} | {row['clean_ap50_mean']:.6f} | "
            f"{row['edge_ap75_mean']:.6f} | {row['inside_ap75_mean']:.6f} | {row['near_ap75_mean']:.6f} |"
        )

    lines.extend(["", "## Amp Causality Pairs", ""])
    lines.append("| real | control | lambda | policy | KL | seed | mean AP75 delta | positive scenes | gate |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for pair in amp_pairs:
        lines.append(
            f"| {pair['name']} | {pair['control']} | {float(pair['reward_lambda']):.3g} | "
            f"{float(pair['policy_loss_weight']):.3g} | {float(pair['baseline_kl_weight']):.3g} | "
            f"{int(pair['seed'])} | {pair['mean_ap75_delta']:.6f} | "
            f"{pair['positive_scene_count']} | {pair['gate_reason']} |"
        )

    passed = [pair for pair in amp_pairs if pair["gate_passed"]]
    lines.extend(["", "## Decision", ""])
    if passed:
        best = passed[0]
        lines.append(
            f"Amp causality gate passed. Best pair: {best['name']} over {best['control']} "
            f"with mean AP75 delta {best['mean_ap75_delta']:.6f}."
        )
    else:
        lines.append("Amp causality gate did not pass. Treat amplitude as promising but not proven causal.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--output", default="docs/rlvr_plan30_amp_causality_report.md")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path_str in args.results:
        rows.extend(load_jsonl(Path(path_str)))
    report = render_report(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create initial report file**

Create `docs/rlvr_plan30_amp_causality_report.md`:

````markdown
# Plan 3.0 Amp Causality Report

Generate this report after Phase A, then refresh it after Phase B and Phase C:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.analysis.summarize_round30_results --results runs/nni_rlvr_round30_phaseA/nni_rlvr_results.jsonl --output docs/rlvr_plan30_amp_causality_report.md
```
````

- [ ] **Step 5: Run tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round30_results.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add spectral_detection_posttrain/analysis/summarize_round30_results.py tests/test_round30_results.py docs/rlvr_plan30_amp_causality_report.md
git commit -m "feat: summarize round30 causality results"
```

---

## Task 8: Smoke Test The Round 3.0 Pipeline

**Files:**
- Uses implemented files.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round30_training_diagnostics.py tests/test_round30_results.py tests/test_round30_search_space.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Generate a small two-trial smoke search space**

Create `tmp_round30_smoke_search.json`:

```json
{
  "preset": {
    "_type": "choice",
    "_value": [
      {
        "name": "smoke_amp_seed42",
        "signal": "ramp",
        "reward_lambda": 0.1,
        "struct_weight": 0.0,
        "policy_loss_weight": 0.0003,
        "det_loss_weight": 0.0,
        "baseline_kl_weight": 10.0,
        "box_loss_weight": 0.0,
        "unfreeze": "cls",
        "optimizer": "adamw",
        "temperature": 1.0,
        "max_candidates": 20,
        "reward_score_threshold": 0.2,
        "rollout_source": "baseline",
        "policy_objective": "signed",
        "seed": 42
      },
      {
        "name": "smoke_shuffled_amp_seed42",
        "signal": "shuffled_amp",
        "reward_lambda": 0.1,
        "struct_weight": 0.0,
        "policy_loss_weight": 0.0003,
        "det_loss_weight": 0.0,
        "baseline_kl_weight": 10.0,
        "box_loss_weight": 0.0,
        "unfreeze": "cls",
        "optimizer": "adamw",
        "temperature": 1.0,
        "max_candidates": 20,
        "reward_score_threshold": 0.2,
        "rollout_source": "baseline",
        "policy_objective": "signed",
        "seed": 42
      }
    ]
  }
}
```

- [ ] **Step 3: Run one amp smoke trial directly**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.nni_rlvr_trial --config spectral_detection_posttrain/configs/mvp.yaml --run-prefix nni_rlvr_round30_smoke --params-json "{\"preset\":{\"name\":\"smoke_amp_seed42\",\"signal\":\"ramp\",\"reward_lambda\":0.1,\"struct_weight\":0.0,\"policy_loss_weight\":0.0003,\"det_loss_weight\":0.0,\"baseline_kl_weight\":10.0,\"box_loss_weight\":0.0,\"unfreeze\":\"cls\",\"optimizer\":\"adamw\",\"temperature\":1.0,\"max_candidates\":20,\"reward_score_threshold\":0.2,\"rollout_source\":\"baseline\",\"policy_objective\":\"signed\",\"seed\":42}}" --limit-train 4 --limit-val 4 --rlvr-epochs 1 --early-stopping-patience 1
```

Expected:

```text
runs/nni_rlvr_round30_smoke/nni_rlvr_results.jsonl exists
result row contains clean_ap75, edge_ap75, inside_ap75, near_ap75
metrics_train.jsonl contains amp_mean and reward_mean
```

- [ ] **Step 4: Run summarizer on smoke output**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.analysis.summarize_round30_results --results runs/nni_rlvr_round30_smoke/nni_rlvr_results.jsonl --output docs/rlvr_plan30_amp_causality_report.md
```

Expected:

```text
docs/rlvr_plan30_amp_causality_report.md exists
report contains "Top Hyperparameter Groups"
```

- [ ] **Step 5: Commit smoke fixes if code changed**

If the smoke run required code edits:

```powershell
git add spectral_detection_posttrain tests nni_configs docs
git commit -m "fix: pass round30 smoke pipeline"
```

If no code edits were made, skip this commit.

---

## Task 9: Run Phase A Amp Causality Grid

**Files:**
- Produces: `runs/nni_rlvr_round30_phaseA/nni_rlvr_results.jsonl`
- Updates: `docs/rlvr_plan30_amp_causality_report.md`

- [ ] **Step 1: Launch Phase A**

Run:

```powershell
.\run_nni_rlvr_round30_phaseA.bat
```

Expected:

```text
NNI starts on port 8100
maxTrialNumber = 68
```

- [ ] **Step 2: Wait for all Phase A trials**

Use NNI UI or terminal logs until all 68 trials finish. Required file:

```text
runs/nni_rlvr_round30_phaseA/nni_rlvr_results.jsonl
```

- [ ] **Step 3: Generate Phase A report**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.analysis.summarize_round30_results --results runs/nni_rlvr_round30_phaseA/nni_rlvr_results.jsonl --output docs/rlvr_plan30_amp_causality_report.md
```

Expected:

```text
Amp Causality Pairs table is populated
Decision section says whether amp causality gate passed
```

- [ ] **Step 4: Decide Phase B launch condition**

Phase B should run only if at least one amp pair has:

```text
mean AP75 delta over shuffled_amp >= 0.003
```

If no pair reaches `0.003`, stop and write a negative Phase A report. Do not spend Phase B budget.

- [ ] **Step 5: Commit Phase A report**

```powershell
git add docs/rlvr_plan30_amp_causality_report.md
git commit -m "docs: report round30 phaseA amp causality"
```

---

## Task 10: Run Phase B Amp Hyperparameter Tuning

**Files:**
- Produces: `runs/nni_rlvr_round30_phaseB/nni_rlvr_results.jsonl`
- Updates: `docs/rlvr_plan30_amp_causality_report.md`

- [ ] **Step 1: Choose Phase B reward lambda**

Open the Phase A report and choose the best `reward_lambda` from the top passed or near-passed amp pair. If the report has no passed pair, use the best positive mean AP75 delta. Write the chosen value into the command below.

Example command for `0.1`:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.analysis.build_round30_search_space --phase B --best-reward-lambda 0.1 --output nni_configs/rlvr_round30_phaseB_search_space.json
```

Expected:

```text
wrote 54 presets to nni_configs/rlvr_round30_phaseB_search_space.json
```

- [ ] **Step 2: Launch Phase B**

Run:

```powershell
.\run_nni_rlvr_round30_phaseB.bat
```

Expected:

```text
NNI starts on port 8101
maxTrialNumber = 54
```

- [ ] **Step 3: Generate combined Phase A+B report**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.analysis.summarize_round30_results --results runs/nni_rlvr_round30_phaseA/nni_rlvr_results.jsonl runs/nni_rlvr_round30_phaseB/nni_rlvr_results.jsonl --output docs/rlvr_plan30_amp_causality_report.md
```

Expected:

```text
Top Hyperparameter Groups table ranks Phase B configs
```

- [ ] **Step 4: Commit Phase B report**

```powershell
git add nni_configs/rlvr_round30_phaseB_search_space.json docs/rlvr_plan30_amp_causality_report.md
git commit -m "docs: report round30 phaseB amp tuning"
```

---

## Task 11: Run Phase C Structure Recheck

**Files:**
- Produces: `runs/nni_rlvr_round30_phaseC/nni_rlvr_results.jsonl`
- Updates: `docs/rlvr_plan30_amp_causality_report.md`

- [ ] **Step 1: Generate Phase C search space**

Use the same best amplitude lambda chosen for Phase B. Example for `0.1`:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.analysis.build_round30_search_space --phase C --best-reward-lambda 0.1 --output nni_configs/rlvr_round30_phaseC_search_space.json
```

Expected:

```text
wrote 24 presets to nni_configs/rlvr_round30_phaseC_search_space.json
```

- [ ] **Step 2: Launch Phase C**

Run:

```powershell
.\run_nni_rlvr_round30_phaseC.bat
```

Expected:

```text
NNI starts on port 8102
maxTrialNumber = 24
```

- [ ] **Step 3: Generate final report**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.analysis.summarize_round30_results --results runs/nni_rlvr_round30_phaseA/nni_rlvr_results.jsonl runs/nni_rlvr_round30_phaseB/nni_rlvr_results.jsonl runs/nni_rlvr_round30_phaseC/nni_rlvr_results.jsonl --output docs/rlvr_plan30_amp_causality_report.md
```

Expected:

```text
Final report includes Top Hyperparameter Groups and Amp Causality Pairs
```

- [ ] **Step 4: Commit final report**

```powershell
git add nni_configs/rlvr_round30_phaseC_search_space.json docs/rlvr_plan30_amp_causality_report.md
git commit -m "docs: report round30 final search"
```

---

## Task 12: Final Verification Checklist

**Files:**
- Uses result files and docs.

- [ ] **Step 1: Verify all expected result files exist**

Run:

```powershell
Get-Item runs\nni_rlvr_round30_phaseA\nni_rlvr_results.jsonl
Get-Item runs\nni_rlvr_round30_phaseB\nni_rlvr_results.jsonl
Get-Item runs\nni_rlvr_round30_phaseC\nni_rlvr_results.jsonl
Get-Item docs\rlvr_plan30_amp_causality_report.md
```

Expected: all four files are printed.

- [ ] **Step 2: Count result rows**

Run:

```powershell
(Get-Content runs\nni_rlvr_round30_phaseA\nni_rlvr_results.jsonl).Count
(Get-Content runs\nni_rlvr_round30_phaseB\nni_rlvr_results.jsonl).Count
(Get-Content runs\nni_rlvr_round30_phaseC\nni_rlvr_results.jsonl).Count
```

Expected:

```text
68
54
24
```

- [ ] **Step 3: Verify result rows have four scene metrics**

Run:

```powershell
@'
import json
from pathlib import Path
for path in [
    Path("runs/nni_rlvr_round30_phaseA/nni_rlvr_results.jsonl"),
    Path("runs/nni_rlvr_round30_phaseB/nni_rlvr_results.jsonl"),
    Path("runs/nni_rlvr_round30_phaseC/nni_rlvr_results.jsonl"),
]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        for key in ["clean_ap75", "edge_ap75", "inside_ap75", "near_ap75"]:
            assert row.get(key) is not None, (path, row.get("name"), key)
    print(path, len(rows), "ok")
'@ | E:\anaconda\01\envs\RLimage\python.exe -
```

Expected:

```text
runs\nni_rlvr_round30_phaseA\nni_rlvr_results.jsonl 68 ok
runs\nni_rlvr_round30_phaseB\nni_rlvr_results.jsonl 54 ok
runs\nni_rlvr_round30_phaseC\nni_rlvr_results.jsonl 24 ok
```

- [ ] **Step 4: Verify final tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round30_training_diagnostics.py tests/test_round30_results.py tests/test_round30_search_space.py tests/test_rlvr_verifier.py tests/test_rlvr_policy_objective.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit verification note if report changed**

If `docs/rlvr_plan30_amp_causality_report.md` changed after verification:

```powershell
git add docs/rlvr_plan30_amp_causality_report.md
git commit -m "docs: finalize round30 verification"
```

If no files changed, skip this commit.

---

## Success Criteria

Plan 3.0 is successful if:

```text
1. Phase A includes real amp and shuffled_amp controls with matched hyperparameters and seeds.
2. Every trial writes clean, edge, inside, and near metrics.
3. Training logs include amp_mean, structure_mean, reward_mean, and reward_std.
4. Phase A report decides whether amp is causal, not merely better than baseline.
5. Phase B identifies a constrained best amp configuration, or is skipped because Phase A failed the launch condition.
6. Phase C determines whether structure adds value beyond amp and beyond shuffled controls.
7. Final report states one of:
   - Amp causal and best config selected
   - Amp promising but not causal
   - RLVR shell stable but verifier signal unproven
```

---

## Final Interpretation Rules

Use these exact interpretations:

| Result | Interpretation | Next Step |
|---|---|---|
| Amp beats shuffled_amp in >= 3 scenes and mean AP75 delta >= 0.005 | Amplitude verifier has causal value | Run 5-epoch confirmation and VOC person subset |
| Amp beats baseline but not shuffled_amp | RLVR perturbation helps, amplitude causality unproven | Build learned verifier or stronger negative scenes |
| Structure beats shuffled_structure on edge/inside | Structure branch has localized value | Search lower struct weights and consider bbox head update |
| Amp+Struct does not beat Amp | Structure is not complementary yet | Keep Amp-only for next confirmation |
| Best config improves AP75 but worsens ECE | Verifier is localization-biased | Add ECE or high-conf FP penalty to objective |
| Any config collapses AP50/recall | Stability regression | Inspect KL, candidate count, and freeze-state logs before more search |

---

## Self-Review Checklist

- Spec coverage: includes next improvement, causal control, and large hyperparameter search.
- Search size: Phase A/B/C total 146 trials.
- Controls: includes IoU-only, null, amp, shuffled_amp, structure, shuffled_structure, amp_structure, shuffled_amp_structure.
- Evaluation: every Round 3.0 row must include clean, edge, inside, and near scenes.
- Causality: final decision depends on real-vs-shuffled pair deltas.
- Stability: preserves baseline rollout, signed objective, KL, and `det_loss_weight=0`.
- Result readability: final report gives top configs and causal-pair table.
