# Round 2.4 Fill Result Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the two missing Round 2.3 result gaps before making any R_amp causality claim: `det_only_cls` must produce a readable result row, and `signed_ramp_0003_kl10` must have a valid clean evaluation.

**Architecture:** Round 2.4 does not introduce a new RLVR method or a larger search. It is a result-completion and audit pass over the existing Round 2.3 matrix. It fixes the failure paths that allowed a trial directory to exist without `rlvr_result.json`, and an eval directory to exist without `eval_metrics.json`; then it reruns only the missing pieces and writes an auditable consolidated report.

**Tech Stack:** Python, PyTorch, TorchVision Faster R-CNN, existing `spectral_detection_posttrain` package, pytest, local `runs/nni_rlvr_round23` outputs.

---

## Round 2.3 Gaps To Close

Round 2.3 made the RLVR training stable, but two result gaps remain:

1. `det_only_cls` gap
   - Directory exists: `runs/nni_rlvr_round23/rlvr_det_only_cls_cls_adamw`
   - `initial_sanity.json` exists.
   - `rlvr_result.json` is missing.
   - No clean/edge eval rows were written.
   - This prevents judging whether supervised cls continuation is still destructive.

2. `signed_ramp_0003_kl10` clean eval gap
   - Directory exists: `runs/nni_rlvr_round23/rlvr_signed_ramp_0003_kl10_cls_adamw`
   - `rlvr_result.json` exists.
   - Edge eval exists.
   - Clean eval directory exists but only has `config.yaml`; `eval_metrics.json` is missing.
   - This prevents comparing R_amp vs IoU-only vs shuffled R_amp.

Round 2.4 success is intentionally narrow:

```text
nni_rlvr_results.jsonl or a repaired result file has all 6 Round 2.3 preset names.
det_only_cls has clean and edge metrics.
signed_ramp_0003_kl10 has clean and edge metrics.
No new matrix is launched.
```

---

## Files

- Modify: `spectral_detection_posttrain/nni_rlvr_trial.py`
  Add repair helpers that can rebuild a readable row from an existing run directory and eval outputs.
- Create: `spectral_detection_posttrain/analysis/repair_round23_results.py`
  Reconstructs `runs/nni_rlvr_round23/nni_rlvr_results_repaired.jsonl` from existing Round 2.3 run directories and rerun eval outputs.
- Create: `tests/test_round24_result_repair.py`
  Tests rebuilding readable rows from partial outputs and detecting missing clean/edge eval.
- Create: `docs/rlvr_round24_gap_closure.md`
  Final short report after the two gaps are closed.

---

## Task 1: Add Repairable Result Row Builder

**Files:**
- Modify: `spectral_detection_posttrain/nni_rlvr_trial.py`
- Create: `tests/test_round24_result_repair.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_round24_result_repair.py`:

```python
from spectral_detection_posttrain.nni_rlvr_trial import (
    build_round23_result_row,
    collect_eval_status,
)


def test_collect_eval_status_ok_when_clean_and_edge_exist():
    metrics = {
        "clean": {"ap50": 0.87},
        "object_edge_checkerboard": {"ap50": 0.86},
    }

    assert collect_eval_status(metrics) == "ok"


def test_collect_eval_status_names_missing_clean():
    metrics = {
        "object_edge_checkerboard": {"ap50": 0.86},
    }

    assert collect_eval_status(metrics) == "missing_clean"


def test_repaired_row_keeps_name_even_when_eval_missing():
    row = build_round23_result_row(
        params={"name": "signed_ramp_0003_kl10", "signal": "ramp"},
        metrics={"object_edge_checkerboard": {"ap50": 0.87}},
        objective={"default": -1.0, "constraint_failed": "missing_clean"},
        run_name="rlvr_signed_ramp_0003_kl10_cls_adamw",
        checkpoint="runs/x/checkpoint_best.pth",
        eval_status="missing_clean",
    )

    assert row["name"] == "signed_ramp_0003_kl10"
    assert row["eval_status"] == "missing_clean"
    assert row["clean_ap50"] is None
    assert row["edge_ap50"] == 0.87
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round24_result_repair.py -v
```

Expected: fails because `collect_eval_status` does not exist.

- [ ] **Step 3: Implement `collect_eval_status`**

Add to `spectral_detection_posttrain/nni_rlvr_trial.py`:

```python
def collect_eval_status(metrics: dict) -> str:
    has_clean = bool(metrics.get("clean"))
    has_edge = bool(metrics.get("object_edge_checkerboard"))
    if has_clean and has_edge:
        return "ok"
    if not has_clean and not has_edge:
        return "missing_clean_and_edge"
    if not has_clean:
        return "missing_clean"
    return "missing_edge"
```

When building objective for repair:

```python
status = collect_eval_status(metrics)
if status != "ok":
    objective = {"default": -1.0, "constraint_failed": status}
else:
    objective = compute_round23_objective(metrics, baseline)
```

- [ ] **Step 4: Run tests and commit**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round24_result_repair.py tests/test_round23_readable_results.py -v
git add spectral_detection_posttrain/nni_rlvr_trial.py tests/test_round24_result_repair.py
git commit -m "fix: classify missing RLVR eval outputs"
```

Expected: tests pass and commit succeeds.

---

## Task 2: Create Round 2.3 Result Repair Script

**Files:**
- Create: `spectral_detection_posttrain/analysis/repair_round23_results.py`

- [ ] **Step 1: Create analysis package if missing**

Create `spectral_detection_posttrain/analysis/__init__.py`:

```python
"""Analysis helpers for RLVR experiment outputs."""
```

- [ ] **Step 2: Implement repair script**

Create `spectral_detection_posttrain/analysis/repair_round23_results.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spectral_detection_posttrain.nni_rlvr_trial import (
    build_round23_result_row,
    collect_eval_status,
    compute_round23_objective,
)


ROUND23_PRESETS = {
    "null_no_update": "rlvr_null_no_update_cls_adamw",
    "det_only_cls": "rlvr_det_only_cls_cls_adamw",
    "signed_iou_0003_kl10": "rlvr_signed_iou_0003_kl10_cls_adamw",
    "signed_ramp_0003_kl10": "rlvr_signed_ramp_0003_kl10_cls_adamw",
    "signed_shuffled_0003_kl10": "rlvr_signed_shuffled_0003_kl10_cls_adamw",
    "weighted_ce_iou_0003_kl10": "rlvr_weighted_ce_iou_0003_kl10_cls_adamw",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair Round 2.3 RLVR result rows from run outputs.")
    parser.add_argument("--run-prefix", default="nni_rlvr_round23")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _params_from_result(name: str, result: dict) -> dict:
    return {
        "name": name,
        "signal": result.get("signal", "none"),
        "reward_lambda": result.get("reward_lambda", 0.0),
        "policy_loss_weight": result.get("policy_loss_weight", 0.0),
        "det_loss_weight": result.get("det_loss_weight", 0.0),
        "baseline_kl_weight": result.get("baseline_kl_weight", 0.0),
        "box_loss_weight": result.get("box_loss_weight", 0.0),
        "unfreeze": result.get("unfreeze", "cls"),
        "optimizer": result.get("optimizer", "adamw"),
        "temperature": result.get("temperature", 1.0),
        "max_candidates": result.get("max_candidates", 40),
        "reward_score_threshold": result.get("reward_score_threshold", 0.2),
        "rollout_source": result.get("rollout_source", "baseline"),
        "policy_objective": result.get("policy_objective", "signed"),
    }


def _load_eval_metrics(runs_root: Path, run_dir_name: str) -> dict:
    metrics = {}
    clean_path = runs_root / f"{run_dir_name}_eval_clean" / "eval_metrics.json"
    edge_path = runs_root / f"{run_dir_name}_eval_object_edge_checkerboard" / "eval_metrics.json"
    if clean_path.exists():
        metrics["clean"] = _load_json(clean_path)
    if edge_path.exists():
        metrics["object_edge_checkerboard"] = _load_json(edge_path)
    return metrics


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    round_root = runs_root / args.run_prefix
    baseline = _load_json(round_root / "baseline_metrics.json")
    output = Path(args.output) if args.output else round_root / "nni_rlvr_results_repaired.jsonl"

    rows = []
    for preset_name, run_dir_name in ROUND23_PRESETS.items():
        run_dir = round_root / run_dir_name
        result = _load_json(run_dir / "rlvr_result.json")
        params = _params_from_result(preset_name, result)
        metrics = _load_eval_metrics(runs_root, run_dir_name)
        status = collect_eval_status(metrics)
        if status == "ok":
            objective = compute_round23_objective(metrics, baseline)
        else:
            objective = {"default": -1.0, "constraint_failed": status}
        checkpoint = run_dir / "checkpoint_best.pth"
        if not checkpoint.exists():
            checkpoint = run_dir / "checkpoint_last.pth"
        row = build_round23_result_row(
            params=params,
            metrics=metrics,
            objective=objective,
            run_name=run_dir_name,
            checkpoint=str(checkpoint) if checkpoint.exists() else "",
            eval_status=status,
        )
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print({"output": str(output), "rows": len(rows), "statuses": {row["name"]: row["eval_status"] for row in rows}})


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit script**

Run:

```powershell
git add spectral_detection_posttrain/analysis/__init__.py spectral_detection_posttrain/analysis/repair_round23_results.py
git commit -m "feat: add Round 2.3 result repair script"
```

Expected: commit succeeds.

---

## Task 3: Complete Missing Evaluations Only

**Files:**
- Runtime only under `runs/`

- [ ] **Step 1: Confirm existing checkpoints**

Run:

```powershell
Get-ChildItem runs\nni_rlvr_round23\rlvr_det_only_cls_cls_adamw
Get-ChildItem runs\nni_rlvr_round23\rlvr_signed_ramp_0003_kl10_cls_adamw
```

Expected:

```text
det_only_cls may be missing checkpoint/result.
signed_ramp should have checkpoint_best.pth or checkpoint_last.pth.
```

- [ ] **Step 2: If `det_only_cls` has no checkpoint, rerun that single preset**

Create a temporary params file `runs/nni_rlvr_round23/det_only_params.json`:

```json
{
  "preset": {
    "name": "det_only_cls",
    "signal": "none",
    "reward_lambda": 0.0,
    "policy_loss_weight": 0.0,
    "det_loss_weight": 1.0,
    "baseline_kl_weight": 0.0,
    "box_loss_weight": 0.0,
    "unfreeze": "cls",
    "optimizer": "adamw",
    "temperature": 1.0,
    "max_candidates": 40,
    "reward_score_threshold": 0.2,
    "rollout_source": "baseline",
    "policy_objective": "signed"
  }
}
```

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.nni_rlvr_trial --config spectral_detection_posttrain/configs/mvp.yaml --run-prefix nni_rlvr_round23 --rlvr-epochs 3 --early-stopping-patience 2 --params-file runs/nni_rlvr_round23/det_only_params.json
```

Expected:

```text
runs/nni_rlvr_round23/rlvr_det_only_cls_cls_adamw/rlvr_result.json exists.
runs/rlvr_det_only_cls_cls_adamw_eval_clean/eval_metrics.json exists.
runs/rlvr_det_only_cls_cls_adamw_eval_object_edge_checkerboard/eval_metrics.json exists.
```

- [ ] **Step 3: Rerun only `signed_ramp` clean eval**

Find checkpoint:

```powershell
Get-ChildItem runs\nni_rlvr_round23\rlvr_signed_ramp_0003_kl10_cls_adamw\checkpoint_best.pth
```

Then run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.eval.eval_detector --config spectral_detection_posttrain/configs/mvp.yaml --checkpoint runs/nni_rlvr_round23/rlvr_signed_ramp_0003_kl10_cls_adamw/checkpoint_best.pth --run-name rlvr_signed_ramp_0003_kl10_cls_adamw_eval_clean --patch-mode none --patch-type random
```

Expected:

```text
runs/rlvr_signed_ramp_0003_kl10_cls_adamw_eval_clean/eval_metrics.json exists.
```

- [ ] **Step 4: Rebuild repaired result file**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.analysis.repair_round23_results --run-prefix nni_rlvr_round23
```

Expected:

```text
runs/nni_rlvr_round23/nni_rlvr_results_repaired.jsonl
6 rows
all statuses are ok
```

---

## Task 4: Write Gap Closure Report

**Files:**
- Create: `docs/rlvr_round24_gap_closure.md`

- [ ] **Step 1: Create report**

Create `docs/rlvr_round24_gap_closure.md`:

```markdown
# Round 2.4 Gap Closure

## Purpose

Round 2.4 closes two Round 2.3 result gaps without launching a new matrix:

1. Complete `det_only_cls`.
2. Complete clean eval for `signed_ramp_0003_kl10`.

## Repaired Result File

`runs/nni_rlvr_round23/nni_rlvr_results_repaired.jsonl`

## Trial Table

| name | status | clean AP50 | clean AP75 | clean precision | clean num pred | edge AP50 | edge AP75 | edge num pred | failed constraint |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| null_no_update | | | | | | | | | |
| det_only_cls | | | | | | | | | |
| signed_iou_0003_kl10 | | | | | | | | | |
| signed_ramp_0003_kl10 | | | | | | | | | |
| signed_shuffled_0003_kl10 | | | | | | | | | |
| weighted_ce_iou_0003_kl10 | | | | | | | | | |

## Interpretation

1. `det_only_cls` tells whether supervised cls continuation remains harmful.
2. `signed_ramp` can now be compared against `signed_iou` and `signed_shuffled`.
3. If `signed_ramp` is not better than shuffled on the same metric, R_amp is not yet proven as a useful verifier signal.
```

- [ ] **Step 2: Commit outputs**

Run:

```powershell
git add runs/nni_rlvr_round23/nni_rlvr_results_repaired.jsonl docs/rlvr_round24_gap_closure.md
git commit -m "docs: close Round 2.3 result gaps"
```

Expected: commit succeeds.

---

## Round 2.4 Success Criteria

Round 2.4 is successful only if:

```text
runs/nni_rlvr_round23/nni_rlvr_results_repaired.jsonl has exactly 6 rows.
All rows have non-empty `name`.
All rows have eval_status == "ok".
det_only_cls has clean_ap50 and edge_ap50.
signed_ramp_0003_kl10 has clean_ap50 and edge_ap50.
```

After this, analysis can answer:

```text
Does det-only cls continuation still hurt?
Does signed_ramp beat signed_iou?
Does signed_ramp beat signed_shuffled?
```

Do not start Round 2.5 until these answers are written down.
