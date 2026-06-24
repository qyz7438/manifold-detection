# Round 2.3 Freeze-State Readable RLVR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Round 2.2 result readability and training-state contamination so null/no-update, frozen-baseline KL, and signed RLVR can be interpreted before any R_amp causal claim is made.

**Architecture:** Round 2.3 has two gates. Gate A makes every trial auditable: each result row records the preset name, all loss weights, rollout source, objective type, checkpoint path, eval status, clean metrics, edge metrics, and failure reason. Gate B fixes model state: policy/KL trials must keep frozen detector modules in eval mode and avoid BatchNorm running-stat drift; initial current-vs-baseline ROI KL must be near zero before training. Only after Gate A and Gate B pass do we rerun a reduced RLVR matrix.

**Tech Stack:** Python, PyTorch, TorchVision Faster R-CNN, Penn-Fudan, NNI GridSearch, pytest, existing `spectral_detection_posttrain` package.

---

## Why Round 2.3 Exists

Round 2.2 exposed two separate failures:

1. Result readability failed.
   - The JSONL rows did not include `name`, `det_loss_weight`, `baseline_kl_weight`, `policy_objective`, `rollout_source`, `num_predictions`, `precision`, or checkpoint path.
   - `null_no_update` did not produce a run directory, so Level 0 sanity could not be judged.
   - The result file had fewer rows than the planned matrix.

2. Training-state sanity failed.
   - KL-stabilized trials produced about 1150 predictions vs baseline about 119.
   - Precision fell to about 0.058.
   - Initial KL was not near zero, even though current and baseline models loaded the same checkpoint.

The most likely implementation cause is that `requires_grad=False` froze parameters but not module state. `model.train()` can still update BatchNorm running stats in frozen modules. Round 2.3 therefore treats model mode as part of the experiment, not an implementation detail.

---

## Files

- Modify: `spectral_detection_posttrain/nni_rlvr_trial.py`
  Add readable result rows, expected preset audit, no-missing-row behavior, and Round 2.3 objective.
- Modify: `spectral_detection_posttrain/train/posttrain_rlvr.py`
  Add null no-update fast path, freeze-state control, initial KL sanity logging, and no-BN-drift policy/KL loop.
- Modify: `spectral_detection_posttrain/models/build_detector.py`
  Add helpers to set RLVR train/eval state without updating frozen BatchNorm modules.
- Modify: `spectral_detection_posttrain/rlvr/roi_policy_loss.py`
  Add a helper for ROI logit parity diagnostics.
- Create: `tests/test_round23_readable_results.py`
  Tests row flattening, required result columns, missing eval rows, and expected preset audit.
- Create: `tests/test_round23_freeze_state.py`
  Tests frozen BatchNorm modules remain eval and initial ROI KL is near zero for identical checkpoints.
- Create: `nni_configs/rlvr_round23_search_space.json`
  Reduced matrix with null, det-only, signed IoU, signed R_amp, shuffled R_amp, and weighted CE control.
- Create: `nni_configs/rlvr_round23_config.yml`
  Runs the Round 2.3 matrix.

---

## Required Result Row Schema

Every row in `runs/nni_rlvr_round23/nni_rlvr_results.jsonl` must include these fields:

```python
REQUIRED_RESULT_FIELDS = [
    "name",
    "default",
    "constraint_failed",
    "run_name",
    "checkpoint",
    "eval_status",
    "signal",
    "reward_lambda",
    "policy_loss_weight",
    "det_loss_weight",
    "baseline_kl_weight",
    "box_loss_weight",
    "unfreeze",
    "optimizer",
    "temperature",
    "max_candidates",
    "reward_score_threshold",
    "rollout_source",
    "policy_objective",
    "clean_ap50",
    "clean_ap75",
    "clean_precision",
    "clean_recall",
    "clean_num_predictions",
    "clean_high_conf_fp_count",
    "clean_ece",
    "edge_ap50",
    "edge_ap75",
    "edge_precision",
    "edge_recall",
    "edge_num_predictions",
    "edge_high_conf_fp_count",
    "edge_ece",
]
```

No row may rely on implicit preset order. If an eval fails, the row still exists with `eval_status="failed"` and numeric metric fields set to `None`.

---

## Task 1: Readable Result Rows

**Files:**
- Modify: `spectral_detection_posttrain/nni_rlvr_trial.py`
- Create: `tests/test_round23_readable_results.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_round23_readable_results.py`:

```python
from spectral_detection_posttrain.nni_rlvr_trial import (
    REQUIRED_ROUND23_RESULT_FIELDS,
    build_round23_result_row,
    validate_expected_presets,
)


def test_round23_result_row_contains_identity_params_and_metrics():
    params = {
        "name": "signed_iou_0003_kl10",
        "signal": "none",
        "reward_lambda": 0.0,
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
    }
    metrics = {
        "clean": {
            "ap50": 0.85,
            "ap75": 0.60,
            "precision": 0.66,
            "recall": 0.87,
            "num_predictions": 120,
            "high_conf_fp_count": 2,
            "ece": 0.06,
        },
        "object_edge_checkerboard": {
            "ap50": 0.84,
            "ap75": 0.55,
            "precision": 0.63,
            "recall": 0.86,
            "num_predictions": 130,
            "high_conf_fp_count": 4,
            "ece": 0.07,
        },
    }
    row = build_round23_result_row(
        params=params,
        metrics=metrics,
        objective={"default": 2.1, "constraint_failed": ""},
        run_name="nni_rlvr_round23/rlvr_signed_iou_0003_kl10_cls_adamw",
        checkpoint="runs/x/checkpoint_best.pth",
        eval_status="ok",
    )

    for field in REQUIRED_ROUND23_RESULT_FIELDS:
        assert field in row
    assert row["name"] == "signed_iou_0003_kl10"
    assert row["clean_num_predictions"] == 120
    assert row["edge_ap50"] == 0.84


def test_round23_result_row_survives_missing_eval():
    params = {"name": "broken", "signal": "none"}
    row = build_round23_result_row(
        params=params,
        metrics={},
        objective={"default": -1.0, "constraint_failed": "eval_missing"},
        run_name="runs/broken",
        checkpoint="",
        eval_status="failed",
    )

    assert row["name"] == "broken"
    assert row["eval_status"] == "failed"
    assert row["clean_ap50"] is None


def test_validate_expected_presets_detects_missing_name():
    expected = ["null_no_update", "det_only_cls", "signed_iou"]
    rows = [{"name": "null_no_update"}, {"name": "signed_iou"}]

    result = validate_expected_presets(expected, rows)

    assert result["missing"] == ["det_only_cls"]
    assert result["complete"] is False
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round23_readable_results.py -v
```

Expected: fails because the helpers do not exist.

- [ ] **Step 3: Implement readable row helpers**

Add to `spectral_detection_posttrain/nni_rlvr_trial.py`:

```python
REQUIRED_ROUND23_RESULT_FIELDS = [
    "name", "default", "constraint_failed", "run_name", "checkpoint", "eval_status",
    "signal", "reward_lambda", "policy_loss_weight", "det_loss_weight",
    "baseline_kl_weight", "box_loss_weight", "unfreeze", "optimizer",
    "temperature", "max_candidates", "reward_score_threshold", "rollout_source",
    "policy_objective",
    "clean_ap50", "clean_ap75", "clean_precision", "clean_recall",
    "clean_num_predictions", "clean_high_conf_fp_count", "clean_ece",
    "edge_ap50", "edge_ap75", "edge_precision", "edge_recall",
    "edge_num_predictions", "edge_high_conf_fp_count", "edge_ece",
]


def _metric(metrics: dict, scene: str, key: str):
    return metrics.get(scene, {}).get(key)


def build_round23_result_row(
    params: dict,
    metrics: dict,
    objective: dict,
    run_name: str,
    checkpoint: str,
    eval_status: str,
) -> dict:
    row = {
        "name": params.get("name", ""),
        "default": objective.get("default", -1.0),
        "constraint_failed": objective.get("constraint_failed", ""),
        "run_name": run_name,
        "checkpoint": checkpoint,
        "eval_status": eval_status,
        "signal": params.get("signal", ""),
        "reward_lambda": float(params.get("reward_lambda", 0.0)),
        "policy_loss_weight": float(params.get("policy_loss_weight", 0.0)),
        "det_loss_weight": float(params.get("det_loss_weight", 0.0)),
        "baseline_kl_weight": float(params.get("baseline_kl_weight", 0.0)),
        "box_loss_weight": float(params.get("box_loss_weight", 0.0)),
        "unfreeze": params.get("unfreeze", ""),
        "optimizer": params.get("optimizer", ""),
        "temperature": float(params.get("temperature", 1.0)),
        "max_candidates": int(params.get("max_candidates", 0)),
        "reward_score_threshold": float(params.get("reward_score_threshold", 0.0)),
        "rollout_source": params.get("rollout_source", ""),
        "policy_objective": params.get("policy_objective", ""),
        "clean_ap50": _metric(metrics, "clean", "ap50"),
        "clean_ap75": _metric(metrics, "clean", "ap75"),
        "clean_precision": _metric(metrics, "clean", "precision"),
        "clean_recall": _metric(metrics, "clean", "recall"),
        "clean_num_predictions": _metric(metrics, "clean", "num_predictions"),
        "clean_high_conf_fp_count": _metric(metrics, "clean", "high_conf_fp_count"),
        "clean_ece": _metric(metrics, "clean", "ece"),
        "edge_ap50": _metric(metrics, "object_edge_checkerboard", "ap50"),
        "edge_ap75": _metric(metrics, "object_edge_checkerboard", "ap75"),
        "edge_precision": _metric(metrics, "object_edge_checkerboard", "precision"),
        "edge_recall": _metric(metrics, "object_edge_checkerboard", "recall"),
        "edge_num_predictions": _metric(metrics, "object_edge_checkerboard", "num_predictions"),
        "edge_high_conf_fp_count": _metric(metrics, "object_edge_checkerboard", "high_conf_fp_count"),
        "edge_ece": _metric(metrics, "object_edge_checkerboard", "ece"),
    }
    for field in REQUIRED_ROUND23_RESULT_FIELDS:
        row.setdefault(field, None)
    return row


def validate_expected_presets(expected_names: list[str], rows: list[dict]) -> dict:
    seen = {row.get("name") for row in rows}
    missing = [name for name in expected_names if name not in seen]
    extra = sorted(name for name in seen if name and name not in set(expected_names))
    return {"complete": not missing and not extra, "missing": missing, "extra": extra}
```

- [ ] **Step 4: Use readable rows in `main()`**

In `main()`, after eval and objective calculation, build the row with `build_round23_result_row()` and append that row instead of the old sparse row.

When eval raises, set:

```python
metrics = {}
objective = {"default": -1.0, "constraint_failed": "eval_missing"}
eval_status = "failed"
```

When eval succeeds for both scenes:

```python
eval_status = "ok"
```

- [ ] **Step 5: Run tests and commit**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round23_readable_results.py -v
git add spectral_detection_posttrain/nni_rlvr_trial.py tests/test_round23_readable_results.py
git commit -m "fix: make RLVR NNI results auditable"
```

Expected: tests pass and commit succeeds.

---

## Task 2: Freeze Detector State For Policy/KL Trials

**Files:**
- Modify: `spectral_detection_posttrain/models/build_detector.py`
- Modify: `spectral_detection_posttrain/train/posttrain_rlvr.py`
- Create: `tests/test_round23_freeze_state.py`

- [ ] **Step 1: Write failing freeze-state tests**

Create `tests/test_round23_freeze_state.py`:

```python
import torch

from spectral_detection_posttrain.models.build_detector import (
    set_detector_eval_except_trainable,
)


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
        self.head = torch.nn.Linear(4, 2)

    def forward(self, x):
        return x


def test_set_detector_eval_except_trainable_keeps_batchnorm_eval():
    model = TinyModule()
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True

    set_detector_eval_except_trainable(model)

    assert model.training is False
    assert model.bn.training is False
    assert model.head.training is True
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round23_freeze_state.py -v
```

Expected: fails because `set_detector_eval_except_trainable` does not exist.

- [ ] **Step 3: Implement state helper**

Add to `spectral_detection_posttrain/models/build_detector.py`:

```python
def set_detector_eval_except_trainable(model: torch.nn.Module) -> None:
    """Keep frozen detector state stable while allowing trainable leaf modules to get gradients."""
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    for module in model.modules():
        has_trainable_param = any(parameter.requires_grad for parameter in module.parameters(recurse=False))
        if has_trainable_param and not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.train()
```

- [ ] **Step 4: Use helper in policy/KL training loop**

In `spectral_detection_posttrain/train/posttrain_rlvr.py`:

Import:

```python
from spectral_detection_posttrain.models.build_detector import set_detector_eval_except_trainable
```

After `set_rlvr_trainable_params(model, mode=args.unfreeze)`:

```python
set_detector_eval_except_trainable(model)
```

Inside the epoch loop:

```python
if args.det_loss_weight > 0:
    model.train()
    loss_dict = model(device_images, device_targets)
    loss_det = sum(v for v in loss_dict.values())
    set_detector_eval_except_trainable(model)
else:
    set_detector_eval_except_trainable(model)
    loss_det = torch.tensor(0.0, device=device)
```

Before `extract_roi_head_outputs_for_boxes(model, ...)`, call:

```python
set_detector_eval_except_trainable(model)
```

Never call plain `model.train()` for policy/KL-only trials.

- [ ] **Step 5: Run tests and commit**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round23_freeze_state.py -v
git add spectral_detection_posttrain/models/build_detector.py spectral_detection_posttrain/train/posttrain_rlvr.py tests/test_round23_freeze_state.py
git commit -m "fix: prevent frozen detector state drift"
```

Expected: tests pass and commit succeeds.

---

## Task 3: Initial KL And No-Update Sanity

**Files:**
- Modify: `spectral_detection_posttrain/train/posttrain_rlvr.py`
- Modify: `spectral_detection_posttrain/rlvr/roi_policy_loss.py`
- Modify: `tests/test_round23_freeze_state.py`

- [ ] **Step 1: Add ROI KL diagnostic helper**

Append to `spectral_detection_posttrain/rlvr/roi_policy_loss.py`:

```python
@torch.no_grad()
def roi_logit_max_abs_diff(current_logits: torch.Tensor, baseline_logits: torch.Tensor) -> float:
    if current_logits.numel() == 0:
        return 0.0
    return float((current_logits - baseline_logits.to(current_logits.device)).abs().max().item())
```

- [ ] **Step 2: Log initial KL before optimizer step**

In `posttrain_rlvr.py`, before the first training batch update, compute one batch of rollout boxes from `baseline_model`, then:

```python
set_detector_eval_except_trainable(model)
baseline_model.eval()
with torch.no_grad():
    current_logits, _, _, _ = extract_roi_head_outputs_for_boxes(model, device_images, proposal_boxes)
    baseline_logits, _, _, _ = extract_roi_head_outputs_for_boxes(baseline_model, device_images, proposal_boxes)
initial_kl = float(baseline_kl_loss(current_logits, baseline_logits).item())
initial_logit_max_abs_diff = roi_logit_max_abs_diff(current_logits, baseline_logits)
```

Write:

```python
save_json(
    {
        "initial_roi_kl": initial_kl,
        "initial_logit_max_abs_diff": initial_logit_max_abs_diff,
        "rollout_source": args.rollout_source,
    },
    run_dir / "initial_sanity.json",
)
```

- [ ] **Step 3: Add no-update fast path**

If all update weights are zero:

```python
is_no_update = (
    args.det_loss_weight == 0
    and args.policy_loss_weight == 0
    and args.baseline_kl_weight == 0
    and args.box_loss_weight == 0
)
```

Then skip optimizer creation and training updates. Save the loaded baseline as both checkpoint files:

```python
save_checkpoint(model, run_dir / "checkpoint_best.pth", {"epoch": 0, "run_name": args.run_name, "val_ap50": None})
save_checkpoint(model, run_dir / "checkpoint_last.pth", {"epoch": 0, "run_name": args.run_name, "val_ap50": None})
save_json(
    {
        "best_epoch": 0,
        "best_val_ap50": None,
        "total_epochs": 0,
        "no_update": True,
        "initial_roi_kl": initial_kl,
        "initial_logit_max_abs_diff": initial_logit_max_abs_diff,
        "signal": args.signal,
        "unfreeze": args.unfreeze,
        "optimizer": args.optimizer,
        "reward_lambda": args.reward_lambda,
        "policy_loss_weight": args.policy_loss_weight,
        "baseline_kl_weight": args.baseline_kl_weight,
        "det_loss_weight": args.det_loss_weight,
    },
    run_dir / "rlvr_result.json",
)
return
```

- [ ] **Step 4: Add sanity thresholds**

If `args.rollout_source == "baseline"` and `args.det_loss_weight == 0`, require:

```python
if initial_kl > 1e-5 or initial_logit_max_abs_diff > 1e-4:
    raise RuntimeError(
        f"Initial baseline/current ROI logits differ: kl={initial_kl}, "
        f"max_abs={initial_logit_max_abs_diff}"
    )
```

This makes state contamination fail fast instead of producing another useless NNI matrix.

- [ ] **Step 5: Run smoke no-update command**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.train.posttrain_rlvr --config spectral_detection_posttrain/configs/mvp.yaml --baseline runs/nni_rlvr_round22/baseline/checkpoint_last.pth --run-name smoke_round23_null_no_update --signal none --unfreeze cls --optimizer adamw --reward-lambda 0.0 --policy-loss-weight 0.0 --box-loss-weight 0.0 --det-loss-weight 0.0 --baseline-kl-weight 0.0 --rollout-source baseline --policy-objective signed --epochs 3 --max-candidates 40 --reward-score-threshold 0.2
```

Expected:

```text
runs/smoke_round23_null_no_update/rlvr_result.json
runs/smoke_round23_null_no_update/initial_sanity.json
```

`initial_roi_kl` must be below `1e-5`.

- [ ] **Step 6: Commit**

Run:

```powershell
git add spectral_detection_posttrain/train/posttrain_rlvr.py spectral_detection_posttrain/rlvr/roi_policy_loss.py tests/test_round23_freeze_state.py
git commit -m "fix: add RLVR initial sanity and no-update path"
```

Expected: commit succeeds.

---

## Task 4: Round 2.3 Objective And Matrix

**Files:**
- Modify: `spectral_detection_posttrain/nni_rlvr_trial.py`
- Create: `nni_configs/rlvr_round23_search_space.json`
- Create: `nni_configs/rlvr_round23_config.yml`
- Modify: `tests/test_round23_readable_results.py`

- [ ] **Step 1: Add Round 2.3 objective test**

Append to `tests/test_round23_readable_results.py`:

```python
from spectral_detection_posttrain.nni_rlvr_trial import compute_round23_objective


def test_round23_objective_rejects_prediction_explosion():
    baseline = {
        "clean": {"ap50": 0.86, "ap75": 0.63, "precision": 0.67, "recall": 0.88, "num_predictions": 119, "ece": 0.06},
        "object_edge_checkerboard": {"ap50": 0.87, "ap75": 0.58, "precision": 0.66, "recall": 0.89, "num_predictions": 122, "ece": 0.06},
    }
    metrics = {
        "clean": {"ap50": 0.84, "ap75": 0.60, "precision": 0.20, "recall": 0.86, "num_predictions": 500, "ece": 0.08},
        "object_edge_checkerboard": {"ap50": 0.84, "ap75": 0.55, "precision": 0.20, "recall": 0.86, "num_predictions": 500, "ece": 0.08},
    }

    result = compute_round23_objective(metrics, baseline)

    assert result["default"] == -1.0
    assert result["constraint_failed"] == "clean_num_predictions"
```

- [ ] **Step 2: Implement `compute_round23_objective`**

Add to `nni_rlvr_trial.py`:

```python
def compute_round23_objective(metrics: dict, baseline: dict) -> dict:
    clean = metrics.get("clean", {})
    edge = metrics.get("object_edge_checkerboard", {})
    if not clean or not edge:
        return {"default": -1.0, "constraint_failed": "eval_missing"}
    base_clean = baseline["clean"]
    base_edge = baseline["object_edge_checkerboard"]

    checks = [
        ("clean_ap50", clean.get("ap50", 0) >= base_clean["ap50"] - 0.03),
        ("clean_ap75", clean.get("ap75", 0) >= base_clean["ap75"] - 0.08),
        ("clean_recall", clean.get("recall", 0) >= base_clean["recall"] - 0.04),
        ("clean_precision", clean.get("precision", 0) >= base_clean["precision"] - 0.08),
        ("clean_num_predictions", clean.get("num_predictions", 10**9) <= base_clean["num_predictions"] * 1.20),
        ("edge_ap50", edge.get("ap50", 0) >= base_edge["ap50"] - 0.06),
        ("edge_num_predictions", edge.get("num_predictions", 10**9) <= base_edge["num_predictions"] * 1.25),
    ]
    for name, ok in checks:
        if not ok:
            return {"default": -1.0, "constraint_failed": name}

    score = (
        clean["ap50"]
        + 0.5 * clean["ap75"]
        + edge["ap50"]
        + 0.5 * edge["ap75"]
        - 0.2 * clean.get("ece", 0.0)
        - 0.2 * edge.get("ece", 0.0)
    )
    return {"default": float(score), "constraint_failed": ""}
```

Use `compute_round23_objective` in `main()` when `args.run_prefix` contains `round23`; otherwise keep older objectives for older configs.

- [ ] **Step 3: Create Round 2.3 search space**

Create `nni_configs/rlvr_round23_search_space.json`:

```json
{
  "preset": {
    "_type": "choice",
    "_value": [
      {
        "name": "null_no_update",
        "signal": "none",
        "reward_lambda": 0.0,
        "policy_loss_weight": 0.0,
        "det_loss_weight": 0.0,
        "baseline_kl_weight": 0.0,
        "box_loss_weight": 0.0,
        "unfreeze": "cls",
        "optimizer": "adamw",
        "temperature": 1.0,
        "max_candidates": 40,
        "reward_score_threshold": 0.2,
        "rollout_source": "baseline",
        "policy_objective": "signed"
      },
      {
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
      },
      {
        "name": "signed_iou_0003_kl10",
        "signal": "none",
        "reward_lambda": 0.0,
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
        "policy_objective": "signed"
      },
      {
        "name": "signed_ramp_0003_kl10",
        "signal": "ramp",
        "reward_lambda": 0.1,
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
        "policy_objective": "signed"
      },
      {
        "name": "signed_shuffled_0003_kl10",
        "signal": "shuffled_ramp",
        "reward_lambda": 0.1,
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
        "policy_objective": "signed"
      },
      {
        "name": "weighted_ce_iou_0003_kl10",
        "signal": "none",
        "reward_lambda": 0.0,
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
        "policy_objective": "weighted_ce"
      }
    ]
  }
}
```

Round 2.3 intentionally uses 6 trials. Do not add more until `null_no_update` passes.

- [ ] **Step 4: Create NNI config**

Create `nni_configs/rlvr_round23_config.yml`:

```yaml
experimentName: rlvr_round23_freeze_state_readable
experimentWorkingDirectory: E:/CLIproject/RLimage/nni_experiments
trialCommand: E:/anaconda/01/envs/RLimage/python.exe -m spectral_detection_posttrain.nni_rlvr_trial --config spectral_detection_posttrain/configs/mvp.yaml --run-prefix nni_rlvr_round23 --rlvr-epochs 3 --early-stopping-patience 2
trialCodeDirectory: E:/CLIproject/RLimage
searchSpaceFile: rlvr_round23_search_space.json
trialConcurrency: 1
maxTrialNumber: 6
maxExperimentDuration: 48h
tuner:
  name: GridSearch
trainingService:
  platform: local
```

- [ ] **Step 5: Run tests and commit**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round23_readable_results.py -v
git add spectral_detection_posttrain/nni_rlvr_trial.py tests/test_round23_readable_results.py nni_configs/rlvr_round23_search_space.json nni_configs/rlvr_round23_config.yml
git commit -m "feat: add Round 2.3 readable matrix"
```

Expected: tests pass and commit succeeds.

---

## Task 5: Round 2.3 Execution And Report

**Files:**
- Runtime: `runs/nni_rlvr_round23/`
- Create: `docs/rlvr_round23_results.md`

- [ ] **Step 1: Run focused tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round23_readable_results.py tests/test_round23_freeze_state.py tests/test_rlvr_policy_objective.py tests/test_rlvr_verifier.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Run null smoke first**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.train.posttrain_rlvr --config spectral_detection_posttrain/configs/mvp.yaml --baseline runs/nni_rlvr_round22/baseline/checkpoint_last.pth --run-name smoke_round23_null_no_update --signal none --unfreeze cls --optimizer adamw --reward-lambda 0.0 --policy-loss-weight 0.0 --box-loss-weight 0.0 --det-loss-weight 0.0 --baseline-kl-weight 0.0 --rollout-source baseline --policy-objective signed --epochs 3 --max-candidates 40 --reward-score-threshold 0.2
```

Then evaluate:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.eval.eval_detector --config spectral_detection_posttrain/configs/mvp.yaml --checkpoint runs/smoke_round23_null_no_update/checkpoint_best.pth --run-name smoke_round23_null_eval_clean --patch-mode none
```

Expected:

```text
abs(smoke AP50 - baseline AP50) <= 0.01
initial_roi_kl <= 1e-5
```

If this fails, stop. Do not run NNI.

- [ ] **Step 3: Run NNI**

Run:

```powershell
nnictl create --config nni_configs/rlvr_round23_config.yml
```

Expected:

```text
runs/nni_rlvr_round23/nni_rlvr_results.jsonl has exactly 6 rows.
Every row has all REQUIRED_ROUND23_RESULT_FIELDS.
```

- [ ] **Step 4: Write report**

Create `docs/rlvr_round23_results.md`:

```markdown
# RLVR Round 2.3 Results

## Baseline

| split | AP50 | AP75 | Precision | Recall | Num predictions | ECE |
|---|---:|---:|---:|---:|---:|---:|
| clean | | | | | | |
| object_edge_checkerboard | | | | | | |

## Trial Results

| name | det | KL | policy | objective | signal | clean AP50 | clean AP75 | clean precision | clean recall | clean num pred | edge AP50 | edge AP75 | edge num pred | failed constraint |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|

## Interpretation

1. Null no-update sanity:
2. Det-only continuation:
3. Signed vs weighted CE:
4. R_amp vs IoU-only:
5. R_amp vs shuffled R_amp:
```

- [ ] **Step 5: Commit results**

Run:

```powershell
git add runs/nni_rlvr_round23/nni_rlvr_results.jsonl runs/nni_rlvr_round23/baseline_metrics.json docs/rlvr_round23_results.md
git commit -m "docs: report Round 2.3 freeze-state RLVR results"
```

Expected: commit succeeds.

---

## Round 2.3 Success Criteria

Level 0: readability and plumbing

```text
nni_rlvr_results.jsonl has exactly 6 rows
all rows include name, loss weights, objective, rollout_source, clean metrics, edge metrics
null_no_update exists and is evaluated
```

Level 1: no-update sanity

```text
null_no_update clean AP50 within 0.01 of baseline clean AP50
null_no_update clean num_predictions within 10% of baseline
initial_roi_kl <= 1e-5
```

Level 2: state-safe RLVR

At least one signed-policy trial satisfies:

```text
clean_ap50 >= baseline.clean.ap50 - 0.03
clean_ap75 >= baseline.clean.ap75 - 0.08
clean_precision >= baseline.clean.precision - 0.08
clean_num_predictions <= baseline.clean.num_predictions * 1.20
edge_ap50 >= baseline.edge.ap50 - 0.06
edge_num_predictions <= baseline.edge.num_predictions * 1.25
```

Level 3: verifier value

Only after Level 2:

```text
signed_ramp_0003_kl10 >= signed_iou_0003_kl10 on edge AP75 or edge ECE
signed_ramp_0003_kl10 > signed_shuffled_0003_kl10 on the same metric
```

If Level 0 or Level 1 fails, do not discuss R_amp. Fix pipeline state and result logging first.

If Level 2 fails but Level 1 passes, the RLVR objective remains too destructive; reduce policy weight further or switch to logit-space reward shaping.

If Level 2 and Level 3 pass, Round 2.4 can test `rollout_source=current` and then native RPN proposal localization.
