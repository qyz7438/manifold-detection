# Round 2.2 Diagnostic KL-Stabilized RLVR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose why Round 2.1 cls-only RLVR still collapses detector precision, then replace weighted CE policy training with a KL-stabilized signed-advantage objective that changes detector behavior without destroying the baseline score distribution.

**Architecture:** Round 2.2 separates three effects that were mixed in Round 2.1: supervised detector continuation loss, reward policy loss, and rollout distribution drift. It adds `det_loss_weight`, frozen-baseline rollout/logit replay, signed GRPO-style ROI policy loss, baseline KL regularization, and strict score-distribution diagnostics before any R_amp causal claim is made.

**Tech Stack:** Python, PyTorch, TorchVision Faster R-CNN, Penn-Fudan, NNI GridSearch, pytest, existing `spectral_detection_posttrain` package.

---

## Why Round 2.2 Exists

Round 2.1 fixed several engineering bugs, but the result still failed:

- baseline clean AP50: about 0.884
- best Round 2.1 clean AP50: about 0.623
- baseline clean precision: about 0.701
- Round 2.1 precision: about 0.21
- baseline clean predictions: about 122
- Round 2.1 predictions: about 300

The failure mode changed from high-confidence FP collapse to low-confidence prediction explosion. This means the cls score distribution was damaged. Because `policy_loss_weight=0.005` still collapsed AP after epoch 1, Round 2.2 must not simply reduce the weight again. It must isolate the source of drift.

Round 2.1 used a positive weighted CE loss:

```text
TP -> person
FP -> background
all samples get positive weights
```

That is not a real GRPO/RLVR objective. A detector action should be the model's sampled/predicted label for a candidate. High reward should increase the probability of that action; low reward should decrease the probability of that same action. This requires signed advantages, not positive CE weights.

Round 2.2 objective:

```text
L = det_loss_weight * L_det
    + policy_loss_weight * L_signed_policy
    + baseline_kl_weight * L_baseline_kl
    + recovery_loss_weight * L_recovery
```

Round 2.2 default:

```text
det_loss_weight = 0.0
policy_loss_weight = 0.001 or 0.003
baseline_kl_weight = 0.5 or 1.0
recovery_loss_weight = 0.0
box_loss_weight = 0.0
unfreeze = cls
rollout_source = baseline
```

Only after this is stable should `rollout_source=current` or bbox updates be reintroduced.

---

## Files

- Modify: `spectral_detection_posttrain/train/posttrain_rlvr.py`
  Add `det_loss_weight`, `baseline_kl_weight`, `recovery_loss_weight`, `rollout_source`, frozen baseline model loading, diagnostics, and signed policy objective integration.
- Modify: `spectral_detection_posttrain/rlvr/detection_verifier.py`
  Return policy action labels, signed rewards, signed advantages, matched masks, and diagnostics instead of only positive CE labels and positive weights.
- Modify: `spectral_detection_posttrain/rlvr/roi_policy_loss.py`
  Add signed GRPO-style ROI policy loss and baseline KL loss while keeping old weighted CE available for comparison.
- Modify: `spectral_detection_posttrain/nni_rlvr_trial.py`
  Pass new parameters, add full result rows when eval fails, and compute Round 2.2 hard constraints including prediction-count drift.
- Create: `nni_configs/rlvr_round22_search_space.json`
  Defines diagnostic and KL-stabilized presets.
- Create: `nni_configs/rlvr_round22_config.yml`
  Runs the Round 2.2 matrix.
- Create: `tests/test_rlvr_policy_objective.py`
  Tests signed policy objective and baseline KL.
- Modify: `tests/test_rlvr_verifier.py`
  Tests returned action labels, signed advantages, and diagnostic fields.
- Create: `tests/test_nni_rlvr_round22.py`
  Tests preset expansion, eval failure handling, and hard constraints.

---

## Task 1: Add Signed Policy Objective Tests

**Files:**
- Create: `tests/test_rlvr_policy_objective.py`
- Modify: `spectral_detection_posttrain/rlvr/roi_policy_loss.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_rlvr_policy_objective.py`:

```python
import torch

from spectral_detection_posttrain.rlvr.roi_policy_loss import (
    baseline_kl_loss,
    signed_roi_policy_loss,
)


def test_signed_policy_loss_rewards_high_advantage_action_probability():
    low_action_logit = torch.tensor([[0.0, 0.0]], requires_grad=True)
    high_action_logit = torch.tensor([[0.0, 2.0]], requires_grad=True)
    action_labels = torch.tensor([1])
    advantages = torch.tensor([1.0])

    low_loss = signed_roi_policy_loss(low_action_logit, action_labels, advantages)
    high_loss = signed_roi_policy_loss(high_action_logit, action_labels, advantages)

    assert high_loss.item() < low_loss.item()


def test_signed_policy_loss_penalizes_low_reward_action_probability():
    low_action_logit = torch.tensor([[0.0, 0.0]], requires_grad=True)
    high_action_logit = torch.tensor([[0.0, 2.0]], requires_grad=True)
    action_labels = torch.tensor([1])
    advantages = torch.tensor([-1.0])

    low_loss = signed_roi_policy_loss(low_action_logit, action_labels, advantages)
    high_loss = signed_roi_policy_loss(high_action_logit, action_labels, advantages)

    assert high_loss.item() > low_loss.item()


def test_baseline_kl_loss_is_zero_for_identical_logits():
    logits = torch.tensor([[1.0, 0.5], [0.2, 2.0]], requires_grad=True)
    loss = baseline_kl_loss(logits, logits.detach())

    assert loss.item() < 1e-7


def test_baseline_kl_loss_positive_for_different_logits():
    current = torch.tensor([[2.0, 0.0]], requires_grad=True)
    baseline = torch.tensor([[0.0, 2.0]])

    loss = baseline_kl_loss(current, baseline)

    assert loss.item() > 0.1
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_rlvr_policy_objective.py -v
```

Expected: fails because `signed_roi_policy_loss` and `baseline_kl_loss` do not exist.

- [ ] **Step 3: Implement signed objective and KL**

Append to `spectral_detection_posttrain/rlvr/roi_policy_loss.py`:

```python
def signed_roi_policy_loss(
    class_logits: torch.Tensor,
    action_labels: torch.Tensor,
    advantages: torch.Tensor,
    max_abs_advantage: float = 3.0,
) -> torch.Tensor:
    if action_labels.numel() == 0:
        return class_logits.sum() * 0.0
    action_labels = action_labels.to(class_logits.device).long()
    advantages = advantages.to(class_logits.device).float().clamp(
        min=-float(max_abs_advantage),
        max=float(max_abs_advantage),
    )
    log_probs = F.log_softmax(class_logits, dim=1)
    selected = log_probs[torch.arange(action_labels.numel(), device=class_logits.device), action_labels]
    return -(advantages.detach() * selected).mean()


def baseline_kl_loss(current_logits: torch.Tensor, baseline_logits: torch.Tensor) -> torch.Tensor:
    if current_logits.numel() == 0:
        return current_logits.sum() * 0.0
    log_current = F.log_softmax(current_logits, dim=1)
    baseline_prob = F.softmax(baseline_logits.to(current_logits.device), dim=1)
    return F.kl_div(log_current, baseline_prob, reduction="batchmean")
```

- [ ] **Step 4: Run tests and commit**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_rlvr_policy_objective.py -v
git add spectral_detection_posttrain/rlvr/roi_policy_loss.py tests/test_rlvr_policy_objective.py
git commit -m "feat: add signed ROI policy objective"
```

Expected: tests pass and commit succeeds.

---

## Task 2: Return Policy Actions And Signed Advantages From Verifier

**Files:**
- Modify: `spectral_detection_posttrain/rlvr/detection_verifier.py`
- Modify: `tests/test_rlvr_verifier.py`

- [ ] **Step 1: Add verifier tests**

Append to `tests/test_rlvr_verifier.py`:

```python
def test_rewarded_actions_include_predicted_policy_labels_and_signed_advantages():
    from spectral_detection_posttrain.rlvr.detection_verifier import build_rewarded_roi_actions

    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]),
        "labels": torch.tensor([1, 1]),
        "scores": torch.tensor([0.9, 0.9]),
    }
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([1]),
    }
    actions = build_rewarded_roi_actions(prediction, target, num_classes=2, max_candidates=8)

    assert "policy_labels" in actions
    assert "advantages" in actions
    assert "matched" in actions
    assert actions["policy_labels"].tolist()[:2] == [1, 1]
    assert actions["advantages"][0] > actions["advantages"][1]
    assert actions["matched"].tolist()[:2] == [True, False]


def test_reward_score_threshold_filters_amp_with_same_mask():
    from spectral_detection_posttrain.rlvr.detection_verifier import build_rewarded_roi_actions

    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        "labels": torch.tensor([1, 1]),
        "scores": torch.tensor([0.95, 0.10]),
    }
    target = {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])}
    s_amp = torch.tensor([0.7, 0.2])

    actions = build_rewarded_roi_actions(
        prediction,
        target,
        num_classes=2,
        reward_score_threshold=0.2,
        s_amp=s_amp,
    )

    assert actions["boxes"].shape[0] >= 1
    assert actions["amp_values"][0].item() == 0.7
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_rlvr_verifier.py::test_rewarded_actions_include_predicted_policy_labels_and_signed_advantages tests/test_rlvr_verifier.py::test_reward_score_threshold_filters_amp_with_same_mask -v
```

Expected: fails because returned fields are missing.

- [ ] **Step 3: Change verifier output**

Modify `build_rewarded_roi_actions()` so it returns both supervised labels and policy labels:

```python
policy_labels = pred_labels.clamp(min=0, max=num_classes - 1)
supervised_labels = torch.zeros_like(pred_labels)
supervised_labels[matched] = gt_labels[best_gt[matched]].clamp(max=num_classes - 1)
```

Use signed advantages:

```python
rewards = compute_box_rewards(cfg, best_iou, class_correct, scores, matched, s_amp=amp)
std = rewards.std(unbiased=False)
advantages = (rewards - rewards.mean()) / (std + 1e-6)
advantages = advantages / max(float(cfg.temperature), 1e-6)
advantages = advantages.clamp(-3.0, 3.0)
```

Do not convert advantages with `softplus` for the signed policy objective. Keep `weights = softplus(...)` only for the legacy CE objective.

Return:

```python
return {
    "boxes": boxes,
    "labels": supervised_labels.long(),
    "policy_labels": policy_labels.long(),
    "matched_gt_boxes": matched_gt_boxes,
    "weights": weights.float(),
    "advantages": advantages.float(),
    "rewards": rewards.float(),
    "matched": matched.bool(),
    "scores": scores.float(),
    "amp_values": amp.float(),
}
```

Recovery boxes are not model actions. In Round 2.2 default they must not enter `policy_labels` or `advantages`. If recovery is needed later, put it behind `recovery_loss_weight > 0`.

- [ ] **Step 4: Run verifier tests and commit**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_rlvr_verifier.py -v
git add spectral_detection_posttrain/rlvr/detection_verifier.py tests/test_rlvr_verifier.py
git commit -m "feat: return signed RLVR actions"
```

Expected: tests pass and commit succeeds.

---

## Task 3: Add Frozen Baseline Logit Replay

**Files:**
- Modify: `spectral_detection_posttrain/train/posttrain_rlvr.py`
- Modify: `spectral_detection_posttrain/rlvr/roi_policy_loss.py`
- Test: `tests/test_rlvr_policy_objective.py`

- [ ] **Step 1: Add CLI arguments**

Update `parse_args()` in `spectral_detection_posttrain/train/posttrain_rlvr.py`:

```python
parser.add_argument("--det-loss-weight", type=float, default=0.0)
parser.add_argument("--baseline-kl-weight", type=float, default=1.0)
parser.add_argument("--recovery-loss-weight", type=float, default=0.0)
parser.add_argument("--policy-objective", default="signed", choices=["signed", "weighted_ce"])
parser.add_argument("--rollout-source", default="baseline", choices=["baseline", "current"])
```

- [ ] **Step 2: Load a frozen baseline model**

After loading the trainable model, create a frozen copy:

```python
baseline_model = build_detector(model_cfg).to(device)
load_checkpoint(baseline_model, args.baseline, device)
baseline_model.eval()
for parameter in baseline_model.parameters():
    parameter.requires_grad = False
```

- [ ] **Step 3: Generate rollouts from the requested source**

Replace the current prediction generation:

```python
rollout_model = baseline_model if args.rollout_source == "baseline" else model
rollout_model.eval()
with torch.no_grad():
    predictions = rollout_model(device_images)
```

Round 2.2 NNI must use `rollout_source=baseline`. This prevents candidate distribution drift while testing the policy objective.

- [ ] **Step 4: Compute baseline logits on the same ROI boxes**

After current model ROI outputs are computed:

```python
class_logits, box_regression, scaled_boxes, transformed_image_sizes = extract_roi_head_outputs_for_boxes(
    model,
    device_images,
    proposal_boxes,
)
with torch.no_grad():
    baseline_logits, _, _, _ = extract_roi_head_outputs_for_boxes(
        baseline_model,
        device_images,
        proposal_boxes,
    )
```

- [ ] **Step 5: Use signed policy plus baseline KL**

Collect labels and advantages:

```python
policy_labels = torch.cat([a["policy_labels"] for a in actions], dim=0).to(device)
advantages = torch.cat([a["advantages"] for a in actions], dim=0).to(device)
```

Then compute:

```python
if args.policy_objective == "signed":
    loss_policy = signed_roi_policy_loss(class_logits, policy_labels, advantages)
else:
    loss_policy = legacy_weighted_ce_loss

loss_kl = baseline_kl_loss(class_logits, baseline_logits)
loss_det = sum(loss_dict.values()) if args.det_loss_weight > 0 else class_logits.sum() * 0.0
loss = (
    args.det_loss_weight * loss_det
    + args.policy_loss_weight * loss_policy
    + args.baseline_kl_weight * loss_kl
)
```

Do not call `model(device_images, device_targets)` when `det_loss_weight == 0`. This avoids BatchNorm/dropout mode churn and avoids supervised continuation drift.

- [ ] **Step 6: Log score-distribution diagnostics**

Each epoch row in `metrics_train.jsonl` must include:

```python
{
    "loss_det": ...,
    "loss_policy": ...,
    "loss_kl": ...,
    "candidate_count": ...,
    "matched_tp_count": ...,
    "fp_count": ...,
    "advantage_mean": ...,
    "advantage_std": ...,
    "policy_label_person_rate": ...,
    "amp_norm_mean": ...,
    "amp_norm_std": ...,
}
```

- [ ] **Step 7: Run a one-epoch smoke command**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.train.posttrain_rlvr --config spectral_detection_posttrain/configs/mvp.yaml --baseline runs/nni_rlvr_round21/baseline/checkpoint_last.pth --run-name smoke_round22_signed_iou --signal none --unfreeze cls --optimizer adamw --reward-lambda 0.0 --policy-loss-weight 0.001 --box-loss-weight 0.0 --det-loss-weight 0.0 --baseline-kl-weight 1.0 --rollout-source baseline --policy-objective signed --epochs 1 --max-candidates 40 --reward-score-threshold 0.2
```

Expected: writes `runs/smoke_round22_signed_iou/rlvr_result.json` and `metrics_train.jsonl` with `loss_kl` and diagnostics.

- [ ] **Step 8: Commit**

Run:

```powershell
git add spectral_detection_posttrain/train/posttrain_rlvr.py spectral_detection_posttrain/rlvr/roi_policy_loss.py
git commit -m "feat: add KL-stabilized signed RLVR training"
```

Expected: commit succeeds.

---

## Task 4: Add Null Diagnostics

**Files:**
- Modify: `spectral_detection_posttrain/nni_rlvr_trial.py`
- Create: `tests/test_nni_rlvr_round22.py`

- [ ] **Step 1: Write tests for Round 2.2 constraints**

Create `tests/test_nni_rlvr_round22.py`:

```python
from spectral_detection_posttrain.nni_rlvr_trial import compute_round22_objective, expand_preset


def test_expand_round22_preset_preserves_new_controls():
    params = {
        "preset": {
            "name": "signed_iou_001",
            "signal": "none",
            "policy_loss_weight": 0.001,
            "det_loss_weight": 0.0,
            "baseline_kl_weight": 1.0,
            "rollout_source": "baseline",
            "policy_objective": "signed",
        }
    }
    expanded = expand_preset(params)

    assert expanded["det_loss_weight"] == 0.0
    assert expanded["baseline_kl_weight"] == 1.0
    assert expanded["rollout_source"] == "baseline"
    assert expanded["policy_objective"] == "signed"


def test_round22_objective_rejects_prediction_explosion():
    baseline = {
        "clean": {"ap50": 0.88, "ap75": 0.64, "recall": 0.90, "precision": 0.70, "high_conf_fp_count": 2, "num_predictions": 122},
        "object_edge_checkerboard": {"ap50": 0.86, "ap75": 0.50, "recall": 0.88, "precision": 0.67, "high_conf_fp_count": 4, "num_predictions": 125},
    }
    metrics = {
        "clean": {"ap50": 0.84, "ap75": 0.60, "recall": 0.87, "precision": 0.30, "high_conf_fp_count": 2, "num_predictions": 300},
        "object_edge_checkerboard": {"ap50": 0.82, "ap75": 0.47, "recall": 0.84, "precision": 0.31, "high_conf_fp_count": 4, "num_predictions": 290},
    }

    result = compute_round22_objective(metrics, baseline)

    assert result["default"] == -1.0
    assert result["constraint_failed"] == "num_predictions_clean"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_nni_rlvr_round22.py -v
```

Expected: fails because `compute_round22_objective` does not exist.

- [ ] **Step 3: Pass new parameters through NNI trial**

In `_run_rlvr()` in `spectral_detection_posttrain/nni_rlvr_trial.py`, read and pass:

```python
det_loss_weight = float(params.get("det_loss_weight", 0.0))
baseline_kl_weight = float(params.get("baseline_kl_weight", 1.0))
recovery_loss_weight = float(params.get("recovery_loss_weight", 0.0))
rollout_source = str(params.get("rollout_source", "baseline"))
policy_objective = str(params.get("policy_objective", "signed"))
```

Append CLI args:

```python
"--det-loss-weight", str(det_loss_weight),
"--baseline-kl-weight", str(baseline_kl_weight),
"--recovery-loss-weight", str(recovery_loss_weight),
"--rollout-source", rollout_source,
"--policy-objective", policy_objective,
```

- [ ] **Step 4: Save full eval result rows even when one eval fails**

In `main()` after eval, do not skip a trial if `object_edge_checkerboard` metrics are missing. Save:

```python
if "clean" not in metrics or "object_edge_checkerboard" not in metrics:
    result = {
        "default": -1.0,
        "constraint_failed": "eval_missing",
        "eval_keys": sorted(metrics.keys()),
    }
```

Still append it to `nni_rlvr_results.jsonl`. This prevents Round 2.1's 5-of-6-row problem.

- [ ] **Step 5: Implement Round 2.2 objective**

Add:

```python
def compute_round22_objective(metrics: dict, baseline: dict) -> dict:
    clean = metrics.get("clean", {})
    edge = metrics.get("object_edge_checkerboard", {})
    if not clean or not edge:
        return {"default": -1.0, "constraint_failed": "eval_missing"}
    base_clean = baseline["clean"]
    base_edge = baseline["object_edge_checkerboard"]

    if clean.get("ap50", 0) < base_clean["ap50"] - 0.05:
        return {"default": -1.0, "constraint_failed": "ap50_clean"}
    if clean.get("recall", 0) < base_clean["recall"] - 0.05:
        return {"default": -1.0, "constraint_failed": "recall_clean"}
    if clean.get("ap75", 0) < base_clean["ap75"] - 0.10:
        return {"default": -1.0, "constraint_failed": "ap75_clean"}
    if clean.get("num_predictions", 10**9) > base_clean["num_predictions"] * 1.30:
        return {"default": -1.0, "constraint_failed": "num_predictions_clean"}
    if clean.get("precision", 0) < base_clean["precision"] - 0.10:
        return {"default": -1.0, "constraint_failed": "precision_clean"}
    if edge.get("ap50", 0) < base_edge["ap50"] - 0.08:
        return {"default": -1.0, "constraint_failed": "ap50_object_edge_checkerboard"}

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

Also update baseline metric saving to include `num_predictions` and `precision`.

- [ ] **Step 6: Run tests and commit**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_nni_rlvr_round22.py -v
git add spectral_detection_posttrain/nni_rlvr_trial.py tests/test_nni_rlvr_round22.py
git commit -m "feat: add Round 2.2 diagnostics objective"
```

Expected: tests pass and commit succeeds.

---

## Task 5: Create Round 2.2 NNI Matrix

**Files:**
- Create: `nni_configs/rlvr_round22_search_space.json`
- Create: `nni_configs/rlvr_round22_config.yml`

- [ ] **Step 1: Create search space**

Create `nni_configs/rlvr_round22_search_space.json`:

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
        "name": "signed_iou_001_kl1",
        "signal": "none",
        "reward_lambda": 0.0,
        "policy_loss_weight": 0.001,
        "det_loss_weight": 0.0,
        "baseline_kl_weight": 1.0,
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
        "name": "signed_ramp_001_kl1",
        "signal": "ramp",
        "reward_lambda": 0.1,
        "policy_loss_weight": 0.001,
        "det_loss_weight": 0.0,
        "baseline_kl_weight": 1.0,
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
        "name": "signed_shuffled_001_kl1",
        "signal": "shuffled_ramp",
        "reward_lambda": 0.1,
        "policy_loss_weight": 0.001,
        "det_loss_weight": 0.0,
        "baseline_kl_weight": 1.0,
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
        "name": "signed_ramp_003_kl1",
        "signal": "ramp",
        "reward_lambda": 0.1,
        "policy_loss_weight": 0.003,
        "det_loss_weight": 0.0,
        "baseline_kl_weight": 1.0,
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
        "name": "signed_ramp_001_kl05",
        "signal": "ramp",
        "reward_lambda": 0.1,
        "policy_loss_weight": 0.001,
        "det_loss_weight": 0.0,
        "baseline_kl_weight": 0.5,
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
        "name": "weighted_ce_iou_001_kl1",
        "signal": "none",
        "reward_lambda": 0.0,
        "policy_loss_weight": 0.001,
        "det_loss_weight": 0.0,
        "baseline_kl_weight": 1.0,
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

- [ ] **Step 2: Create NNI config**

Create `nni_configs/rlvr_round22_config.yml`:

```yaml
experimentName: rlvr_round22_diagnostic_kl
experimentWorkingDirectory: E:/CLIproject/RLimage/nni_experiments
trialCommand: E:/anaconda/01/envs/RLimage/python.exe -m spectral_detection_posttrain.nni_rlvr_trial --config spectral_detection_posttrain/configs/mvp.yaml --run-prefix nni_rlvr_round22 --rlvr-epochs 3 --early-stopping-patience 2
trialCodeDirectory: E:/CLIproject/RLimage
searchSpaceFile: rlvr_round22_search_space.json
trialConcurrency: 1
maxTrialNumber: 8
maxExperimentDuration: 48h
tuner:
  name: GridSearch
trainingService:
  platform: local
```

- [ ] **Step 3: Commit configs**

Run:

```powershell
git add nni_configs/rlvr_round22_search_space.json nni_configs/rlvr_round22_config.yml
git commit -m "feat: add Round 2.2 NNI matrix"
```

Expected: commit succeeds.

---

## Task 6: Run Round 2.2 And Interpret Results

**Files:**
- Runtime: `runs/nni_rlvr_round22/`
- Create: `docs/rlvr_round22_results.md`

- [ ] **Step 1: Run focused tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_rlvr_policy_objective.py tests/test_rlvr_verifier.py tests/test_nni_rlvr_round22.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Run one smoke trial**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.train.posttrain_rlvr --config spectral_detection_posttrain/configs/mvp.yaml --baseline runs/nni_rlvr_round21/baseline/checkpoint_last.pth --run-name smoke_round22_no_update --signal none --unfreeze cls --optimizer adamw --reward-lambda 0.0 --policy-loss-weight 0.0 --box-loss-weight 0.0 --det-loss-weight 0.0 --baseline-kl-weight 0.0 --rollout-source baseline --policy-objective signed --epochs 1 --max-candidates 40 --reward-score-threshold 0.2
```

Expected: resulting eval should match baseline within AP50 0.01. If this fails, stop and debug checkpoint loading or eval path.

- [ ] **Step 3: Run NNI**

Run:

```powershell
nnictl create --config nni_configs/rlvr_round22_config.yml
```

Expected: `runs/nni_rlvr_round22/nni_rlvr_results.jsonl` has exactly 8 lines.

- [ ] **Step 4: Write report**

Create `docs/rlvr_round22_results.md`:

```markdown
# RLVR Round 2.2 Results

## Baseline

| split | AP50 | AP75 | Precision | Recall | Num predictions | High-conf FP | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | | | | | | | |
| object_edge_checkerboard | | | | | | | |

## Trial Results

| preset | AP50 clean | AP75 clean | Precision clean | Recall clean | Num pred clean | AP50 edge | AP75 edge | passed | failed constraint |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|

## Interpretation

1. Null no-update check:
2. Det-only continuation check:
3. Signed policy vs weighted CE:
4. R_amp vs IoU-only:
5. R_amp vs shuffled R_amp:
```

- [ ] **Step 5: Commit results**

Run:

```powershell
git add runs/nni_rlvr_round22/nni_rlvr_results.jsonl runs/nni_rlvr_round22/baseline_metrics.json docs/rlvr_round22_results.md
git commit -m "docs: report Round 2.2 RLVR diagnostics"
```

Expected: commit succeeds.

---

## Round 2.2 Success Criteria

Round 2.2 has three levels of success.

Level 0: pipeline sanity

```text
null_no_update AP50 clean within 0.01 of baseline
nni_rlvr_results.jsonl has exactly 8 rows
no eval_missing rows
```

Level 1: stability

At least one signed-policy trial satisfies:

```text
AP50_clean >= baseline.clean.ap50 - 0.05
Recall_clean >= baseline.clean.recall - 0.05
AP75_clean >= baseline.clean.ap75 - 0.10
Precision_clean >= baseline.clean.precision - 0.10
Num_predictions_clean <= baseline.clean.num_predictions * 1.30
AP50_object_edge >= baseline.edge.ap50 - 0.08
```

Level 2: verifier value

Only after Level 1:

```text
signed_ramp_001_kl1 >= signed_iou_001_kl1 on AP75 edge or ECE edge
signed_ramp_001_kl1 > signed_shuffled_001_kl1 on the same metric
```

If Level 0 fails, debug checkpoint/eval plumbing.

If Level 1 fails but null no-update passes, the RLVR objective is still too destructive. Do not discuss R_amp causality.

If Level 1 passes and Level 2 fails, the RLVR framework is stabilizing but R_amp is not adding reliable verifier value on Penn-Fudan.

If Level 1 and Level 2 pass, Round 2.3 can test `rollout_source=current` and then a separate localization objective on native RPN proposals.
