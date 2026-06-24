# Detection RLVR Verifiable Reward Post-Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an RLVR-style post-training framework for object detection so the detector itself learns from verifiable rewards which boxes are trustworthy, where to attend, and how to localize more accurately under patch and checkerboard perturbations.

**Architecture:** The baseline detector remains a pretrained TorchVision Faster R-CNN/Mask R-CNN fine-tuned on Penn-Fudan. RLVR post-training generates detector rollouts, scores each predicted region with an explicit detection verifier made from IoU, class correctness, high-confidence error penalties, and optional spectral evidence such as R_amp. Rewards are converted into weighted ROI classification and box-regression policy losses, so the detector ROI heads are updated directly; spectral quality reranking is kept only as an analysis baseline, not the main method.

**Tech Stack:** Python, PyTorch, TorchVision detection models, Penn-Fudan dataset, NNI GridSearch, pytest, existing `spectral_detection_posttrain` package.

---

## Corrected Project Framing

This project is not a plain ROI reranking project and not only detector score calibration.

The target contribution is:

```text
baseline detector
  -> rollout predictions
  -> verifiable detection reward
  -> GRPO/RLVR-style advantage estimation
  -> weighted ROI classification + localization update
  -> detector produces fewer bad boxes and better boxes by itself
```

`R_amp` is not the final method by itself. It is one verifier component inside a broader detection reward. The key experimental question is whether adding region-frequency evidence to an IoU/class verifier improves RLVR post-training beyond IoU-only and shuffled-frequency controls.

## Files

- Modify: `spectral_detection_posttrain/train/posttrain_rlvr.py`
  Replaces pseudo-GRPO whole-detector-loss weighting with detector rollout generation, verifier scoring, and rewarded ROI policy loss.
- Create: `spectral_detection_posttrain/rlvr/__init__.py`
  Exports RLVR verifier and ROI policy loss helpers.
- Create: `spectral_detection_posttrain/rlvr/detection_verifier.py`
  Computes per-box rewards, image-level penalties, group advantages, and spectral-control variants.
- Create: `spectral_detection_posttrain/rlvr/roi_policy_loss.py`
  Extracts differentiable ROI logits/box deltas for fixed rollout boxes and computes weighted classification/regression losses.
- Modify: `spectral_detection_posttrain/models/build_detector.py`
  Adds a small public helper for selecting trainable detector modules for RLVR.
- Modify: `spectral_detection_posttrain/spectral/rlvr_reward.py`
  Keeps R_amp feature computation but changes normalization to min-max clamp for verifier use.
- Modify: `spectral_detection_posttrain/datasets/patch_transform.py`
  Adds explicit `object_edge`, `object_inside`, and `near_object` patch placements.
- Modify: `spectral_detection_posttrain/eval/eval_detector.py`
  Evaluates raw detector checkpoints without reranking on clean and placement-specific patch modes.
- Modify: `spectral_detection_posttrain/eval/detection_metrics.py`
  Adds AP75, IoU histogram, score-IoU correlation, high-confidence FP count/rate, and ECE reporting.
- Modify: `spectral_detection_posttrain/nni_rlvr_trial.py`
  Runs Round 2 RLVR presets, saves unified baseline metrics, applies hard constraints, and reports objective.
- Create: `spectral_detection_posttrain/nni_configs/rlvr_round2_search_space.json`
  Defines an 8-preset causal matrix for IoU-only, R_amp, shuffled R_amp, policy strength, and temperature variants.
- Create: `spectral_detection_posttrain/nni_configs/rlvr_round2_config.yml`
  Runs the 8 presets with GridSearch.
- Test: `tests/test_rlvr_verifier.py`
  Verifier reward, R_amp control, clamp, and advantage behavior.
- Test: `tests/test_roi_policy_loss.py`
  Differentiable ROI policy loss shape, weighting, no-candidate fallback, and box target encoding.
- Test: `tests/test_detection_patch.py`
  Extends existing patch placement coverage.
- Test: `tests/test_nni_rlvr_round2.py`
  NNI preset expansion, baseline metric schema, hard-constraint objective.

---

## Reward Design

Per prediction box reward:

```python
r_box = (
    w_iou * iou
    + w_cls * class_correct
    + w_amp * s_amp
    + w_struct * s_struct
    - w_hconf_fp * high_conf_fp
)
```

Rules:

- Matched TP: same class and IoU >= 0.5, reward uses IoU, class correctness, and enabled verifier signals.
- Low-IoU matched proposal: gets lower IoU reward and still receives box-regression supervision toward GT when it is the best proposal for a GT.
- Unmatched FP: target label becomes background and penalty weight increases when confidence is high.
- Missed GT: add the GT box and one jittered GT box as recovery proposals so the model learns where to look.
- `signal=none`: `w_amp = 0`, `w_struct = 0`, IoU/class verifier only.
- `signal=ramp`: use min-max normalized R_amp from train-set statistics.
- `signal=shuffled_ramp`: shuffle R_amp only among matched TP boxes in the batch; FP R_amp stays zero.
- `signal=structure`: use edge/low-frequency structure vector after R_amp is already working.

Group advantage:

```python
advantage = (reward - group_mean) / (group_std + 1e-6)
policy_weight = clamp(softplus(advantage / temperature), min=0.05, max=5.0)
```

This replaces the old bug where the same detector loss was multiplied by rollout weights. The new weight is attached to fixed ROI actions and therefore changes the gradient direction.

---

## Task 1: Verifier Unit Tests

**Files:**
- Create: `tests/test_rlvr_verifier.py`
- Create: `spectral_detection_posttrain/rlvr/__init__.py`
- Create: `spectral_detection_posttrain/rlvr/detection_verifier.py`

- [ ] **Step 1: Write verifier tests**

```python
import torch

from spectral_detection_posttrain.rlvr.detection_verifier import (
    DetectionVerifierConfig,
    compute_box_rewards,
    normalize_group_advantages,
    shuffle_tp_ramp,
)


def test_compute_box_rewards_rewards_tp_and_penalizes_high_conf_fp():
    cfg = DetectionVerifierConfig(signal="ramp", w_iou=1.0, w_cls=0.2, w_amp=0.1, w_hconf_fp=0.5)
    ious = torch.tensor([0.8, 0.0])
    class_correct = torch.tensor([1.0, 0.0])
    scores = torch.tensor([0.9, 0.95])
    matched = torch.tensor([True, False])
    s_amp = torch.tensor([0.7, 0.0])

    rewards = compute_box_rewards(cfg, ious, class_correct, scores, matched, s_amp=s_amp)

    assert rewards.shape == (2,)
    assert rewards[0] > 0.9
    assert rewards[1] < 0.0


def test_shuffle_tp_ramp_preserves_fp_zero_and_changes_tp_order():
    values = torch.tensor([0.1, 0.2, 0.3, 0.0])
    matched = torch.tensor([True, True, True, False])
    shuffled = shuffle_tp_ramp(values, matched, seed=7)

    assert torch.equal(shuffled[~matched], torch.tensor([0.0]))
    assert sorted(shuffled[matched].tolist()) == sorted(values[matched].tolist())
    assert not torch.equal(shuffled[matched], values[matched])


def test_normalize_group_advantages_has_nonzero_weights_for_all_boxes():
    rewards = torch.tensor([0.2, 0.5, -0.1])
    weights = normalize_group_advantages(rewards, temperature=1.0)

    assert weights.shape == rewards.shape
    assert torch.all(weights > 0)
    assert weights[1] > weights[0] > weights[2]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_rlvr_verifier.py -v
```

Expected: fails because `spectral_detection_posttrain.rlvr.detection_verifier` does not exist.

- [ ] **Step 3: Implement verifier helpers**

Create `spectral_detection_posttrain/rlvr/__init__.py`:

```python
"""RLVR helpers for detection post-training."""
```

Create `spectral_detection_posttrain/rlvr/detection_verifier.py` with these public functions:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DetectionVerifierConfig:
    signal: str = "none"
    w_iou: float = 1.0
    w_cls: float = 0.2
    w_amp: float = 0.1
    w_struct: float = 0.0
    w_hconf_fp: float = 0.5
    high_conf_threshold: float = 0.8


def compute_box_rewards(
    cfg: DetectionVerifierConfig,
    ious: torch.Tensor,
    class_correct: torch.Tensor,
    scores: torch.Tensor,
    matched: torch.Tensor,
    s_amp: torch.Tensor | None = None,
    s_struct: torch.Tensor | None = None,
) -> torch.Tensor:
    amp = torch.zeros_like(ious) if s_amp is None else s_amp.to(ious.device).float()
    struct = torch.zeros_like(ious) if s_struct is None else s_struct.to(ious.device).float()
    amp_weight = cfg.w_amp if cfg.signal in {"ramp", "shuffled_ramp", "structure"} else 0.0
    struct_weight = cfg.w_struct if cfg.signal == "structure" else 0.0
    high_conf_fp = ((~matched) & (scores >= cfg.high_conf_threshold)).float()
    reward = cfg.w_iou * ious + cfg.w_cls * class_correct + amp_weight * amp + struct_weight * struct
    reward = reward - cfg.w_hconf_fp * high_conf_fp
    return reward


def normalize_group_advantages(rewards: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if rewards.numel() == 0:
        return rewards
    std = rewards.std(unbiased=False)
    advantages = (rewards - rewards.mean()) / (std + 1e-6)
    return F.softplus(advantages / max(float(temperature), 1e-6)).clamp(0.05, 5.0)


def shuffle_tp_ramp(values: torch.Tensor, matched: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    out = values.clone()
    tp_idx = torch.where(matched)[0]
    if tp_idx.numel() < 2:
        return out
    generator = torch.Generator(device=values.device)
    if seed is not None:
        generator.manual_seed(seed)
    perm = tp_idx[torch.randperm(tp_idx.numel(), generator=generator, device=values.device)]
    out[tp_idx] = values[perm]
    out[~matched] = 0.0
    return out
```

- [ ] **Step 4: Run verifier tests and commit**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_rlvr_verifier.py -v
git add spectral_detection_posttrain/rlvr tests/test_rlvr_verifier.py
git commit -m "feat: add detection RLVR verifier"
```

Expected: verifier tests pass and commit succeeds.

---

## Task 2: Differentiable ROI Policy Loss

**Files:**
- Create: `tests/test_roi_policy_loss.py`
- Create: `spectral_detection_posttrain/rlvr/roi_policy_loss.py`

- [ ] **Step 1: Write ROI policy loss tests**

```python
import torch

from spectral_detection_posttrain.rlvr.roi_policy_loss import (
    resize_boxes_to_image,
    weighted_fastrcnn_policy_loss,
)


def test_resize_boxes_to_image_scales_xyxy_coordinates():
    boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    scaled = resize_boxes_to_image(boxes, original_size=(100, 200), new_size=(200, 400))
    assert torch.allclose(scaled, torch.tensor([[20.0, 40.0, 60.0, 80.0]]))


def test_weighted_fastrcnn_policy_loss_handles_no_candidates():
    class_logits = torch.empty((0, 2), requires_grad=True)
    box_regression = torch.empty((0, 8), requires_grad=True)
    labels = torch.empty((0,), dtype=torch.long)
    regression_targets = torch.empty((0, 4))
    weights = torch.empty((0,))

    loss = weighted_fastrcnn_policy_loss(class_logits, box_regression, labels, regression_targets, weights)

    assert loss["loss_roi_policy_cls"].item() == 0.0
    assert loss["loss_roi_policy_box"].item() == 0.0


def test_weighted_fastrcnn_policy_loss_backpropagates():
    class_logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True)
    box_regression = torch.zeros((2, 8), requires_grad=True)
    labels = torch.tensor([1, 0])
    regression_targets = torch.zeros((2, 4))
    weights = torch.tensor([2.0, 1.0])

    loss = weighted_fastrcnn_policy_loss(class_logits, box_regression, labels, regression_targets, weights)
    total = loss["loss_roi_policy_cls"] + loss["loss_roi_policy_box"]
    total.backward()

    assert class_logits.grad is not None
    assert box_regression.grad is not None
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_roi_policy_loss.py -v
```

Expected: fails because `roi_policy_loss.py` does not exist.

- [ ] **Step 3: Implement ROI policy primitives**

Create `spectral_detection_posttrain/rlvr/roi_policy_loss.py` with:

```python
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F


def resize_boxes_to_image(boxes: torch.Tensor, original_size: tuple[int, int], new_size: tuple[int, int]) -> torch.Tensor:
    ratio_h = float(new_size[0]) / float(original_size[0])
    ratio_w = float(new_size[1]) / float(original_size[1])
    ratios = boxes.new_tensor([ratio_w, ratio_h, ratio_w, ratio_h])
    return boxes * ratios


def weighted_fastrcnn_policy_loss(
    class_logits: torch.Tensor,
    box_regression: torch.Tensor,
    labels: torch.Tensor,
    regression_targets: torch.Tensor,
    weights: torch.Tensor,
    box_loss_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    if labels.numel() == 0:
        zero = class_logits.sum() * 0.0 + box_regression.sum() * 0.0
        return {"loss_roi_policy_cls": zero, "loss_roi_policy_box": zero}

    weights = weights.to(class_logits.device).float().clamp_min(0.0)
    normalizer = weights.sum().clamp_min(1.0)
    cls_loss = F.cross_entropy(class_logits, labels.to(class_logits.device), reduction="none")
    cls_loss = (cls_loss * weights).sum() / normalizer

    pos_inds = torch.where(labels > 0)[0]
    if pos_inds.numel() == 0:
        box_loss = box_regression.sum() * 0.0
    else:
        labels_pos = labels[pos_inds].to(box_regression.device)
        box_regression = box_regression.reshape(box_regression.shape[0], -1, 4)
        target = regression_targets[pos_inds].to(box_regression.device)
        raw_box_loss = F.smooth_l1_loss(
            box_regression[pos_inds, labels_pos],
            target,
            beta=1.0 / 9.0,
            reduction="none",
        ).sum(dim=1)
        box_loss = (raw_box_loss * weights[pos_inds].to(box_regression.device)).sum() / normalizer

    return {"loss_roi_policy_cls": cls_loss, "loss_roi_policy_box": box_loss * float(box_loss_weight)}
```

Then add `extract_roi_head_outputs_for_boxes()` in the same file after the tested helpers:

```python
def extract_roi_head_outputs_for_boxes(model, images: list[torch.Tensor], boxes: list[torch.Tensor]):
    original_sizes = [tuple(img.shape[-2:]) for img in images]
    transformed, _ = model.transform(images, None)
    features = model.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    scaled_boxes = [
        resize_boxes_to_image(b.to(transformed.tensors.device), original, new)
        for b, original, new in zip(boxes, original_sizes, transformed.image_sizes)
    ]
    box_features = model.roi_heads.box_roi_pool(features, scaled_boxes, transformed.image_sizes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)
    return class_logits, box_regression, scaled_boxes, transformed.image_sizes
```

- [ ] **Step 4: Run ROI policy tests and commit**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_roi_policy_loss.py -v
git add spectral_detection_posttrain/rlvr/roi_policy_loss.py tests/test_roi_policy_loss.py
git commit -m "feat: add rewarded ROI policy loss"
```

Expected: ROI policy tests pass and commit succeeds.

---

## Task 3: Convert Rollout Predictions Into Trainable ROI Actions

**Files:**
- Modify: `spectral_detection_posttrain/rlvr/detection_verifier.py`
- Test: `tests/test_rlvr_verifier.py`

- [ ] **Step 1: Add action-building tests**

Append to `tests/test_rlvr_verifier.py`:

```python
from spectral_detection_posttrain.rlvr.detection_verifier import build_rewarded_roi_actions


def test_build_rewarded_roi_actions_marks_fp_as_background_and_adds_missed_gt():
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]),
        "labels": torch.tensor([1, 1]),
        "scores": torch.tensor([0.9, 0.95]),
    }
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        "labels": torch.tensor([1, 1]),
    }
    actions = build_rewarded_roi_actions(prediction, target, num_classes=2, max_candidates=8)

    assert actions["boxes"].shape[0] == actions["labels"].shape[0]
    assert 0 in actions["labels"].tolist()
    assert actions["labels"].tolist().count(1) >= 2
    assert torch.all(actions["weights"] > 0)
```

- [ ] **Step 2: Run targeted test and verify failure**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_rlvr_verifier.py::test_build_rewarded_roi_actions_marks_fp_as_background_and_adds_missed_gt -v
```

Expected: fails because `build_rewarded_roi_actions` does not exist.

- [ ] **Step 3: Implement `build_rewarded_roi_actions`**

Add to `spectral_detection_posttrain/rlvr/detection_verifier.py`:

```python
def _box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    return inter / (area1[:, None] + area2 - inter).clamp_min(1e-6)


def build_rewarded_roi_actions(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    num_classes: int,
    max_candidates: int = 80,
    verifier_cfg: DetectionVerifierConfig | None = None,
    s_amp: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    cfg = verifier_cfg or DetectionVerifierConfig()
    boxes = prediction["boxes"][:max_candidates].detach()
    pred_labels = prediction["labels"][:max_candidates].detach()
    scores = prediction["scores"][:max_candidates].detach()
    gt_boxes = target["boxes"].detach()
    gt_labels = target["labels"].detach()

    if boxes.numel() == 0:
        return {
            "boxes": gt_boxes,
            "labels": gt_labels.clamp(max=num_classes - 1),
            "matched_gt_boxes": gt_boxes,
            "weights": torch.ones((gt_boxes.shape[0],), dtype=torch.float32, device=gt_boxes.device),
        }

    ious = _box_iou_matrix(boxes, gt_boxes)
    best_iou, best_gt = ious.max(dim=1)
    matched = best_iou >= 0.5
    labels = torch.zeros_like(pred_labels)
    labels[matched] = gt_labels[best_gt[matched]].clamp(max=num_classes - 1)
    matched_gt_boxes = gt_boxes[best_gt]
    class_correct = (pred_labels == labels).float() * matched.float()
    amp = torch.zeros_like(best_iou) if s_amp is None else s_amp[: boxes.shape[0]].to(best_iou.device)
    rewards = compute_box_rewards(cfg, best_iou, class_correct, scores, matched, s_amp=amp)
    weights = normalize_group_advantages(rewards)

    covered_gt = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool, device=gt_boxes.device)
    if matched.any():
        covered_gt[best_gt[matched]] = True
    missed_gt = torch.where(~covered_gt)[0]
    if missed_gt.numel() > 0:
        recovery_boxes = gt_boxes[missed_gt]
        boxes = torch.cat([boxes, recovery_boxes], dim=0)
        labels = torch.cat([labels, gt_labels[missed_gt].clamp(max=num_classes - 1)], dim=0)
        matched_gt_boxes = torch.cat([matched_gt_boxes, recovery_boxes], dim=0)
        recovery_weights = torch.ones((missed_gt.numel(),), device=weights.device) * weights.mean().clamp_min(1.0)
        weights = torch.cat([weights, recovery_weights], dim=0)

    return {
        "boxes": boxes,
        "labels": labels.long(),
        "matched_gt_boxes": matched_gt_boxes,
        "weights": weights.float(),
    }
```

- [ ] **Step 4: Run verifier tests and commit**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_rlvr_verifier.py -v
git add spectral_detection_posttrain/rlvr/detection_verifier.py tests/test_rlvr_verifier.py
git commit -m "feat: build reward-weighted ROI actions"
```

Expected: all verifier tests pass and commit succeeds.

---

## Task 4: Replace Pseudo-GRPO Training Loop

**Files:**
- Modify: `spectral_detection_posttrain/train/posttrain_rlvr.py`
- Modify: `spectral_detection_posttrain/models/build_detector.py`
- Test: `tests/test_rlvr.py`

- [ ] **Step 1: Add a smoke test for the new training mode**

Append to `tests/test_rlvr.py`:

```python
def test_rlvr_training_args_accept_verifier_policy_mode():
    from spectral_detection_posttrain.train.posttrain_rlvr import parse_args

    args = parse_args([
        "--config", "spectral_detection_posttrain/configs/smoke.yaml",
        "--checkpoint", "runs/baseline/checkpoint.pt",
        "--run-name", "smoke",
        "--signal", "none",
        "--unfreeze", "box",
        "--optimizer", "adamw",
        "--reward-lambda", "0.0",
        "--alpha", "0.1",
        "--beta", "0.05",
        "--epochs", "1",
        "--policy-loss-weight", "0.3",
    ])

    assert args.policy_loss_weight == 0.3
    assert args.signal == "none"
```

- [ ] **Step 2: Run targeted test and verify failure**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_rlvr.py::test_rlvr_training_args_accept_verifier_policy_mode -v
```

Expected: fails because `--policy-loss-weight` is not accepted.

- [ ] **Step 3: Change parser and trainable module selection**

In `spectral_detection_posttrain/train/posttrain_rlvr.py`, expose `parse_args(argv=None)` and add:

```python
parser.add_argument("--policy-loss-weight", type=float, default=0.3)
parser.add_argument("--box-loss-weight", type=float, default=1.0)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--signal", required=True, choices=["none", "ramp", "shuffled_ramp", "structure"])
```

In `spectral_detection_posttrain/models/build_detector.py`, add a helper:

```python
def set_rlvr_trainable_params(model: torch.nn.Module, mode: str = "box") -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if mode in {"box", "roi"}:
        for parameter in model.roi_heads.box_head.parameters():
            parameter.requires_grad = True
        for parameter in model.roi_heads.box_predictor.parameters():
            parameter.requires_grad = True
    elif mode == "cls":
        for parameter in model.roi_heads.box_predictor.cls_score.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"Unknown RLVR trainable mode: {mode}")
```

- [ ] **Step 4: Replace same-loss weighting with ROI policy update**

In `spectral_detection_posttrain/train/posttrain_rlvr.py`, the per-batch logic must follow this sequence:

```python
model.train()
loss_dict = model(images, targets)
loss_det = sum(loss for loss in loss_dict.values())

model.eval()
with torch.no_grad():
    predictions = model(images)

actions = [
    build_rewarded_roi_actions(pred, tgt, num_classes=2, verifier_cfg=verifier_cfg, max_candidates=max_candidates)
    for pred, tgt in zip(predictions, targets)
]
proposal_boxes = [item["boxes"] for item in actions]
proposal_labels = [item["labels"] for item in actions]
matched_gt_boxes = [item["matched_gt_boxes"] for item in actions]
proposal_weights = [item["weights"] for item in actions]

model.train()
class_logits, box_regression, scaled_boxes, transformed_image_sizes = extract_roi_head_outputs_for_boxes(model, images, proposal_boxes)
labels = torch.cat(proposal_labels, dim=0).to(device)
weights = torch.cat(proposal_weights, dim=0).to(device)
scaled_gt_boxes = [
    resize_boxes_to_image(gt.to(device), tuple(img.shape[-2:]), image_size)
    for gt, img, image_size in zip(matched_gt_boxes, images, transformed_image_sizes)
]
regression_targets = model.roi_heads.box_coder.encode(scaled_gt_boxes, scaled_boxes)
policy_losses = weighted_fastrcnn_policy_loss(
    class_logits,
    box_regression,
    labels,
    regression_targets,
    weights,
    box_loss_weight=args.box_loss_weight,
)
loss_policy = policy_losses["loss_roi_policy_cls"] + policy_losses["loss_roi_policy_box"]
loss = loss_det + args.policy_loss_weight * loss_policy
```

Use `transformed_image_sizes` returned by `extract_roi_head_outputs_for_boxes()`; do not encode original-scale GT boxes against transformed-scale proposals.

- [ ] **Step 5: Run tests and a one-batch smoke command**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_rlvr.py tests/test_rlvr_verifier.py tests/test_roi_policy_loss.py -v
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.train.posttrain_rlvr --config spectral_detection_posttrain/configs/smoke.yaml --checkpoint runs/baseline_detector/checkpoint.pt --run-name smoke_rlvr_policy --signal none --unfreeze box --optimizer adamw --reward-lambda 0.0 --alpha 0.1 --beta 0.05 --epochs 1 --policy-loss-weight 0.3
```

Expected: tests pass. The smoke command writes `runs/smoke_rlvr_policy/rlvr_result.json` with `loss_det`, `loss_policy`, `val_ap50`, and `checkpoint`.

- [ ] **Step 6: Commit**

Run:

```powershell
git add spectral_detection_posttrain/train/posttrain_rlvr.py spectral_detection_posttrain/models/build_detector.py tests/test_rlvr.py
git commit -m "fix: train detector with reward-weighted ROI policy loss"
```

Expected: commit succeeds.

---

## Task 5: Patch Placement Evaluation Modes

**Files:**
- Modify: `spectral_detection_posttrain/datasets/patch_transform.py`
- Modify: `spectral_detection_posttrain/eval/eval_detector.py`
- Modify: `spectral_detection_posttrain/spectral/roi_spectral_dataset.py`
- Test: `tests/test_detection_patch.py`

- [ ] **Step 1: Add placement tests**

Append to `tests/test_detection_patch.py`:

```python
import torch

from spectral_detection_posttrain.datasets.patch_transform import add_detection_patch


def test_object_inside_object_edge_and_near_object_patch_modes_change_image():
    image = torch.zeros((3, 64, 64), dtype=torch.float32)
    target = {"boxes": torch.tensor([[16.0, 16.0, 48.0, 48.0]]), "labels": torch.tensor([1])}

    for placement in ["object_inside", "object_edge", "near_object"]:
        patched = add_detection_patch(image, target, placement=placement, patch_type="checkerboard", patch_size=12)
        assert patched.shape == image.shape
        assert torch.sum(torch.abs(patched - image)) > 0
```

- [ ] **Step 2: Run targeted patch test and verify failure**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_detection_patch.py::test_object_inside_object_edge_and_near_object_patch_modes_change_image -v
```

Expected: fails for at least one missing placement mode.

- [ ] **Step 3: Implement placement aliases and near-object placement**

In `spectral_detection_posttrain/datasets/patch_transform.py`, keep existing modes and add:

```python
placement = {
    "object": "object_inside",
    "edge": "object_edge",
}.get(placement, placement)
```

For `object_inside`, place the patch center inside the first GT box.  
For `object_edge`, place it so half overlaps the GT box boundary.  
For `near_object`, place it adjacent to the GT box with no intentional overlap and clamp to image bounds.

- [ ] **Step 4: Extend evaluator choices**

In `eval_detector.py` and `roi_spectral_dataset.py`, change choices to:

```python
choices=["none", "background", "object_inside", "object_edge", "near_object", "random", "object", "edge"]
```

- [ ] **Step 5: Run patch tests and commit**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_detection_patch.py tests/test_patch.py -v
git add spectral_detection_posttrain/datasets/patch_transform.py spectral_detection_posttrain/eval/eval_detector.py spectral_detection_posttrain/spectral/roi_spectral_dataset.py tests/test_detection_patch.py
git commit -m "feat: add object-aware checkerboard patch modes"
```

Expected: tests pass and commit succeeds.

---

## Task 6: Unified Metrics For RLVR, Not Reranking

**Files:**
- Modify: `spectral_detection_posttrain/eval/detection_metrics.py`
- Modify: `spectral_detection_posttrain/eval/eval_detector.py`
- Test: `tests/test_detection_metrics.py`

- [ ] **Step 1: Add metric tests**

Append to `tests/test_detection_metrics.py`:

```python
from spectral_detection_posttrain.eval.detection_metrics import summarize_iou_diagnostics


def test_summarize_iou_diagnostics_reports_ap75_related_fields():
    summary = summarize_iou_diagnostics(
        matched_ious=[0.9, 0.76, 0.4],
        matched_scores=[0.95, 0.7, 0.8],
    )

    assert summary["tp_iou_mean"] > 0.0
    assert summary["tp_iou_ge_075_rate"] == 2 / 3
    assert "score_iou_corr" in summary
```

- [ ] **Step 2: Run metric test and verify failure**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_detection_metrics.py::test_summarize_iou_diagnostics_reports_ap75_related_fields -v
```

Expected: fails because `summarize_iou_diagnostics` does not exist.

- [ ] **Step 3: Implement raw-detector diagnostics**

Add `summarize_iou_diagnostics()` to `detection_metrics.py`:

```python
def summarize_iou_diagnostics(matched_ious: list[float], matched_scores: list[float]) -> dict[str, float]:
    if not matched_ious:
        return {"tp_iou_mean": 0.0, "tp_iou_median": 0.0, "tp_iou_ge_075_rate": 0.0, "score_iou_corr": 0.0}
    ious = torch.tensor(matched_ious, dtype=torch.float32)
    scores = torch.tensor(matched_scores, dtype=torch.float32)
    corr = 0.0
    if ious.numel() > 1 and float(scores.std(unbiased=False)) > 0 and float(ious.std(unbiased=False)) > 0:
        corr = float(torch.corrcoef(torch.stack([scores, ious]))[0, 1].item())
    return {
        "tp_iou_mean": float(ious.mean().item()),
        "tp_iou_median": float(ious.median().item()),
        "tp_iou_ge_075_rate": float((ious >= 0.75).float().mean().item()),
        "score_iou_corr": corr,
    }
```

Update `eval_detector.py` so every eval JSON includes:

```json
{
  "ap50": 0.0,
  "ap75": 0.0,
  "precision": 0.0,
  "recall": 0.0,
  "ece": 0.0,
  "high_conf_fp_count": 0,
  "high_conf_fp_rate": 0.0,
  "tp_iou_mean": 0.0,
  "tp_iou_median": 0.0,
  "tp_iou_ge_075_rate": 0.0,
  "score_iou_corr": 0.0,
  "patch_mode": "clean"
}
```

- [ ] **Step 4: Run metric tests and commit**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_detection_metrics.py tests/test_metrics.py -v
git add spectral_detection_posttrain/eval/detection_metrics.py spectral_detection_posttrain/eval/eval_detector.py tests/test_detection_metrics.py
git commit -m "feat: add localization diagnostics for RLVR eval"
```

Expected: metric tests pass and commit succeeds.

---

## Task 7: NNI Round 2 RLVR Matrix

**Files:**
- Modify: `spectral_detection_posttrain/nni_rlvr_trial.py`
- Create: `spectral_detection_posttrain/nni_configs/rlvr_round2_search_space.json`
- Create: `spectral_detection_posttrain/nni_configs/rlvr_round2_config.yml`
- Test: `tests/test_nni_rlvr_round2.py`

- [ ] **Step 1: Write NNI preset tests**

Create `tests/test_nni_rlvr_round2.py`:

```python
from spectral_detection_posttrain.nni_rlvr_trial import compute_round2_objective, expand_preset


def test_expand_preset_returns_single_trial_dict():
    params = {"preset": {"name": "ramp_mid", "signal": "ramp", "reward_lambda": 0.1, "policy_loss_weight": 0.3}}
    expanded = expand_preset(params)

    assert expanded["name"] == "ramp_mid"
    assert expanded["signal"] == "ramp"
    assert expanded["reward_lambda"] == 0.1


def test_compute_round2_objective_rejects_ap50_collapse():
    baseline = {
        "clean": {"ap50": 0.86, "ap75": 0.62, "recall": 0.88, "ece": 0.06},
        "object_edge_checkerboard": {"ap50": 0.84, "ap75": 0.55, "recall": 0.86, "ece": 0.05},
    }
    metrics = {
        "clean": {"ap50": 0.70, "ap75": 0.50, "recall": 0.80, "ece": 0.04},
        "object_edge_checkerboard": {"ap50": 0.83, "ap75": 0.57, "recall": 0.86, "ece": 0.04},
    }

    assert compute_round2_objective(metrics, baseline)["default"] == -1.0
```

- [ ] **Step 2: Run NNI test and verify failure**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_nni_rlvr_round2.py -v
```

Expected: fails because helper functions are missing.

- [ ] **Step 3: Implement preset expansion and hard constraints**

In `nni_rlvr_trial.py`, add:

```python
def expand_preset(params: dict) -> dict:
    preset = params.get("preset", params)
    if not isinstance(preset, dict):
        raise TypeError("NNI preset must be a dict")
    return dict(preset)


def compute_round2_objective(metrics: dict, baseline: dict) -> dict:
    clean = metrics["clean"]
    edge = metrics["object_edge_checkerboard"]
    base_clean = baseline["clean"]
    base_edge = baseline["object_edge_checkerboard"]
    if clean["ap50"] < base_clean["ap50"] - 0.07:
        return {"default": -1.0, "constraint_failed": "ap50_clean"}
    if edge["ap50"] < base_edge["ap50"] - 0.08:
        return {"default": -1.0, "constraint_failed": "ap50_object_edge_checkerboard"}
    if clean["recall"] < base_clean["recall"] - 0.04:
        return {"default": -1.0, "constraint_failed": "recall_clean"}
    if clean["ap75"] < base_clean["ap75"] - 0.07:
        return {"default": -1.0, "constraint_failed": "ap75_clean"}
    score = clean["ap50"] + edge["ap50"] + 0.5 * edge["ap75"] - 0.2 * clean["ece"] - 0.2 * edge["ece"]
    return {"default": float(score), "constraint_failed": ""}
```

Use nested metric keys consistently; do not mix `baseline["ap50_clean"]` with `baseline["clean"]["ap50"]`.

- [ ] **Step 4: Create Round 2 search space**

Create `spectral_detection_posttrain/nni_configs/rlvr_round2_search_space.json`:

```json
{
  "preset": {
    "_type": "choice",
    "_value": [
      {"name": "iou_only", "signal": "none", "reward_lambda": 0.0, "policy_loss_weight": 0.3, "box_loss_weight": 1.0, "temperature": 1.0},
      {"name": "ramp_low", "signal": "ramp", "reward_lambda": 0.05, "policy_loss_weight": 0.3, "box_loss_weight": 1.0, "temperature": 1.0},
      {"name": "ramp_mid", "signal": "ramp", "reward_lambda": 0.1, "policy_loss_weight": 0.3, "box_loss_weight": 1.0, "temperature": 1.0},
      {"name": "ramp_high", "signal": "ramp", "reward_lambda": 0.2, "policy_loss_weight": 0.3, "box_loss_weight": 1.0, "temperature": 1.0},
      {"name": "shuffled_ramp_mid", "signal": "shuffled_ramp", "reward_lambda": 0.1, "policy_loss_weight": 0.3, "box_loss_weight": 1.0, "temperature": 1.0},
      {"name": "ramp_mid_low_policy", "signal": "ramp", "reward_lambda": 0.1, "policy_loss_weight": 0.1, "box_loss_weight": 1.0, "temperature": 1.0},
      {"name": "ramp_mid_high_policy", "signal": "ramp", "reward_lambda": 0.1, "policy_loss_weight": 0.5, "box_loss_weight": 1.0, "temperature": 1.0},
      {"name": "ramp_mid_cool_temp", "signal": "ramp", "reward_lambda": 0.1, "policy_loss_weight": 0.3, "box_loss_weight": 1.0, "temperature": 0.5}
    ]
  }
}
```

- [ ] **Step 5: Create Round 2 NNI config**

Create `spectral_detection_posttrain/nni_configs/rlvr_round2_config.yml`:

```yaml
experimentName: rlvr_round2_detection_verifier
trialConcurrency: 1
maxTrialNumber: 8
searchSpaceFile: spectral_detection_posttrain/nni_configs/rlvr_round2_search_space.json
trialCommand: >-
  E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.nni_rlvr_trial
  --config spectral_detection_posttrain/configs/mvp.yaml
  --run-prefix nni_rlvr_round2
  --rlvr-epochs 5
  --baseline-epochs 3
tuner:
  name: GridSearch
trainingService:
  platform: local
```

- [ ] **Step 6: Run tests and commit**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_nni_rlvr_round2.py -v
git add spectral_detection_posttrain/nni_rlvr_trial.py spectral_detection_posttrain/nni_configs/rlvr_round2_search_space.json spectral_detection_posttrain/nni_configs/rlvr_round2_config.yml tests/test_nni_rlvr_round2.py
git commit -m "feat: add RLVR Round 2 NNI matrix"
```

Expected: NNI tests pass and commit succeeds.

---

## Task 8: Baseline Metrics Before NNI

**Files:**
- Modify: `spectral_detection_posttrain/nni_rlvr_trial.py`

- [ ] **Step 1: Add baseline generation behavior**

Before any trial starts, `nni_rlvr_trial.py` must generate or load:

```text
runs/nni_rlvr_round2/baseline_metrics.json
```

with this exact schema:

```json
{
  "clean": {"ap50": 0.0, "ap75": 0.0, "precision": 0.0, "recall": 0.0, "ece": 0.0, "high_conf_fp_count": 0},
  "object_edge_checkerboard": {"ap50": 0.0, "ap75": 0.0, "precision": 0.0, "recall": 0.0, "ece": 0.0, "high_conf_fp_count": 0},
  "object_inside_checkerboard": {"ap50": 0.0, "ap75": 0.0, "precision": 0.0, "recall": 0.0, "ece": 0.0, "high_conf_fp_count": 0},
  "near_object_checkerboard": {"ap50": 0.0, "ap75": 0.0, "precision": 0.0, "recall": 0.0, "ece": 0.0, "high_conf_fp_count": 0}
}
```

- [ ] **Step 2: Use raw detector eval commands**

The baseline generator must call `eval_detector.py`, not `eval_rerank.py`, for:

```text
clean
object_edge_checkerboard
object_inside_checkerboard
near_object_checkerboard
```

Patch command mapping:

```text
clean -> --patch-mode none
object_edge_checkerboard -> --patch-mode object_edge --patch-type checkerboard
object_inside_checkerboard -> --patch-mode object_inside --patch-type checkerboard
near_object_checkerboard -> --patch-mode near_object --patch-type checkerboard
```

- [ ] **Step 3: Run a baseline-only dry command**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.nni_rlvr_trial --config spectral_detection_posttrain/configs/mvp.yaml --run-prefix nni_rlvr_round2_dry --baseline-epochs 1 --rlvr-epochs 1 --baseline-only
```

Expected: writes `runs/nni_rlvr_round2_dry/baseline_metrics.json` with the nested schema above.

- [ ] **Step 4: Commit**

Run:

```powershell
git add spectral_detection_posttrain/nni_rlvr_trial.py
git commit -m "fix: save unified raw-detector baseline metrics"
```

Expected: commit succeeds.

---

## Task 9: Round 2 Experiment Execution

**Files:**
- Runtime outputs under `runs/nni_rlvr_round2/`
- Documentation: `docs/rlvr_round2_results.md`

- [ ] **Step 1: Run unit tests before experiment**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_rlvr.py tests/test_rlvr_verifier.py tests/test_roi_policy_loss.py tests/test_detection_patch.py tests/test_detection_metrics.py tests/test_nni_rlvr_round2.py -v
```

Expected: all listed tests pass.

- [ ] **Step 2: Run one 1-epoch functional trial**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.train.posttrain_rlvr --config spectral_detection_posttrain/configs/mvp.yaml --checkpoint runs/baseline_detector/checkpoint.pt --run-name rlvr_round2_one_epoch_iou_only --signal none --unfreeze box --optimizer adamw --reward-lambda 0.0 --alpha 0.1 --beta 0.05 --epochs 1 --policy-loss-weight 0.3 --temperature 1.0
```

Expected: `runs/rlvr_round2_one_epoch_iou_only/rlvr_result.json` exists and contains a checkpoint path.

- [ ] **Step 3: Run NNI matrix**

Run:

```powershell
nnictl create --config spectral_detection_posttrain/nni_configs/rlvr_round2_config.yml
```

Expected: 8 trials run under `runs/nni_rlvr_round2/`, and `runs/nni_rlvr_round2/nni_rlvr_results.jsonl` has 8 JSON lines.

- [ ] **Step 4: Write result report**

Create `docs/rlvr_round2_results.md` with this table:

```markdown
# RLVR Round 2 Results

| preset | signal | reward_lambda | AP50 clean | AP75 clean | AP50 edge | AP75 edge | Recall clean | ECE clean | High-conf FP edge | passed constraints |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
```

Then add three short conclusions:

```markdown
## Conclusions

1. IoU-only RLVR compared with baseline:
2. R_amp compared with IoU-only:
3. R_amp compared with shuffled R_amp:
```

- [ ] **Step 5: Commit results**

Run:

```powershell
git add runs/nni_rlvr_round2/nni_rlvr_results.jsonl runs/nni_rlvr_round2/baseline_metrics.json docs/rlvr_round2_results.md
git commit -m "docs: report RLVR Round 2 matrix results"
```

Expected: commit succeeds.

---

## Success Criteria

The Round 2 plan is considered successful only if all of these hold:

1. The detector checkpoint is evaluated directly by `eval_detector.py`; no reranking output is used as the main result.
2. At least one RLVR trial satisfies:
   - `AP50_clean >= baseline.clean.ap50 - 0.07`
   - `AP50_object_edge_checkerboard >= baseline.object_edge_checkerboard.ap50 - 0.08`
   - `Recall_clean >= baseline.clean.recall - 0.04`
   - `AP75_clean >= baseline.clean.ap75 - 0.07`
3. `ramp_mid` or another R_amp trial beats `iou_only` on at least one localization or robustness metric:
   - AP75 object-edge checkerboard
   - TP IoU mean
   - TP IoU >= 0.75 rate
   - high-confidence FP object-edge checkerboard
4. `ramp_mid` beats `shuffled_ramp_mid` on at least one of the same metrics.
5. The report explicitly says when a gain is detector behavior change versus score calibration.

If these criteria fail, the next plan should reduce the RLVR update strength or move the verifier to a learned reward model, but it should not return to pure reranking as the main contribution.
