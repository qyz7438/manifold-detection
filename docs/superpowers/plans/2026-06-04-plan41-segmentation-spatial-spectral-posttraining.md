# Plan 4.1 Segmentation Spatial-Spectral Post-Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the real spatial-domain plus frequency-domain post-training path in semantic segmentation, where mask-level supervision gives magnitude and phase/structure verifiers a dense, stable target.

**Architecture:** Build on Plan 4.0's `spectral_segmentation_posttrain` package. Train or load a binary person segmentation baseline, then run a second-stage post-training checkpoint update with the encoder mostly frozen and the decoder/head plus optional AFM verifier modules trainable. Spatial rewards optimize mask IoU, Dice, boundary F1, and connected structure; spectral rewards optimize foreground-region amplitude consistency and low-frequency phase/edge structure consistency with shuffled controls.

**Tech Stack:** Python, PyTorch, TorchVision FCN/DeepLab segmentation models, Penn-Fudan masks, pytest, existing `spectral_segmentation_posttrain` package from Plan 4.0.

---

## Why Plan 4.1 Exists

Detection-side experiments showed two negative facts:

```text
handwritten box-level spectral verifier: no robust causal signal on Penn-Fudan detection
in-network AFM on detector ROI features: FFT gates did not activate; gains came from head adaptation
```

Segmentation is the correct place to continue the user's real objective:

```text
spatial domain: dense mask shape, boundary, connected components
frequency domain: foreground amplitude profile, low-frequency phase/edge structure
post-training: update a pretrained/baseline model from a checkpoint using verifiable rewards
```

Plan 4.1 is not a generic supervised segmentation plan. It is the mainline for true verifiable-reward post-training with spatial and spectral evidence.

---

## Experiment Count

Small Plan 4.1 MVP: **7 groups**.

| ID | Group | Second-Stage Signal |
|---|---|---|
| S1 | `seg41_s1_baseline_eval` | evaluate baseline checkpoint |
| S2 | `seg41_s2_posttrain_supervised_only` | supervised CE/Dice post-training only |
| S3 | `seg41_s3_posttrain_spatial` | supervised + spatial verifier |
| S4 | `seg41_s4_posttrain_amp` | supervised + amplitude verifier |
| S5 | `seg41_s5_posttrain_structure` | supervised + boundary/phase-structure verifier |
| S6 | `seg41_s6_posttrain_spatial_amp_structure` | supervised + spatial + amplitude + structure |
| S7 | `seg41_s7_posttrain_shuffled_amp_structure` | same as S6, but shuffled spectral controls |

Promotion rule:

```text
S6 must beat S3, and S7 must not match S6, before claiming spectral evidence has causal value.
```

---

## File Map

- Create: `spectral_segmentation_posttrain/rlvr/spatial_rewards.py`
  Dense mask IoU, Dice, boundary, connected-component rewards.

- Create: `spectral_segmentation_posttrain/spectral/mask_frequency.py`
  Foreground amplitude profiles, low-frequency phase/edge structure, shuffled controls.

- Create: `spectral_segmentation_posttrain/rlvr/spatial_spectral_objective.py`
  Combines supervised loss, KL anchor, spatial reward loss, spectral reward loss, and shuffled controls.

- Modify: `spectral_segmentation_posttrain/train/posttrain_rlvr.py`
  Adds Plan 4.1 signal modes and logs reward terms separately.

- Create: `scripts/seg41_run_matrix.py`
  Runs S1-S7.

- Create: `scripts/seg41_eval_patch.py`
  Evaluates clean, checkerboard, object-inside, and boundary patch scenes for key groups.

- Create: `scripts/seg41_summarize.py`
  Writes `runs/seg41_summary.json` and `docs/seg41_results.md`.

- Create tests:
  `tests/test_seg41_spatial_rewards.py`,
  `tests/test_seg41_mask_frequency.py`,
  `tests/test_seg41_objective.py`.

---

## Task 1: Spatial Mask Rewards

**Files:**
- Create: `spectral_segmentation_posttrain/rlvr/spatial_rewards.py`
- Create: `tests/test_seg41_spatial_rewards.py`

- [ ] **Step 1: Add tests**

Create `tests/test_seg41_spatial_rewards.py`:

```python
import torch

from spectral_segmentation_posttrain.rlvr.spatial_rewards import boundary_reward, connected_component_reward, dice_reward, mask_iou_reward


def test_mask_iou_and_dice_perfect_are_one():
    mask = torch.tensor([[0, 1], [0, 1]], dtype=torch.bool)
    assert mask_iou_reward(mask, mask) == 1.0
    assert dice_reward(mask, mask) == 1.0


def test_boundary_reward_is_bounded():
    pred = torch.zeros((16, 16), dtype=torch.bool)
    target = torch.zeros((16, 16), dtype=torch.bool)
    pred[4:12, 4:12] = True
    target[5:13, 5:13] = True
    value = boundary_reward(pred, target, tolerance=2)
    assert 0.0 <= value <= 1.0


def test_connected_component_reward_penalizes_fragmentation():
    target = torch.zeros((16, 16), dtype=torch.bool)
    pred = torch.zeros((16, 16), dtype=torch.bool)
    target[2:14, 2:14] = True
    pred[2:6, 2:6] = True
    pred[10:14, 10:14] = True
    assert connected_component_reward(pred, target) < 1.0
```

- [ ] **Step 2: Implement rewards**

Create `spectral_segmentation_posttrain/rlvr/spatial_rewards.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def mask_iou_reward(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = pred.bool()
    target = target.bool()
    inter = (pred & target).sum().float()
    union = (pred | target).sum().float()
    return float((inter / union.clamp_min(eps)).item())


def dice_reward(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = pred.bool()
    target = target.bool()
    inter = (pred & target).sum().float()
    denom = pred.sum().float() + target.sum().float()
    return float(((2.0 * inter) / denom.clamp_min(eps)).item())


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    x = mask.float().view(1, 1, *mask.shape)
    eroded = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    return (x - eroded).squeeze(0).squeeze(0) > 0


def _dilate(mask: torch.Tensor, tolerance: int) -> torch.Tensor:
    x = mask.float().view(1, 1, *mask.shape)
    size = 2 * tolerance + 1
    return F.max_pool2d(x, kernel_size=size, stride=1, padding=tolerance).squeeze(0).squeeze(0) > 0


def boundary_reward(pred: torch.Tensor, target: torch.Tensor, tolerance: int = 2, eps: float = 1e-6) -> float:
    pred_b = _boundary(pred.bool())
    target_b = _boundary(target.bool())
    if pred_b.sum() == 0 and target_b.sum() == 0:
        return 1.0
    precision = (pred_b & _dilate(target_b, tolerance)).sum().float() / pred_b.sum().float().clamp_min(eps)
    recall = (target_b & _dilate(pred_b, tolerance)).sum().float() / target_b.sum().float().clamp_min(eps)
    return float((2.0 * precision * recall / (precision + recall).clamp_min(eps)).item())


def connected_component_reward(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.bool()
    target = target.bool()
    if target.sum() == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    overlap = (pred & target).sum().float() / target.sum().float().clamp_min(1.0)
    extra = (pred & ~target).sum().float() / pred.sum().float().clamp_min(1.0)
    return float((overlap * (1.0 - extra)).clamp(0.0, 1.0).item())
```

- [ ] **Step 3: Verify**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_seg41_spatial_rewards.py -v
```

Expected:

```text
3 passed
```

- [ ] **Step 4: Commit**

Run:

```powershell
git add spectral_segmentation_posttrain/rlvr/spatial_rewards.py tests/test_seg41_spatial_rewards.py
git commit -m "feat: add segmentation spatial rewards"
```

---

## Task 2: Mask Frequency Evidence

**Files:**
- Create: `spectral_segmentation_posttrain/spectral/mask_frequency.py`
- Create: `tests/test_seg41_mask_frequency.py`

- [ ] **Step 1: Add tests**

Create `tests/test_seg41_mask_frequency.py`:

```python
import torch

from spectral_segmentation_posttrain.spectral.mask_frequency import (
    foreground_amplitude_profile,
    lowfreq_phase_structure,
    profile_similarity,
    shuffle_profiles,
)


def test_foreground_amplitude_profile_shape():
    image = torch.rand(3, 32, 32)
    mask = torch.zeros((32, 32), dtype=torch.bool)
    mask[8:24, 8:24] = True
    profile = foreground_amplitude_profile(image, mask, bins=8)
    assert profile.shape == (8,)
    assert torch.isfinite(profile).all()


def test_profile_similarity_identical_is_one():
    profile = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(profile_similarity(profile, profile), torch.tensor(1.0), atol=1e-5)


def test_lowfreq_phase_structure_is_finite():
    image = torch.rand(3, 32, 32)
    mask = torch.ones((32, 32), dtype=torch.bool)
    value = lowfreq_phase_structure(image, mask, size=8)
    assert value.shape == (8, 8)
    assert torch.isfinite(value).all()


def test_shuffle_profiles_preserves_shape():
    profiles = torch.rand(4, 8)
    shuffled = shuffle_profiles(profiles)
    assert shuffled.shape == profiles.shape
```

- [ ] **Step 2: Implement frequency utilities**

Create `spectral_segmentation_posttrain/spectral/mask_frequency.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_gray(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    gray = image.float().mean(dim=0)
    return gray * mask.float()


def foreground_amplitude_profile(image: torch.Tensor, mask: torch.Tensor, bins: int = 16) -> torch.Tensor:
    roi = _masked_gray(image, mask)
    fft = torch.fft.fftshift(torch.fft.fft2(roi, norm="ortho"))
    amp = torch.log1p(torch.abs(fft))
    h, w = amp.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=amp.device), torch.arange(w, device=amp.device), indexing="ij")
    radius = torch.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    radius = radius / radius.max().clamp_min(1e-6)
    values = []
    for idx in range(bins):
        keep = (radius >= idx / bins) & (radius < (idx + 1) / bins)
        values.append(amp[keep].mean() if keep.any() else amp.new_tensor(0.0))
    return torch.stack(values)


def profile_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).clamp(-1.0, 1.0)


def lowfreq_phase_structure(image: torch.Tensor, mask: torch.Tensor, size: int = 8) -> torch.Tensor:
    roi = _masked_gray(image, mask)
    fft = torch.fft.fftshift(torch.fft.fft2(roi, norm="ortho"))
    phase = torch.angle(fft)
    h, w = phase.shape
    top = max(0, h // 2 - size // 2)
    left = max(0, w // 2 - size // 2)
    return phase[top:top + size, left:left + size]


def shuffle_profiles(profiles: torch.Tensor) -> torch.Tensor:
    if len(profiles) <= 1:
        return profiles
    index = torch.arange(len(profiles), device=profiles.device).roll(1)
    return profiles[index]
```

- [ ] **Step 3: Verify**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_seg41_mask_frequency.py -v
```

Expected:

```text
4 passed
```

- [ ] **Step 4: Commit**

Run:

```powershell
git add spectral_segmentation_posttrain/spectral/mask_frequency.py tests/test_seg41_mask_frequency.py
git commit -m "feat: add segmentation mask frequency evidence"
```

---

## Task 3: Spatial-Spectral Objective

**Files:**
- Create: `spectral_segmentation_posttrain/rlvr/spatial_spectral_objective.py`
- Create: `tests/test_seg41_objective.py`

- [ ] **Step 1: Add tests**

Create `tests/test_seg41_objective.py`:

```python
import torch

from spectral_segmentation_posttrain.rlvr.spatial_spectral_objective import compute_reward_terms, reward_to_loss


def test_reward_to_loss_has_gradient_through_logits():
    logits = torch.randn(1, 2, 8, 8, requires_grad=True)
    reward = torch.tensor([0.8])
    loss = reward_to_loss(logits, reward)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_compute_reward_terms_contains_required_keys():
    image = torch.rand(3, 16, 16)
    pred = torch.zeros((16, 16), dtype=torch.bool)
    target = torch.zeros((16, 16), dtype=torch.bool)
    pred[4:12, 4:12] = True
    target[4:12, 4:12] = True
    terms = compute_reward_terms(image, pred, target)
    assert {"iou", "dice", "boundary", "component", "amplitude", "phase_structure"}.issubset(terms.keys())
```

- [ ] **Step 2: Implement objective**

Create `spectral_segmentation_posttrain/rlvr/spatial_spectral_objective.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F

from spectral_segmentation_posttrain.rlvr.spatial_rewards import boundary_reward, connected_component_reward, dice_reward, mask_iou_reward
from spectral_segmentation_posttrain.spectral.mask_frequency import foreground_amplitude_profile, lowfreq_phase_structure, profile_similarity


def compute_reward_terms(image: torch.Tensor, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> dict[str, float]:
    pred_profile = foreground_amplitude_profile(image, pred_mask)
    target_profile = foreground_amplitude_profile(image, target_mask)
    pred_phase = lowfreq_phase_structure(image, pred_mask)
    target_phase = lowfreq_phase_structure(image, target_mask)
    return {
        "iou": mask_iou_reward(pred_mask, target_mask),
        "dice": dice_reward(pred_mask, target_mask),
        "boundary": boundary_reward(pred_mask, target_mask),
        "component": connected_component_reward(pred_mask, target_mask),
        "amplitude": float(profile_similarity(pred_profile, target_profile).item()),
        "phase_structure": float(profile_similarity(pred_phase, target_phase).item()),
    }


def combine_terms(terms: dict[str, float], mode: str) -> float:
    if mode == "supervised":
        return 0.0
    if mode == "spatial":
        return 0.4 * terms["iou"] + 0.3 * terms["dice"] + 0.2 * terms["boundary"] + 0.1 * terms["component"]
    if mode == "amp":
        return terms["amplitude"]
    if mode == "structure":
        return 0.5 * terms["boundary"] + 0.5 * terms["phase_structure"]
    if mode == "spatial_amp_structure":
        return 0.5 * combine_terms(terms, "spatial") + 0.25 * terms["amplitude"] + 0.25 * terms["phase_structure"]
    raise ValueError(f"Unknown reward mode: {mode}")


def reward_to_loss(logits: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, 1]
    confidence = probs.mean(dim=(1, 2))
    advantage = reward.to(logits.device).float() - reward.float().mean().to(logits.device)
    return -(advantage.detach() * torch.log(confidence.clamp_min(1e-6))).mean()


def kl_anchor_loss(current_logits: torch.Tensor, baseline_logits: torch.Tensor) -> torch.Tensor:
    current_logp = torch.log_softmax(current_logits, dim=1)
    baseline_p = torch.softmax(baseline_logits.detach(), dim=1)
    return F.kl_div(current_logp, baseline_p, reduction="batchmean")
```

- [ ] **Step 3: Verify**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_seg41_objective.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 4: Commit**

Run:

```powershell
git add spectral_segmentation_posttrain/rlvr/spatial_spectral_objective.py tests/test_seg41_objective.py
git commit -m "feat: add segmentation spatial spectral objective"
```

---

## Task 4: Extend Segmentation Post-Training

**Files:**
- Modify: `spectral_segmentation_posttrain/train/posttrain_rlvr.py`

- [ ] **Step 1: Add Plan 4.1 signal modes**

Extend CLI choices to:

```text
supervised
spatial
amp
structure
spatial_amp_structure
shuffled_amp_structure
```

For every training step:

```text
load baseline logits with no grad
compute current logits
compute CE/Dice supervised loss
sample or threshold predicted masks
compute reward terms per image
combine rewards by signal mode
add reward_to_loss
add KL anchor
log each reward term separately
```

Default weights:

```text
supervised_weight = 1.0
reward_weight = 0.05
kl_weight = 10.0
```

For `shuffled_amp_structure`, shuffle amplitude/phase terms across the batch before combining.

- [ ] **Step 2: Compile check**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m py_compile spectral_segmentation_posttrain/train/posttrain_rlvr.py
```

Expected:

```text
exit code 0
```

- [ ] **Step 3: Commit**

Run:

```powershell
git add spectral_segmentation_posttrain/train/posttrain_rlvr.py
git commit -m "feat: extend segmentation posttraining for Plan 4.1"
```

---

## Task 5: Plan 4.1 Matrix Runner

**Files:**
- Create: `scripts/seg41_run_matrix.py`

- [ ] **Step 1: Add matrix runner**

Create `scripts/seg41_run_matrix.py`.

It must:

```text
train baseline if runs/seg41/baseline/checkpoint_last.pth is missing
run S1-S7
write eval_metrics.json for every group
write reward_logs.json for every post-training group
```

Group mapping:

```python
GROUPS = [
    ("seg41_s1_baseline_eval", "eval_only"),
    ("seg41_s2_posttrain_supervised_only", "supervised"),
    ("seg41_s3_posttrain_spatial", "spatial"),
    ("seg41_s4_posttrain_amp", "amp"),
    ("seg41_s5_posttrain_structure", "structure"),
    ("seg41_s6_posttrain_spatial_amp_structure", "spatial_amp_structure"),
    ("seg41_s7_posttrain_shuffled_amp_structure", "shuffled_amp_structure"),
]
```

Use:

```text
baseline config = spectral_segmentation_posttrain/configs/penn_fudan_smoke.yaml for first smoke
baseline epochs = 1
posttrain epochs = 1
limit_train = 16
limit_val = 16
```

- [ ] **Step 2: Run matrix smoke**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/seg41_run_matrix.py --config spectral_segmentation_posttrain/configs/penn_fudan_smoke.yaml --limit-train 16 --limit-val 16 --baseline-epochs 1 --posttrain-epochs 1
```

Expected:

```text
runs/seg41_s1_baseline_eval/eval_metrics.json
runs/seg41_s2_posttrain_supervised_only/eval_metrics.json
runs/seg41_s3_posttrain_spatial/eval_metrics.json
runs/seg41_s4_posttrain_amp/eval_metrics.json
runs/seg41_s5_posttrain_structure/eval_metrics.json
runs/seg41_s6_posttrain_spatial_amp_structure/eval_metrics.json
runs/seg41_s7_posttrain_shuffled_amp_structure/eval_metrics.json
```

- [ ] **Step 3: Commit**

Run:

```powershell
git add scripts/seg41_run_matrix.py
git commit -m "feat: add Plan 4.1 segmentation matrix runner"
```

---

## Task 6: Patch Evaluation And Summary

**Files:**
- Create: `scripts/seg41_eval_patch.py`
- Create: `scripts/seg41_summarize.py`
- Create: `docs/seg41_results.md`

- [ ] **Step 1: Add patch evaluator**

Create `scripts/seg41_eval_patch.py`.

Evaluate:

```text
clean
checkerboard_patch
object_inside_patch
boundary_patch
```

Key groups:

```python
KEY_GROUPS = [
    "seg41_s1_baseline_eval",
    "seg41_s3_posttrain_spatial",
    "seg41_s6_posttrain_spatial_amp_structure",
    "seg41_s7_posttrain_shuffled_amp_structure",
]
```

Metrics:

```text
mIoU, Dice, Boundary F1, pixel ECE, high-conf false foreground, high-conf false background
```

- [ ] **Step 2: Add summarizer**

Create `scripts/seg41_summarize.py`. It must write:

```text
runs/seg41_summary.json
docs/seg41_results.md
```

Verdict rules:

```text
If S3 beats S2, spatial post-training is viable.
If S6 beats S3 and S7 does not match S6, spectral evidence has incremental causal value.
If S4/S5 beat shuffled controls only weakly, keep spectral as analysis signal and move to larger data.
If clean improves but boundary patch worsens, do not promote to Plan 3.x scale.
If S6 and S7 are tied, spectral evidence is not causal.
```

- [ ] **Step 3: Run patch and summary**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/seg41_eval_patch.py
E:\anaconda\01\envs\RLimage\python.exe scripts/seg41_summarize.py
```

Expected:

```text
runs/seg41_summary.json exists
docs/seg41_results.md exists
```

- [ ] **Step 4: Commit**

Run:

```powershell
git add scripts/seg41_eval_patch.py scripts/seg41_summarize.py docs/seg41_results.md runs/seg41_summary.json
git commit -m "docs: report Plan 4.1 segmentation posttraining"
```

---

## Success Criteria

Engineering completion:

```text
1. Spatial reward tests pass.
2. Frequency evidence tests pass.
3. Objective tests pass.
4. S1-S7 outputs exist.
5. Patch eval outputs exist for key groups.
6. docs/seg41_results.md exists.
```

Scientific completion:

```text
1. Post-training starts from a segmentation baseline checkpoint.
2. Spatial verifier improves or stabilizes mask metrics relative to supervised-only post-training.
3. Spectral evidence is accepted only if real spectral groups beat shuffled spectral controls.
4. Magnitude and phase/structure are reported separately.
5. Boundary quality is evaluated directly, not inferred from mIoU alone.
```

---

## Assumptions And Defaults

```text
dataset = Penn-Fudan binary person segmentation
seed = 42
baseline_epochs = 1 for smoke, 5 for full Plan 4.1 later
posttrain_epochs = 1 for smoke, 3 for full Plan 4.1 later
model = FCN/DeepLab from Plan 4.0
freeze = encoder/backbone mostly frozen
trainable = decoder/head plus optional AFM/verifier modules
primary metrics = mIoU, Dice, Boundary F1, pixel ECE
spectral claim requires real-vs-shuffled separation
```

