# Plan 4.0 Semantic Segmentation RLVR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the RLVR project from target detection to semantic segmentation, where amplitude and phase/structure verifier signals can act on dense masks instead of noisy box candidates.

**Architecture:** Create a new `spectral_segmentation_posttrain` package instead of mutating the detection package. Train a supervised binary person segmentation baseline, then run KL-stabilized signed RLVR over mask rollouts using verifiable rewards: Dice/IoU, boundary F1, foreground amplitude consistency, and phase/edge structure consistency. Keep the detector experiments as prior evidence and make segmentation the new mainline.

**Tech Stack:** Python, PyTorch, TorchVision segmentation models, Penn-Fudan masks, optional VOC person subset, pytest, NNI GridSearch, PowerShell/Windows batch, conda env `E:\anaconda\01\envs\RLimage`.

---

## Why Plan 4.0 Exists

Round 2.3-2.5 proved that the detection RLVR shell can be stable, but handcrafted box-level spectral verifier signals are too weak on Penn-Fudan detection:

```text
stable shell:
  KL = 10
  policy_loss_weight = 0.0003
  frozen baseline rollout
  signed objective

detection failure mode:
  high policy weight collapses detector
  low policy weight neutralizes weak verifier signal
  box-level phase/structure is noisy under crop shifts
```

Semantic segmentation is a better home for the MPLSeg idea:

```text
magnitude -> foreground semantic region consistency
phase/structure -> boundary and shape consistency
dense GT mask -> direct verifier target
no NMS/proposal matching noise
```

Plan 4.0 therefore stops expanding detection hyperparameter search and starts a segmentation-first RLVR line.

---

## Research Hypothesis

```text
In semantic segmentation, dense mask supervision makes verifiable rewards stronger than box-level ROI rewards.
Magnitude evidence should improve foreground semantic stability.
Phase/edge structure evidence should improve boundary quality and patch robustness.
```

The first claim to test is not "MPLSeg reproduced." The first claim is:

```text
A KL-stabilized RLVR shell for binary segmentation can run without mIoU/Dice collapse,
and mask-level verifier signals can be distinguished from shuffled controls.
```

---

## Experiment Phases

```text
Phase 0: Smoke
  Penn-Fudan binary segmentation, 8 train / 8 val images, 1 epoch baseline, 1 epoch RLVR.

Phase 1: Penn-Fudan MVP
  Full Penn-Fudan split, supervised baseline, Dice-only RLVR, Amp vs shuffled_amp, Struct vs shuffled_struct.

Phase 2: Larger Dataset
  Pascal VOC person subset or COCO person 1k-5k subset.
  Same verifier matrix, no detector code.

Phase 3: Learned Verifier
  Only after Phase 1/2 shows that handcrafted verifier is noisy or partial.
```

Plan 4.0 implements Phase 0 and Phase 1 completely, plus the interfaces needed for Phase 2.

---

## File Map

- Create: `spectral_segmentation_posttrain/__init__.py`
  New package marker.

- Create: `spectral_segmentation_posttrain/configs/penn_fudan_smoke.yaml`
  Tiny smoke config.

- Create: `spectral_segmentation_posttrain/configs/penn_fudan_mvp.yaml`
  Full Penn-Fudan binary segmentation config.

- Create: `spectral_segmentation_posttrain/datasets/penn_fudan_seg.py`
  Loads RGB image and binary person mask from Penn-Fudan `PedMasks`.

- Create: `spectral_segmentation_posttrain/datasets/patch_transform.py`
  Applies random, checkerboard, object-inside, boundary, and background patches to segmentation inputs.

- Create: `spectral_segmentation_posttrain/models/build_segmenter.py`
  Builds TorchVision FCN/DeepLab binary segmentation models and freeze helpers.

- Create: `spectral_segmentation_posttrain/losses/segmentation_losses.py`
  Dice loss, BCE/CE wrapper, supervised loss.

- Create: `spectral_segmentation_posttrain/eval/segmentation_metrics.py`
  mIoU, Dice, Boundary F1, pixel ECE, high-confidence false foreground/background.

- Create: `spectral_segmentation_posttrain/spectral/mask_fft.py`
  Foreground amplitude profile, phase/edge structure similarity, shuffled controls.

- Create: `spectral_segmentation_posttrain/rlvr/mask_rewards.py`
  Mask-level verifier reward and normalization.

- Create: `spectral_segmentation_posttrain/rlvr/mask_policy_loss.py`
  Signed mask policy loss and pixel KL.

- Create: `spectral_segmentation_posttrain/train/train_baseline.py`
  Supervised segmentation baseline trainer.

- Create: `spectral_segmentation_posttrain/train/posttrain_rlvr.py`
  KL-stabilized mask-level RLVR trainer.

- Create: `spectral_segmentation_posttrain/eval/eval_segmenter.py`
  Clean and patch evaluation.

- Create: `spectral_segmentation_posttrain/nni_seg_rlvr_trial.py`
  Baseline -> RLVR -> four-scene eval trial runner.

- Create: `spectral_segmentation_posttrain/analysis/summarize_seg_rlvr.py`
  Aggregates segmentation RLVR results and real-vs-shuffled deltas.

- Create: `nni_configs/seg_rlvr_plan40_search_space.json`
  Phase 1 verifier matrix.

- Create: `nni_configs/seg_rlvr_plan40_config.yml`
  NNI GridSearch config.

- Create: `run_seg_plan40_smoke.bat`
  One-command smoke run.

- Create: `run_nni_seg_plan40.bat`
  One-command NNI launch.

- Create: `tests/test_seg_penn_fudan_dataset.py`
- Create: `tests/test_seg_metrics.py`
- Create: `tests/test_seg_spectral_verifier.py`
- Create: `tests/test_seg_policy_loss.py`
- Create: `tests/test_seg_nni_trial.py`

- Create: `docs/seg_plan40_report.md`
  Final report generated after smoke and NNI runs.

---

## Task 1: Create Package And Configs

**Files:**
- Create: `spectral_segmentation_posttrain/__init__.py`
- Create: `spectral_segmentation_posttrain/configs/penn_fudan_smoke.yaml`
- Create: `spectral_segmentation_posttrain/configs/penn_fudan_mvp.yaml`

- [ ] **Step 1: Create package marker**

Create `spectral_segmentation_posttrain/__init__.py`:

```python
"""Semantic segmentation RLVR with mask-level spectral verifiers."""
```

- [ ] **Step 2: Create smoke config**

Create `spectral_segmentation_posttrain/configs/penn_fudan_smoke.yaml`:

```yaml
seed: 42
device: auto

data:
  root: ./data
  download: true
  max_size: 320
  train_fraction: 0.8
  num_workers: 0

model:
  name: fcn_resnet50
  num_classes: 2
  pretrained: true
  pretrained_backbone: true

train:
  epochs: 1
  batch_size: 2
  lr: 0.0005
  weight_decay: 0.0001
  dice_weight: 1.0
  ce_weight: 1.0

eval:
  threshold: 0.5
  high_conf_threshold: 0.8
  boundary_tolerance: 2

patch:
  patch_size: 48
  patch_type: checkerboard

rlvr:
  epochs: 1
  batch_size: 1
  lr: 0.0001
  policy_loss_weight: 0.0003
  baseline_kl_weight: 10.0
  reward_temperature: 1.0
  rollout_count: 4
  rollout_thresholds: [0.35, 0.45, 0.55, 0.65]
  unfreeze: head
```

- [ ] **Step 3: Create MVP config**

Create `spectral_segmentation_posttrain/configs/penn_fudan_mvp.yaml`:

```yaml
seed: 42
device: auto

data:
  root: ./data
  download: true
  max_size: 480
  train_fraction: 0.8
  num_workers: 0

model:
  name: deeplabv3_resnet50
  num_classes: 2
  pretrained: true
  pretrained_backbone: true

train:
  epochs: 5
  batch_size: 2
  lr: 0.0003
  weight_decay: 0.0001
  dice_weight: 1.0
  ce_weight: 1.0

eval:
  threshold: 0.5
  high_conf_threshold: 0.8
  boundary_tolerance: 2

patch:
  patch_size: 64
  patch_type: checkerboard

rlvr:
  epochs: 3
  batch_size: 1
  lr: 0.0001
  policy_loss_weight: 0.0003
  baseline_kl_weight: 10.0
  reward_temperature: 1.0
  rollout_count: 4
  rollout_thresholds: [0.35, 0.45, 0.55, 0.65]
  unfreeze: head
```

- [ ] **Step 4: Commit**

```powershell
git add spectral_segmentation_posttrain/__init__.py spectral_segmentation_posttrain/configs/penn_fudan_smoke.yaml spectral_segmentation_posttrain/configs/penn_fudan_mvp.yaml
git commit -m "feat: add segmentation rlvr configs"
```

---

## Task 2: Add Penn-Fudan Binary Segmentation Dataset

**Files:**
- Create: `spectral_segmentation_posttrain/datasets/__init__.py`
- Create: `spectral_segmentation_posttrain/datasets/penn_fudan_seg.py`
- Create: `tests/test_seg_penn_fudan_dataset.py`

- [ ] **Step 1: Write dataset tests**

Create `tests/test_seg_penn_fudan_dataset.py`:

```python
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from spectral_segmentation_posttrain.datasets.penn_fudan_seg import (
    PennFudanSegmentationDataset,
    segmentation_collate,
)


def _write_fake_penn_fudan(root: Path) -> None:
    image_dir = root / "PennFudanPed" / "PNGImages"
    mask_dir = root / "PennFudanPed" / "PedMasks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    image = Image.fromarray(np.full((16, 20, 3), 127, dtype=np.uint8))
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[4:12, 6:14] = 1
    image.save(image_dir / "FudanPed00001.png")
    Image.fromarray(mask).save(mask_dir / "FudanPed00001_mask.png")


def test_penn_fudan_segmentation_dataset_returns_binary_mask(tmp_path):
    _write_fake_penn_fudan(tmp_path)
    dataset = PennFudanSegmentationDataset(tmp_path, download=False)

    image, target = dataset[0]

    assert image.shape == (3, 16, 20)
    assert target["mask"].shape == (16, 20)
    assert target["mask"].dtype == torch.long
    assert set(target["mask"].unique().tolist()) == {0, 1}


def test_segmentation_collate_keeps_lists(tmp_path):
    _write_fake_penn_fudan(tmp_path)
    dataset = PennFudanSegmentationDataset(tmp_path, download=False)
    batch = segmentation_collate([dataset[0], dataset[0]])

    assert len(batch[0]) == 2
    assert len(batch[1]) == 2
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_seg_penn_fudan_dataset.py -v
```

Expected: fails because `spectral_segmentation_posttrain.datasets.penn_fudan_seg` does not exist.

- [ ] **Step 3: Implement dataset**

Create `spectral_segmentation_posttrain/datasets/__init__.py`:

```python
from .penn_fudan_seg import PennFudanSegmentationDataset, build_penn_fudan_seg_loaders, segmentation_collate

__all__ = ["PennFudanSegmentationDataset", "build_penn_fudan_seg_loaders", "segmentation_collate"]
```

Create `spectral_segmentation_posttrain/datasets/penn_fudan_seg.py`:

```python
from __future__ import annotations

import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import functional as F


PENN_FUDAN_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"


def _download_penn_fudan(root: Path) -> None:
    dataset_dir = root / "PennFudanPed"
    if dataset_dir.exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "PennFudanPed.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(PENN_FUDAN_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)


class PennFudanSegmentationDataset(Dataset):
    def __init__(self, root: str | Path, download: bool = True, max_size: int | None = None) -> None:
        self.root = Path(root)
        self.max_size = max_size
        if download:
            _download_penn_fudan(self.root)
        self.dataset_dir = self.root / "PennFudanPed"
        self.image_dir = self.dataset_dir / "PNGImages"
        self.mask_dir = self.dataset_dir / "PedMasks"
        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(f"Penn-Fudan dataset not found under {self.dataset_dir}")
        self.images = sorted(self.image_dir.glob("*.png"))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        mask_path = self.mask_dir / image_path.name.replace(".png", "_mask.png")
        image = F.to_tensor(Image.open(image_path).convert("RGB"))
        mask_np = np.array(Image.open(mask_path))
        mask = torch.as_tensor((mask_np > 0).astype(np.int64), dtype=torch.long)
        if self.max_size:
            image, mask = resize_image_and_mask(image, mask, int(self.max_size))
        return image, {"mask": mask, "image_id": torch.tensor([idx])}


def resize_image_and_mask(image: torch.Tensor, mask: torch.Tensor, max_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    _, height, width = image.shape
    largest_side = max(height, width)
    if largest_side <= max_size:
        return image, mask
    scale = max_size / float(largest_side)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    image_out = F_torch.interpolate(image.unsqueeze(0), size=(new_height, new_width), mode="bilinear", align_corners=False).squeeze(0)
    mask_out = F_torch.interpolate(mask.float().view(1, 1, height, width), size=(new_height, new_width), mode="nearest").squeeze(0).squeeze(0).long()
    return image_out, mask_out


def segmentation_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def _subset(dataset: Dataset, indices: list[int], limit: int | None) -> Dataset:
    if limit is not None:
        indices = indices[: min(limit, len(indices))]
    return Subset(dataset, indices)


def build_penn_fudan_seg_loaders(config: dict, limit_train: int | None = None, limit_val: int | None = None, batch_size: int | None = None):
    data_cfg = config["data"]
    dataset = PennFudanSegmentationDataset(
        data_cfg.get("root", "./data"),
        download=bool(data_cfg.get("download", True)),
        max_size=data_cfg.get("max_size"),
    )
    indices = list(range(len(dataset)))
    rng = random.Random(int(config.get("seed", 42)))
    rng.shuffle(indices)
    split = int(len(indices) * float(data_cfg.get("train_fraction", 0.8)))
    train_set = _subset(dataset, indices[:split], limit_train)
    val_set = _subset(dataset, indices[split:], limit_val)
    bs = int(batch_size or config["train"].get("batch_size", 2))
    num_workers = int(data_cfg.get("num_workers", 0))
    return (
        DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=num_workers, collate_fn=segmentation_collate),
        DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=num_workers, collate_fn=segmentation_collate),
    )
```

- [ ] **Step 4: Run dataset tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_seg_penn_fudan_dataset.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add spectral_segmentation_posttrain/datasets tests/test_seg_penn_fudan_dataset.py
git commit -m "feat: add Penn-Fudan segmentation dataset"
```

---

## Task 3: Add Segmentation Metrics

**Files:**
- Create: `spectral_segmentation_posttrain/eval/__init__.py`
- Create: `spectral_segmentation_posttrain/eval/segmentation_metrics.py`
- Create: `tests/test_seg_metrics.py`

- [ ] **Step 1: Write metric tests**

Create `tests/test_seg_metrics.py`:

```python
import pytest
import torch

from spectral_segmentation_posttrain.eval.segmentation_metrics import (
    binary_dice,
    binary_iou,
    boundary_f1,
    high_confusion_counts,
    pixel_ece,
)


def test_binary_iou_and_dice_are_one_for_perfect_mask():
    pred = torch.tensor([[0, 1], [0, 1]], dtype=torch.bool)
    target = torch.tensor([[0, 1], [0, 1]], dtype=torch.bool)

    assert binary_iou(pred, target) == pytest.approx(1.0)
    assert binary_dice(pred, target) == pytest.approx(1.0)


def test_binary_iou_handles_partial_overlap():
    pred = torch.tensor([[1, 1], [0, 0]], dtype=torch.bool)
    target = torch.tensor([[1, 0], [1, 0]], dtype=torch.bool)

    assert binary_iou(pred, target) == pytest.approx(1.0 / 3.0)


def test_pixel_ece_is_bounded():
    probs = torch.tensor([[0.9, 0.8], [0.2, 0.1]])
    target = torch.tensor([[1, 1], [0, 0]])

    value = pixel_ece(probs, target, n_bins=5)

    assert 0.0 <= value <= 1.0


def test_high_confusion_counts_detect_high_conf_errors():
    probs = torch.tensor([[0.95, 0.90], [0.05, 0.10]])
    target = torch.tensor([[0, 1], [1, 0]])

    counts = high_confusion_counts(probs, target, threshold=0.8)

    assert counts["high_conf_false_foreground"] == 1
    assert counts["high_conf_false_background"] == 1


def test_boundary_f1_perfect_is_one():
    mask = torch.zeros((16, 16), dtype=torch.bool)
    mask[4:12, 4:12] = True

    assert boundary_f1(mask, mask, tolerance=1) == pytest.approx(1.0)
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_seg_metrics.py -v
```

Expected: fails because `segmentation_metrics.py` does not exist.

- [ ] **Step 3: Implement metrics**

Create `spectral_segmentation_posttrain/eval/__init__.py`:

```python
"""Segmentation evaluation utilities."""
```

Create `spectral_segmentation_posttrain/eval/segmentation_metrics.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred_bool = pred.bool()
    target_bool = target.bool()
    intersection = (pred_bool & target_bool).sum().float()
    union = (pred_bool | target_bool).sum().float()
    return float((intersection / union.clamp_min(eps)).item())


def binary_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred_bool = pred.bool()
    target_bool = target.bool()
    intersection = (pred_bool & target_bool).sum().float()
    denom = pred_bool.sum().float() + target_bool.sum().float()
    return float(((2.0 * intersection) / denom.clamp_min(eps)).item())


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    x = mask.float().view(1, 1, *mask.shape)
    eroded = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    return (x - eroded).squeeze(0).squeeze(0) > 0


def _dilate(mask: torch.Tensor, tolerance: int) -> torch.Tensor:
    if tolerance <= 0:
        return mask.bool()
    x = mask.float().view(1, 1, *mask.shape)
    size = 2 * tolerance + 1
    return F.max_pool2d(x, kernel_size=size, stride=1, padding=tolerance).squeeze(0).squeeze(0) > 0


def boundary_f1(pred: torch.Tensor, target: torch.Tensor, tolerance: int = 2, eps: float = 1e-6) -> float:
    pred_b = _boundary(pred.bool())
    target_b = _boundary(target.bool())
    if pred_b.sum() == 0 and target_b.sum() == 0:
        return 1.0
    pred_match = pred_b & _dilate(target_b, tolerance)
    target_match = target_b & _dilate(pred_b, tolerance)
    precision = pred_match.sum().float() / pred_b.sum().float().clamp_min(eps)
    recall = target_match.sum().float() / target_b.sum().float().clamp_min(eps)
    return float((2.0 * precision * recall / (precision + recall).clamp_min(eps)).item())


def pixel_ece(probs: torch.Tensor, target: torch.Tensor, n_bins: int = 15) -> float:
    conf = torch.maximum(probs, 1.0 - probs).flatten()
    pred = (probs >= 0.5).long().flatten()
    truth = target.long().flatten()
    correct = (pred == truth).float()
    ece = torch.tensor(0.0, dtype=torch.float32)
    for idx in range(n_bins):
        lo = idx / n_bins
        hi = (idx + 1) / n_bins
        mask = (conf >= lo) & (conf < hi if idx < n_bins - 1 else conf <= hi)
        if mask.any():
            ece = ece + mask.float().mean() * (correct[mask].mean() - conf[mask].mean()).abs()
    return float(ece.item())


def high_confusion_counts(probs: torch.Tensor, target: torch.Tensor, threshold: float = 0.8) -> dict[str, int]:
    pred_fg = probs >= threshold
    pred_bg = probs <= (1.0 - threshold)
    target_fg = target.bool()
    return {
        "high_conf_false_foreground": int((pred_fg & ~target_fg).sum().item()),
        "high_conf_false_background": int((pred_bg & target_fg).sum().item()),
    }
```

- [ ] **Step 4: Run metric tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_seg_metrics.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add spectral_segmentation_posttrain/eval tests/test_seg_metrics.py
git commit -m "feat: add segmentation metrics"
```

---

## Task 4: Add Segmentation Model Builder And Losses

**Files:**
- Create: `spectral_segmentation_posttrain/models/__init__.py`
- Create: `spectral_segmentation_posttrain/models/build_segmenter.py`
- Create: `spectral_segmentation_posttrain/losses/__init__.py`
- Create: `spectral_segmentation_posttrain/losses/segmentation_losses.py`
- Create: `tests/test_seg_policy_loss.py`

- [ ] **Step 1: Write loss tests**

Create `tests/test_seg_policy_loss.py`:

```python
import torch

from spectral_segmentation_posttrain.losses.segmentation_losses import dice_loss, supervised_segmentation_loss


def test_dice_loss_is_low_for_perfect_logits():
    mask = torch.tensor([[[1, 0], [1, 0]]], dtype=torch.long)
    logits = torch.tensor([[[[-5.0, 5.0], [-5.0, 5.0]], [[5.0, -5.0], [5.0, -5.0]]]])

    loss = dice_loss(logits, mask)

    assert loss.item() < 0.05


def test_supervised_segmentation_loss_returns_scalar():
    mask = torch.tensor([[[1, 0], [1, 0]]], dtype=torch.long)
    logits = torch.randn((1, 2, 2, 2))

    loss = supervised_segmentation_loss(logits, mask, ce_weight=1.0, dice_weight=1.0)

    assert loss.ndim == 0
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_seg_policy_loss.py -v
```

Expected: fails because losses do not exist.

- [ ] **Step 3: Implement losses**

Create `spectral_segmentation_posttrain/losses/__init__.py`:

```python
from .segmentation_losses import dice_loss, supervised_segmentation_loss

__all__ = ["dice_loss", "supervised_segmentation_loss"]
```

Create `spectral_segmentation_posttrain/losses/segmentation_losses.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, 1]
    target_fg = target.float()
    intersection = (probs * target_fg).sum(dim=(-2, -1))
    denom = probs.sum(dim=(-2, -1)) + target_fg.sum(dim=(-2, -1))
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def supervised_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target.long())
    return ce_weight * ce + dice_weight * dice_loss(logits, target)
```

- [ ] **Step 4: Implement model builder**

Create `spectral_segmentation_posttrain/models/__init__.py`:

```python
from .build_segmenter import build_segmenter, freeze_backbone, set_segmentation_trainable

__all__ = ["build_segmenter", "freeze_backbone", "set_segmentation_trainable"]
```

Create `spectral_segmentation_posttrain/models/build_segmenter.py`:

```python
from __future__ import annotations

import torch
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    FCN_ResNet50_Weights,
    deeplabv3_resnet50,
    fcn_resnet50,
)


def _weights(name: str, pretrained: bool):
    if not pretrained:
        return None
    if name == "fcn_resnet50":
        return FCN_ResNet50_Weights.DEFAULT
    if name == "deeplabv3_resnet50":
        return DeepLabV3_ResNet50_Weights.DEFAULT
    raise ValueError(f"Unsupported segmentation model: {name}")


def build_segmenter(config: dict) -> torch.nn.Module:
    model_cfg = config["model"]
    name = str(model_cfg.get("name", "fcn_resnet50"))
    num_classes = int(model_cfg.get("num_classes", 2))
    weights = _weights(name, bool(model_cfg.get("pretrained", True)))
    if name == "fcn_resnet50":
        model = fcn_resnet50(weights=weights, weights_backbone=None, num_classes=num_classes)
    elif name == "deeplabv3_resnet50":
        model = deeplabv3_resnet50(weights=weights, weights_backbone=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported segmentation model: {name}")
    return model


def freeze_backbone(model: torch.nn.Module) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = False


def set_segmentation_trainable(model: torch.nn.Module, mode: str = "head") -> None:
    for param in model.parameters():
        param.requires_grad = False
    if mode == "head":
        for param in model.classifier.parameters():
            param.requires_grad = True
        if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
            for param in model.aux_classifier.parameters():
                param.requires_grad = True
        return
    if mode == "all":
        for param in model.parameters():
            param.requires_grad = True
        return
    raise ValueError(f"Unsupported trainable mode: {mode}")
```

- [ ] **Step 5: Run tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_seg_policy_loss.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add spectral_segmentation_posttrain/models spectral_segmentation_posttrain/losses tests/test_seg_policy_loss.py
git commit -m "feat: add segmentation models and losses"
```

---

## Task 5: Add Mask Spectral Verifier

**Files:**
- Create: `spectral_segmentation_posttrain/spectral/__init__.py`
- Create: `spectral_segmentation_posttrain/spectral/mask_fft.py`
- Create: `spectral_segmentation_posttrain/rlvr/__init__.py`
- Create: `spectral_segmentation_posttrain/rlvr/mask_rewards.py`
- Create: `tests/test_seg_spectral_verifier.py`

- [ ] **Step 1: Write verifier tests**

Create `tests/test_seg_spectral_verifier.py`:

```python
import torch

from spectral_segmentation_posttrain.rlvr.mask_rewards import MaskRewardConfig, compute_mask_reward
from spectral_segmentation_posttrain.spectral.mask_fft import foreground_amplitude_similarity, structure_similarity


def _sample():
    image = torch.zeros((3, 32, 32))
    image[:, 8:24, 10:22] = 1.0
    mask = torch.zeros((32, 32), dtype=torch.bool)
    mask[8:24, 10:22] = True
    shifted = torch.zeros((32, 32), dtype=torch.bool)
    shifted[10:26, 12:24] = True
    return image, mask, shifted


def test_foreground_amplitude_similarity_is_higher_for_same_mask():
    image, mask, shifted = _sample()

    same = foreground_amplitude_similarity(image, mask, mask)
    moved = foreground_amplitude_similarity(image, shifted, mask)

    assert same >= moved
    assert 0.0 <= same <= 1.0


def test_structure_similarity_is_bounded():
    _, mask, shifted = _sample()

    value = structure_similarity(mask, shifted)

    assert 0.0 <= value <= 1.0


def test_mask_reward_combines_dice_amp_and_structure():
    image, mask, shifted = _sample()
    cfg = MaskRewardConfig(signal="amp_structure", w_dice=1.0, w_iou=1.0, w_amp=0.2, w_struct=0.3)

    reward_same = compute_mask_reward(image, mask, mask, cfg)
    reward_shifted = compute_mask_reward(image, shifted, mask, cfg)

    assert reward_same > reward_shifted
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_seg_spectral_verifier.py -v
```

Expected: fails because spectral verifier files do not exist.

- [ ] **Step 3: Implement spectral mask features**

Create `spectral_segmentation_posttrain/spectral/__init__.py`:

```python
"""Mask-level spectral features."""
```

Create `spectral_segmentation_posttrain/spectral/mask_fft.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def _crop_to_mask(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    coords = torch.nonzero(mask.bool(), as_tuple=False)
    if coords.numel() == 0:
        return torch.zeros((image.shape[0], 32, 32), dtype=image.dtype, device=image.device)
    y1, x1 = coords.min(dim=0).values
    y2, x2 = coords.max(dim=0).values + 1
    crop = image[:, int(y1):int(y2), int(x1):int(x2)]
    return F.interpolate(crop.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False).squeeze(0)


def _amplitude_profile(crop: torch.Tensor, bins: int = 32) -> torch.Tensor:
    gray = crop.mean(dim=0)
    h, w = gray.shape
    window = torch.outer(torch.hann_window(h, device=gray.device), torch.hann_window(w, device=gray.device))
    fft = torch.fft.fftshift(torch.fft.fft2(gray * window))
    amp = torch.log1p(torch.abs(fft))
    amp = (amp - amp.min()) / (amp.max() - amp.min()).clamp_min(1e-6)
    y, x = torch.meshgrid(torch.arange(h, device=gray.device), torch.arange(w, device=gray.device), indexing="ij")
    radius = torch.sqrt((y - (h - 1) / 2.0) ** 2 + (x - (w - 1) / 2.0) ** 2)
    radius = radius / radius.max().clamp_min(1e-6)
    values = []
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        mask = (radius >= lo) & (radius < hi if idx < bins - 1 else radius <= hi)
        values.append(amp[mask].mean() if mask.any() else torch.tensor(0.0, device=gray.device))
    return torch.stack(values)


def foreground_amplitude_similarity(image: torch.Tensor, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    pred_profile = _amplitude_profile(_crop_to_mask(image, pred_mask))
    gt_profile = _amplitude_profile(_crop_to_mask(image, gt_mask))
    cosine = F.cosine_similarity(pred_profile, gt_profile, dim=0).clamp(-1.0, 1.0)
    return float(((cosine + 1.0) * 0.5).item())


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    x = mask.float().view(1, 1, *mask.shape)
    eroded = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    return (x - eroded).squeeze(0).squeeze(0) > 0


def structure_similarity(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    pred_b = _boundary(pred_mask.bool()).float().flatten()
    gt_b = _boundary(gt_mask.bool()).float().flatten()
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0
    cosine = F.cosine_similarity(pred_b, gt_b, dim=0).clamp(-1.0, 1.0)
    return float(((cosine + 1.0) * 0.5).item())
```

- [ ] **Step 4: Implement reward**

Create `spectral_segmentation_posttrain/rlvr/__init__.py`:

```python
"""Mask-level RLVR utilities."""
```

Create `spectral_segmentation_posttrain/rlvr/mask_rewards.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch

from spectral_segmentation_posttrain.eval.segmentation_metrics import binary_dice, binary_iou, boundary_f1
from spectral_segmentation_posttrain.spectral.mask_fft import foreground_amplitude_similarity, structure_similarity


AMP_SIGNALS = {"amp", "shuffled_amp", "amp_structure", "shuffled_amp_structure"}
STRUCT_SIGNALS = {"structure", "shuffled_structure", "amp_structure", "shuffled_amp_structure"}


@dataclass(frozen=True)
class MaskRewardConfig:
    signal: str = "dice"
    w_iou: float = 1.0
    w_dice: float = 1.0
    w_boundary: float = 0.5
    w_amp: float = 0.0
    w_struct: float = 0.0


def compute_mask_reward(image: torch.Tensor, pred_mask: torch.Tensor, gt_mask: torch.Tensor, cfg: MaskRewardConfig) -> float:
    reward = cfg.w_iou * binary_iou(pred_mask, gt_mask)
    reward += cfg.w_dice * binary_dice(pred_mask, gt_mask)
    reward += cfg.w_boundary * boundary_f1(pred_mask, gt_mask)
    if cfg.signal in AMP_SIGNALS:
        reward += cfg.w_amp * foreground_amplitude_similarity(image, pred_mask, gt_mask)
    if cfg.signal in STRUCT_SIGNALS:
        reward += cfg.w_struct * structure_similarity(pred_mask, gt_mask)
    return float(reward)


def normalize_advantages(rewards: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if rewards.numel() == 0:
        return rewards
    centered = rewards - rewards.mean()
    std = rewards.std(unbiased=False).clamp_min(1e-6)
    return (centered / std / max(float(temperature), 1e-6)).clamp(-3.0, 3.0)
```

- [ ] **Step 5: Run verifier tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_seg_spectral_verifier.py tests/test_seg_metrics.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add spectral_segmentation_posttrain/spectral spectral_segmentation_posttrain/rlvr tests/test_seg_spectral_verifier.py
git commit -m "feat: add mask spectral rewards"
```

---

## Task 6: Add Mask Policy Loss

**Files:**
- Create: `spectral_segmentation_posttrain/rlvr/mask_policy_loss.py`
- Modify: `tests/test_seg_policy_loss.py`

- [ ] **Step 1: Add policy loss tests**

Append to `tests/test_seg_policy_loss.py`:

```python
from spectral_segmentation_posttrain.rlvr.mask_policy_loss import mask_kl_loss, signed_mask_policy_loss


def test_signed_mask_policy_loss_rewards_high_advantage_mask():
    logits = torch.tensor([[[[0.0, 0.0]], [[2.0, -2.0]]]], requires_grad=True)
    action_mask = torch.tensor([[[1, 0]]], dtype=torch.long)
    advantages = torch.tensor([1.0])

    loss = signed_mask_policy_loss(logits, action_mask, advantages)

    assert loss.item() < 0.2


def test_mask_kl_loss_zero_for_identical_logits():
    logits = torch.randn((1, 2, 4, 4))

    loss = mask_kl_loss(logits, logits.detach())

    assert loss.item() < 1e-6
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_seg_policy_loss.py -v
```

Expected: fails because `mask_policy_loss.py` does not exist.

- [ ] **Step 3: Implement policy loss**

Create `spectral_segmentation_posttrain/rlvr/mask_policy_loss.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def signed_mask_policy_loss(logits: torch.Tensor, action_masks: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    ce = F.cross_entropy(logits, action_masks.long(), reduction="none")
    per_item = ce.flatten(1).mean(dim=1)
    return (per_item * (-advantages.float())).mean()


def mask_kl_loss(current_logits: torch.Tensor, baseline_logits: torch.Tensor) -> torch.Tensor:
    current_logp = F.log_softmax(current_logits, dim=1)
    baseline_p = F.softmax(baseline_logits.detach(), dim=1)
    return F.kl_div(current_logp, baseline_p, reduction="batchmean")
```

- [ ] **Step 4: Run policy tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_seg_policy_loss.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add spectral_segmentation_posttrain/rlvr/mask_policy_loss.py tests/test_seg_policy_loss.py
git commit -m "feat: add segmentation mask policy loss"
```

---

## Task 7: Add Baseline Trainer And Evaluator

**Files:**
- Create: `spectral_segmentation_posttrain/utils/__init__.py`
- Create: `spectral_segmentation_posttrain/utils/config.py`
- Create: `spectral_segmentation_posttrain/utils/io.py`
- Create: `spectral_segmentation_posttrain/utils/seed.py`
- Create: `spectral_segmentation_posttrain/train/__init__.py`
- Create: `spectral_segmentation_posttrain/train/train_baseline.py`
- Create: `spectral_segmentation_posttrain/eval/eval_segmenter.py`

- [ ] **Step 1: Create utils**

Create `spectral_segmentation_posttrain/utils/config.py`:

```python
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
```

Create `spectral_segmentation_posttrain/utils/io.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import torch


def ensure_run_dir(run_name: str) -> Path:
    path = Path("runs") / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_checkpoint(model: torch.nn.Module, path: str | Path, meta: dict) -> None:
    torch.save({"model": model.state_dict(), "meta": meta}, path)


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    return payload.get("meta", {})
```

Create `spectral_segmentation_posttrain/utils/seed.py`:

```python
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config: dict) -> torch.device:
    requested = str(config.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
```

Create `spectral_segmentation_posttrain/utils/__init__.py`:

```python
"""Shared segmentation utilities."""
```

- [ ] **Step 2: Create train package**

Create `spectral_segmentation_posttrain/train/__init__.py`:

```python
"""Segmentation training entrypoints."""
```

- [ ] **Step 3: Implement baseline trainer**

Create `spectral_segmentation_posttrain/train/train_baseline.py`:

```python
from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from spectral_segmentation_posttrain.datasets import build_penn_fudan_seg_loaders
from spectral_segmentation_posttrain.losses import supervised_segmentation_loss
from spectral_segmentation_posttrain.models import build_segmenter
from spectral_segmentation_posttrain.utils.config import load_config
from spectral_segmentation_posttrain.utils.io import ensure_run_dir, save_checkpoint, save_json
from spectral_segmentation_posttrain.utils.seed import resolve_device, set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    return parser.parse_args(argv)


def _stack_masks(targets: list[dict], device: torch.device) -> torch.Tensor:
    return torch.stack([target["mask"].to(device) for target in targets], dim=0)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config)
    train_cfg = config["train"]
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    run_dir = ensure_run_dir(args.run_name)
    train_loader, _ = build_penn_fudan_seg_loaders(config, limit_train=args.limit_train, limit_val=args.limit_val)
    model = build_segmenter(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 0.0003)), weight_decay=float(train_cfg.get("weight_decay", 0.0001)))

    history = []
    for epoch in range(1, int(train_cfg.get("epochs", 1)) + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for images, targets in tqdm(train_loader, desc=f"baseline epoch {epoch}"):
            batch = torch.stack([image.to(device) for image in images], dim=0)
            masks = _stack_masks(targets, device)
            logits = model(batch)["out"]
            loss = supervised_segmentation_loss(logits, masks, ce_weight=float(train_cfg.get("ce_weight", 1.0)), dice_weight=float(train_cfg.get("dice_weight", 1.0)))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(images)
            total_seen += len(images)
        row = {"epoch": epoch, "train_loss": total_loss / max(1, total_seen)}
        history.append(row)
        save_checkpoint(model, run_dir / "checkpoint_last.pth", {"epoch": epoch})
    save_json({"history": history}, run_dir / "train_result.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement evaluator**

Create `spectral_segmentation_posttrain/eval/eval_segmenter.py`:

```python
from __future__ import annotations

import argparse

import torch

from spectral_segmentation_posttrain.datasets import build_penn_fudan_seg_loaders
from spectral_segmentation_posttrain.eval.segmentation_metrics import binary_dice, binary_iou, boundary_f1, high_confusion_counts, pixel_ece
from spectral_segmentation_posttrain.models import build_segmenter
from spectral_segmentation_posttrain.utils.config import load_config
from spectral_segmentation_posttrain.utils.io import ensure_run_dir, load_checkpoint, save_json
from spectral_segmentation_posttrain.utils.seed import resolve_device, set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit-val", type=int, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config)
    _, val_loader = build_penn_fudan_seg_loaders(config, limit_val=args.limit_val)
    model = build_segmenter(config).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    ious = []
    dices = []
    bf1s = []
    eces = []
    false_fg = 0
    false_bg = 0
    threshold = float(config["eval"].get("threshold", 0.5))
    high_conf = float(config["eval"].get("high_conf_threshold", 0.8))
    tolerance = int(config["eval"].get("boundary_tolerance", 2))
    with torch.no_grad():
        for images, targets in val_loader:
            batch = torch.stack([image.to(device) for image in images], dim=0)
            logits = model(batch)["out"]
            probs = torch.softmax(logits, dim=1)[:, 1].cpu()
            for prob, target in zip(probs, targets):
                mask = target["mask"].cpu()
                pred = prob >= threshold
                ious.append(binary_iou(pred, mask))
                dices.append(binary_dice(pred, mask))
                bf1s.append(boundary_f1(pred, mask, tolerance=tolerance))
                eces.append(pixel_ece(prob, mask))
                counts = high_confusion_counts(prob, mask, threshold=high_conf)
                false_fg += counts["high_conf_false_foreground"]
                false_bg += counts["high_conf_false_background"]
    metrics = {
        "miou": sum(ious) / max(1, len(ious)),
        "dice": sum(dices) / max(1, len(dices)),
        "boundary_f1": sum(bf1s) / max(1, len(bf1s)),
        "pixel_ece": sum(eces) / max(1, len(eces)),
        "high_conf_false_foreground": false_fg,
        "high_conf_false_background": false_bg,
        "num_images": len(ious),
    }
    run_dir = ensure_run_dir(args.run_name)
    save_json(metrics, run_dir / "eval_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run import checks**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m py_compile spectral_segmentation_posttrain/train/train_baseline.py spectral_segmentation_posttrain/eval/eval_segmenter.py
```

Expected: exits with code 0.

- [ ] **Step 6: Commit**

```powershell
git add spectral_segmentation_posttrain/utils spectral_segmentation_posttrain/train spectral_segmentation_posttrain/eval/eval_segmenter.py
git commit -m "feat: add segmentation baseline and eval"
```

---

## Task 8: Add Patch Transform For Segmentation

**Files:**
- Create: `spectral_segmentation_posttrain/datasets/patch_transform.py`
- Modify: `tests/test_seg_penn_fudan_dataset.py`

- [ ] **Step 1: Add patch tests**

Append to `tests/test_seg_penn_fudan_dataset.py`:

```python
from spectral_segmentation_posttrain.datasets.patch_transform import add_segmentation_patch


def test_add_segmentation_patch_changes_image_not_mask():
    image = torch.zeros((3, 32, 32))
    mask = torch.zeros((32, 32), dtype=torch.long)
    mask[8:24, 8:24] = 1

    patched, patched_mask = add_segmentation_patch(image, mask, placement="object_inside", patch_type="checkerboard", patch_size=8)

    assert not torch.equal(patched, image)
    assert torch.equal(patched_mask, mask)
```

- [ ] **Step 2: Run failing patch test**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_seg_penn_fudan_dataset.py::test_add_segmentation_patch_changes_image_not_mask -v
```

Expected: fails because `patch_transform.py` does not exist.

- [ ] **Step 3: Implement patch transform**

Create `spectral_segmentation_posttrain/datasets/patch_transform.py`:

```python
from __future__ import annotations

import torch


def _make_patch(channels: int, size: int, patch_type: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if patch_type == "random":
        return torch.rand((channels, size, size), device=device, dtype=dtype)
    if patch_type == "checkerboard":
        yy, xx = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing="ij")
        return ((yy + xx) % 2).to(dtype=dtype).view(1, size, size).repeat(channels, 1, 1)
    raise ValueError(f"Unknown patch_type: {patch_type}")


def _mask_location(mask: torch.Tensor, size: int, placement: str) -> tuple[int, int]:
    height, width = mask.shape
    coords = torch.nonzero(mask.bool(), as_tuple=False)
    if coords.numel() == 0:
        return 0, 0
    y1, x1 = coords.min(dim=0).values
    y2, x2 = coords.max(dim=0).values
    if placement == "object_inside":
        top = int(((y1 + y2) // 2 - size // 2).clamp(0, max(0, height - size)).item())
        left = int(((x1 + x2) // 2 - size // 2).clamp(0, max(0, width - size)).item())
        return top, left
    if placement == "boundary":
        top = int((y1 - size // 2).clamp(0, max(0, height - size)).item())
        left = int((x1 - size // 2).clamp(0, max(0, width - size)).item())
        return top, left
    if placement == "background":
        return 0, 0
    top = int(torch.randint(0, max(1, height - size + 1), (1,)).item())
    left = int(torch.randint(0, max(1, width - size + 1), (1,)).item())
    return top, left


def add_segmentation_patch(
    image: torch.Tensor,
    mask: torch.Tensor,
    placement: str = "random",
    patch_type: str = "checkerboard",
    patch_size: int = 48,
) -> tuple[torch.Tensor, torch.Tensor]:
    channels, height, width = image.shape
    size = min(int(patch_size), height - 1, width - 1)
    if size <= 0:
        return image.clone(), mask.clone()
    top, left = _mask_location(mask, size, placement)
    out = image.clone()
    out[:, top:top + size, left:left + size] = _make_patch(channels, size, patch_type, image.device, image.dtype)
    return out, mask.clone()
```

- [ ] **Step 4: Run tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_seg_penn_fudan_dataset.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add spectral_segmentation_posttrain/datasets/patch_transform.py tests/test_seg_penn_fudan_dataset.py
git commit -m "feat: add segmentation patch transforms"
```

---

## Task 9: Add Segmentation RLVR Post-Training

**Files:**
- Create: `spectral_segmentation_posttrain/train/posttrain_rlvr.py`

- [ ] **Step 1: Implement post-training entrypoint**

Create `spectral_segmentation_posttrain/train/posttrain_rlvr.py`:

```python
from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from spectral_segmentation_posttrain.datasets import build_penn_fudan_seg_loaders
from spectral_segmentation_posttrain.models import build_segmenter, set_segmentation_trainable
from spectral_segmentation_posttrain.rlvr.mask_policy_loss import mask_kl_loss, signed_mask_policy_loss
from spectral_segmentation_posttrain.rlvr.mask_rewards import MaskRewardConfig, compute_mask_reward, normalize_advantages
from spectral_segmentation_posttrain.utils.config import load_config
from spectral_segmentation_posttrain.utils.io import ensure_run_dir, load_checkpoint, save_checkpoint, save_json
from spectral_segmentation_posttrain.utils.seed import resolve_device, set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--signal", required=True, choices=["dice", "amp", "shuffled_amp", "structure", "shuffled_structure", "amp_structure", "shuffled_amp_structure"])
    parser.add_argument("--reward-lambda", type=float, default=0.0)
    parser.add_argument("--struct-weight", type=float, default=0.0)
    parser.add_argument("--policy-loss-weight", type=float, default=0.0003)
    parser.add_argument("--baseline-kl-weight", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    return parser.parse_args(argv)


def _rollout_masks(probs: torch.Tensor, thresholds: list[float]) -> torch.Tensor:
    return torch.stack([(probs >= threshold).long() for threshold in thresholds], dim=1)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config)
    rlvr_cfg = config["rlvr"]
    if args.epochs is not None:
        rlvr_cfg["epochs"] = args.epochs
    run_dir = ensure_run_dir(args.run_name)
    train_loader, _ = build_penn_fudan_seg_loaders(config, limit_train=args.limit_train, limit_val=args.limit_val, batch_size=int(rlvr_cfg.get("batch_size", 1)))

    model = build_segmenter(config).to(device)
    baseline = build_segmenter(config).to(device)
    load_checkpoint(model, args.baseline, device)
    load_checkpoint(baseline, args.baseline, device)
    baseline.eval()
    for param in baseline.parameters():
        param.requires_grad = False
    set_segmentation_trainable(model, mode=str(rlvr_cfg.get("unfreeze", "head")))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(rlvr_cfg.get("lr", 0.0001)))
    thresholds = [float(x) for x in rlvr_cfg.get("rollout_thresholds", [0.35, 0.45, 0.55, 0.65])]
    reward_cfg = MaskRewardConfig(signal=args.signal, w_amp=args.reward_lambda, w_struct=args.struct_weight)

    history = []
    for epoch in range(1, int(rlvr_cfg.get("epochs", 1)) + 1):
        model.train()
        total_loss = 0.0
        total_reward = 0.0
        total_seen = 0
        for images, targets in tqdm(train_loader, desc=f"seg rlvr epoch {epoch}"):
            batch = torch.stack([image.to(device) for image in images], dim=0)
            gt_masks = torch.stack([target["mask"].to(device) for target in targets], dim=0)
            with torch.no_grad():
                baseline_logits = baseline(batch)["out"]
                baseline_probs = torch.softmax(baseline_logits, dim=1)[:, 1]
                action_masks = _rollout_masks(baseline_probs.cpu(), thresholds)
            rewards = []
            flat_actions = []
            for item_idx, image in enumerate(images):
                item_rewards = []
                for action in action_masks[item_idx]:
                    item_rewards.append(compute_mask_reward(image.cpu(), action.cpu().bool(), gt_masks[item_idx].cpu().bool(), reward_cfg))
                    flat_actions.append(action)
                rewards.extend(item_rewards)
            reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            advantages = normalize_advantages(reward_tensor, temperature=float(rlvr_cfg.get("reward_temperature", 1.0)))
            current_logits = model(batch)["out"]
            repeated_logits = current_logits.repeat_interleave(len(thresholds), dim=0)
            action_tensor = torch.stack(flat_actions, dim=0).to(device)
            policy_loss = signed_mask_policy_loss(repeated_logits, action_tensor, advantages)
            kl_loss = mask_kl_loss(current_logits, baseline_logits)
            loss = args.policy_loss_weight * policy_loss + args.baseline_kl_weight * kl_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(images)
            total_reward += float(reward_tensor.mean().item()) * len(images)
            total_seen += len(images)
        row = {"epoch": epoch, "loss": total_loss / max(1, total_seen), "reward_mean": total_reward / max(1, total_seen)}
        history.append(row)
        save_checkpoint(model, run_dir / "checkpoint_last.pth", {"epoch": epoch})
    save_json({"history": history, "signal": args.signal}, run_dir / "rlvr_result.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run compile check**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m py_compile spectral_segmentation_posttrain/train/posttrain_rlvr.py
```

Expected: exits with code 0.

- [ ] **Step 3: Commit**

```powershell
git add spectral_segmentation_posttrain/train/posttrain_rlvr.py
git commit -m "feat: add segmentation rlvr posttraining"
```

---

## Task 10: Add NNI Trial, Search Space, And Reports

**Files:**
- Create: `spectral_segmentation_posttrain/nni_seg_rlvr_trial.py`
- Create: `nni_configs/seg_rlvr_plan40_search_space.json`
- Create: `nni_configs/seg_rlvr_plan40_config.yml`
- Create: `run_nni_seg_plan40.bat`
- Create: `run_seg_plan40_smoke.bat`
- Create: `docs/seg_plan40_report.md`

- [ ] **Step 1: Create search space**

Create `nni_configs/seg_rlvr_plan40_search_space.json`:

```json
{
  "preset": {
    "_type": "choice",
    "_value": [
      {"name": "dice_only", "signal": "dice", "reward_lambda": 0.0, "struct_weight": 0.0, "policy_loss_weight": 0.0003, "baseline_kl_weight": 10.0},
      {"name": "amp_005", "signal": "amp", "reward_lambda": 0.05, "struct_weight": 0.0, "policy_loss_weight": 0.0003, "baseline_kl_weight": 10.0},
      {"name": "amp_010", "signal": "amp", "reward_lambda": 0.1, "struct_weight": 0.0, "policy_loss_weight": 0.0003, "baseline_kl_weight": 10.0},
      {"name": "shuffled_amp_010", "signal": "shuffled_amp", "reward_lambda": 0.1, "struct_weight": 0.0, "policy_loss_weight": 0.0003, "baseline_kl_weight": 10.0},
      {"name": "structure_010", "signal": "structure", "reward_lambda": 0.0, "struct_weight": 0.1, "policy_loss_weight": 0.0003, "baseline_kl_weight": 10.0},
      {"name": "shuffled_structure_010", "signal": "shuffled_structure", "reward_lambda": 0.0, "struct_weight": 0.1, "policy_loss_weight": 0.0003, "baseline_kl_weight": 10.0},
      {"name": "amp_structure_005_010", "signal": "amp_structure", "reward_lambda": 0.05, "struct_weight": 0.1, "policy_loss_weight": 0.0003, "baseline_kl_weight": 10.0},
      {"name": "shuffled_amp_structure_005_010", "signal": "shuffled_amp_structure", "reward_lambda": 0.05, "struct_weight": 0.1, "policy_loss_weight": 0.0003, "baseline_kl_weight": 10.0}
    ]
  }
}
```

- [ ] **Step 2: Create NNI config**

Create `nni_configs/seg_rlvr_plan40_config.yml`:

```yaml
experimentName: seg_plan40_rlvr
experimentWorkingDirectory: E:/CLIproject/RLimage/nni_experiments
trialCommand: E:/anaconda/01/envs/RLimage/python.exe -m spectral_segmentation_posttrain.nni_seg_rlvr_trial --config spectral_segmentation_posttrain/configs/penn_fudan_mvp.yaml --run-prefix seg_plan40 --baseline-epochs 5 --rlvr-epochs 3
trialCodeDirectory: E:/CLIproject/RLimage
searchSpaceFile: seg_rlvr_plan40_search_space.json
trialConcurrency: 1
maxTrialNumber: 8
maxExperimentDuration: 48h
tuner:
  name: GridSearch
trainingService:
  platform: local
```

- [ ] **Step 3: Create trial runner**

Create `spectral_segmentation_posttrain/nni_seg_rlvr_trial.py`:

```python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-prefix", default="seg_plan40")
    parser.add_argument("--params-json", default=None)
    parser.add_argument("--baseline-epochs", type=int, default=5)
    parser.add_argument("--rlvr-epochs", type=int, default=3)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    return parser.parse_args()


def _python() -> str:
    return sys.executable


def _run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _params(params_json: str | None) -> dict:
    if params_json:
        return json.loads(params_json)["preset"]
    try:
        import nni
        return dict(nni.get_next_parameter())["preset"]
    except Exception:
        return {"name": "dice_only", "signal": "dice", "reward_lambda": 0.0, "struct_weight": 0.0, "policy_loss_weight": 0.0003, "baseline_kl_weight": 10.0}


def main() -> None:
    args = parse_args()
    params = _params(args.params_json)
    baseline_run = f"{args.run_prefix}/baseline"
    baseline_ckpt = Path("runs") / baseline_run / "checkpoint_last.pth"
    limit_args = []
    if args.limit_train is not None:
        limit_args.extend(["--limit-train", str(args.limit_train)])
    if args.limit_val is not None:
        limit_args.extend(["--limit-val", str(args.limit_val)])
    if not baseline_ckpt.exists():
        _run([_python(), "-m", "spectral_segmentation_posttrain.train.train_baseline", "--config", args.config, "--run-name", baseline_run, "--epochs", str(args.baseline_epochs), *limit_args])
    rlvr_run = f"{args.run_prefix}/rlvr_{params['name']}"
    _run([
        _python(), "-m", "spectral_segmentation_posttrain.train.posttrain_rlvr",
        "--config", args.config,
        "--baseline", str(baseline_ckpt),
        "--run-name", rlvr_run,
        "--signal", str(params["signal"]),
        "--reward-lambda", str(params.get("reward_lambda", 0.0)),
        "--struct-weight", str(params.get("struct_weight", 0.0)),
        "--policy-loss-weight", str(params.get("policy_loss_weight", 0.0003)),
        "--baseline-kl-weight", str(params.get("baseline_kl_weight", 10.0)),
        "--epochs", str(args.rlvr_epochs),
        *limit_args,
    ])
    eval_run = f"{args.run_prefix}/eval_{params['name']}"
    _run([_python(), "-m", "spectral_segmentation_posttrain.eval.eval_segmenter", "--config", args.config, "--checkpoint", str(Path("runs") / rlvr_run / "checkpoint_last.pth"), "--run-name", eval_run])
    metrics_path = Path("runs") / eval_run / "eval_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    row = {"name": params["name"], **params, **metrics}
    out = Path("runs") / args.run_prefix / "seg_rlvr_results.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(out.read_text(encoding="utf-8") + json.dumps(row) + "\n" if out.exists() else json.dumps(row) + "\n", encoding="utf-8")
    try:
        import nni
        nni.report_final_result(row)
    except Exception:
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create run scripts**

Create `run_seg_plan40_smoke.bat`:

```bat
@echo off
cd /d E:\CLIproject\RLimage
E:\anaconda\01\envs\RLimage\python.exe -m spectral_segmentation_posttrain.train.train_baseline --config spectral_segmentation_posttrain/configs/penn_fudan_smoke.yaml --run-name seg_plan40_smoke/baseline --epochs 1 --limit-train 8 --limit-val 8
E:\anaconda\01\envs\RLimage\python.exe -m spectral_segmentation_posttrain.train.posttrain_rlvr --config spectral_segmentation_posttrain/configs/penn_fudan_smoke.yaml --baseline runs/seg_plan40_smoke/baseline/checkpoint_last.pth --run-name seg_plan40_smoke/rlvr_amp --signal amp --reward-lambda 0.05 --epochs 1 --limit-train 8 --limit-val 8
E:\anaconda\01\envs\RLimage\python.exe -m spectral_segmentation_posttrain.eval.eval_segmenter --config spectral_segmentation_posttrain/configs/penn_fudan_smoke.yaml --checkpoint runs/seg_plan40_smoke/rlvr_amp/checkpoint_last.pth --run-name seg_plan40_smoke/eval_amp --limit-val 8
```

Create `run_nni_seg_plan40.bat`:

```bat
@echo off
cd /d E:\CLIproject\RLimage
E:\anaconda\01\envs\RLimage\nni.exe experiment create --config nni_configs\seg_rlvr_plan40_config.yml --port 8110
```

- [ ] **Step 5: Create report scaffold**

Create `docs/seg_plan40_report.md`:

````markdown
# Plan 4.0 Semantic Segmentation RLVR Report

Run smoke:

```powershell
.\run_seg_plan40_smoke.bat
```

Run NNI:

```powershell
.\run_nni_seg_plan40.bat
```

Primary metrics:

```text
mIoU
Dice
Boundary F1
Pixel ECE
High-confidence false foreground
High-confidence false background
```
````

- [ ] **Step 6: Compile checks**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m py_compile spectral_segmentation_posttrain/nni_seg_rlvr_trial.py
```

Expected: exits with code 0.

- [ ] **Step 7: Commit**

```powershell
git add spectral_segmentation_posttrain/nni_seg_rlvr_trial.py nni_configs/seg_rlvr_plan40_search_space.json nni_configs/seg_rlvr_plan40_config.yml run_seg_plan40_smoke.bat run_nni_seg_plan40.bat docs/seg_plan40_report.md
git commit -m "feat: add segmentation rlvr experiment runner"
```

---

## Task 11: Smoke Test Plan 4.0

**Files:**
- Uses implemented files.

- [ ] **Step 1: Run unit tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_seg_penn_fudan_dataset.py tests/test_seg_metrics.py tests/test_seg_spectral_verifier.py tests/test_seg_policy_loss.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Run smoke script**

Run:

```powershell
.\run_seg_plan40_smoke.bat
```

Expected files:

```text
runs/seg_plan40_smoke/baseline/checkpoint_last.pth
runs/seg_plan40_smoke/rlvr_amp/checkpoint_last.pth
runs/seg_plan40_smoke/eval_amp/eval_metrics.json
```

- [ ] **Step 3: Inspect smoke metrics**

Run:

```powershell
Get-Content runs\seg_plan40_smoke\eval_amp\eval_metrics.json
```

Expected keys:

```text
miou
dice
boundary_f1
pixel_ece
high_conf_false_foreground
high_conf_false_background
```

- [ ] **Step 4: Commit smoke report if changed**

If `docs/seg_plan40_report.md` is updated with smoke numbers:

```powershell
git add docs/seg_plan40_report.md
git commit -m "docs: report segmentation rlvr smoke"
```

If no docs changed, skip this commit.

---

## Task 12: Run Plan 4.0 Phase 1 NNI Matrix

**Files:**
- Produces: `runs/seg_plan40/seg_rlvr_results.jsonl`
- Updates: `docs/seg_plan40_report.md`

- [ ] **Step 1: Launch NNI**

Run:

```powershell
.\run_nni_seg_plan40.bat
```

Expected:

```text
NNI starts on port 8110
maxTrialNumber = 8
```

- [ ] **Step 2: Wait for all trials**

Expected file:

```text
runs/seg_plan40/seg_rlvr_results.jsonl
```

Expected row count:

```powershell
(Get-Content runs\seg_plan40\seg_rlvr_results.jsonl).Count
```

Expected:

```text
8
```

- [ ] **Step 3: Compare real signals to shuffled controls**

Run:

```powershell
Get-Content runs\seg_plan40\seg_rlvr_results.jsonl
```

Manual decision gates:

```text
amp_010 should beat shuffled_amp_010 on Dice or Boundary F1 by >= 0.005 to claim amplitude causality.
structure_010 should beat shuffled_structure_010 on Boundary F1 by >= 0.005 to claim structure causality.
amp_structure_005_010 should beat both amp_005 and shuffled_amp_structure_005_010 to claim complementarity.
```

- [ ] **Step 4: Update report**

Write these rows into `docs/seg_plan40_report.md`:

```markdown
## Phase 1 Penn-Fudan Results

| method | mIoU | Dice | Boundary F1 | Pixel ECE | HC false FG | HC false BG |
|---|---:|---:|---:|---:|---:|---:|
```

Add interpretation:

```text
Passed gates:
Failed gates:
Next dataset:
```

- [ ] **Step 5: Commit report**

```powershell
git add docs/seg_plan40_report.md
git commit -m "docs: report Plan 4.0 segmentation matrix"
```

---

## Success Criteria

Plan 4.0 succeeds if:

```text
1. Penn-Fudan binary segmentation dataset loads RGB image + binary mask.
2. Supervised baseline trains and evaluates.
3. RLVR post-training runs without mIoU/Dice collapse.
4. Evaluation reports mIoU, Dice, Boundary F1, pixel ECE, and high-confidence pixel errors.
5. Phase 1 includes real-vs-shuffled controls for amplitude and structure.
6. The final report states whether verifier causality is supported, unproven, or negative.
```

---

## Interpretation Rules

| Outcome | Meaning | Next Step |
|---|---|---|
| Amp beats shuffled_amp on Dice/mIoU | Magnitude verifier has segmentation-level causal value | Move to VOC person subset |
| Struct beats shuffled_structure on Boundary F1 | Phase/edge structure verifier has boundary value | Search lower structure weights |
| Amp+Struct beats Amp and shuffled_amp_structure | MPLSeg-style decoupling is useful | Run larger VOC/COCO subset |
| Dice improves but shuffled controls also improve | RLVR shell helps, verifier causality unproven | Try learned verifier |
| Boundary F1 improves but mIoU drops | Structure reward is over-sharpening masks | Reduce `struct_weight` and add Dice floor |
| mIoU/Dice collapses | Policy update too strong | Lower `policy_loss_weight`, increase KL, freeze more layers |

---

## Self-Review Checklist

- Spec coverage: migrates project from detection to semantic segmentation.
- Main research line: keeps RLVR/verifiable reward, not ordinary supervised fine-tuning only.
- MPLSeg connection: magnitude handles foreground semantics; phase/structure handles boundaries.
- Controls: includes shuffled amplitude and shuffled structure.
- Minimal data path: Penn-Fudan masks already exist in current dataset.
- Larger data path: report decides whether to move to VOC/COCO subset.
- Stability: preserves KL-stabilized conservative post-training.
- Result format: report must compare real-vs-shuffled, not only baseline-vs-method.
