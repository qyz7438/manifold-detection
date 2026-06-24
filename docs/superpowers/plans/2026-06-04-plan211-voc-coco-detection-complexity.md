# Plan 2.11 VOC Spectral Signal Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether the handwritten ROI Fourier verifier becomes measurably useful on a harder VOC detection subset before any new detection post-training matrix is attempted.

**Architecture:** Reuse the existing `scripts/round28_train_eval.py` training and evaluation path instead of creating a new Round 2.11 training pipeline. Add a VOC dataset selector, train one VOC 3-class baseline, measure real `R_amp` TP/FP separation against a shuffled control, and use a fixed gate to decide whether a later Plan 2.12 should run VOC/COCO post-training.

**Tech Stack:** Python, PyTorch, TorchVision Faster R-CNN MobileNetV3 FPN, TorchVision VOCDetection XML layout, pytest, existing `spectral_detection_posttrain` metrics, matching, and spectral reward utilities.

---

## Why This Plan Was Repaired

The previous Plan 2.11 tried to build a full new VOC/COCO pipeline with separate baseline and post-training scripts. That made the plan larger than a Round 2.x sanity check and left the most important runner code underspecified.

This repaired Plan 2.11 is narrower:

```text
Question: Did Penn-Fudan fail because the task was too simple?
Test: Run one VOC baseline and measure whether R_amp separates TP from FP better than shuffled control.
Decision: Only write a later post-training plan if the spectral signal is real.
```

This plan intentionally does not perform second-stage post-training. The current evidence says handwritten detector-side spectral rewards are neutral on Penn-Fudan. Before spending more compute, Plan 2.11 must first prove that VOC provides a stronger signal.

---

## Experiment Count

Plan 2.11 contains **3 validation units** and **0 post-training groups**.

| ID | Name | Action | Output |
|---|---|---|---|
| E1 | `round211_voc_baseline_1ep` | Train/evaluate one VOC baseline through `round28_train_eval.py` | `runs/round211_voc_baseline_1ep/eval_metrics.json` |
| E2 | `round211_voc_spectral_gap` | Measure `R_amp` TP/FP gap and shuffled-control AUC | `runs/round211_voc_spectral_gap/spectral_gap_metrics.json` |
| E3 | `round211_voc_gate_summary` | Write the go/no-go decision report | `docs/round211_results.md` |

No NNI, no multi-seed, no COCO full run, no new detector post-training in this plan.

---

## Fixed Decision Gate

Use these thresholds exactly:

```text
promote_to_plan212 if:
  tp_fp_gap > 0.02
  and auc_real is not null
  and auc_shuffled is not null
  and auc_real - auc_shuffled >= 0.03

otherwise:
  do not run detection spectral post-training on VOC/COCO yet
```

Interpretation:

```text
gap around 0.008: same as Penn-Fudan, handwritten Fourier verifier remains unsupported
gap above 0.02 with shuffled failure: VOC complexity exposes a real signal, write Plan 2.12
insufficient TP/FP counts: rerun baseline with a larger VOC val limit before deciding
```

---

## File Map

- Create: `spectral_detection_posttrain/datasets/voc_detection.py`
  Loads VOC XML annotations into Faster R-CNN image/target pairs for selected classes.

- Modify: `spectral_detection_posttrain/datasets/__init__.py`
  Exports the VOC loader alongside the existing Penn-Fudan loader.

- Modify: `scripts/round28_train_eval.py`
  Adds `--dataset penn_fudan|voc`, VOC root/year/split/class CLI options, and loader selection.

- Create: `scripts/round211_voc_spectral_gap.py`
  Loads the VOC baseline checkpoint and computes real-vs-shuffled `R_amp` TP/FP gap.

- Create: `scripts/round211_run_gate.py`
  Runs the baseline and spectral-gap scripts in sequence.

- Create: `scripts/round211_summarize.py`
  Writes the Round 2.11 decision report.

- Create: `tests/test_round211_voc_dataset.py`
  Unit tests for VOC annotation parsing and dataset target format.

- Create: `tests/test_round211_spectral_gap.py`
  Unit tests for the shuffled-control and decision-gate helpers.

- Create after execution: `docs/round211_results.md`
  Human-readable results and decision.

---

## Task 1: VOC Dataset Loader

**Files:**
- Create: `spectral_detection_posttrain/datasets/voc_detection.py`
- Modify: `spectral_detection_posttrain/datasets/__init__.py`
- Test: `tests/test_round211_voc_dataset.py`

- [ ] **Step 1: Write the failing VOC dataset test**

Create `tests/test_round211_voc_dataset.py`:

```python
from pathlib import Path

import torch
from PIL import Image

from spectral_detection_posttrain.datasets.voc_detection import VOC_CLASS_TO_LABEL, VOCDetectionSubset, parse_voc_annotation


def _write_voc_sample(root: Path) -> None:
    voc_root = root / "VOCdevkit" / "VOC2007"
    (voc_root / "JPEGImages").mkdir(parents=True)
    (voc_root / "Annotations").mkdir(parents=True)
    (voc_root / "ImageSets" / "Main").mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(127, 127, 127)).save(voc_root / "JPEGImages" / "000001.jpg")
    (voc_root / "Annotations" / "000001.xml").write_text(
        """
<annotation>
  <object>
    <name>person</name>
    <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>10</xmax><ymax>20</ymax></bndbox>
  </object>
  <object>
    <name>bottle</name>
    <bndbox><xmin>3</xmin><ymin>4</ymin><xmax>8</xmax><ymax>9</ymax></bndbox>
  </object>
</annotation>
""",
        encoding="utf-8",
    )
    (voc_root / "ImageSets" / "Main" / "train.txt").write_text("000001\n", encoding="utf-8")


def test_parse_voc_annotation_keeps_selected_classes(tmp_path: Path) -> None:
    _write_voc_sample(tmp_path)
    xml_path = tmp_path / "VOCdevkit" / "VOC2007" / "Annotations" / "000001.xml"
    target = parse_voc_annotation(xml_path, classes=["person", "car", "dog"])

    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [VOC_CLASS_TO_LABEL["person"]]
    assert target["area"].tolist() == [171.0]


def test_voc_subset_returns_faster_rcnn_target(tmp_path: Path) -> None:
    _write_voc_sample(tmp_path)
    dataset = VOCDetectionSubset(tmp_path, year="2007", image_set="train", classes=["person", "car", "dog"], download=False)

    image, target = dataset[0]

    assert image.shape == (3, 32, 32)
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].dtype == torch.int64
    assert target["image_id"].item() == 0
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round211_voc_dataset.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'spectral_detection_posttrain.datasets.voc_detection'
```

- [ ] **Step 3: Implement the VOC dataset loader**

Create `spectral_detection_posttrain/datasets/voc_detection.py`:

```python
from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn.functional as F_torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import VOCDetection
from torchvision.transforms import functional as F_vision

from spectral_detection_posttrain.datasets.penn_fudan import detection_collate


VOC_CLASS_TO_LABEL = {"person": 1, "car": 2, "dog": 3}


def parse_voc_annotation(xml_path: str | Path, classes: list[str]) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes: list[list[float]] = []
    labels: list[int] = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in classes:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(VOC_CLASS_TO_LABEL[name])

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    if boxes_tensor.numel() == 0:
        boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)
    area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]).clamp_min(0) * (boxes_tensor[:, 3] - boxes_tensor[:, 1]).clamp_min(0)
    return {
        "boxes": boxes_tensor,
        "labels": labels_tensor,
        "area": area,
        "iscrowd": torch.zeros((len(labels_tensor),), dtype=torch.int64),
    }


class VOCDetectionSubset(Dataset):
    def __init__(
        self,
        root: str | Path,
        year: str = "2007",
        image_set: str = "train",
        classes: list[str] | None = None,
        download: bool = True,
        max_size: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.year = year
        self.image_set = image_set
        self.classes = classes or ["person", "car", "dog"]
        self.max_size = max_size
        if download:
            VOCDetection(str(self.root), year=year, image_set=image_set, download=True)
        self.voc_root = self.root / "VOCdevkit" / f"VOC{year}"
        ids_path = self.voc_root / "ImageSets" / "Main" / f"{image_set}.txt"
        if not ids_path.exists():
            raise FileNotFoundError(f"VOC image set file not found: {ids_path}")
        image_ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.samples: list[str] = []
        for image_id in image_ids:
            annotation_path = self.voc_root / "Annotations" / f"{image_id}.xml"
            target = parse_voc_annotation(annotation_path, self.classes)
            if len(target["boxes"]) > 0:
                self.samples.append(image_id)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_id = self.samples[index]
        image = Image.open(self.voc_root / "JPEGImages" / f"{image_id}.jpg").convert("RGB")
        image_tensor = F_vision.to_tensor(image)
        target = parse_voc_annotation(self.voc_root / "Annotations" / f"{image_id}.xml", self.classes)
        target["image_id"] = torch.tensor([index], dtype=torch.int64)
        if self.max_size is not None:
            image_tensor, target = _resize_image_and_target(image_tensor, target, int(self.max_size))
        return image_tensor, target


def _resize_image_and_target(image: torch.Tensor, target: dict, max_size: int) -> tuple[torch.Tensor, dict]:
    _, height, width = image.shape
    largest = max(height, width)
    if largest <= max_size:
        return image, target
    scale = max_size / float(largest)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    resized = F_torch.interpolate(
        image.unsqueeze(0),
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    scaled = dict(target)
    boxes = target["boxes"].clone()
    boxes[:, [0, 2]] *= new_width / float(width)
    boxes[:, [1, 3]] *= new_height / float(height)
    scaled["boxes"] = boxes
    scaled["area"] = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    return resized, scaled


def _subset_indices(length: int, limit: int | None, seed: int) -> list[int]:
    indices = list(range(length))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if limit is not None:
        indices = indices[: int(limit)]
    return indices


def build_voc_detection_loaders(
    config: dict,
    limit_train: int | None = None,
    limit_val: int | None = None,
    batch_size: int | None = None,
):
    data_cfg = config["data"]
    train_set = VOCDetectionSubset(
        root=data_cfg.get("root", "./data"),
        year=str(data_cfg.get("year", "2007")),
        image_set=str(data_cfg.get("train_set", "train")),
        classes=list(data_cfg.get("classes", ["person", "car", "dog"])),
        download=bool(data_cfg.get("download", True)),
        max_size=data_cfg.get("max_size"),
    )
    val_set = VOCDetectionSubset(
        root=data_cfg.get("root", "./data"),
        year=str(data_cfg.get("year", "2007")),
        image_set=str(data_cfg.get("val_set", "val")),
        classes=list(data_cfg.get("classes", ["person", "car", "dog"])),
        download=bool(data_cfg.get("download", True)),
        max_size=data_cfg.get("max_size"),
    )
    seed = int(config.get("seed", 42))
    train_subset = Subset(train_set, _subset_indices(len(train_set), limit_train, seed))
    val_subset = Subset(val_set, _subset_indices(len(val_set), limit_val, seed + 1))
    resolved_batch_size = int(batch_size or config["train"].get("batch_size", 2))
    num_workers = int(data_cfg.get("num_workers", 0))
    return (
        DataLoader(train_subset, batch_size=resolved_batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate),
        DataLoader(val_subset, batch_size=resolved_batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate),
    )
```

- [ ] **Step 4: Export the VOC loader**

Modify `spectral_detection_posttrain/datasets/__init__.py` to:

```python
from .penn_fudan import PennFudanDetectionDataset, build_penn_fudan_loaders
from .voc_detection import VOC_CLASS_TO_LABEL, VOCDetectionSubset, build_voc_detection_loaders

__all__ = [
    "PennFudanDetectionDataset",
    "VOC_CLASS_TO_LABEL",
    "VOCDetectionSubset",
    "build_penn_fudan_loaders",
    "build_voc_detection_loaders",
]
```

- [ ] **Step 5: Verify the VOC loader**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round211_voc_dataset.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add spectral_detection_posttrain/datasets/voc_detection.py spectral_detection_posttrain/datasets/__init__.py tests/test_round211_voc_dataset.py
git commit -m "feat: add VOC loader for Round 2.11"
```

---

## Task 2: Add VOC Selection To The Existing Round 2.8 Runner

**Files:**
- Modify: `scripts/round28_train_eval.py`

- [ ] **Step 1: Add the VOC loader import**

Replace the dataset import near the top of `scripts/round28_train_eval.py`:

```python
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders, build_voc_detection_loaders
```

- [ ] **Step 2: Add CLI arguments**

Add these parser arguments after `--seed`:

```python
    parser.add_argument("--dataset", default="penn_fudan", choices=["penn_fudan", "voc"])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--voc-year", default="2007")
    parser.add_argument("--voc-train-set", default="train")
    parser.add_argument("--voc-val-set", default="val")
    parser.add_argument("--voc-classes", default="person,car,dog")
```

- [ ] **Step 3: Build the dataset-aware config**

Replace the existing hardcoded `config = { ... }` block with:

```python
    voc_classes = [item.strip() for item in args.voc_classes.split(",") if item.strip()]
    num_classes = 2 if args.dataset == "penn_fudan" else len(voc_classes) + 1
    data_config = {
        "root": args.data_root,
        "download": True,
        "max_size": 320,
        "train_fraction": 0.8,
        "num_workers": 0,
    }
    if args.dataset == "voc":
        data_config.update({
            "year": args.voc_year,
            "train_set": args.voc_train_set,
            "val_set": args.voc_val_set,
            "classes": voc_classes,
        })

    config = {
        "seed": args.seed,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data": data_config,
        "model": {
            "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "pretrained": True,
            "num_classes": num_classes,
            "min_size": 320,
            "max_size": 320,
            "afm_channels": 256 if args.afm_type != "none" else 0,
            "afm_type": args.afm_type,
            "afm_residual_mode": args.afm_residual_mode,
        },
        "train": {"batch_size": 2, "lr": 0.003, "momentum": 0.9, "weight_decay": 0.0005},
        "matching": {"iou_threshold": 0.5, "score_threshold": 0.05},
        "eval": {"batch_size": 2, "high_conf_threshold": 0.7},
    }
```

- [ ] **Step 4: Select the loader**

Replace:

```python
    train_loader, val_loader = build_penn_fudan_loaders(config, limit_train=args.limit_train, limit_val=args.limit_val)
```

with:

```python
    if args.dataset == "voc":
        train_loader, val_loader = build_voc_detection_loaders(config, limit_train=args.limit_train, limit_val=args.limit_val)
    else:
        train_loader, val_loader = build_penn_fudan_loaders(config, limit_train=args.limit_train, limit_val=args.limit_val)
```

- [ ] **Step 5: Include dataset metadata in metrics**

In both metrics update blocks, add:

```python
                        "dataset": args.dataset, "data_root": args.data_root,
                        "voc_classes": voc_classes if args.dataset == "voc" else ["person"],
```

The final metrics JSON for VOC must contain:

```json
{
  "dataset": "voc",
  "voc_classes": ["person", "car", "dog"]
}
```

- [ ] **Step 6: Compile the runner**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m py_compile scripts/round28_train_eval.py
```

Expected:

```text
exit code 0
```

- [ ] **Step 7: Commit**

Run:

```powershell
git add scripts/round28_train_eval.py
git commit -m "feat: let Round 2.8 runner use VOC"
```

---

## Task 3: Spectral Gap Script

**Files:**
- Create: `scripts/round211_voc_spectral_gap.py`
- Test: `tests/test_round211_spectral_gap.py`

- [ ] **Step 1: Write gate-helper tests**

Create `tests/test_round211_spectral_gap.py`:

```python
from scripts.round211_voc_spectral_gap import decide_gate, shuffled_auc


def test_decide_gate_promotes_only_with_real_gap() -> None:
    decision = decide_gate(tp_fp_gap=0.031, auc_real=0.71, auc_shuffled=0.61, num_tp=30, num_fp=30)
    assert decision == "promote_to_plan212"


def test_decide_gate_rejects_small_gap() -> None:
    decision = decide_gate(tp_fp_gap=0.008, auc_real=0.55, auc_shuffled=0.53, num_tp=30, num_fp=30)
    assert decision == "stop_detection_spectral_reward"


def test_decide_gate_marks_insufficient_counts() -> None:
    decision = decide_gate(tp_fp_gap=0.04, auc_real=0.8, auc_shuffled=0.5, num_tp=4, num_fp=30)
    assert decision == "insufficient_predictions"


def test_shuffled_auc_is_deterministic() -> None:
    first = shuffled_auc([0.9, 0.8, 0.7], [0.2, 0.3, 0.4], seed=42)
    second = shuffled_auc([0.9, 0.8, 0.7], [0.2, 0.3, 0.4], seed=42)
    assert first == second
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round211_spectral_gap.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'scripts.round211_voc_spectral_gap'
```

- [ ] **Step 3: Implement the spectral gap script**

Create `scripts/round211_voc_spectral_gap.py`:

```python
"""Measure whether VOC exposes a real R_amp TP/FP signal for Round 2.11."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from spectral_detection_posttrain.datasets import build_voc_detection_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.spectral.spectral_reward import auc_tp_vs_fp, compute_prediction_rewards
from spectral_detection_posttrain.utils.io import load_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def shuffled_auc(tp_values: list[float], fp_values: list[float], seed: int) -> float | None:
    if not tp_values or not fp_values:
        return None
    combined = tp_values + fp_values
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(combined), generator=generator).tolist()
    shuffled = [combined[index] for index in order]
    shuffled_tp = shuffled[: len(tp_values)]
    shuffled_fp = shuffled[len(tp_values) :]
    return auc_tp_vs_fp(shuffled_tp, shuffled_fp)


def decide_gate(tp_fp_gap: float, auc_real: float | None, auc_shuffled: float | None, num_tp: int, num_fp: int) -> str:
    if num_tp < 10 or num_fp < 10:
        return "insufficient_predictions"
    if auc_real is None or auc_shuffled is None:
        return "insufficient_predictions"
    if tp_fp_gap > 0.02 and (auc_real - auc_shuffled) >= 0.03:
        return "promote_to_plan212"
    return "stop_detection_spectral_reward"


def build_voc_config(args: argparse.Namespace) -> dict:
    classes = [item.strip() for item in args.voc_classes.split(",") if item.strip()]
    return {
        "seed": args.seed,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data": {
            "root": args.data_root,
            "download": True,
            "max_size": 320,
            "year": args.voc_year,
            "train_set": args.voc_train_set,
            "val_set": args.voc_val_set,
            "classes": classes,
            "num_workers": 0,
        },
        "model": {
            "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "pretrained": False,
            "num_classes": len(classes) + 1,
            "min_size": 320,
            "max_size": 320,
            "afm_channels": 0,
            "afm_type": "none",
            "afm_residual_mode": "current",
        },
        "train": {"batch_size": 2, "lr": 0.003, "momentum": 0.9, "weight_decay": 0.0005},
        "matching": {"iou_threshold": 0.5, "score_threshold": 0.05},
        "eval": {"batch_size": 2, "high_conf_threshold": 0.7},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", default="round211_voc_spectral_gap")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--voc-year", default="2007")
    parser.add_argument("--voc-train-set", default="train")
    parser.add_argument("--voc-val-set", default="val")
    parser.add_argument("--voc-classes", default="person,car,dog")
    parser.add_argument("--limit-val", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    config = build_voc_config(args)
    device = resolve_device(config)
    _, val_loader = build_voc_detection_loaders(config, limit_train=1, limit_val=args.limit_val)
    model = build_detector(config).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    predictions: list[dict] = []
    targets_list: list[dict] = []
    tp_values: list[float] = []
    fp_values: list[float] = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=args.run_name):
            outputs = model([image.to(device) for image in images])
            for image, output, target in zip(images, outputs, targets):
                prediction = {key: value.detach().cpu() for key, value in output.items()}
                target_cpu = {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()}
                predictions.append(prediction)
                targets_list.append(target_cpu)
                rewards = compute_prediction_rewards(
                    image.cpu(),
                    prediction,
                    target_cpu,
                    iou_threshold=float(config["matching"]["iou_threshold"]),
                    score_threshold=float(config["matching"]["score_threshold"]),
                )
                tp_values.extend(rewards["tp_r_amp"])
                fp_values.extend(rewards["fp_r_amp"])

    tp_mean = float(torch.tensor(tp_values).mean().item()) if tp_values else 0.0
    fp_mean = float(torch.tensor(fp_values).mean().item()) if fp_values else 0.0
    gap = tp_mean - fp_mean
    auc_real = auc_tp_vs_fp(tp_values, fp_values)
    auc_control = shuffled_auc(tp_values, fp_values, seed=args.seed)
    decision = decide_gate(gap, auc_real, auc_control, len(tp_values), len(fp_values))
    detection_metrics = evaluate_detection_predictions(
        predictions,
        targets_list,
        iou_threshold=float(config["matching"]["iou_threshold"]),
        score_threshold=float(config["matching"]["score_threshold"]),
        high_conf_threshold=float(config["eval"]["high_conf_threshold"]),
    )

    result = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "dataset": "voc",
        "classes": config["data"]["classes"],
        "limit_val": args.limit_val,
        "num_tp": len(tp_values),
        "num_fp": len(fp_values),
        "tp_r_amp_mean": tp_mean,
        "fp_r_amp_mean": fp_mean,
        "tp_fp_gap": gap,
        "auc_real": auc_real,
        "auc_shuffled": auc_control,
        "auc_real_minus_shuffled": None if auc_real is None or auc_control is None else auc_real - auc_control,
        "decision": decision,
        "detection_metrics": detection_metrics,
    }
    output = Path("runs") / args.run_name / "spectral_gap_metrics.json"
    save_json(result, output)
    print(result)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify helper tests and compile**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round211_spectral_gap.py -v
E:\anaconda\01\envs\RLimage\python.exe -m py_compile scripts/round211_voc_spectral_gap.py
```

Expected:

```text
4 passed
exit code 0
```

- [ ] **Step 5: Commit**

Run:

```powershell
git add scripts/round211_voc_spectral_gap.py tests/test_round211_spectral_gap.py
git commit -m "feat: add VOC spectral gap gate"
```

---

## Task 4: Round 2.11 Gate Runner

**Files:**
- Create: `scripts/round211_run_gate.py`

- [ ] **Step 1: Implement the gate runner**

Create `scripts/round211_run_gate.py`:

```python
"""Run the repaired Plan 2.11 VOC spectral signal gate."""
from __future__ import annotations

import subprocess
import sys


PYTHON = sys.executable
BASELINE_RUN = "round211_voc_baseline_1ep"
GAP_RUN = "round211_voc_spectral_gap"


COMMANDS = [
    [
        PYTHON,
        "scripts/round28_train_eval.py",
        "--run-name",
        BASELINE_RUN,
        "--dataset",
        "voc",
        "--data-root",
        "./data",
        "--voc-year",
        "2007",
        "--voc-train-set",
        "train",
        "--voc-val-set",
        "val",
        "--voc-classes",
        "person,car,dog",
        "--afm-type",
        "none",
        "--afm-residual-mode",
        "current",
        "--trainable-mode",
        "full",
        "--epochs",
        "1",
        "--seed",
        "42",
        "--limit-train",
        "300",
        "--limit-val",
        "150",
    ],
    [
        PYTHON,
        "scripts/round211_voc_spectral_gap.py",
        "--checkpoint",
        f"runs/{BASELINE_RUN}/checkpoint_last.pth",
        "--run-name",
        GAP_RUN,
        "--data-root",
        "./data",
        "--voc-year",
        "2007",
        "--voc-train-set",
        "train",
        "--voc-val-set",
        "val",
        "--voc-classes",
        "person,car,dog",
        "--limit-val",
        "150",
        "--seed",
        "42",
    ],
    [PYTHON, "scripts/round211_summarize.py"],
]


def main() -> None:
    for command in COMMANDS:
        print("RUN", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Compile the runner**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m py_compile scripts/round211_run_gate.py
```

Expected:

```text
exit code 0
```

- [ ] **Step 3: Commit**

Run:

```powershell
git add scripts/round211_run_gate.py
git commit -m "feat: add Round 2.11 gate runner"
```

---

## Task 5: Summary Report

**Files:**
- Create: `scripts/round211_summarize.py`
- Create after execution: `docs/round211_results.md`

- [ ] **Step 1: Implement the summarizer**

Create `scripts/round211_summarize.py`:

```python
"""Summarize Round 2.11 and write the decision report."""
from __future__ import annotations

import json
from pathlib import Path


BASELINE_METRICS = Path("runs/round211_voc_baseline_1ep/eval_metrics.json")
GAP_METRICS = Path("runs/round211_voc_spectral_gap/spectral_gap_metrics.json")
REPORT = Path("docs/round211_results.md")


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    baseline = _load(BASELINE_METRICS)
    gap = _load(GAP_METRICS)
    decision = gap["decision"]
    if decision == "promote_to_plan212":
        verdict = "VOC produced a non-shuffled spectral signal. Write Plan 2.12 for VOC/COCO post-training."
    elif decision == "insufficient_predictions":
        verdict = "The gate did not have enough TP/FP samples. Rerun Plan 2.11 with a larger validation limit."
    else:
        verdict = "VOC did not rescue the handwritten Fourier verifier. Do not run detection spectral reward post-training yet."

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "\n".join(
            [
                "# Round 2.11 VOC Spectral Signal Gate Results",
                "",
                "## Baseline",
                "",
                "| metric | value |",
                "|---|---:|",
                f"| AP50 | {baseline.get('ap50', 0):.6f} |",
                f"| AP75 | {baseline.get('ap75', 0):.6f} |",
                f"| Precision | {baseline.get('precision', 0):.6f} |",
                f"| Recall | {baseline.get('recall', 0):.6f} |",
                f"| ECE | {baseline.get('ece', 0):.6f} |",
                f"| High-conf FP count | {baseline.get('high_conf_fp_count', 0)} |",
                "",
                "## Spectral Gate",
                "",
                "| metric | value |",
                "|---|---:|",
                f"| num TP | {gap.get('num_tp', 0)} |",
                f"| num FP | {gap.get('num_fp', 0)} |",
                f"| TP R_amp mean | {gap.get('tp_r_amp_mean', 0):.6f} |",
                f"| FP R_amp mean | {gap.get('fp_r_amp_mean', 0):.6f} |",
                f"| TP-FP gap | {gap.get('tp_fp_gap', 0):.6f} |",
                f"| real AUC | {gap.get('auc_real')} |",
                f"| shuffled AUC | {gap.get('auc_shuffled')} |",
                f"| real minus shuffled | {gap.get('auc_real_minus_shuffled')} |",
                "",
                "## Decision",
                "",
                f"Decision: `{decision}`",
                "",
                verdict,
                "",
                "## Gate Rule",
                "",
                "Promote only when `tp_fp_gap > 0.02` and `auc_real - auc_shuffled >= 0.03`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print({"report": str(REPORT), "decision": decision})


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Compile the summarizer**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m py_compile scripts/round211_summarize.py
```

Expected:

```text
exit code 0
```

- [ ] **Step 3: Commit**

Run:

```powershell
git add scripts/round211_summarize.py
git commit -m "feat: add Round 2.11 summary report"
```

---

## Task 6: Execute The Gate

**Files:**
- Reads: `scripts/round211_run_gate.py`
- Produces: `runs/round211_voc_baseline_1ep/eval_metrics.json`
- Produces: `runs/round211_voc_spectral_gap/spectral_gap_metrics.json`
- Produces: `docs/round211_results.md`

- [ ] **Step 1: Run the repaired Plan 2.11 gate**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/round211_run_gate.py
```

Expected files:

```text
runs/round211_voc_baseline_1ep/checkpoint_last.pth
runs/round211_voc_baseline_1ep/eval_metrics.json
runs/round211_voc_spectral_gap/spectral_gap_metrics.json
docs/round211_results.md
```

- [ ] **Step 2: Inspect the gate decision**

Run:

```powershell
Get-Content -Path docs\round211_results.md
```

Expected decision cases:

```text
Decision: `promote_to_plan212`
```

or:

```text
Decision: `stop_detection_spectral_reward`
```

or:

```text
Decision: `insufficient_predictions`
```

- [ ] **Step 3: Commit the results**

Run:

```powershell
git add docs/round211_results.md runs/round211_voc_baseline_1ep/eval_metrics.json runs/round211_voc_spectral_gap/spectral_gap_metrics.json
git commit -m "docs: report Round 2.11 VOC spectral gate"
```

---

## Success Criteria

Engineering success:

```text
1. VOC dataset unit tests pass.
2. round28_train_eval.py still supports Penn-Fudan by default.
3. round28_train_eval.py supports VOC through --dataset voc.
4. One VOC baseline checkpoint is produced.
5. spectral_gap_metrics.json contains num_tp, num_fp, tp_fp_gap, auc_real, auc_shuffled, and decision.
6. docs/round211_results.md states the fixed gate decision.
```

Scientific success:

```text
1. Plan 2.11 directly tests whether task complexity creates an R_amp TP/FP signal.
2. The real spectral signal is compared against a shuffled control.
3. No detection post-training is run before signal existence is established.
4. Promotion to Plan 2.12 is thresholded, not based on subjective inspection.
```

---

## Explicit Non-Goals

```text
No second-stage post-training in Plan 2.11.
No separate Round 2.11 baseline/post-training runner pair.
No NNI matrix.
No multi-seed.
No full COCO.
No claim that Fourier reward works in detection unless the gate passes.
```

---

## Follow-Up Rules

If the decision is `promote_to_plan212`:

```text
Write Plan 2.12 for VOC/COCO small-subset RLVR post-training.
Use detection-only, spatial-only, spatial+spectral, and shuffled-spectral controls.
Keep 2.12 small until the post-training path is stable.
```

If the decision is `stop_detection_spectral_reward`:

```text
Close the detector-side handwritten Fourier reward branch as unsupported.
Move effort to Plan 4.x semantic segmentation, where pixel masks provide denser spatial supervision.
```

If the decision is `insufficient_predictions`:

```text
Rerun only E1 and E2 with limit_val=300.
Do not add new reward designs before the sample-count problem is resolved.
```

---

## Self-Review Checklist

```text
Spec coverage: The plan answers whether VOC complexity reveals a spectral signal before VOC/COCO post-training.
Placeholder scan: The plan contains no underspecified runner task.
Type consistency: Dataset config, runner CLI arguments, and spectral-gap script all use the same class list and VOC split fields.
Scope check: The plan remains Round 2.x scale with one seed and one baseline epoch.
```
