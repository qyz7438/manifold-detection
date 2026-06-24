# Plan 2.19: Cross-Dataset / Cross-Backbone Small-Scale Validation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test mid06 MPLSeg-style AFM on 2 new combinations — VOC 3-class (person/car/dog) with MobileNetV3, and Penn-Fudan with ResNet50 — to check whether the AP75 improvement generalizes beyond the original dataset+backbone pair.

**Architecture:** Extend `build_detector` to support `model_name: fasterrcnn_resnet50_fpn`. Extend `round28_train_eval.py` to support `--dataset voc`. Reuse all existing AFM types (mplseg_mid, mplseg, etc). 12 groups total: 2 combinations × (baseline + mid06) × 3 seeds × 3 epochs. Deterministic (cudnn.benchmark=False). Single GPU, 8GB VRAM.

**Tech Stack:** Python, PyTorch, TorchVision Faster R-CNN, Penn-Fudan, VOC2012, existing `spectral_detection_posttrain` package.

---

## Starting Point

```
Penn-Fudan + MobileNetV3:  2.16 baseline + mid06 3ep ✓ already have data
Penn-Fudan + ResNet50:     NEW — backbone generalization test
VOC + MobileNetV3:         NEW — dataset generalization test
```

Available assets:
- VOC dataset loader: `spectral_detection_posttrain/datasets/voc_detection.py` (from Plan 2.11)
- VOC data: `E:/pythonProject1/VOCdevkit/VOC2012/` (17k images)
- AFM variants: mplseg_mid, mplseg, mplseg_weak, mplseg_frozen, etc (from Plans 2.14-2.15)
- Deterministic training: `cudnn.benchmark=False` (from Plan 2.16)

---

## Experiment Matrix (12 groups)

### Penn-Fudan + ResNet50 (6 groups)

| Group | AFM | Seed | Epoch | Question |
|-------|-----|------|-------|----------|
| r50_pf_baseline_s{42,123,456} | none | 42,123,456 | 3 | ResNet50 baseline on PF |
| r50_pf_mid06_s{42,123,456} | mplseg_mid | 42,123,456 | 3 | AFM on ResNet50 backbone |

### VOC 3-class + MobileNetV3 (6 groups)

| Group | AFM | Seed | Epoch | Question |
|-------|-----|------|-------|----------|
| voc_mob_baseline_s{42,123,456} | none | 42,123,456 | 3 | MobileNetV3 baseline on VOC |
| voc_mob_mid06_s{42,123,456} | mplseg_mid | 42,123,456 | 3 | AFM on multi-class VOC |

---

## Implementation

### Task 1: Extend `build_detector` to support ResNet50

**Files:**
- Modify: `spectral_detection_posttrain/models/build_detector.py`

`build_detector` currently hardcodes `fasterrcnn_mobilenet_v3_large_320_fpn`. Need to add:

```python
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn,
)

MODEL_REGISTRY = {
    "fasterrcnn_mobilenet_v3_large_320_fpn": (
        fasterrcnn_mobilenet_v3_large_320_fpn,
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    ),
    "fasterrcnn_resnet50_fpn": (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    ),
}
```

In `build_detector`:
```python
model_name = str(model_cfg.get("model_name", "fasterrcnn_mobilenet_v3_large_320_fpn"))
build_fn, weights_cls = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["fasterrcnn_mobilenet_v3_large_320_fpn"])

weights = weights_cls.DEFAULT if pretrained else None
model = build_fn(weights=weights, **model_kwargs)
```

AFM insertion logic (lines 53-74) is already model-agnostic — it just wraps `box_head`. No changes needed there.

For ResNet50, `afm_channels` should be set to 1024 (the box_head fc7 output dim of ResNet50). This is already handled by reading `config["model"]["afm_channels"]`. The config just needs to set the right value.

### Task 2: Extend `round28_train_eval.py` to support VOC dataset

**Files:**
- Modify: `scripts/round28_train_eval.py`

Current code:
```python
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
...
train_loader, val_loader = build_penn_fudan_loaders(config, ...)
```

Add:
```python
parser.add_argument("--dataset", default="penn_fudan", choices=["penn_fudan", "voc"])

# In main():
if args.dataset == "voc":
    from spectral_detection_posttrain.datasets.voc_detection import build_voc_detection_loaders
    train_loader, val_loader = build_voc_detection_loaders(config, ...)
else:
    train_loader, val_loader = build_penn_fudan_loaders(config, ...)
```

VOC needs `limit_train`/`limit_val` (from Plan 2.11: 300 train, 150 val). Use existing `--limit-train`/`--limit-val` args.

Also set `model.num_classes` correctly for VOC (4 classes: background + person/car/dog). This is already handled by `config["model"]["num_classes"]`.

### Task 3: Run Penn-Fudan + ResNet50 (6 groups)

**Files:**
- Create: `scripts/round219_r50_pf.py` (runner script)

For each seed (42, 123, 456), run:
```bash
# Baseline
python scripts/round28_train_eval.py \
  --run-name round219_r50_pf_baseline_s{seed} \
  --afm-type none --trainable-mode full \
  --epochs 3 --seed {seed} --dataset penn_fudan

# mid06
python scripts/round28_train_eval.py \
  --run-name round219_r50_pf_mid06_s{seed} \
  --afm-type mplseg_mid --trainable-mode full \
  --epochs 3 --seed {seed} --dataset penn_fudan
```

Config overrides (in runner script):
```python
config["model"]["model_name"] = "fasterrcnn_resnet50_fpn"
config["model"]["afm_channels"] = 1024  # ResNet50 box_head fc7 output dim
config["model"]["min_size"] = 480
config["model"]["max_size"] = 480
```

ResNet50 needs larger min_size (at least 480) because it uses a different FPN architecture with larger feature strides.

### Task 4: Run VOC + MobileNetV3 (6 groups)

**Files:**
- Create: `scripts/round219_voc_mob.py` (runner script)

For each seed (42, 123, 456), run:
```bash
# Baseline
python scripts/round28_train_eval.py \
  --run-name round219_voc_mob_baseline_s{seed} \
  --afm-type none --trainable-mode full \
  --epochs 3 --seed {seed} --dataset voc \
  --limit-train 300 --limit-val 150

# mid06
python scripts/round28_train_eval.py \
  --run-name round219_voc_mob_mid06_s{seed} \
  --afm-type mplseg_mid --trainable-mode full \
  --epochs 3 --seed {seed} --dataset voc \
  --limit-train 300 --limit-val 150
```

Config overrides:
```python
config["model"]["num_classes"] = 4  # background + person/car/dog
config["model"]["afm_channels"] = 256  # MobileNetV3 box_head
config["model"]["max_size"] = 480
config["data"]["root"] = "E:/pythonProject1"
config["data"]["year"] = "2012"
config["data"]["classes"] = ["person", "car", "dog"]
config["data"]["download"] = False
```

### Task 5: Summarize

**Files:**
- Create: `scripts/round219_summarize.py`

Read all 12 eval_metrics.json, compute per-config means, send to Feishu.

---

## Success Criteria

```text
1. ResNet50 baseline on Penn-Fudan trains without crash (AP50 > 0.5).
2. mid06 on ResNet50 shows any AP75 improvement over baseline (ΔAP75 > 0).
3. VOC baseline on MobileNetV3 trains without crash (AP50 > 0.2).
4. mid06 on VOC shows positive signal on at least 2/3 seeds.
5. All 12 groups complete, no NaN.
```

## Interpretation Rules

```text
If mid06 AP75 improves on BOTH new combinations (2/2):
  → AFM generalizes across backbones and datasets. Worth COCO-scale validation.

If mid06 AP75 improves on 1/2:
  → AFM benefit depends on backbone or dataset. The failing combination tells us where the limit is.

If mid06 AP75 does not improve on either:
  → AFM benefit is specific to Penn-Fudan + MobileNetV3. Not generalizable at current scale.
```
