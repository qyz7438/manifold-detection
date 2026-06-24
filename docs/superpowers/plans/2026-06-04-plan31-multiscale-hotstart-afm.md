# Plan 3.1: Hot-Start + Multi-Scale MicroAFM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Round 2.6's single-layer, zero-initialized MicroAFM with a hot-started, multi-scale AFM across FPN levels, verifying that FFT/iFFT as an in-network transform can match or exceed baseline AP50 on Penn-Fudan.

**Architecture:** Task 1 upgrades MicroAFM to MultiScaleAFM — apply independent AFM blocks at FPN P2-P5 feature levels, then sum-fuse. Task 2 implements hot-start: freeze backbone + box_head, train only AFM blocks with L2 identity regularization for K warmup epochs, then unfreeze for joint fine-tuning. Task 3 runs a 4-group comparison (baseline / singleAFM / multiAFM cold / multiAFM hot-start) on full Penn-Fudan.

**Tech Stack:** Python, PyTorch, TorchVision Faster R-CNN MobileNetV3, Penn-Fudan, pytest, existing `spectral_detection_posttrain` package.

---

## Why Plan 3.1 Exists

Round 2.6 proved FFT/iFFT in a feature path doesn't crash, but two limitations were exposed:

1. **Zero-init kills AP50 by 3.6%**. sigmoid(0)=0.5 halves the magnitude spectrum, scrambling pretrained box_head statistics. Hot-start lets AFM first learn a near-identity transform before joint training.

2. **Single-scale misses MPLSeg's core design**. MPLSeg applies AFM at 4 FPN scales (P2-P5) with progressive upsampling fusion. Single box_roi_pool features (7×7) have too little spatial resolution for meaningful frequency decomposition. Multi-scale FPN features (200×200 down to 25×25) provide rich multi-frequency information.

---

## Non-Goals

- Do not add RLVR policy loss or reward signals.
- Do not modify the detector backbone weights.
- Do not run NNI or hyperparameter search.
- Do not change FFT/iFFT core logic (already verified in Round 2.6).

---

## File Map

- Modify: `spectral_detection_posttrain/models/micro_afm.py`
  Replace single-scale MicroAFM with MultiScaleAFM supporting per-FPN-level AFM blocks.

- Modify: `spectral_detection_posttrain/models/build_detector.py`
  Replace box_head hook with FPN-level injection via backbone forward patching.

- Create: `scripts/round31_hotstart_smoke.py`
  Hot-start training: freeze backbone+box_head, train only AFM with detection loss.

- Create: `tests/test_multiscale_afm.py`
  Tests for MultiScaleAFM: output shapes match input, gradient flow, fusion correctness.

- Modify: `tests/test_micro_afm.py`
  Update existing tests for renamed MicroAFM → AFMBlock.

---

## Task 1: Multi-Scale AFM With Per-Level Blocks

**Files:**
- Modify: `spectral_detection_posttrain/models/micro_afm.py`
- Create: `tests/test_multiscale_afm.py`
- Modify: `tests/test_micro_afm.py`

- [ ] **Step 1: Write failing multi-scale tests**

Create `tests/test_multiscale_afm.py`:

```python
import pytest
import torch

from spectral_detection_posttrain.models.micro_afm import MultiScaleAFM


def test_multiscale_afm_preserves_shape_per_level():
    afm = MultiScaleAFM(channels=[256, 512, 1024, 1024])
    x_p2 = torch.randn(1, 256, 200, 200)
    x_p5 = torch.randn(1, 1024, 25, 25)
    out_p2 = afm(x_p2, level=0)
    out_p5 = afm(x_p5, level=3)
    assert out_p2.shape == x_p2.shape
    assert out_p5.shape == x_p5.shape


def test_multiscale_afm_gradient_flows_per_level():
    afm = MultiScaleAFM(channels=[256, 512])
    x = torch.randn(1, 256, 32, 32, requires_grad=True)
    out = afm(x, level=0)
    loss = out.mean()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_multiscale_afm_blocks_are_independent():
    afm = MultiScaleAFM(channels=[16, 32])
    x0 = torch.randn(1, 16, 8, 8, requires_grad=True)
    x1 = torch.randn(1, 32, 4, 4, requires_grad=True)
    out0 = afm(x0, level=0)
    out1 = afm(x1, level=1)
    (out0.mean() + out1.mean()).backward()
    assert x0.grad is not None and x1.grad is not None
    assert not torch.equal(x0.grad, x1.grad)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_multiscale_afm.py -v
```

Expected: fails because `MultiScaleAFM` does not exist.

- [ ] **Step 3: Rename MicroAFM → AFMBlock, add MultiScaleAFM**

In `spectral_detection_posttrain/models/micro_afm.py`:

```python
class AFMBlock(nn.Module):  # was MicroAFM
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mag_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.phase_res = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.Tanh(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Tanh(),
        )
        self._eps = 1e-3
        self._init_near_zero()

    def _init_near_zero(self):
        for module in [self.mag_gate, self.phase_res]:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    nn.init.zeros_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        F_repr = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(F_repr)
        pha = torch.angle(F_repr + self._eps)
        mag_log = torch.log1p(mag)
        mag = mag * (1.0 - self.mag_gate(mag_log))
        pha = pha + self.phase_res(pha)
        F_mod = mag * torch.exp(1j * pha)
        out = torch.fft.irfft2(F_mod, s=x.shape[-2:], norm="ortho")
        return F.relu(out, inplace=True)


class MultiScaleAFM(nn.Module):
    def __init__(self, channels: list[int], reduction: int = 4):
        super().__init__()
        self.blocks = nn.ModuleDict({
            str(i): AFMBlock(channels=c, reduction=reduction)
            for i, c in enumerate(channels)
        })

    def forward(self, feature_map: torch.Tensor, level: int) -> torch.Tensor:
        return self.blocks[str(level)](feature_map)


MicroAFM = AFMBlock  # backward compat
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_micro_afm.py tests/test_multiscale_afm.py -v
```

Expected: all tests pass (5 from micro_afm + 3 from multiscale = 8 passed).

- [ ] **Step 5: Commit**

```bash
git add spectral_detection_posttrain/models/micro_afm.py tests/test_multiscale_afm.py tests/test_micro_afm.py
git commit -m "feat: rename MicroAFM to AFMBlock, add MultiScaleAFM"
```

---

## Task 2: Hot-Start Training Pipeline

**Files:**
- Modify: `spectral_detection_posttrain/models/build_detector.py`
- Create: `scripts/round31_hotstart_smoke.py`

- [ ] **Step 1: Add MultiScaleAFM FPN injection to build_detector**

Replace the existing `afm_channels` injection block in `spectral_detection_posttrain/models/build_detector.py` with:

```python
    afm_channels = int(model_cfg.get("afm_channels", 0))
    afm_fpn = bool(model_cfg.get("afm_fpn", False))

    if afm_fpn:
        from spectral_detection_posttrain.models.micro_afm import MultiScaleAFM
        fpn_channels = [256, 256, 256, 256]
        multi_afm = MultiScaleAFM(channels=fpn_channels)
        original_backbone_forward = model.backbone.forward

        def _patched_backbone_forward(x):
            features = original_backbone_forward(x)
            if isinstance(features, torch.Tensor):
                features = {"0": features}
            out = {}
            for i, (key, feat) in enumerate(features.items()):
                out[key] = multi_afm(feat, level=i)
            return out

        model.backbone.forward = _patched_backbone_forward
        model._multi_afm = multi_afm

    elif afm_channels > 0:
        ...  # existing single-scale code
```

- [ ] **Step 2: Create hot-start smoke script**

Create `scripts/round31_hotstart_smoke.py`:

```python
"""Hot-start MultiScaleAFM: freeze backbone+box_head, train AFM, then joint FT."""
from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def _to_device(targets, device):
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]


def _eval_model(model, val_loader, device, run_dir, suffix=""):
    model.eval()
    predictions, targets_list = [], []
    for images, batch_targets in val_loader:
        outputs = model([img.to(device) for img in images])
        predictions.extend([{k: v.detach().cpu() for k, v in output.items()} for output in outputs])
        targets_list.extend([{k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in t.items()} for t in batch_targets])
    metrics = evaluate_detection_predictions(
        predictions, targets_list, iou_threshold=0.5, score_threshold=0.05, high_conf_threshold=0.7,
    )
    save_json(metrics, run_dir / f"eval_metrics{suffix}.json")
    print(metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--afm-mode", default="none", choices=["none", "single", "multi"])
    parser.add_argument("--hotstart-epochs", type=int, default=2)
    parser.add_argument("--full-epochs", type=int, default=3)
    parser.add_argument("--run-name", default="round31_hotstart")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    args = parser.parse_args()

    config = {
        "seed": 42, "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data": {"root": "./data", "download": True, "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
        "model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn", "pretrained": True,
                  "num_classes": 2, "min_size": 320, "max_size": 320,
                  "afm_channels": 256 if args.afm_mode == "single" else 0,
                  "afm_fpn": args.afm_mode == "multi"},
        "train": {"batch_size": 2, "epochs": 1, "lr": 0.003, "momentum": 0.9, "weight_decay": 0.0005},
        "matching": {"iou_threshold": 0.5, "score_threshold": 0.05},
        "eval": {"batch_size": 2, "high_conf_threshold": 0.7},
    }
    set_seed(config["seed"])
    device = resolve_device(config)
    run_dir = ensure_run_dir(args.run_name)

    train_loader, val_loader = build_penn_fudan_loaders(
        config, limit_train=args.limit_train, limit_val=args.limit_val,
    )
    model = build_detector(config).to(device)

    if args.afm_mode in ("single", "multi"):
        # Phase 1: hot-start — freeze all, train only AFM blocks
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "_multi_afm"):
            for param in model._multi_afm.parameters():
                param.requires_grad = True

        opt_hs = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0003)
        for epoch in range(1, args.hotstart_epochs + 1):
            model.train()
            total_loss = 0.0
            total_seen = 0
            for images, targets in tqdm(train_loader, desc=f"hotstart epoch {epoch}"):
                images = [img.to(device) for img in images]
                targets = _to_device(targets, device)
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                opt_hs.zero_grad(set_to_none=True)
                loss.backward()
                opt_hs.step()
                total_loss += float(loss.item()) * len(images)
                total_seen += len(images)
            print(f"hotstart epoch {epoch}: loss={total_loss / max(1, total_seen):.4f}")

        # Phase 2: joint fine-tuning
        for param in model.roi_heads.box_head.parameters():
            param.requires_grad = True
        for param in model.roi_heads.box_predictor.parameters():
            param.requires_grad = True

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.003, momentum=0.9, weight_decay=0.0005,
    )
    for epoch in range(1, args.full_epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for images, targets in tqdm(train_loader, desc=f"joint epoch {epoch}"):
            images = [img.to(device) for img in images]
            targets = _to_device(targets, device)
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(images)
            total_seen += len(images)
        print(f"joint epoch {epoch}: loss={total_loss / max(1, total_seen):.4f}")
        save_checkpoint(model, run_dir / "checkpoint_last.pth", {"epoch": epoch})

    _eval_model(model, val_loader, device, run_dir)
    print(f"run: {args.run_name} mode: {args.afm_mode}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run smoke: singleAFM hot-start + multiAFM hot-start**

Run:

```powershell
cd E:/CLIproject/RLimage
$env:PYTHONPATH = "E:/CLIproject/RLimage"
E:\anaconda\01\envs\RLimage\python.exe scripts/round31_hotstart_smoke.py --afm-mode single --hotstart-epochs 2 --full-epochs 1 --run-name round31_single_hot_smoke --limit-train 4 --limit-val 4
E:\anaconda\01\envs\RLimage\python.exe scripts/round31_hotstart_smoke.py --afm-mode multi --hotstart-epochs 2 --full-epochs 1 --run-name round31_multi_hot_smoke --limit-train 4 --limit-val 4
```

Expected: both complete without NaN. AP50 may vary widely on 4-image smoke (just verify no crash).

- [ ] **Step 4: Commit**

```bash
git add spectral_detection_posttrain/models/build_detector.py scripts/round31_hotstart_smoke.py
git commit -m "feat: add MultiScaleAFM FPN injection and hot-start training"
```

---

## Task 3: Full Penn-Fudan 4-Group Comparison

**Files:**
- Uses `scripts/round31_hotstart_smoke.py`

- [ ] **Step 1: Run 4 groups**

```powershell
cd E:/CLIproject/RLimage; $env:PYTHONPATH = "E:/CLIproject/RLimage"

# G1: baseline (no AFM, 3 epochs)
E:\anaconda\01\envs\RLimage\python.exe scripts/round31_hotstart_smoke.py --afm-mode none --hotstart-epochs 0 --full-epochs 3 --run-name round31_group1_baseline

# G2: single AFM, cold start (3 epochs)
E:\anaconda\01\envs\RLimage\python.exe scripts/round31_hotstart_smoke.py --afm-mode single --hotstart-epochs 0 --full-epochs 3 --run-name round31_group2_single_cold

# G3: multi AFM, cold start (3 epochs)
E:\anaconda\01\envs\RLimage\python.exe scripts/round31_hotstart_smoke.py --afm-mode multi --hotstart-epochs 0 --full-epochs 3 --run-name round31_group3_multi_cold

# G4: multi AFM, hot-start (2 warmup + 3 joint = 5 total)
E:\anaconda\01\envs\RLimage\python.exe scripts/round31_hotstart_smoke.py --afm-mode multi --hotstart-epochs 2 --full-epochs 3 --run-name round31_group4_multi_hot
```

Expected: 4 `eval_metrics.json` files produced.

- [ ] **Step 2: Build comparison table**

```powershell
PYTHONPATH=E:/CLIproject/RLimage E:\anaconda\01\envs\RLimage\python.exe -c "
import json
for g in ['group1_baseline','group2_single_cold','group3_multi_cold','group4_multi_hot']:
    m = json.load(open(f'runs/round31_{g}/eval_metrics.json'))
    print(f'{g}: AP50={m[\"ap50\"]:.4f} AP75={m[\"ap75\"]:.4f} prec={m[\"precision\"]:.4f} preds={m[\"num_predictions\"]} FP={m[\"high_conf_fp_count\"]}')
"
```

- [ ] **Step 3: Write results report**

Create `docs/round31_results.md`:

```markdown
# Plan 3.1 Multi-Scale Hot-Start AFM Results

| group | AP50 | AP75 | precision | num_pred | high_conf_FP | ECE |
|-------|------|------|-----------|----------|-------------|-----|
| baseline | | | | | | |
| single_cold | | | | | | |
| multi_cold | | | | | | |
| multi_hot | | | | | | |

## Verdict
- [ ] Multi-scale beats single-scale on AP75
- [ ] Hot-start beats cold-start on precision
- [ ] Best AFM config within 2% of baseline AP50
```

- [ ] **Step 4: Commit**

```bash
git add docs/round31_results.md
git commit -m "docs: report Plan 3.1 multi-scale hot-start AFM"
```

---

## Success Criteria

```text
1. MultiScaleAFM tests pass (gradient flow, shape preservation, independent blocks).
2. Multi-scale FPN injection does not crash on 4-level feature dict.
3. Hot-start training completes without NaN loss.
4. At least one AFM config achieves AP50 >= baseline - 2% on full Penn-Fudan.
5. Multi-scale configuration beats single-scale on AP75 or precision.
```

---

## Plan 位置

`E:/CLIproject/RLimage/docs/superpowers/plans/2026-06-04-plan31-multiscale-hotstart-afm.md`
