# Round 2.6 MicroAFM Feature-Map FFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify that FFT/iFFT as an in-network feature transform (not external verifier) can run without numerical instability, gradient collapse, or loss explosion on Penn-Fudan.

**Architecture:** Create a micro `MicroAFM` module that runs `rFFT2 → magnitude gate + phase residual → iRFFT2` on a single feature map. Test gradient flow on random tensors, then insert it before the Faster R-CNN ROI classifier head and verify a single supervised training step completes without NaN and AP50 stays within 5% of baseline after 1 epoch.

**Tech Stack:** Python, PyTorch, TorchVision Faster R-CNN, Penn-Fudan, pytest, existing `spectral_detection_posttrain` package.

---

## Why Round 2.6 Exists

Round 2.3-2.5 proved the RLVR shell (KL=10, policy=0.0003, signed objective) is stable but handcrafted spectral verifiers have no causal signal. The root cause: R_amp is a scalar reward computed on image ROI crops, not a feature-level transform.

MPLSeg's key insight is different: FFT is an **in-network module**, not an external verifier. The AFM block applies `rFFT2 → magnitude gate + phase residual → iRFFT2` directly on intermediate feature maps, with full gradient flow through the FFT/iFFT operations.

Round 2.6 tests the most minimal version of this idea: does the FFT/iFFT path work at all in our pipeline?

---

## Non-Goals

- Do not add RLVR policy loss, KL regularization, or any reward signal.
- Do not add multi-layer FPN-style frequency fusion.
- Do not run NNI or any hyperparameter search.
- Do not modify the detector backbone, RPN, or box regression head.
- Do not claim MPLSeg reproduction.
- Do not build segmentation pipeline.

---

## File Map

- Create: `spectral_detection_posttrain/models/micro_afm.py`
  MicroAFM module with magnitude gate and phase residual.

- Create: `tests/test_micro_afm.py`
  Gradient flow, FFT identity, numerical stability, and training sanity tests.

- Create: `scripts/round26_micro_afm_smoke.py`
  Smoke script: insert MicroAFM into detector, run 1 epoch supervised training, eval.

- Modify: `spectral_detection_posttrain/models/build_detector.py`
  Add optional `afm_channels` parameter; when set, insert MicroAFM before ROI predictor.

---

## Task 1: MicroAFM Module And Gradient Tests

**Files:**
- Create: `spectral_detection_posttrain/models/micro_afm.py`
- Create: `tests/test_micro_afm.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_micro_afm.py`:

```python
import pytest
import torch

from spectral_detection_posttrain.models.micro_afm import MicroAFM


def test_micro_afm_output_shape_matches_input():
    afm = MicroAFM(channels=16)
    x = torch.randn(2, 16, 32, 32)
    out = afm(x)

    assert out.shape == x.shape


@pytest.mark.parametrize("channels", [16, 64, 256])
def test_micro_afm_gradient_flows_to_input(channels: int):
    afm = MicroAFM(channels=channels)
    x = torch.randn(1, channels, 16, 16, requires_grad=True)

    out = afm(x)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_micro_afm_preserves_input_when_gate_and_res_zero():
    afm = MicroAFM(channels=16)
    for param in afm.parameters():
        param.data.zero_()
    x = torch.randn(2, 16, 32, 32)
    out = afm(x)

    assert torch.allclose(out, x, atol=1e-4)


def test_micro_afm_no_nan_on_wide_value_range():
    afm = MicroAFM(channels=16)
    x = torch.randn(2, 16, 32, 32) * 100.0

    out = afm(x)

    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_micro_afm.py -v
```

Expected: fails because `spectral_detection_posttrain.models.micro_afm` does not exist.

- [ ] **Step 3: Implement MicroAFM**

Create `spectral_detection_posttrain/models/micro_afm.py`:

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroAFM(nn.Module):
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
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_micro_afm.py -v
```

Expected: all 5 tests pass (1 parametrized = 3 + 3 = 5 passed).

- [ ] **Step 5: Commit**

```bash
git add spectral_detection_posttrain/models/micro_afm.py tests/test_micro_afm.py
git commit -m "feat: add MicroAFM feature-map FFT module"
```

---

## Task 2: Insert MicroAFM Into Detector And Verify Training Sanity

**Files:**
- Modify: `spectral_detection_posttrain/models/build_detector.py`
- Create: `scripts/round26_micro_afm_smoke.py`

- [ ] **Step 1: Add MicroAFM injection point to build_detector**

In `spectral_detection_posttrain/models/build_detector.py`, after building the model and before returning, add:

```python
afm_channels = int(model_cfg.get("afm_channels", 0))
if afm_channels > 0:
    from spectral_detection_posttrain.models.micro_afm import MicroAFM

    afm = MicroAFM(channels=afm_channels)
    original_forward = model.roi_heads.box_head.forward

    def _patched_forward(x):
        features = original_forward(x)
        return afm(features)

    model.roi_heads.box_head.forward = _patched_forward
```

- [ ] **Step 2: Create smoke training script**

Create `scripts/round26_micro_afm_smoke.py`:

```python
"""Smoke test: MicroAFM inserted into detector, 1 epoch supervised training, eval."""
from __future__ import annotations

import argparse

import torch
import yaml
from tqdm import tqdm

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def _to_device(targets: list[dict], device: torch.device) -> list[dict]:
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--afm-channels", type=int, default=0)
    parser.add_argument("--run-name", default="round26_micro_afm_smoke")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    args = parser.parse_args()

    config = {
        "seed": 42, "device": "auto",
        "data": {"root": "./data", "download": True, "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
        "model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn", "pretrained": True,
                  "num_classes": 2, "min_size": 320, "max_size": 320,
                  "afm_channels": args.afm_channels},
        "train": {"batch_size": 2, "epochs": args.epochs, "lr": 0.003, "momentum": 0.9, "weight_decay": 0.0005},
        "matching": {"iou_threshold": 0.5, "score_threshold": 0.05},
        "eval": {"batch_size": 2, "high_conf_threshold": 0.7},
    }
    set_seed(int(config["seed"]))
    device = resolve_device(config)
    run_dir = ensure_run_dir(args.run_name)

    train_loader, val_loader = build_penn_fudan_loaders(config, limit_train=args.limit_train, limit_val=args.limit_val)

    model = build_detector(config).to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["train"]["lr"]), momentum=float(config["train"]["momentum"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for images, targets in tqdm(train_loader, desc=f"epoch {epoch}"):
            images = [img.to(device) for img in images]
            targets = _to_device(targets, device)
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(images)
            total_seen += len(images)

        avg_loss = total_loss / max(1, total_seen)
        print(f"epoch {epoch}: train_loss={avg_loss:.4f}")
        save_checkpoint(model, run_dir / "checkpoint_last.pth", {"epoch": epoch})

    model.eval()
    predictions = []
    targets_list = []
    for images, batch_targets in val_loader:
        outputs = model([img.to(device) for img in images])
        predictions.extend([{k: v.detach().cpu() for k, v in output.items()} for output in outputs])
        targets_list.extend([{k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in t.items()} for t in batch_targets])

    metrics = evaluate_detection_predictions(
        predictions, targets_list,
        iou_threshold=float(config["matching"]["iou_threshold"]),
        score_threshold=float(config["matching"]["score_threshold"]),
        high_conf_threshold=float(config["eval"]["high_conf_threshold"]),
    )
    metrics["afm_channels"] = args.afm_channels
    save_json(metrics, run_dir / "eval_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run baseline smoke (AFM disabled)**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/round26_micro_afm_smoke.py --afm-channels 0 --run-name round26_baseline_smoke --limit-train 4 --limit-val 4
```

Expected: `runs/round26_baseline_smoke/eval_metrics.json` shows AP50 around 0.8-0.88 (smoke with 4 images, normal variance).

- [ ] **Step 4: Run MicroAFM smoke (AFM enabled)**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/round26_micro_afm_smoke.py --afm-channels 256 --run-name round26_afm_smoke --limit-train 4 --limit-val 4
```

Expected: `runs/round26_afm_smoke/eval_metrics.json` shows AP50 within 5% of baseline smoke. No NaN in loss values.

- [ ] **Step 5: Commit smoke script**

```bash
git add spectral_detection_posttrain/models/build_detector.py scripts/round26_micro_afm_smoke.py
git commit -m "feat: add MicroAFM injection point and smoke training script"
```

---

## Task 3: Run Full Penn-Fudan 1-Epoch Comparison

**Files:**
- Uses `scripts/round26_micro_afm_smoke.py`

- [ ] **Step 1: Run full baseline 1-epoch (AFM disabled)**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/round26_micro_afm_smoke.py --afm-channels 0 --run-name round26_baseline_full --epochs 1
```

Expected: AP50 around 0.86-0.88.

- [ ] **Step 2: Run full MicroAFM 1-epoch (AFM enabled, channels=256)**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/round26_micro_afm_smoke.py --afm-channels 256 --run-name round26_afm_full_256 --epochs 1
```

Expected: AP50 within 5% of baseline, no NaN in loss.

- [ ] **Step 3: Write short report to `docs/round26_results.md`**

```markdown
# Round 2.6 MicroAFM Results

| config | AP50 | AP75 | ECE | num_pred |
|--------|------|------|-----|----------|
| baseline | | | | |
| MicroAFM-256 | | | | |

## Verdict
- [ ] Gradient flow verified
- [ ] No NaN/Inf
- [ ] AP50 within 5% of baseline
- [ ] Ready for RLVR integration in Round 2.7
```

- [ ] **Step 4: Commit report**

```bash
git add docs/round26_results.md
git commit -m "docs: report Round 2.6 MicroAFM sanity check"
```

---

## Success Criteria

```text
1. tests/test_micro_afm.py all pass (gradient flow, identity, NaN safety, shape preservation).
2. MicroAFM smoke training completes without NaN loss.
3. MicroAFM full 1-epoch AP50 within 5% of baseline 1-epoch AP50.
4. No crash in FFT/iFFT path during training or eval.
```

If any criterion fails, do not add more complexity. Debug the MicroAFM gradient path or reduce channels.

---

## Verdict

Round 2.6 is a **technology readiness check**, not a method experiment. Success means "FFT/iFFT as in-network module is safe and does not destroy detector training." It gives us permission to replace external verifiers with internal frequency transforms in Round 2.7.
