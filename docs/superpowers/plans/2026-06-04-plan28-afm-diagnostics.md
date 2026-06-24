# Plan 2.8 AFM Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose whether AFM improves detection through useful frequency features or merely perturbs proposal/score distributions, using a normal small-scale Round 2.x experiment suite.

**Architecture:** Keep Penn-Fudan + TorchVision Faster R-CNN as the small validation environment. Add explicit AFM variants, detector-level frozen parity, fair same-script comparisons, residual-form ablations, trainable-parameter ablations, and AP75/threshold/IoU diagnostics. Do not run large multi-seed or patch robustness sweeps here; those belong to Plan 3.x after this diagnostic closes.

**Tech Stack:** Python, PyTorch, TorchVision Faster R-CNN MobileNetV3 FPN, Penn-Fudan, pytest, existing `spectral_detection_posttrain` package.

---

## Why Plan 2.8 Exists

Round 2.7 fixed the obvious identity bug in AFM:

```text
old: mag = mag * (1 - sigmoid(gate)); sigmoid(0)=0.5
fixed: out = x + residual_scale * freq_out; residual_scale=0
```

That fixed AP50 parity:

```text
round27 baseline:  AP50=0.8770 AP75=0.6524 precision=0.6667 pred=123
round27 AFM fixed: AP50=0.8761 AP75=0.5367 precision=0.5226 pred=155
```

But AP75 and precision still degraded, while predictions increased. Plan 2.8 therefore answers four questions:

1. Is identity AFM truly a no-op at detector level before training?
2. Was Plan 3.1's apparent gain caused by useful frequency modeling or by non-identity feature perturbation?
3. Does the current residual form `x + s * freq_out` create feature-scale drift?
4. Is the AP75/precision drop caused by localization error, duplicate predictions, or score-threshold calibration?

---

## Experiment Matrix

Plan 2.8 is a normal small-scale diagnostic, not a minimal run. It contains:

```text
1 frozen parity experiment, no training
9 training/eval groups, each 1 epoch on full Penn-Fudan split
3 post-hoc diagnostics, no retraining
1 markdown result report
```

### Frozen Parity

| group | train | purpose |
|-------|-------|---------|
| parity_no_afm | no | baseline detector predictions |
| parity_identity_afm | no | same weights loaded into identity AFM detector |

Pass condition:

```text
max_box_diff <= 1e-2
max_score_diff <= 1e-2
num_prediction_diff == 0 on the same images
```

### Training Groups

| group | AFM type | residual mode | trainable mode | epochs |
|-------|----------|---------------|----------------|--------|
| round28_g01_baseline_full | none | none | full | 1 |
| round28_g02_old_afm_full | old | old | full | 1 |
| round28_g03_identity_current_full | identity | current | full | 1 |
| round28_g04_identity_delta_full | identity | delta | full | 1 |
| round28_g05_identity_norm_delta_full | identity | norm_delta | full | 1 |
| round28_g06_baseline_box_head_only | none | none | box_head_only | 1 |
| round28_g07_identity_current_afm_only | identity | current | afm_only | 1 |
| round28_g08_identity_current_afm_box_head | identity | current | afm_box_head | 1 |
| round28_g09_identity_delta_afm_box_head | identity | delta | afm_box_head | 1 |

Interpretation rules:

```text
If old_afm_full improves AP75 but precision drops, the Plan 3.1 gain is likely perturbation-driven.
If identity_delta_full beats identity_current_full on AP75/precision, current residual form caused scale drift.
If afm_only degrades precision, AFM itself disrupts features.
If afm_box_head recovers AP75, the box head needs adaptation to AFM features.
If baseline_box_head_only also changes prediction count strongly, training-scope effects dominate the AFM interpretation.
```

### Post-Hoc Diagnostics

Run these diagnostics on all 9 trained groups:

```text
score thresholds: 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90
IoU thresholds: AP50 and AP75 plus matched TP IoU statistics
localization diagnostics: center error, width/height error, duplicate prediction count
```

---

## Non-Goals

- Do not add RLVR objective.
- Do not add GRPO or policy optimization.
- Do not run NNI.
- Do not run multi-seed or multi-epoch large matrix.
- Do not migrate to segmentation in this plan.
- Do not claim AFM is effective unless AP75, precision, and prediction count all support the claim.

---

## File Map

- Modify: `spectral_detection_posttrain/models/micro_afm.py`
  Add explicit `OldAFMBlock`, identity AFM residual modes, and variant factory.

- Modify: `spectral_detection_posttrain/models/build_detector.py`
  Add config-driven AFM selection: `afm_type`, `afm_residual_mode`.

- Create: `tests/test_round28_afm_variants.py`
  Validate old AFM is non-identity, identity AFM is identity, delta variants preserve shape and gradients.

- Create: `scripts/round28_frozen_parity.py`
  Load the same detector state into no-AFM and identity-AFM models and compare predictions.

- Create: `scripts/round28_train_eval.py`
  Single-run training/eval entry point for one matrix group.

- Create: `scripts/round28_run_matrix.py`
  Execute the normal 9-group Round 2.8 matrix.

- Create: `scripts/round28_diagnostics.py`
  Threshold curves, matched IoU statistics, center/size error, and duplicate counts.

- Create: `scripts/round28_summarize.py`
  Produce `runs/round28_summary.json` and `docs/round28_results.md`.

---

## Task 1: Add AFM Variants

**Files:**
- Modify: `spectral_detection_posttrain/models/micro_afm.py`
- Create: `tests/test_round28_afm_variants.py`

- [ ] **Step 1: Add failing tests for AFM variants**

Create `tests/test_round28_afm_variants.py`:

```python
import torch

from spectral_detection_posttrain.models.micro_afm import AFMBlock, OldAFMBlock, build_afm_block


def test_identity_afm_is_identity_at_init():
    afm = AFMBlock(channels=16, residual_mode="current")
    x = torch.randn(2, 16, 24, 24)
    assert torch.allclose(afm(x), x, atol=1e-3)


def test_old_afm_is_not_identity_at_init():
    afm = OldAFMBlock(channels=16)
    x = torch.randn(2, 16, 24, 24)
    diff = (afm(x) - x).abs().mean().item()
    assert diff > 1e-3


def test_delta_residual_is_identity_at_init():
    afm = AFMBlock(channels=16, residual_mode="delta")
    x = torch.randn(2, 16, 24, 24)
    assert torch.allclose(afm(x), x, atol=1e-3)


def test_norm_delta_residual_is_identity_at_init():
    afm = AFMBlock(channels=16, residual_mode="norm_delta")
    x = torch.randn(2, 16, 24, 24)
    assert torch.allclose(afm(x), x, atol=1e-3)


def test_afm_factory_builds_expected_variants():
    assert isinstance(build_afm_block("old", channels=16), OldAFMBlock)
    assert isinstance(build_afm_block("identity", channels=16, residual_mode="delta"), AFMBlock)
    assert build_afm_block("none", channels=16) is None


def test_delta_variants_have_gradient_flow():
    for residual_mode in ["current", "delta", "norm_delta"]:
        afm = AFMBlock(channels=16, residual_mode=residual_mode)
        x = torch.randn(1, 16, 16, 16, requires_grad=True)
        out = afm(x)
        out.mean().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round28_afm_variants.py -v
```

Expected:

```text
FAIL because OldAFMBlock/build_afm_block/residual_mode do not exist yet.
```

- [ ] **Step 3: Implement `OldAFMBlock`, residual modes, and factory**

In `spectral_detection_posttrain/models/micro_afm.py`, update the file to include:

```python
class OldAFMBlock(nn.Module):
    """Round 2.6/3.1 non-identity AFM kept only for controlled diagnostics."""

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
        for module in [self.mag_gate, self.phase_res]:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    nn.init.zeros_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_repr = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(f_repr)
        pha = torch.angle(f_repr + self._eps)
        mag = mag * (1.0 - self.mag_gate(torch.log1p(mag)))
        pha = pha + self.phase_res(pha)
        f_mod = mag * torch.exp(1j * pha)
        out = torch.fft.irfft2(f_mod, s=x.shape[-2:], norm="ortho")
        return F.relu(out, inplace=True)
```

Update `AFMBlock.__init__`:

```python
def __init__(self, channels: int, reduction: int = 4, residual_mode: str = "current"):
    super().__init__()
    if residual_mode not in {"current", "delta", "norm_delta"}:
        raise ValueError(f"Unknown AFM residual_mode: {residual_mode}")
    self.residual_mode = residual_mode
```

Update `AFMBlock.forward` after `freq_out`:

```python
if self.residual_mode == "current":
    residual = freq_out
elif self.residual_mode == "delta":
    residual = freq_out - x
else:
    residual = freq_out - x
    denom = residual.detach().flatten(1).norm(dim=1).clamp_min(1e-6)
    view_shape = [residual.shape[0]] + [1] * (residual.ndim - 1)
    residual = residual / denom.view(*view_shape)
return x + self.residual_scale * residual
```

Add factory:

```python
def build_afm_block(afm_type: str, channels: int, residual_mode: str = "current") -> nn.Module | None:
    if afm_type == "none":
        return None
    if afm_type == "old":
        return OldAFMBlock(channels=channels)
    if afm_type == "identity":
        return AFMBlock(channels=channels, residual_mode=residual_mode)
    raise ValueError(f"Unknown afm_type: {afm_type}")
```

- [ ] **Step 4: Run AFM variant tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_micro_afm.py tests/test_multiscale_afm.py tests/test_round28_afm_variants.py -v
```

Expected:

```text
All tests pass.
```

- [ ] **Step 5: Commit**

```bash
git add spectral_detection_posttrain/models/micro_afm.py tests/test_round28_afm_variants.py
git commit -m "feat: add AFM variants for Round 2.8 diagnostics"
```

---

## Task 2: Make Detector Build Config-Driven For AFM Type

**Files:**
- Modify: `spectral_detection_posttrain/models/build_detector.py`
- Test: `tests/test_round28_afm_variants.py`

- [ ] **Step 1: Extend detector builder config**

In `spectral_detection_posttrain/models/build_detector.py`, replace the single-AFM injection block:

```python
elif afm_channels > 0:
    from spectral_detection_posttrain.models.micro_afm import MicroAFM
```

with:

```python
elif afm_channels > 0:
    from spectral_detection_posttrain.models.micro_afm import build_afm_block

    afm_type = str(model_cfg.get("afm_type", "identity"))
    afm_residual_mode = str(model_cfg.get("afm_residual_mode", "current"))
    afm = build_afm_block(
        afm_type=afm_type,
        channels=afm_channels,
        residual_mode=afm_residual_mode,
    )
    if afm is None:
        return model
```

Keep the existing `AFMThenHead` wrapper, but set metadata:

```python
model._afm_type = afm_type
model._afm_residual_mode = afm_residual_mode
```

- [ ] **Step 2: Run tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round28_afm_variants.py -v
```

Expected:

```text
All tests pass.
```

- [ ] **Step 3: Commit**

```bash
git add spectral_detection_posttrain/models/build_detector.py
git commit -m "feat: configure AFM detector variants"
```

---

## Task 3: Add Frozen Detector Parity Script

**Files:**
- Create: `scripts/round28_frozen_parity.py`

- [ ] **Step 1: Create script**

Create `scripts/round28_frozen_parity.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def _config(afm_type: str = "none") -> dict:
    return {
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data": {"root": "./data", "download": True, "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
        "model": {
            "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "pretrained": True,
            "num_classes": 2,
            "min_size": 320,
            "max_size": 320,
            "afm_channels": 256 if afm_type != "none" else 0,
            "afm_type": afm_type,
            "afm_residual_mode": "current",
        },
        "train": {"batch_size": 2},
        "eval": {"batch_size": 1},
    }


def _strip_afm_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in state_dict.items() if ".afm." not in key}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="round28_frozen_parity")
    parser.add_argument("--limit-val", type=int, default=8)
    args = parser.parse_args()

    set_seed(42)
    cfg_no = _config("none")
    device = resolve_device(cfg_no)
    run_dir = ensure_run_dir(args.run_name)

    model_no = build_detector(cfg_no).to(device).eval()
    cfg_afm = _config("identity")
    model_afm = build_detector(cfg_afm).to(device).eval()
    missing, unexpected = model_afm.load_state_dict(_strip_afm_state(model_no.state_dict()), strict=False)

    _, val_loader = build_penn_fudan_loaders(cfg_no, limit_val=args.limit_val, batch_size=1)
    max_box_diff = 0.0
    max_score_diff = 0.0
    max_count_diff = 0
    per_image = []

    with torch.no_grad():
        for image_idx, (images, _) in enumerate(val_loader):
            images = [image.to(device) for image in images]
            pred_no = model_no(images)[0]
            pred_afm = model_afm(images)[0]
            count_diff = abs(len(pred_no["scores"]) - len(pred_afm["scores"]))
            max_count_diff = max(max_count_diff, count_diff)
            common = min(len(pred_no["scores"]), len(pred_afm["scores"]))
            box_diff = 0.0
            score_diff = 0.0
            if common > 0:
                box_diff = float((pred_no["boxes"][:common] - pred_afm["boxes"][:common]).abs().max().item())
                score_diff = float((pred_no["scores"][:common] - pred_afm["scores"][:common]).abs().max().item())
            max_box_diff = max(max_box_diff, box_diff)
            max_score_diff = max(max_score_diff, score_diff)
            per_image.append({
                "image_idx": image_idx,
                "box_diff": box_diff,
                "score_diff": score_diff,
                "count_no_afm": int(len(pred_no["scores"])),
                "count_afm": int(len(pred_afm["scores"])),
            })

    metrics = {
        "max_box_diff": max_box_diff,
        "max_score_diff": max_score_diff,
        "max_count_diff": max_count_diff,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "per_image": per_image,
        "pass": max_box_diff <= 1e-2 and max_score_diff <= 1e-2 and max_count_diff == 0,
    }
    save_json(metrics, Path(run_dir) / "parity_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run parity script**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/round28_frozen_parity.py --run-name round28_frozen_parity --limit-val 8
```

Expected:

```text
runs/round28_frozen_parity/parity_metrics.json exists and pass=true.
```

- [ ] **Step 3: Commit**

```bash
git add scripts/round28_frozen_parity.py
git commit -m "test: add Round 2.8 frozen AFM parity check"
```

---

## Task 4: Add One-Group Train/Eval Script

**Files:**
- Create: `scripts/round28_train_eval.py`

- [ ] **Step 1: Create trainable-mode helpers**

Create `scripts/round28_train_eval.py` with:

```python
from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def _to_device(targets: list[dict], device: torch.device) -> list[dict]:
    return [{key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()} for target in targets]


def _set_trainable(model: torch.nn.Module, mode: str) -> None:
    for param in model.parameters():
        param.requires_grad = mode == "full"
    if mode == "full":
        return
    if mode == "box_head_only":
        for param in model.roi_heads.box_head.parameters():
            param.requires_grad = True
        for param in model.roi_heads.box_predictor.parameters():
            param.requires_grad = True
        return
    if mode == "afm_only":
        if hasattr(model.roi_heads.box_head, "afm"):
            for param in model.roi_heads.box_head.afm.parameters():
                param.requires_grad = True
        return
    if mode == "afm_box_head":
        if hasattr(model.roi_heads.box_head, "afm"):
            for param in model.roi_heads.box_head.afm.parameters():
                param.requires_grad = True
        for param in model.roi_heads.box_head.parameters():
            param.requires_grad = True
        for param in model.roi_heads.box_predictor.parameters():
            param.requires_grad = True
        return
    raise ValueError(f"Unknown trainable mode: {mode}")
```

- [ ] **Step 2: Add main training/eval body**

Append:

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--afm-type", default="none", choices=["none", "old", "identity"])
    parser.add_argument("--afm-residual-mode", default="current", choices=["current", "delta", "norm_delta"])
    parser.add_argument("--trainable-mode", default="full", choices=["full", "box_head_only", "afm_only", "afm_box_head"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    args = parser.parse_args()

    config = {
        "seed": args.seed,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data": {"root": "./data", "download": True, "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
        "model": {
            "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "pretrained": True,
            "num_classes": 2,
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

    set_seed(args.seed)
    device = resolve_device(config)
    run_dir = ensure_run_dir(args.run_name)
    train_loader, val_loader = build_penn_fudan_loaders(config, limit_train=args.limit_train, limit_val=args.limit_val)
    model = build_detector(config).to(device)
    _set_trainable(model, args.trainable_mode)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"No trainable params for mode={args.trainable_mode}")

    optimizer = torch.optim.SGD(
        trainable_params,
        lr=float(config["train"]["lr"]),
        momentum=float(config["train"]["momentum"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for images, targets in tqdm(train_loader, desc=f"{args.run_name} epoch {epoch}"):
            images = [image.to(device) for image in images]
            targets = _to_device(targets, device)
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(images)
            total_seen += len(images)
        avg_loss = total_loss / max(1, total_seen)
        history.append({"epoch": epoch, "train_loss": avg_loss})
        save_checkpoint(model, run_dir / "checkpoint_last.pth", {"epoch": epoch})

    model.eval()
    predictions = []
    targets_list = []
    for images, batch_targets in val_loader:
        outputs = model([image.to(device) for image in images])
        predictions.extend([{key: value.detach().cpu() for key, value in output.items()} for output in outputs])
        targets_list.extend([
            {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()}
            for target in batch_targets
        ])

    metrics = evaluate_detection_predictions(
        predictions,
        targets_list,
        iou_threshold=float(config["matching"]["iou_threshold"]),
        score_threshold=float(config["matching"]["score_threshold"]),
        high_conf_threshold=float(config["eval"]["high_conf_threshold"]),
    )
    metrics.update({
        "run_name": args.run_name,
        "afm_type": args.afm_type,
        "afm_residual_mode": args.afm_residual_mode,
        "trainable_mode": args.trainable_mode,
        "epochs": args.epochs,
        "seed": args.seed,
        "history": history,
    })
    save_json(metrics, run_dir / "eval_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke run one tiny group**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/round28_train_eval.py --run-name round28_smoke_identity_delta --afm-type identity --afm-residual-mode delta --trainable-mode afm_box_head --epochs 1 --limit-train 4 --limit-val 4
```

Expected:

```text
runs/round28_smoke_identity_delta/eval_metrics.json exists.
```

- [ ] **Step 4: Commit**

```bash
git add scripts/round28_train_eval.py
git commit -m "feat: add Round 2.8 train eval runner"
```

---

## Task 5: Add Matrix Runner

**Files:**
- Create: `scripts/round28_run_matrix.py`

- [ ] **Step 1: Create matrix runner**

Create `scripts/round28_run_matrix.py`:

```python
from __future__ import annotations

import subprocess
import sys


GROUPS = [
    ("round28_g01_baseline_full", "none", "current", "full"),
    ("round28_g02_old_afm_full", "old", "current", "full"),
    ("round28_g03_identity_current_full", "identity", "current", "full"),
    ("round28_g04_identity_delta_full", "identity", "delta", "full"),
    ("round28_g05_identity_norm_delta_full", "identity", "norm_delta", "full"),
    ("round28_g06_baseline_box_head_only", "none", "current", "box_head_only"),
    ("round28_g07_identity_current_afm_only", "identity", "current", "afm_only"),
    ("round28_g08_identity_current_afm_box_head", "identity", "current", "afm_box_head"),
    ("round28_g09_identity_delta_afm_box_head", "identity", "delta", "afm_box_head"),
]


def main() -> None:
    for run_name, afm_type, residual_mode, trainable_mode in GROUPS:
        cmd = [
            sys.executable,
            "scripts/round28_train_eval.py",
            "--run-name", run_name,
            "--afm-type", afm_type,
            "--afm-residual-mode", residual_mode,
            "--trainable-mode", trainable_mode,
            "--epochs", "1",
            "--seed", "42",
        ]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run matrix**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/round28_run_matrix.py
```

Expected:

```text
9 eval files exist:
runs/round28_g01_baseline_full/eval_metrics.json
runs/round28_g02_old_afm_full/eval_metrics.json
runs/round28_g03_identity_current_full/eval_metrics.json
runs/round28_g04_identity_delta_full/eval_metrics.json
runs/round28_g05_identity_norm_delta_full/eval_metrics.json
runs/round28_g06_baseline_box_head_only/eval_metrics.json
runs/round28_g07_identity_current_afm_only/eval_metrics.json
runs/round28_g08_identity_current_afm_box_head/eval_metrics.json
runs/round28_g09_identity_delta_afm_box_head/eval_metrics.json
```

- [ ] **Step 3: Commit**

```bash
git add scripts/round28_run_matrix.py
git commit -m "feat: add Round 2.8 AFM diagnostic matrix"
```

---

## Task 6: Add Threshold And Localization Diagnostics

**Files:**
- Create: `scripts/round28_diagnostics.py`

- [ ] **Step 1: Create diagnostics script**

Create `scripts/round28_diagnostics.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.matching.box_iou import box_iou
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import load_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]


def _config(afm_type: str, residual_mode: str) -> dict:
    return {
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data": {"root": "./data", "download": True, "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
        "model": {
            "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "pretrained": True,
            "num_classes": 2,
            "min_size": 320,
            "max_size": 320,
            "afm_channels": 256 if afm_type != "none" else 0,
            "afm_type": afm_type,
            "afm_residual_mode": residual_mode,
        },
        "train": {"batch_size": 2},
        "matching": {"iou_threshold": 0.5, "score_threshold": 0.05},
        "eval": {"batch_size": 2, "high_conf_threshold": 0.7},
    }


def _localization_stats(predictions: list[dict], targets: list[dict], score_threshold: float) -> dict:
    matched_ious = []
    center_errors = []
    size_errors = []
    duplicates = 0
    for prediction, target in zip(predictions, targets):
        boxes = prediction.get("boxes", torch.empty((0, 4)))
        scores = prediction.get("scores", torch.empty((0,)))
        keep = scores >= score_threshold
        boxes = boxes[keep]
        gt_boxes = target.get("boxes", torch.empty((0, 4)))
        if len(boxes) == 0 or len(gt_boxes) == 0:
            continue
        ious = box_iou(boxes, gt_boxes)
        best_iou, best_gt = ious.max(dim=1)
        gt_match_counts: dict[int, int] = {}
        for pred_idx, iou in enumerate(best_iou.tolist()):
            if iou < 0.5:
                continue
            gt_idx = int(best_gt[pred_idx].item())
            gt_match_counts[gt_idx] = gt_match_counts.get(gt_idx, 0) + 1
            pred_box = boxes[pred_idx]
            gt_box = gt_boxes[gt_idx]
            pred_center = torch.stack([(pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2])
            gt_center = torch.stack([(gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2])
            pred_size = torch.stack([(pred_box[2] - pred_box[0]).clamp_min(1), (pred_box[3] - pred_box[1]).clamp_min(1)])
            gt_size = torch.stack([(gt_box[2] - gt_box[0]).clamp_min(1), (gt_box[3] - gt_box[1]).clamp_min(1)])
            center_errors.append(float(torch.norm(pred_center - gt_center).item()))
            size_errors.append(float((pred_size - gt_size).abs().mean().item()))
            matched_ious.append(float(iou))
        duplicates += sum(max(0, count - 1) for count in gt_match_counts.values())
    def mean(values: list[float]) -> float:
        return float(sum(values) / max(1, len(values)))
    return {
        "matched_iou_mean": mean(matched_ious),
        "matched_iou_count": len(matched_ious),
        "center_error_mean": mean(center_errors),
        "size_error_mean": mean(size_errors),
        "duplicate_predictions": duplicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--afm-type", required=True, choices=["none", "old", "identity"])
    parser.add_argument("--afm-residual-mode", default="current", choices=["current", "delta", "norm_delta"])
    args = parser.parse_args()

    set_seed(42)
    config = _config(args.afm_type, args.afm_residual_mode)
    device = resolve_device(config)
    model = build_detector(config).to(device)
    load_checkpoint(model, Path("runs") / args.run_name / "checkpoint_last.pth", device)
    model.eval()
    _, val_loader = build_penn_fudan_loaders(config)
    predictions = []
    targets = []
    with torch.no_grad():
        for images, batch_targets in val_loader:
            outputs = model([image.to(device) for image in images])
            predictions.extend([{key: value.detach().cpu() for key, value in output.items()} for output in outputs])
            targets.extend([
                {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()}
                for target in batch_targets
            ])

    threshold_curve = {}
    for threshold in THRESHOLDS:
        metrics = evaluate_detection_predictions(predictions, targets, score_threshold=threshold)
        metrics.update(_localization_stats(predictions, targets, score_threshold=threshold))
        threshold_curve[str(threshold)] = metrics
    save_json(threshold_curve, Path("runs") / args.run_name / "round28_diagnostics.json")
    print(json.dumps(threshold_curve, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run diagnostics for all groups**

Run these commands after Task 5 completes:

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/round28_diagnostics.py --run-name round28_g01_baseline_full --afm-type none
E:\anaconda\01\envs\RLimage\python.exe scripts/round28_diagnostics.py --run-name round28_g02_old_afm_full --afm-type old
E:\anaconda\01\envs\RLimage\python.exe scripts/round28_diagnostics.py --run-name round28_g03_identity_current_full --afm-type identity --afm-residual-mode current
E:\anaconda\01\envs\RLimage\python.exe scripts/round28_diagnostics.py --run-name round28_g04_identity_delta_full --afm-type identity --afm-residual-mode delta
E:\anaconda\01\envs\RLimage\python.exe scripts/round28_diagnostics.py --run-name round28_g05_identity_norm_delta_full --afm-type identity --afm-residual-mode norm_delta
E:\anaconda\01\envs\RLimage\python.exe scripts/round28_diagnostics.py --run-name round28_g06_baseline_box_head_only --afm-type none
E:\anaconda\01\envs\RLimage\python.exe scripts/round28_diagnostics.py --run-name round28_g07_identity_current_afm_only --afm-type identity --afm-residual-mode current
E:\anaconda\01\envs\RLimage\python.exe scripts/round28_diagnostics.py --run-name round28_g08_identity_current_afm_box_head --afm-type identity --afm-residual-mode current
E:\anaconda\01\envs\RLimage\python.exe scripts/round28_diagnostics.py --run-name round28_g09_identity_delta_afm_box_head --afm-type identity --afm-residual-mode delta
```

Expected:

```text
Each run directory contains round28_diagnostics.json.
```

- [ ] **Step 3: Commit**

```bash
git add scripts/round28_diagnostics.py
git commit -m "feat: add Round 2.8 AP75 and threshold diagnostics"
```

---

## Task 7: Summarize Results And Write Report

**Files:**
- Create: `scripts/round28_summarize.py`
- Create: `docs/round28_results.md`

- [ ] **Step 1: Create summarizer**

Create `scripts/round28_summarize.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from spectral_detection_posttrain.utils.io import save_json


GROUPS = [
    "round28_g01_baseline_full",
    "round28_g02_old_afm_full",
    "round28_g03_identity_current_full",
    "round28_g04_identity_delta_full",
    "round28_g05_identity_norm_delta_full",
    "round28_g06_baseline_box_head_only",
    "round28_g07_identity_current_afm_only",
    "round28_g08_identity_current_afm_box_head",
    "round28_g09_identity_delta_afm_box_head",
]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    rows = []
    for group in GROUPS:
        metrics_path = Path("runs") / group / "eval_metrics.json"
        diagnostics_path = Path("runs") / group / "round28_diagnostics.json"
        metrics = _load_json(metrics_path)
        diagnostics = _load_json(diagnostics_path) if diagnostics_path.exists() else {}
        threshold_005 = diagnostics.get("0.05", {})
        rows.append({
            "group": group,
            "ap50": metrics.get("ap50"),
            "ap75": metrics.get("ap75"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "ece": metrics.get("ece"),
            "high_conf_fp_count": metrics.get("high_conf_fp_count"),
            "num_predictions": metrics.get("num_predictions"),
            "matched_iou_mean": threshold_005.get("matched_iou_mean"),
            "center_error_mean": threshold_005.get("center_error_mean"),
            "size_error_mean": threshold_005.get("size_error_mean"),
            "duplicate_predictions": threshold_005.get("duplicate_predictions"),
        })

    save_json({"rows": rows}, Path("runs") / "round28_summary.json")
    lines = [
        "# Round 2.8 AFM Diagnostics Results",
        "",
        "| group | AP50 | AP75 | precision | recall | ECE | high_FP | pred | mean_IoU | center_err | size_err | dup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {group} | {ap50:.4f} | {ap75:.4f} | {precision:.4f} | {recall:.4f} | {ece:.4f} | {high_conf_fp_count} | {num_predictions} | {matched_iou_mean:.4f} | {center_error_mean:.4f} | {size_error_mean:.4f} | {duplicate_predictions} |".format(
                **row
            )
        )
    lines.extend([
        "",
        "## Verdict Checklist",
        "",
        "- [ ] Frozen parity passed: identity AFM is detector-level no-op before training.",
        "- [ ] Identity delta residual improves AP75/precision over identity current.",
        "- [ ] Old AFM gain, if present, is separated from prediction-count inflation.",
        "- [ ] AFM-only and AFM+box-head training scopes explain whether AFM itself or head adaptation causes drift.",
        "- [ ] Threshold curves identify whether the AP75 drop is localization error or score calibration.",
    ])
    Path("docs/round28_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run summarizer**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe scripts/round28_summarize.py
```

Expected:

```text
runs/round28_summary.json exists.
docs/round28_results.md exists.
```

- [ ] **Step 3: Commit**

```bash
git add scripts/round28_summarize.py docs/round28_results.md runs/round28_summary.json
git commit -m "docs: report Round 2.8 AFM diagnostics"
```

---

## Success Criteria

Plan 2.8 succeeds if it can explain the Round 2.7 vs Plan 3.1 discrepancy, even if AFM itself fails.

```text
Required engineering completion:
1. AFM variant tests pass.
2. Frozen parity result exists.
3. All 9 matrix eval_metrics.json files exist.
4. All 9 round28_diagnostics.json files exist.
5. docs/round28_results.md contains the comparison table and verdict checklist.

Required scientific interpretation:
1. If old AFM improves AP75 but increases predictions or lowers precision, treat Plan 3.1 as perturbation-driven.
2. If identity current hurts AP75 but identity delta does not, continue with delta residual only.
3. If all identity variants hurt AP75/precision, stop detector AFM and move the AFM idea to segmentation Plan 4.x.
4. If identity delta keeps AP50/AP75/precision close to baseline and improves ECE or high-conf FP, promote it to Plan 3.2 large validation.
```

---

## Next Plan Boundary

Only after Plan 2.8 produces an interpretable result should a Round 3.x large validation start.

Recommended next branches:

```text
Plan 3.2 if AFM signal survives:
multi-seed, 1/3/5 epoch, clean + patch scenes, identity-delta AFM only.

Plan 4.1 if detector AFM remains unstable:
semantic segmentation RLVR/AFM implementation based on Plan 4.0.
```

---

## Plan Location

`E:/CLIproject/RLimage/docs/superpowers/plans/2026-06-04-plan28-afm-diagnostics.md`
