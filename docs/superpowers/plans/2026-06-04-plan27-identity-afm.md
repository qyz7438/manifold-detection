# Plan 2.7: Identity-Preserving AFM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the AFM identity problem — zero-init halves magnitude spectrum (sigmoid(0)=0.5) and ReLU clips output — so a frozen detector with AFM produces identical predictions to one without, and training starts from a clean baseline.

**Architecture:** Redesign AFMBlock as residual: `x + residual_scale × freq_out`. Mag: `mag × (1 + mag_scale × tanh(gate))`. Phase: `pha + phase_scale × tanh(res)`. All scales start at 0. At init, AFM(x) = x.

**Tech Stack:** Python, PyTorch, TorchVision Faster R-CNN, Penn-Fudan, pytest, existing package.

---

## Why Plan 2.7 Exists

Current code (broken):
```python
mag = mag * (1.0 - sigmoid(gate))  # sigmoid(0)=0.5 → mag halved
out = F.relu(iFFT(F))              # clips negatives
```

Round 2.6: AP50 0.853 vs baseline 0.885, AP75 0.409 vs 0.645, precision collapse. Round 3.1: cold AFM beats baseline on AP50 but precision drops 0.71→0.62. Both symptoms trace to non-identity init.

Fix: residual + learnable scales at zero → true identity at init.

---

## Non-Goals

No RLVR, no NNI, no multi-scale, no hot-start. Pure identity fix.

---

## File Map

- Modify: `spectral_detection_posttrain/models/micro_afm.py` — residual AFMBlock
- Modify: `tests/test_micro_afm.py` — add identity test
- Create: `scripts/round27_frozen_parity.py` — frozen detector parity
- Create: `docs/round27_results.md` — comparison report

---

## Task 1: Redesign AFMBlock As Residual Identity

**Files:**
- Modify: `spectral_detection_posttrain/models/micro_afm.py`
- Modify: `tests/test_micro_afm.py`

- [ ] **Step 1: Write identity test first**

Replace `tests/test_micro_afm.py`:

```python
import pytest
import torch
from spectral_detection_posttrain.models.micro_afm import AFMBlock

def test_afm_identity_at_init():
    afm = AFMBlock(channels=16)
    x = torch.randn(2, 16, 32, 32)
    out = afm(x)
    assert torch.allclose(out, x, atol=1e-3)

def test_afm_output_shape():
    afm = AFMBlock(channels=16)
    x = torch.randn(2, 16, 32, 32)
    assert afm(x).shape == x.shape

@pytest.mark.parametrize("channels", [16, 64, 256])
def test_afm_gradient_flows(channels: int):
    afm = AFMBlock(channels=channels)
    x = torch.randn(1, channels, 16, 16, requires_grad=True)
    out = afm(x)
    out.mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0

def test_afm_no_nan():
    afm = AFMBlock(channels=16)
    x = torch.randn(2, 16, 32, 32) * 100.0
    out = afm(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
```

- [ ] **Step 2: Run tests and verify failure**

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_micro_afm.py -v
```

Expected: `test_afm_identity_at_init` FAILS.

- [ ] **Step 3: Implement residual AFMBlock**

Replace AFMBlock in `micro_afm.py`:

```python
class AFMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)

        self.mag_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Tanh(),
        )
        self.phase_res = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.Tanh(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Tanh(),
        )
        self.mag_scale = nn.Parameter(torch.zeros(1))
        self.phase_scale = nn.Parameter(torch.zeros(1))
        self.residual_scale = nn.Parameter(torch.zeros(1))
        self._eps = 1e-3

        for module in [self.mag_gate, self.phase_res]:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    nn.init.zeros_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        F_repr = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(F_repr)
        pha = torch.angle(F_repr + self._eps)

        mag_delta = self.mag_gate(torch.log1p(mag))
        mag = mag * (1.0 + self.mag_scale * mag_delta)

        pha_delta = self.phase_res(pha)
        pha = pha + self.phase_scale * pha_delta

        F_mod = mag * torch.exp(1j * pha)
        freq_out = torch.fft.irfft2(F_mod, s=x.shape[-2:], norm="ortho")
        return x + self.residual_scale * freq_out
```

Key changes: tanh gate (zero→0), learnable scales at 0, residual skip, no ReLU.

- [ ] **Step 4: Run tests and verify pass**

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_micro_afm.py tests/test_multiscale_afm.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add spectral_detection_posttrain/models/micro_afm.py tests/test_micro_afm.py
git commit -m "fix: make AFMBlock identity-preserving at init"
```

---

## Task 2: Frozen Detector Parity Smoke

**Files:**
- Create: `scripts/round27_frozen_parity.py`

- [ ] **Step 1: Create parity script**

```python
"""Verify frozen detector with AFM at identity produces same boxes."""
from __future__ import annotations
import torch
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import load_checkpoint
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

def main():
    config = {
        "seed": 42, "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data": {"root": "./data", "download": True, "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
        "model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn", "pretrained": True,
                  "num_classes": 2, "min_size": 320, "max_size": 320, "afm_channels": 0},
        "train": {"batch_size": 2},
        "eval": {"batch_size": 1},
    }
    set_seed(config["seed"])
    device = resolve_device(config)

    model_no_afm = build_detector(config).to(device)
    ckpt_path = "runs/round27_parity_ckpt.pth"
    torch.save({"model": model_no_afm.state_dict()}, ckpt_path)

    config["model"]["afm_channels"] = 256
    model_afm = build_detector(config).to(device)
    load_checkpoint(model_afm, ckpt_path, device)
    model_afm.eval()

    _, val_loader = build_penn_fudan_loaders(config, limit_val=8)
    images, _ = next(iter(val_loader))
    images = [img.to(device) for img in images]

    with torch.no_grad():
        preds_no = model_no_afm(images)
        preds_afm = model_afm(images)

    for i, (p_no, p_afm) in enumerate(zip(preds_no, preds_afm)):
        db = (p_no["boxes"] - p_afm["boxes"]).abs().max().item()
        ds = (p_no["scores"] - p_afm["scores"]).abs().max().item()
        print(f"Img {i}: max box diff={db:.2e}, max score diff={ds:.2e}")
        assert db < 0.01, f"Box mismatch: {db}"
        assert ds < 0.01, f"Score mismatch: {ds}"
    print("PASS: AFM identity preserves frozen detector predictions")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run parity test**

```powershell
cd E:/CLIproject/RLimage; $env:PYTHONPATH = "E:/CLIproject/RLimage"
E:\anaconda\01\envs\RLimage\python.exe scripts/round27_frozen_parity.py
```

Expected: "PASS: AFM identity preserves frozen detector predictions".

- [ ] **Step 3: Commit**

```bash
git add scripts/round27_frozen_parity.py
git commit -m "test: verify AFM identity preserves frozen detector predictions"
```

---

## Task 3: Re-Run 2.6 Comparison With Fixed AFM

**Files:**
- Uses `scripts/round26_micro_afm_smoke.py`

- [ ] **Step 1: Run full comparison**

```powershell
cd E:/CLIproject/RLimage; $env:PYTHONPATH = "E:/CLIproject/RLimage"
E:/anaconda/01/envs/RLimage/python.exe scripts/round26_micro_afm_smoke.py --afm-channels 0 --run-name round27_baseline --epochs 1
E:/anaconda/01/envs/RLimage/python.exe scripts/round26_micro_afm_smoke.py --afm-channels 256 --run-name round27_afm_fixed --epochs 1
```

- [ ] **Step 2: Show comparison**

```powershell
PYTHONPATH=E:/CLIproject/RLimage E:/anaconda/01/envs/RLimage/python.exe -c "
import json
for g in ['round27_baseline','round27_afm_fixed']:
    m = json.load(open(f'runs/{g}/eval_metrics.json'))
    print(f'{g}: AP50={m[\"ap50\"]:.4f} AP75={m[\"ap75\"]:.4f} prec={m[\"precision\"]:.4f} preds={m[\"num_predictions\"]} FP={m[\"high_conf_fp_count\"]}')
"
```

- [ ] **Step 3: Write and commit report**

Create `docs/round27_results.md`:

```markdown
# Plan 2.7 Identity-Preserving AFM Results

| config | AP50 | AP75 | precision | num_pred | high_FP | ECE |
|--------|------|------|-----------|----------|--------|-----|
| baseline | | | | | | |
| AFM fixed | | | | | | |

## Verdict
- [ ] Identity test passes
- [ ] Frozen parity confirmed
- [ ] AP50 within 1% of baseline
- [ ] Precision within 5% of baseline
```

```bash
git add docs/round27_results.md
git commit -m "docs: report Plan 2.7 identity-preserving AFM"
```

---

## Success Criteria

```text
1. test_afm_identity_at_init passes: AFM(x) == x at init.
2. Frozen parity: same weights → same boxes/scores.
3. Fixed AFM AP50 within 1% of baseline, no precision collapse.
```

If 1-2 pass but 3 fails, FFT feature transform doesn't help detection — publishable negative.

---

## Plan 位置

`E:/CLIproject/RLimage/docs/superpowers/plans/2026-06-04-plan27-identity-afm.md`
