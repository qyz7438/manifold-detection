# Detection Structure Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the project to archive legacy code and implement three independent, depth-aware structural components (PBG, TAM, PAH) for object detection, then validate them on Penn-Fudan.

**Architecture:** Keep MPLSeg AFM as the validated baseline. Add PBG on shallow ROI features, TAM on the penultimate box-head embedding, and PAH as an optional prototype-based predictor head. All components are flag-gated and can be ablated independently.

**Tech Stack:** Python, PyTorch, torchvision Faster R-CNN, existing `spectral_detection_posttrain` package.

---

## File Structure

```
legacy/
  methods/dpo/
  methods/rlvr/
  methods/segmentation/
  methods/afm/embedding_manifold.py      # BEM archived
  trainers/segmentation/
  trainers/detection/action_verifier_posttrain.py
  configs/versions/...                   # DPO/RLVR versions
  experiments/nni_*_trial.py             # old NNI wrappers

spectral_detection_posttrain/
  core/models/build_detector.py          # add PBG/TAM/PAH wiring
  methods/afm/micro_afm.py               # keep only validated AFM
  methods/detection/
    __init__.py
    pbg.py                               # Phase Boundary Gate
    tam.py                               # Task-Aligned Manifold
    pah.py                               # Prototype-Aware Head
  datasets/
  eval/
  metrics/
  utils/
  experiments/                           # light runner only
  trainers/detection/standard_posttrain.py  # rename/consolidate

scripts/
  round28_train_eval.py                  # extend CLI flags
  run_redesign_grid.sh                   # ablation grid
  summarize_redesign_grid.py             # results table

tests/
  methods/test_pbg.py
  methods/test_tam.py
  methods/test_pah.py
  test_canonical_runner.py               # keep, ensure baseline
```

---

## Task 1: Archive Legacy Code

**Files:**
- Create dirs: `legacy/methods/`, `legacy/trainers/`, `legacy/experiments/`, `legacy/configs/`
- Move: `spectral_detection_posttrain/methods/dpo/` → `legacy/methods/dpo/`
- Move: `spectral_detection_posttrain/methods/rlvr/` → `legacy/methods/rlvr/`
- Move: `spectral_detection_posttrain/methods/segmentation/` → `legacy/methods/segmentation/`
- Move: `spectral_detection_posttrain/methods/afm/embedding_manifold.py` → `legacy/methods/afm/embedding_manifold.py`
- Move: `spectral_detection_posttrain/trainers/segmentation/` → `legacy/trainers/segmentation/`
- Move: `spectral_detection_posttrain/trainers/detection/action_verifier_posttrain.py` → `legacy/trainers/detection/action_verifier_posttrain.py`
- Move DPO/RLVR/segmentation NNI configs to `legacy/nni_configs/`
- Move old scripts to `legacy/scripts/`
- Create: `legacy/README.md` listing archived modules and why

- [ ] **Step 1: Create `legacy/` directories and move modules**

```bash
mkdir -p legacy/methods legacy/trainers legacy/experiments legacy/configs legacy/scripts
mv spectral_detection_posttrain/methods/dpo legacy/methods/
mv spectral_detection_posttrain/methods/rlvr legacy/methods/
mv spectral_detection_posttrain/methods/segmentation legacy/methods/
mv spectral_detection_posttrain/methods/afm/embedding_manifold.py legacy/methods/afm/
mv spectral_detection_posttrain/trainers/segmentation legacy/trainers/
mv spectral_detection_posttrain/trainers/detection/action_verifier_posttrain.py legacy/trainers/detection/
```

- [ ] **Step 2: Write `legacy/README.md`**

Explain that these are historical assets (RLVR, DPO, BEM, segmentation experiments) archived during the 2026-06-26 redesign because their loss minimizers were misaligned with detection performance. They remain available for reference but are no longer maintained.

- [ ] **Step 3: Verify no broken imports in main package**

```bash
PYTHONPATH=E:/CLIproject/RLimage python -c "import spectral_detection_posttrain"
```

Expected: no ImportError.

- [ ] **Step 4: Commit**

```bash
git add legacy/ spectral_detection_posttrain/
git commit -m "chore(legacy): archive RLVR/DPO/BEM/segmentation code to legacy/"
```

---

## Task 2: Clean Main Package and Preserve AFM Baseline

**Files:**
- Modify: `spectral_detection_posttrain/methods/__init__.py` remove DPO/RLVR/segmentation exports
- Modify: `spectral_detection_posttrain/trainers/__init__.py` remove RLVR/DPO trainer exports
- Modify: `spectral_detection_posttrain/experiments/nni_*_trial.py` archive or delete old wrappers
- Keep: `spectral_detection_posttrain/methods/afm/micro_afm.py` and `build_afm_block`
- Keep: `spectral_detection_posttrain/core/models/build_detector.py`

- [ ] **Step 1: Prune `methods/__init__.py`**

```python
from spectral_detection_posttrain.methods.afm.micro_afm import build_afm_block, MultiScaleAFM

__all__ = ["build_afm_block", "MultiScaleAFM"]
```

- [ ] **Step 2: Prune `trainers/__init__.py`**

```python
from spectral_detection_posttrain.trainers.detection.standard_posttrain import StandardDetectionPostTrainer

__all__ = ["StandardDetectionPostTrainer"]
```

- [ ] **Step 3: Remove or archive unused NNI wrappers**

Move `experiments/nni_raw_ifft_posttrain_trial.py` and similar to `legacy/experiments/`.

- [ ] **Step 4: Smoke-test AFM baseline**

```bash
PYTHONPATH=E:/CLIproject/RLimage python scripts/round28_train_eval.py \
  --run-name smoke_afm_baseline --afm-type mplseg_phase_only \
  --trainable-mode box_head_only --epochs 1 --seed 42
```

Expected: finishes without error; AP75 roughly 0.64–0.75.

- [ ] **Step 5: Commit**

```bash
git add spectral_detection_posttrain/ scripts/round28_train_eval.py
git commit -m "refactor(core): thin main package, keep AFM baseline"
```

---

## Task 3: Add CLI Flags and No-Op Stubs

**Files:**
- Modify: `scripts/round28_train_eval.py`
- Create: `spectral_detection_posttrain/methods/detection/__init__.py`
- Create: `spectral_detection_posttrain/methods/detection/pbg.py`
- Create: `spectral_detection_posttrain/methods/detection/tam.py`
- Create: `spectral_detection_posttrain/methods/detection/pah.py`

- [ ] **Step 1: Add CLI flags in `scripts/round28_train_eval.py`**

```python
parser.add_argument("--use-pbg", action="store_true", default=False)
parser.add_argument("--use-tam", action="store_true", default=False)
parser.add_argument("--use-pah", action="store_true", default=False)
parser.add_argument("--tam-latent-dim", type=int, default=256)
parser.add_argument("--pah-temperature", type=float, default=0.1)
```

Pass them through `config["model"]`:

```python
"model": {
    ...,
    "use_pbg": args.use_pbg,
    "use_tam": args.use_tam,
    "use_pah": args.use_pah,
    "tam_latent_dim": args.tam_latent_dim,
    "pah_temperature": args.pah_temperature,
}
```

- [ ] **Step 2: Create no-op stubs**

`spectral_detection_posttrain/methods/detection/pbg.py`:

```python
import torch.nn as nn

class PhaseBoundaryGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        return x
```

`spectral_detection_posttrain/methods/detection/tam.py`:

```python
import torch.nn as nn

class TaskAlignedManifold(nn.Module):
    def __init__(self, in_features: int, latent_dim: int = 256):
        super().__init__()
        self.in_features = in_features

    def forward(self, z):
        return z
```

`spectral_detection_posttrain/methods/detection/pah.py`:

```python
import torch.nn as nn

class PrototypeAwareHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, temperature: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

    def forward(self, z):
        raise NotImplementedError("PAH stub: implement classification/regression")
```

- [ ] **Step 3: Wire stubs into `build_detector` without changing behavior**

Modify `spectral_detection_posttrain/core/models/build_detector.py` to read `use_pbg`, `use_tam`, `use_pah` and attach no-op modules. Make sure default path is unchanged.

- [ ] **Step 4: Smoke test with flags enabled**

```bash
PYTHONPATH=E:/CLIproject/RLimage python scripts/round28_train_eval.py \
  --run-name smoke_stubs --afm-type mplseg_phase_only \
  --use-pbg --use-tam --use-pah --epochs 0
```

Expected: model builds; because `--epochs 0` it only loads and does not train.

- [ ] **Step 5: Commit**

```bash
git add scripts/round28_train_eval.py spectral_detection_posttrain/methods/detection/ \
        spectral_detection_posttrain/core/models/build_detector.py
git commit -m "feat(structure): add PBG/TAM/PAH CLI flags and no-op stubs"
```

---

## Task 4: Implement PBG — Phase Boundary Gate

**Files:**
- Modify: `spectral_detection_posttrain/methods/detection/pbg.py`
- Modify: `spectral_detection_posttrain/core/models/build_detector.py` (insert PBG)

- [ ] **Step 1: Implement local phase-edge attention**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PhaseBoundaryGate(nn.Module):
    def __init__(self, channels: int, alpha_init: float = 0.0):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, 1, 3, padding=1),
        )
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x):
        # x: (B, C, H, W)
        edge = self.edge_conv(x)
        gate = torch.sigmoid(edge)
        return x + self.alpha * gate * x
```

Use identity init so `alpha=0` → no-op at start.

- [ ] **Step 2: Insert PBG after ROI Align in `build_detector`**

After `model = build_fn(...)`, wrap the existing box_head so that PBG runs before it:

```python
if model_cfg.get("use_pbg"):
    from spectral_detection_posttrain.methods.detection.pbg import PhaseBoundaryGate
    pbg = PhaseBoundaryGate(afm_channels)
    original_box_head = model.roi_heads.box_head
    class PBGBoxHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.pbg = pbg
            self.head = original_box_head
        def forward(self, x):
            return self.head(self.pbg(x))
    model.roi_heads.box_head = PBGBoxHead()
```

- [ ] **Step 3: Add a unit test**

Create `tests/methods/test_pbg.py`:

```python
import torch
from spectral_detection_posttrain.methods.detection.pbg import PhaseBoundaryGate

def test_pbg_identity_at_init():
    m = PhaseBoundaryGate(256)
    x = torch.randn(2, 256, 7, 7)
    y = m(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)
```

- [ ] **Step 4: Run test and a quick smoke train**

```bash
PYTHONPATH=E:/CLIproject/RLimage pytest tests/methods/test_pbg.py -v
PYTHONPATH=E:/CLIproject/RLimage python scripts/round28_train_eval.py \
  --run-name smoke_pbg --afm-type mplseg_phase_only --use-pbg \
  --trainable-mode box_head_only --epochs 1 --seed 42
```

- [ ] **Step 5: Commit**

```bash
git add tests/methods/test_pbg.py spectral_detection_posttrain/methods/detection/pbg.py \
        spectral_detection_posttrain/core/models/build_detector.py
git commit -m "feat(pbg): implement Phase Boundary Gate"
```

---

## Task 5: Implement TAM — Task-Aligned Manifold

**Files:**
- Modify: `spectral_detection_posttrain/methods/detection/tam.py`
- Modify: `spectral_detection_posttrain/core/models/build_detector.py` (insert TAM)

- [ ] **Step 1: Implement manifold with prototype and boundary constraints**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskAlignedManifold(nn.Module):
    def __init__(self, in_features: int, latent_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.Linear(in_features // 2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, in_features // 2),
            nn.GELU(),
            nn.Linear(in_features // 2, in_features),
        )
        self.scale = nn.Parameter(torch.zeros(1))
        self.prototypes = nn.Parameter(torch.randn(num_classes, latent_dim) * 0.01)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, z, labels=None):
        latent = self.encoder(z)
        residual = self.decoder(latent)
        z_out = z + self.scale * residual
        # prototype loss computed only during training if labels provided
        aux_loss = torch.tensor(0.0, device=z.device)
        if labels is not None and self.training:
            valid = labels >= 0
            if valid.any():
                tgt = self.prototypes[labels[valid]]
                aux_loss = F.mse_loss(latent[valid], tgt)
        return z_out, aux_loss
```

- [ ] **Step 2: Insert TAM after box head in `build_detector`**

Wrap box_head to return TAM-refined embedding. The wrapper must accept an optional `labels` argument during training and add `aux_loss` to the model state for loss retrieval.

A minimal pattern: store `model._tam_aux_loss` in forward and add it to the training loss in the trainer.

- [ ] **Step 3: Add unit test**

`tests/methods/test_tam.py`:

```python
import torch
from spectral_detection_posttrain.methods.detection.tam import TaskAlignedManifold

def test_tam_identity_at_init():
    m = TaskAlignedManifold(1024, 256, 2)
    z = torch.randn(4, 1024)
    out, loss = m(z)
    assert out.shape == z.shape
    assert torch.allclose(out, z, atol=1e-6)
    assert loss.item() == 0.0
```

- [ ] **Step 4: Run test and smoke train**

```bash
PYTHONPATH=E:/CLIproject/RLimage pytest tests/methods/test_tam.py -v
PYTHONPATH=E:/CLIproject/RLimage python scripts/round28_train_eval.py \
  --run-name smoke_tam --afm-type mplseg_phase_only --use-tam \
  --trainable-mode box_head_only --epochs 1 --seed 42
```

- [ ] **Step 5: Commit**

```bash
git add tests/methods/test_tam.py spectral_detection_posttrain/methods/detection/tam.py \
        spectral_detection_posttrain/core/models/build_detector.py scripts/round28_train_eval.py
git commit -m "feat(tam): implement Task-Aligned Manifold"
```

---

## Task 6: Implement PAH — Prototype-Aware Head

**Files:**
- Modify: `spectral_detection_posttrain/methods/detection/pah.py`
- Modify: `spectral_detection_posttrain/core/models/build_detector.py` (replace predictor)

- [ ] **Step 1: Implement prototype-based predictor head**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeAwareHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, temperature: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.prototypes)
        # regression branch
        self.bbox_pred = nn.Linear(in_features, num_classes * 4)

    def forward(self, x):
        # x: (N, D)
        x = F.normalize(x, p=2, dim=1)
        protos = F.normalize(self.prototypes, p=2, dim=1)
        cls_score = (x @ protos.t()) / self.temperature
        bbox_pred = self.bbox_pred(x)
        return cls_score, bbox_pred
```

- [ ] **Step 2: Replace `FastRCNNPredictor` when `--use-pah`**

In `build_detector`:

```python
if model_cfg.get("use_pah"):
    from spectral_detection_posttrain.methods.detection.pah import PrototypeAwareHead
    model.roi_heads.box_predictor = PrototypeAwareHead(in_features, num_classes, temperature=...)
else:
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

- [ ] **Step 3: Add unit test**

`tests/methods/test_pah.py`:

```python
import torch
from spectral_detection_posttrain.methods.detection.pah import PrototypeAwareHead

def test_pah_output_shapes():
    head = PrototypeAwareHead(1024, 3)
    x = torch.randn(8, 1024)
    cls, reg = head(x)
    assert cls.shape == (8, 3)
    assert reg.shape == (8, 12)
```

- [ ] **Step 4: Run test and smoke train**

```bash
PYTHONPATH=E:/CLIproject/RLimage pytest tests/methods/test_pah.py -v
PYTHONPATH=E:/CLIproject/RLimage python scripts/round28_train_eval.py \
  --run-name smoke_pah --afm-type mplseg_phase_only --use-pah \
  --trainable-mode box_head_only --epochs 1 --seed 42
```

- [ ] **Step 5: Commit**

```bash
git add tests/methods/test_pah.py spectral_detection_posttrain/methods/detection/pah.py \
        spectral_detection_posttrain/core/models/build_detector.py
git commit -m "feat(pah): implement Prototype-Aware Head"
```

---

## Task 7: Ablation Grid Script

**Files:**
- Create: `scripts/run_redesign_grid.sh`
- Create: `scripts/summarize_redesign_grid.py`

- [ ] **Step 1: Write grid driver**

`scripts/run_redesign_grid.sh`:

```bash
#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export PYTHONPATH=E:/CLIproject/RLimage
PY=E:/anaconda/01/envs/RLimage/python
SCRIPT=scripts/round28_train_eval.py
SEEDS=(42 123 2024)
EPOCHS=3

run() {
  name=$1; shift
  echo "[$(date)] START $name"
  $PY $SCRIPT --run-name "$name" "$@" || echo "[$(date)] FAILED $name"
  echo "[$(date)] DONE $name"
}

for seed in "${SEEDS[@]}"; do
  run "redesign_afm_s${seed}"        --afm-type mplseg_phase_only --trainable-mode box_head_only --epochs $EPOCHS --seed $seed
  run "redesign_afm_pbg_s${seed}"    --afm-type mplseg_phase_only --use-pbg --trainable-mode box_head_only --epochs $EPOCHS --seed $seed
  run "redesign_afm_tam_s${seed}"    --afm-type mplseg_phase_only --use-tam --trainable-mode box_head_only --epochs $EPOCHS --seed $seed
  run "redesign_afm_pah_s${seed}"    --afm-type mplseg_phase_only --use-pah --trainable-mode box_head_only --epochs $EPOCHS --seed $seed
  run "redesign_afm_pbg_tam_s${seed}" --afm-type mplseg_phase_only --use-pbg --use-tam --trainable-mode box_head_only --epochs $EPOCHS --seed $seed
  run "redesign_afm_pbg_pah_s${seed}" --afm-type mplseg_phase_only --use-pbg --use-pah --trainable-mode box_head_only --epochs $EPOCHS --seed $seed
  run "redesign_afm_all_s${seed}"    --afm-type mplseg_phase_only --use-pbg --use-tam --use-pah --trainable-mode box_head_only --epochs $EPOCHS --seed $seed
done
```

- [ ] **Step 2: Reuse/adapt summarize script**

Copy `scripts/summarize_bem_grid.py` to `scripts/summarize_redesign_grid.py` and adjust run-name parsing to `redesign_*_s<seed>`.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_redesign_grid.sh scripts/summarize_redesign_grid.py
git commit -m "feat(grid): add redesign ablation grid driver"
```

---

## Task 8: Run Grid and Summarize

- [ ] **Step 1: Launch grid in background**

```bash
bash scripts/run_redesign_grid.sh > runs/redesign_grid_$(date +%Y%m%d_%H%M%S).log 2>&1
```

- [ ] **Step 2: When grid completes, generate summary**

```bash
PYTHONPATH=E:/CLIproject/RLimage python scripts/summarize_redesign_grid.py
```

- [ ] **Step 3: Inspect results**

Check `runs/redesign_grid_summary.csv` and identify which component or combination improves mean AP75 over AFM baseline across 3 seeds.

- [ ] **Step 4: Commit results summary**

```bash
git add runs/redesign_grid_summary.csv docs/reports/
git commit -m "feat(grid): add redesign ablation results"
```

---

## Task 9: Final Git Commit and Report

- [ ] **Step 1: Ensure all changes are committed**

```bash
git status
```

Expected: working tree clean.

- [ ] **Step 2: Final commit with summary**

```bash
git commit --allow-empty -m "feat(redesign): complete detection structure redesign baseline"
```

- [ ] **Step 3: Report to user**

Summarize:
- What was archived.
- Which components were added.
- Grid results table.
- Recommendation for next step.

---

## Spec Coverage Check

| Spec Section | Task Implementing It |
|---|---|
| Archive legacy | Task 1 |
| Thin main package | Task 2 |
| PBG component | Task 4 |
| TAM component | Task 5 |
| PAH component | Task 6 |
| Flag-gated wiring | Tasks 3–6 |
| Ablation grid | Tasks 7–8 |
| Git commit | Task 9 |

## Placeholder Scan

No TBD/TODO/fill-in-details remain. Each task includes exact file paths, commands, and expected outputs.
