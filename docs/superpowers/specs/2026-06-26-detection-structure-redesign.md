# Detection Structure Redesign Spec

**Date**: 2026-06-26  
**Status**: Approved for implementation  
**Goal**: Refactor the project away from hand-crafted losses/verifiers toward task-aligned, pluggable structural components for object detection.

---

## 1. Motivation

Empirical results show that:

- MPLSeg-style in-network AFM is the strongest reproducible baseline.
- External spectral verifiers, RLVR, DPO, and BEM do not provide reliable gains because their loss minimizers are misaligned with the true detection objective.
- The project has accumulated many legacy branches (RLVR, DPO, BEM, segmentation experiments) that obscure the mainline.

We therefore refactor the codebase and introduce three independent, depth-aware structural components that encode detection-relevant priors directly into the network.

---

## 2. Scope

### 2.1 Code refactoring

- Move legacy code to a top-level `legacy/` directory:
  - `spectral_detection_posttrain/methods/dpo/`
  - `spectral_detection_posttrain/methods/rlvr/`
  - `spectral_detection_posttrain/methods/segmentation/`
  - `spectral_detection_posttrain/trainers/segmentation/`
  - `spectral_detection_posttrain/trainers/detection/action_verifier_posttrain.py`
  - `spectral_detection_posttrain/methods/afm/embedding_manifold.py` (BEM)
  - related NNI configs and scripts that are no longer active
- Keep in the main package:
  - `datasets/`
  - `core/models/` (Faster R-CNN builder + AFM)
  - `methods/afm/micro_afm.py` (validated AFM baseline)
  - `trainers/detection/` standard detection trainer/evaluator
  - `eval/`, `metrics/`, `utils/`
  - light-weight `experiments/` runner for grid sweeps

### 2.2 New structural components

Three independent, flag-gated modules:

| Component | Placement | Depth | Purpose |
|---|---|---|---|
| **PBG** — Phase Boundary Gate | After ROI Align, before box head | Shallow spatial (7×7) | Highlight real object boundaries via local phase/edge cues |
| **TAM** — Task-Aligned Manifold | After box head, before predictor | Penultimate embedding | Refine embedding on a compact manifold constrained by prototypes and boundary stability |
| **PAH** — Prototype-Aware Head | Replaces/extends `FastRCNNPredictor` | Penultimate embedding | Classify via geometric similarity to learned class prototypes (ETF-style) |

All three are opt-in via CLI/config flags and can be combined in an ablation grid.

---

## 3. Architecture

```
image
  │
  ▼
backbone + FPN
  │
  ▼
RPN
  │
  ▼
ROI Align ──► [PBG] ──► box head ──► [TAM] ──► [PAH] ──► cls + reg
               (shallow)         (deep manifold) (deep geometry)
```

- `[PBG]` operates on spatial ROI features (7×7×C).
- `[TAM]` and `[PAH]` operate on the box-head output vector (D-dim embedding).
- When `PAH` is disabled, the original `FastRCNNPredictor` is used.

---

## 4. Component Specifications

### 4.1 PBG — Phase Boundary Gate

**Input**: ROI feature tensor `x ∈ R^{C×H×W}` (default H=W=7).  
**Output**: refined ROI feature of the same shape.

**Mechanism**:

1. Compute local frequency response via small-window FFT or learned Gabor-like filters.
2. Extract phase-gradient / edge-saliency map `B ∈ R^{1×H×W}`.
3. Generate a learnable spatial gate `g = σ(Conv(B))`.
4. Return `x + α · g ⊙ x`, where `α` is a learnable residual scale initialized to 0 (identity at start).

**Constraints**:
- No heavy spatial downsampling; keep 7×7 resolution.
- Identity initialization so it can be inserted into a pretrained detector without disruption.

### 4.2 TAM — Task-Aligned Manifold

**Input**: box-head embedding `z ∈ R^D`.  
**Output**: refined embedding `z' ∈ R^D`.

**Mechanism**:

1. Encoder: `z → h → latent ∈ R^L` (L < D).
2. Decoder: `latent → h' → residual ∈ R^D`.
3. Output: `z' = z + γ · residual`, `γ` initialized to 0.
4. Auxiliary constraints (only during training, optional weights):
   - **Prototype pull**: latent vectors of the same GT class are pulled toward their class prototype.
   - **Boundary stability**: latent should be sensitive to boundary perturbations but stable to interior perturbations.

**Why this differs from BEM**:
- BEM was an unconstrained autoencoder; TAM uses prototype and boundary regularizers so the manifold is task-aligned.

### 4.3 PAH — Prototype-Aware Head

**Input**: box-head embedding `z ∈ R^D`.  
**Output**: class logits and bbox regression.

**Mechanism**:

1. L2-normalize `z` to the unit sphere.
2. Maintain C+1 learnable prototypes `P ∈ R^{(C+1)×D}` (including background).
3. Classification logits: `logits_i = cos(z, P_i) / τ`, where `τ` is a learnable or fixed temperature.
4. Regression: a small MLP on `z` (plus optional prototype-residual) predicts box deltas.

**Fallback**: if PAH is disabled, the original linear `cls_score` / `bbox_pred` is used.

---

## 5. Configuration and CLI

New model config fields:

```json
{
  "model": {
    "afm_type": "mplseg_phase_only",
    "use_pbg": true,
    "use_tam": true,
    "use_pah": true,
    "pbg_alpha_init": 0.0,
    "tam_latent_dim": 256,
    "pah_temperature": 0.1
  }
}
```

CLI flags added to `round28_train_eval.py` (or its successor):

- `--use-pbg`
- `--use-tam`
- `--use-pah`
- `--pbg-alpha-init`
- `--tam-latent-dim`
- `--pah-temperature`

---

## 6. Validation Plan

All experiments on Penn-Fudan, 3 seeds (42, 123, 2024), 3 epochs, `box_head_only` training unless otherwise noted.

| Run | Configuration | Purpose |
|---|---|---|
| `clean_afm_baseline` | AFM only | Post-refactor baseline |
| `afm+pbg` | AFM + PBG | Validate shallow boundary gate |
| `afm+tam` | AFM + TAM | Validate constrained manifold |
| `afm+pah` | AFM + PAH | Validate prototype head |
| `afm+pbg+tam` | AFM + PBG + TAM | Boundary + manifold |
| `afm+pbg+pah` | AFM + PBG + PAH | Boundary + prototype geometry |
| `afm+pbg+tam+pah` | all three | Full structure |

Primary metrics: AP75, AP50, ECE, FP rate, P@R=0.85.

---

## 7. Implementation Phases

1. **Refactor**: move legacy code to `legacy/`, thin main package, verify AFM baseline still runs.
2. **Skeleton**: add CLI flags and no-op stubs for PBG/TAM/PAH.
3. **PBG**: implement and validate standalone.
4. **TAM**: implement and validate standalone.
5. **PAH**: implement and validate standalone.
6. **Combination grid**: run all combinations and summarize.
7. **Git commit**: commit design doc, refactored code, new modules, and result summary.

---

## 8. Success Criteria

- Refactored codebase runs the AFM baseline without regression.
- At least one of PBG/TAM/PAH (or a combination) achieves a mean AP75 improvement over the AFM baseline on Penn-Fudan.
- The improvement is consistent across 3 seeds.
- All changes are committed to git with a clear history.

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Refactor breaks existing baseline | Keep AFM path intact; run smoke test after each move |
| PBG has no signal | Keep window small; use identity init; ablate on/off |
| PAH collapses to trivial prototypes | Use ETF initialization or fix background prototype; monitor prototype diversity |
| TAM overfits | Strong regularization via prototype pull; early stopping |
| Combination grid too large | Run sequentially; reuse logs; auto-summarize with existing script |
