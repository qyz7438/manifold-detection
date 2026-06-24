# Spectral Quality Head Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current Spectral Quality Head result into a controlled calibration/error-suppression study with Pareto curves, oracle bounds, feature ablations, patch-position stress tests, and a small VOC transfer check.

**Architecture:** Keep the baseline detector fixed. Reuse cached detector candidates and train/evaluate lightweight quality heads as reranking/calibration modules; do not update backbone/RPN/box regression. The core outputs are metric tables and curves that show whether `q_spec` is a controllable calibration signal, whether spectral features add value over ROI features, and where the method fails.

**Tech Stack:** Python, PyTorch, TorchVision detection, matplotlib, pandas, pytest, existing `spectral_detection_posttrain/` modules.

---

## Feasibility Assessment

This plan is highly feasible because the hard engineering pieces already exist:

- Penn-Fudan baseline detector checkpoint exists and runs.
- Candidate caching already saves boxes, labels, scores, ROI features, amplitude profiles, structure features, IoU, TP/FP labels, and targets.
- `SpectralQualityHead` already trains offline.
- `eval_rerank.py` already evaluates baseline/oracle/learned reranking.

The main scientific risk is not implementation; it is interpretation. Current results show `q_spec` improves precision/ECE and can reduce high-confidence FP, but it does not yet improve patch AP50. Therefore the next version should not chase AP first. It should prove controllability, calibration, and error suppression under matched Recall/Precision.

Expected implementation time:

- Alpha grid + metric curves: 0.5 day
- Fixed Recall/Precision metrics: 0.5 day
- Oracle upper bound: 0.5 day
- Feature ablation including random amplitude: 1 day
- Patch-position stress tests: 1 day
- Structure/AP75 analysis: 0.5-1 day
- VOC small subset scaffold: 1-2 days

Recommended first execution batch: Tasks 1-5. Task 6 is analysis-heavy but local. Task 7 should wait until Penn-Fudan conclusions are clear.

---

## File Map

Create:

- `spectral_detection_posttrain/eval/score_fusion.py`  
  Shared score fusion utilities: blend, multiply-power, logit fusion, alpha grid.
- `spectral_detection_posttrain/eval/operating_points.py`  
  Metrics at fixed recall/precision and high-conf FP at matched recall.
- `spectral_detection_posttrain/eval/eval_alpha_grid.py`  
  Runs alpha sweeps over cached candidates and writes Pareto CSV/PNG.
- `spectral_detection_posttrain/eval/eval_feature_ablation.py`  
  Compares ROI, Amp, ROI+Amp, ROI+Amp+Struct, RandomAmp.
- `spectral_detection_posttrain/eval/eval_patch_positions.py`  
  Evaluates background/object/edge/near-object patch groups.
- `spectral_detection_posttrain/datasets/voc_person_subset.py`  
  Minimal VOC subset loader for person/dog/car later transfer.
- `tests/test_score_fusion.py`
- `tests/test_operating_points.py`
- `docs/quality_head_calibration_results_2026-06-03.md`
- `run_quality_head_analysis.bat`

Modify:

- `spectral_detection_posttrain/eval/eval_rerank.py`  
  Use shared score fusion and report fixed operating-point metrics.
- `spectral_detection_posttrain/datasets/patch_transform.py`  
  Add `near_object`, `background`, `object`, `edge`, texture/contrast/blob patch types if missing.
- `spectral_detection_posttrain/configs/mvp.yaml`  
  Add analysis grids and patch-position settings.
- `README.md`  
  Document the analysis script and result files.

---

## Task 1: Alpha Grid Search and Pareto Curves

**Purpose:** Replace hand-picked `alpha=0.7/0.9` with a repeatable sweep. Prove `q_spec` is a controllable calibration signal.

**Files:**

- Create: `spectral_detection_posttrain/eval/score_fusion.py`
- Create: `spectral_detection_posttrain/eval/eval_alpha_grid.py`
- Create: `tests/test_score_fusion.py`
- Modify: `spectral_detection_posttrain/eval/eval_rerank.py`
- Modify: `spectral_detection_posttrain/configs/mvp.yaml`

Steps:

- [ ] Add `score_fusion.py` with:
  - `logit_safe(x, eps=1e-6)`
  - `blend_scores(score_cls, q_spec, alpha)`
  - `power_scores(score_cls, q_spec, alpha)`
  - `logit_fusion_scores(score_cls, q_spec, a, b, c)`
  - `fuse_scores(score_cls, q_spec, method, alpha, a=1, b=1, c=0)`

- [ ] Add tests in `tests/test_score_fusion.py`:
  - alpha `1.0` returns detector score for blend/power
  - alpha `0.0` returns `q_spec`
  - fused scores stay in `[0, 1]`
  - higher `q_spec` increases final score when detector score is fixed

- [ ] Modify `eval_rerank.py` to call `fuse_scores()` instead of inline blend/multiply logic.

- [ ] Add `analysis.alpha_grid` to `spectral_detection_posttrain/configs/mvp.yaml`:

```yaml
analysis:
  alpha_grid: [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
  fusion_methods: [blend, power]
```

- [ ] Create `eval_alpha_grid.py`:
  - Inputs: config, candidates, quality checkpoint, run name, fusion method.
  - For each alpha, evaluate AP50, Precision, Recall, ECE, High-conf FP, High-conf FN.
  - Save `alpha_grid_metrics.csv`.
  - Save `alpha_grid_pareto.png` with AP50/Recall/Precision/ECE/High-conf FP vs alpha.

- [ ] Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_score_fusion.py -q
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.eval.eval_alpha_grid --config spectral_detection_posttrain/configs/mvp.yaml --candidates runs/mvp_qh_candidates_val_clean/candidates.pt --quality-checkpoint runs/mvp_qh_roi_amp_structure/quality_head_last.pth --run-name qh_alpha_clean --fusion-method blend
```

Expected:

- Tests pass.
- `runs/qh_alpha_clean/alpha_grid_metrics.csv` exists.
- `runs/qh_alpha_clean/alpha_grid_pareto.png` exists.

- [ ] Commit:

```powershell
git add spectral_detection_posttrain/eval/score_fusion.py spectral_detection_posttrain/eval/eval_alpha_grid.py spectral_detection_posttrain/eval/eval_rerank.py spectral_detection_posttrain/configs/mvp.yaml tests/test_score_fusion.py
git commit -m "feat: add quality-head alpha grid evaluation"
```

---

## Task 2: Calibration and Fixed Operating-Point Metrics

**Purpose:** Change the main claim from "AP improvement" to "calibration and high-confidence error suppression under fair operating points."

**Files:**

- Create: `spectral_detection_posttrain/eval/operating_points.py`
- Create: `tests/test_operating_points.py`
- Modify: `spectral_detection_posttrain/eval/eval_rerank.py`
- Modify: `spectral_detection_posttrain/eval/eval_alpha_grid.py`

Steps:

- [ ] Implement `operating_points.py`:
  - `precision_recall_curve_from_scored_matches(scored, total_gt)`
  - `precision_at_recall(scored, total_gt, target_recall=0.85)`
  - `recall_at_precision(scored, total_gt, target_precision=0.75)`
  - `threshold_for_recall(scored, total_gt, target_recall)`
  - `high_conf_fp_at_threshold(scored, threshold)`
  - `detection_ece(scored, bins=10)`

- [ ] Add tests:
  - Perfect ranking gives Precision@Recall=1.0.
  - Worse ranking lowers Precision@Recall.
  - threshold_for_recall returns lower/equal threshold as target recall increases.
  - ECE is lower for calibrated scores than overconfident wrong scores.

- [ ] Modify `eval_rerank.py` output to include:
  - `precision_at_recall_085`
  - `recall_at_precision_075`
  - `high_conf_fp_at_recall_085`
  - `threshold_at_recall_085`
  - `detection_ece`

- [ ] Modify `eval_alpha_grid.py` to plot:
  - AP50 vs alpha
  - Precision@Recall=0.85 vs alpha
  - High-conf FP at Recall=0.85 vs alpha
  - ECE vs alpha

- [ ] Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_operating_points.py -q
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.eval.eval_rerank --config spectral_detection_posttrain/configs/mvp.yaml --candidates runs/mvp_qh_candidates_val_clean/candidates.pt --quality-checkpoint runs/mvp_qh_roi_amp_structure/quality_head_last.pth --run-name qh_fixedop_clean --method learned --combine blend --alpha 0.9
```

Expected:

- `eval_rerank_metrics.json` includes fixed operating-point metrics.

- [ ] Commit:

```powershell
git add spectral_detection_posttrain/eval/operating_points.py spectral_detection_posttrain/eval/eval_rerank.py spectral_detection_posttrain/eval/eval_alpha_grid.py tests/test_operating_points.py
git commit -m "feat: add fixed operating point calibration metrics"
```

---

## Task 3: Oracle Upper Bound

**Purpose:** Determine whether the spectral verifier can improve AP or is mainly a calibration/error-suppression signal.

**Files:**

- Modify: `spectral_detection_posttrain/eval/eval_rerank.py`
- Modify: `spectral_detection_posttrain/eval/eval_oracle_ramp.py`
- Create or modify: `spectral_detection_posttrain/eval/eval_alpha_grid.py`

Steps:

- [ ] Add oracle modes:
  - `oracle_ramp`: use normalized `raw_r_amp`.
  - `oracle_qtarget`: use `IoU * normalized(raw_r_amp)` for matched TP and `0` for FP.

- [ ] Ensure `oracle_qtarget` is allowed only for analysis, not deployment. Add `"oracle_mode"` to output JSON.

- [ ] Run oracle grid on clean/random/checkerboard:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.eval.eval_alpha_grid --config spectral_detection_posttrain/configs/mvp.yaml --candidates runs/mvp_qh_candidates_val_clean/candidates.pt --normalization-cache runs/mvp_qh_candidates_train_clean/candidates.pt --run-name qh_oracle_clean --method oracle_qtarget --fusion-method blend
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.eval.eval_alpha_grid --config spectral_detection_posttrain/configs/mvp.yaml --candidates runs/mvp_qh_candidates_val_random/candidates.pt --normalization-cache runs/mvp_qh_candidates_train_clean/candidates.pt --run-name qh_oracle_random --method oracle_qtarget --fusion-method blend
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.eval.eval_alpha_grid --config spectral_detection_posttrain/configs/mvp.yaml --candidates runs/mvp_qh_candidates_val_checker/candidates.pt --normalization-cache runs/mvp_qh_candidates_train_clean/candidates.pt --run-name qh_oracle_checker --method oracle_qtarget --fusion-method blend
```

Expected interpretation:

- If oracle improves AP and learned does not, quality-head learning/features need improvement.
- If oracle only improves ECE/FP, the verifier should be positioned as calibration/error suppression.

- [ ] Commit:

```powershell
git add spectral_detection_posttrain/eval/eval_rerank.py spectral_detection_posttrain/eval/eval_oracle_ramp.py spectral_detection_posttrain/eval/eval_alpha_grid.py
git commit -m "feat: add oracle quality reranking upper bound"
```

---

## Task 4: Feature Ablation Including RandomAmp

**Purpose:** Prove whether frequency features add incremental value beyond ROI features.

**Files:**

- Modify: `spectral_detection_posttrain/models/spectral_quality_head.py`
- Modify: `spectral_detection_posttrain/train/train_quality_head.py`
- Create: `spectral_detection_posttrain/eval/eval_feature_ablation.py`
- Add tests in `tests/test_spectral_quality_head.py`

Steps:

- [ ] Ensure `quality_input_dim()` and `build_quality_features()` support:
  - `roi`
  - `amp`
  - `roi_amp`
  - `roi_amp_structure`
  - `roi_random_amp`

- [ ] Implement deterministic RandomAmp:
  - During training/eval, replace each candidate's `amp_profiles` with a different candidate's profile from the same split.
  - Use config seed for reproducibility.

- [ ] Add tests:
  - `roi_amp` input dimension equals ROI dim + amp bins.
  - `amp` mode ignores ROI feature.
  - `roi_random_amp` preserves tensor shape but changes at least one row when there are 2+ candidates.

- [ ] Create `eval_feature_ablation.py`:
  - Train/evaluate or read checkpoints for:
    - `QH-ROI`
    - `QH-Amp`
    - `QH-ROI+Amp`
    - `QH-ROI+Amp+Struct`
    - `QH-RandomAmp`
  - Save `feature_ablation_metrics.csv`.
  - Save `feature_ablation_bars.png` for q AUC, ECE, AP50, Precision@Recall=0.85.

- [ ] Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.eval.eval_feature_ablation --config spectral_detection_posttrain/configs/mvp.yaml --train-candidates runs/mvp_qh_candidates_train_clean/candidates.pt --val-candidates runs/mvp_qh_candidates_val_clean/candidates.pt --run-name qh_feature_ablation --epochs 8
```

Acceptance:

- If `ROI+Amp` > `ROI` or `RandomAmp`, frequency evidence has incremental value.
- If `ROI` ~= `ROI+Amp`, frequency branch should be framed as interpretability/regularization unless larger data shows otherwise.

- [ ] Commit:

```powershell
git add spectral_detection_posttrain/models/spectral_quality_head.py spectral_detection_posttrain/train/train_quality_head.py spectral_detection_posttrain/eval/eval_feature_ablation.py tests/test_spectral_quality_head.py
git commit -m "test: add quality head feature ablations"
```

---

## Task 5: Patch Position and Patch Type Stress Tests

**Purpose:** Find which error type the quality head suppresses: background FP, object FN, edge localization drift, or near-object duplicate detection.

**Files:**

- Modify: `spectral_detection_posttrain/datasets/patch_transform.py`
- Create: `spectral_detection_posttrain/eval/eval_patch_positions.py`
- Modify: `run_quality_head_analysis.bat`
- Add tests in `tests/test_detection_patch.py`

Steps:

- [ ] Add/verify patch placements:
  - `background`
  - `object`
  - `edge`
  - `near_object`
  - `random`

- [ ] Add patch types:
  - `random`
  - `checkerboard`
  - `texture`
  - `contrast`
  - `color_blob`

- [ ] Add tests:
  - object patch overlaps first GT box.
  - edge patch touches GT box boundary.
  - near-object patch is outside but close to GT box when image bounds allow.
  - background patch avoids GT box when possible.

- [ ] Create `eval_patch_positions.py`:
  - Generate or reuse candidate caches for each placement/type.
  - Evaluate baseline and learned QH at alpha grid.
  - Output per-group AP50/AP75/Recall/FP/FN/ECE/high-conf FP.

- [ ] Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.eval.eval_patch_positions --config spectral_detection_posttrain/configs/mvp.yaml --checkpoint runs/mvp_pf_baseline/checkpoint_last.pth --quality-checkpoint runs/mvp_qh_roi_amp_structure/quality_head_last.pth --run-name qh_patch_positions
```

Acceptance:

- Report must identify whether QH helps background FP, object-inside FN, edge localization drift, or near-object duplicate detections.

- [ ] Commit:

```powershell
git add spectral_detection_posttrain/datasets/patch_transform.py spectral_detection_posttrain/eval/eval_patch_positions.py tests/test_detection_patch.py run_quality_head_analysis.bat
git commit -m "feat: add patch position stress tests"
```

---

## Task 6: Structure Branch and AP75/Localization Analysis

**Purpose:** Evaluate the MPLSeg-inspired phase/structure idea where it should matter: localization and edge perturbations.

**Files:**

- Modify: `spectral_detection_posttrain/spectral/fft_features.py`
- Modify: `spectral_detection_posttrain/eval/detection_metrics.py`
- Modify: `spectral_detection_posttrain/eval/eval_rerank.py`
- Modify: `docs/quality_head_calibration_results_2026-06-03.md`

Steps:

- [ ] Add `ap75` to detection metrics by allowing `iou_threshold=0.75` or reporting both AP50/AP75 in one call.

- [ ] Add localization diagnostics:
  - mean IoU of TP detections
  - box center error
  - box size error

- [ ] Improve structure features:
  - Sobel magnitude stats
  - Laplacian variance
  - low-frequency phase mean/std for sin/cos
  - edge density

- [ ] Compare:
  - `QH-ROI+Amp`
  - `QH-ROI+Amp+Struct`
  on clean, edge patch, checkerboard patch.

- [ ] Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.eval.eval_feature_ablation --config spectral_detection_posttrain/configs/mvp.yaml --train-candidates runs/mvp_qh_candidates_train_clean/candidates.pt --val-candidates runs/mvp_qh_candidates_val_clean/candidates.pt --run-name qh_structure_ablation --epochs 8
```

Acceptance:

- Structure branch should be judged mainly by AP75/localization/edge-patch metrics, not only AP50.

- [ ] Commit:

```powershell
git add spectral_detection_posttrain/spectral/fft_features.py spectral_detection_posttrain/eval/detection_metrics.py spectral_detection_posttrain/eval/eval_rerank.py docs/quality_head_calibration_results_2026-06-03.md
git commit -m "feat: add structure localization diagnostics"
```

---

## Task 7: VOC Small Subset Transfer

**Purpose:** Check whether q_spec TP/FP AUC is not just a Penn-Fudan pedestrian artifact.

**Files:**

- Create: `spectral_detection_posttrain/datasets/voc_subset.py`
- Create: `spectral_detection_posttrain/configs/voc_small.yaml`
- Create: `run_voc_quality_head_smoke.bat`
- Add tests for dataset filtering if feasible.

Steps:

- [ ] Implement VOC subset dataset:
  - Classes: `person`, `dog`, `car`
  - Configurable max images per class.
  - Convert annotations to TorchVision detection target format.

- [ ] Add config:

```yaml
data:
  name: voc
  root: ./data
  classes: [person, dog, car]
  max_train_images: 300
  max_val_images: 100
model:
  num_classes: 4
```

- [ ] Train or load a VOC baseline detector for a short run.

- [ ] Cache candidates and evaluate q_spec AUC:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.train.train_baseline --config spectral_detection_posttrain/configs/voc_small.yaml --run-name voc_small_baseline --epochs 1
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.spectral.roi_spectral_dataset --config spectral_detection_posttrain/configs/voc_small.yaml --checkpoint runs/voc_small_baseline/checkpoint_last.pth --split val --run-name voc_small_candidates_val --output runs/voc_small_candidates_val/candidates.pt
```

Acceptance:

- Report q_spec or oracle R_amp TP/FP AUC by class.
- If VOC AUC remains above 0.7, the direction is worth expanding.

- [ ] Commit:

```powershell
git add spectral_detection_posttrain/datasets/voc_subset.py spectral_detection_posttrain/configs/voc_small.yaml run_voc_quality_head_smoke.bat
git commit -m "feat: add VOC small subset quality-head smoke"
```

---

## Final Verification

Run after Tasks 1-6:

```powershell
$env:TEMP='E:\tmp'
$env:TMP='E:\tmp'
$env:MPLCONFIGDIR='E:\tmp\matplotlib'
$env:TORCH_HOME='E:\tmp\torch'
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests -q
run_quality_head_analysis.bat
```

Expected:

- All tests pass.
- `runs/qh_alpha_clean/alpha_grid_metrics.csv` exists.
- `runs/qh_feature_ablation/feature_ablation_metrics.csv` exists.
- `runs/qh_patch_positions/patch_position_metrics.csv` exists.
- `docs/quality_head_calibration_results_2026-06-03.md` summarizes:
  - Pareto curve
  - fixed Recall/Precision metrics
  - oracle upper bound
  - feature ablation
  - patch-position groups

Final commit:

```powershell
git add README.md docs/quality_head_calibration_results_2026-06-03.md run_quality_head_analysis.bat
git commit -m "docs: summarize quality head calibration analysis"
```

Push:

```powershell
$env:GH_CONFIG_DIR='C:\Users\青云志\AppData\Roaming\GitHub CLI'
$token = gh auth token
git push "https://x-access-token:$token@github.com/qyz7438/RLimage.git" main:main
```

---

## Self-Review

Spec coverage:

- Alpha grid and Pareto curves: Task 1.
- Calibration and matched operating-point metrics: Task 2.
- Oracle upper bound: Task 3.
- Feature ablation including RandomAmp: Task 4.
- Phase/structure branch as engineering structure features: Task 6.
- Patch type/position expansion: Task 5.
- VOC small subset transfer: Task 7.
- No end-to-end detector retraining: explicitly excluded until this analysis is complete.

Known limitations:

- VOC task should not be started before Tasks 1-6 clarify Penn-Fudan behavior.
- `q_spec` may remain primarily a calibration signal; this is acceptable if matched-recall FP/ECE improve.
