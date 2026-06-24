# Round 2.5 MPLSeg-Decoupled RLVR Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an MPLSeg-inspired amplitude/phase-structure decoupled verifier to the target-detection RLVR post-training loop, then test whether structure-aware reward improves localization and patch robustness beyond IoU-only and amplitude-only controls.

**Architecture:** Keep the stable Round 2.3 RLVR training shell: frozen baseline rollout, signed ROI policy loss, KL to baseline, `det_loss_weight=0`, and freeze-state sanity gates. Upgrade the verifier from single `R_amp` to two independent ROI evidence branches: amplitude semantic evidence (`S_amp`) and phase/structure localization evidence (`S_struct`). Feed both as non-differentiable, verifiable reward components into `DetectionVerifierConfig`, and evaluate with clean plus object-positioned patch scenes.

**Tech Stack:** Python, PyTorch, TorchVision Faster R-CNN, Penn-Fudan, existing `spectral_detection_posttrain` package, pytest, NNI GridSearch, local conda env `E:\anaconda\01\envs\RLimage`.

---

## Why This Round Exists

MPLSeg's useful idea for this project is not "FFT similarity is better." Its useful idea is the decoupling:

- Magnitude carries semantic-frequency evidence.
- Phase carries spatial structure and localization evidence.

For detection RLVR, that maps to:

```text
S_amp:
  ROI_pred vs ROI_gt amplitude radial profile similarity
  -> "does this candidate region contain target-like semantic frequency evidence?"

S_struct:
  ROI_pred vs ROI_gt phase/edge/structure similarity
  -> "is this candidate region structurally aligned with the object?"
```

Round 2.4 showed that `R_amp` alone does not reliably beat IoU-only or shuffled controls on Penn-Fudan. Round 2.5 therefore stops treating amplitude as the whole verifier and tests the decoupled verifier:

```text
R_box =
    w_iou    * IoU
  + w_cls    * class_correct
  + w_amp    * S_amp
  + w_struct * S_struct
  - w_fp     * high_conf_fp
```

References:

- Paper: https://www.sciencedirect.com/science/article/pii/S1566253524000927
- Official code: https://github.com/qyan0131/MPLSeg

---

## Non-Goals

- Do not insert MPLSeg AFM blocks into the detector backbone.
- Do not train the detector with full supervised detector loss in Round 2.5.
- Do not multiply verifier scores into the whole detector loss.
- Do not compare full phase spectra with L2 distance.
- Do not claim frequency causality unless `Amp+Struct` beats both `IoU-only` and shuffled controls under the same constraints.

---

## Current Constraints To Preserve

Round 2.3/2.4 established these stability rules:

```text
det_loss_weight = 0.0
policy_objective = signed
policy_loss_weight = 0.0003
baseline_kl_weight = 10.0
rollout_source = baseline
unfreeze = cls
optimizer = adamw
early_stopping_patience = 2
```

Do not change these in the first Round 2.5 matrix. If the structure verifier helps, a later round can expand to `unfreeze=roi` or light box-head updates.

---

## File Map

- Modify: `spectral_detection_posttrain/spectral/fft_features.py`
  Add stable phase/structure similarity functions: phase correlation, low-frequency phase-stat similarity, Sobel edge similarity, and combined structure similarity.

- Modify: `spectral_detection_posttrain/spectral/rlvr_reward.py`
  Add `compute_per_box_structure` and shared verifier diagnostics for dynamic range.

- Modify: `spectral_detection_posttrain/rlvr/detection_verifier.py`
  Let `DetectionVerifierConfig.signal` activate amplitude and structure independently. Pass `s_struct` through the same score filter and candidate ordering as `s_amp`.

- Modify: `spectral_detection_posttrain/train/posttrain_rlvr.py`
  Add CLI support for `structure`, `shuffled_structure`, `amp_structure`, and `shuffled_amp_structure`; compute `S_struct` for each rollout box; log reward component diagnostics.

- Modify: `spectral_detection_posttrain/nni_rlvr_trial.py`
  Add Round 2.5 objective, extra patch scenes, `struct_weight`, and complete readable result rows.

- Create: `nni_configs/rlvr_round25_search_space.json`
  Fixed, interpretable Round 2.5 verifier matrix.

- Create: `nni_configs/rlvr_round25_config.yml`
  NNI GridSearch config for the Round 2.5 matrix.

- Create: `run_nni_rlvr_round25.bat`
  One-command Windows entry point.

- Create: `tests/test_phase_structure_features.py`
  Unit tests for phase correlation and structure similarity.

- Create: `tests/test_round25_mplseg_verifier.py`
  Unit tests for reward wiring, signal activation, and result readability.

- Create: `spectral_detection_posttrain/analysis/summarize_round25_results.py`
  Local result summarizer for pairwise deltas and interpretation gates.

- Create: `docs/rlvr_round25_mplseg_decoupled_report.md`
  Final experiment report after the matrix runs.

---

## Task 1: Add Failing Tests For Phase/Structure Features

**Files:**
- Create: `tests/test_phase_structure_features.py`

- [ ] **Step 1: Create the test file**

Create `tests/test_phase_structure_features.py`:

```python
import torch

from spectral_detection_posttrain.spectral.fft_features import (
    compute_structure_similarity,
    edge_similarity_score,
    lowfreq_phase_similarity,
    phase_correlation_score,
)


def _roi_with_square(shift_x: int = 0, shift_y: int = 0) -> torch.Tensor:
    roi = torch.zeros((3, 64, 64), dtype=torch.float32)
    y1 = 18 + shift_y
    y2 = 46 + shift_y
    x1 = 20 + shift_x
    x2 = 44 + shift_x
    roi[:, y1:y2, x1:x2] = 1.0
    return roi


def test_phase_correlation_is_high_for_identical_roi():
    roi = _roi_with_square()

    score = phase_correlation_score(roi, roi)

    assert 0.90 <= float(score.item()) <= 1.0


def test_phase_correlation_tolerates_small_translation_better_than_noise():
    roi = _roi_with_square()
    shifted = _roi_with_square(shift_x=2, shift_y=1)
    noise = torch.rand_like(roi)

    shifted_score = phase_correlation_score(roi, shifted)
    noise_score = phase_correlation_score(roi, noise)

    assert shifted_score > noise_score


def test_edge_similarity_rewards_same_structure():
    roi = _roi_with_square()
    shifted = _roi_with_square(shift_x=3, shift_y=2)
    noise = torch.rand_like(roi)

    same = edge_similarity_score(roi, roi)
    moved = edge_similarity_score(roi, shifted)
    random = edge_similarity_score(roi, noise)

    assert same > moved
    assert moved > random
    assert 0.0 <= float(random.item()) <= 1.0


def test_lowfreq_phase_similarity_is_bounded():
    roi = _roi_with_square()
    noise = torch.rand_like(roi)

    score = lowfreq_phase_similarity(roi, noise)

    assert 0.0 <= float(score.item()) <= 1.0


def test_structure_similarity_combines_phase_and_edges():
    roi = _roi_with_square()
    shifted = _roi_with_square(shift_x=2, shift_y=2)
    noise = torch.rand_like(roi)

    same = compute_structure_similarity(roi, roi)
    moved = compute_structure_similarity(roi, shifted)
    random = compute_structure_similarity(roi, noise)

    assert same > moved
    assert moved > random
    assert 0.0 <= float(random.item()) <= 1.0
    assert 0.0 <= float(same.item()) <= 1.0
```

- [ ] **Step 2: Run the failing tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_phase_structure_features.py -v
```

Expected: fails because `phase_correlation_score`, `edge_similarity_score`, `lowfreq_phase_similarity`, and `compute_structure_similarity` are not defined.

- [ ] **Step 3: Commit the failing tests**

```powershell
git add tests/test_phase_structure_features.py
git commit -m "test: add phase structure verifier expectations"
```

---

## Task 2: Implement Stable Phase/Structure Similarity

**Files:**
- Modify: `spectral_detection_posttrain/spectral/fft_features.py`

- [ ] **Step 1: Add helper functions**

Append these functions to `spectral_detection_posttrain/spectral/fft_features.py`:

```python
def _normalized_sobel_magnitude(roi: torch.Tensor) -> torch.Tensor:
    gray = roi.mean(dim=0, keepdim=True).unsqueeze(0)
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=roi.dtype,
        device=roi.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=roi.dtype,
        device=roi.device,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, sobel_x, padding=1).squeeze()
    grad_y = F.conv2d(gray, sobel_y, padding=1).squeeze()
    mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)
    return (mag - mag.min()) / (mag.max() - mag.min()).clamp(min=1e-6)


def phase_correlation_score(roi_pred: torch.Tensor, roi_gt: torch.Tensor, use_hann: bool = True) -> torch.Tensor:
    pred_gray = _gray_with_optional_hann(roi_pred, use_hann=use_hann)
    gt_gray = _gray_with_optional_hann(roi_gt, use_hann=use_hann)
    pred_fft = torch.fft.fft2(pred_gray, dim=(-2, -1))
    gt_fft = torch.fft.fft2(gt_gray, dim=(-2, -1))
    cross_power = pred_fft * torch.conj(gt_fft)
    cross_power = cross_power / cross_power.abs().clamp(min=1e-6)
    corr = torch.fft.ifft2(cross_power, dim=(-2, -1)).abs()
    return corr.max().clamp(0.0, 1.0)


def edge_similarity_score(roi_pred: torch.Tensor, roi_gt: torch.Tensor) -> torch.Tensor:
    pred_edge = _normalized_sobel_magnitude(roi_pred).flatten()
    gt_edge = _normalized_sobel_magnitude(roi_gt).flatten()
    cosine = F.cosine_similarity(pred_edge, gt_edge, dim=0).clamp(-1.0, 1.0)
    return ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)


def lowfreq_phase_similarity(
    roi_pred: torch.Tensor,
    roi_gt: torch.Tensor,
    radius_ratio: float = 0.25,
    use_hann: bool = True,
) -> torch.Tensor:
    pred_stats = compute_lowfreq_phase_stats(roi_pred, radius_ratio=radius_ratio, use_hann=use_hann)
    gt_stats = compute_lowfreq_phase_stats(roi_gt, radius_ratio=radius_ratio, use_hann=use_hann)
    mse = torch.mean((pred_stats - gt_stats).square())
    return torch.exp(-mse).clamp(0.0, 1.0)


def compute_structure_similarity(
    roi_pred: torch.Tensor,
    roi_gt: torch.Tensor,
    phase_weight: float = 0.45,
    edge_weight: float = 0.35,
    lowfreq_weight: float = 0.20,
) -> torch.Tensor:
    phase = phase_correlation_score(roi_pred, roi_gt)
    edge = edge_similarity_score(roi_pred, roi_gt)
    lowfreq = lowfreq_phase_similarity(roi_pred, roi_gt)
    total_weight = max(phase_weight + edge_weight + lowfreq_weight, 1e-6)
    score = (phase_weight * phase + edge_weight * edge + lowfreq_weight * lowfreq) / total_weight
    return score.clamp(0.0, 1.0)
```

- [ ] **Step 2: Remove repeated low-frequency phase work in `compute_sobel_structure_features`**

Replace the last two calls in `compute_sobel_structure_features`:

```python
            compute_lowfreq_phase_stats(roi)[0],
            compute_lowfreq_phase_stats(roi)[2],
```

with:

```python
            phase_stats[0],
            phase_stats[2],
```

and add before the `return torch.stack(...)` block:

```python
    phase_stats = compute_lowfreq_phase_stats(roi)
```

- [ ] **Step 3: Run feature tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_phase_structure_features.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```powershell
git add spectral_detection_posttrain/spectral/fft_features.py tests/test_phase_structure_features.py
git commit -m "feat: add phase structure verifier features"
```

---

## Task 3: Compute Per-Box Structure Scores

**Files:**
- Modify: `spectral_detection_posttrain/spectral/rlvr_reward.py`
- Create: `tests/test_round25_mplseg_verifier.py`

- [ ] **Step 1: Add failing tests for per-box structure**

Create `tests/test_round25_mplseg_verifier.py`:

```python
import torch

from spectral_detection_posttrain.spectral.rlvr_reward import compute_per_box_structure


def test_compute_per_box_structure_returns_zero_for_unmatched_box():
    image = torch.zeros((3, 64, 64), dtype=torch.float32)
    image[:, 16:48, 18:46] = 1.0
    pred_boxes = torch.tensor([[18.0, 16.0, 46.0, 48.0], [0.0, 0.0, 8.0, 8.0]])
    gt_boxes = torch.tensor([[18.0, 16.0, 46.0, 48.0]])
    best_gt_indices = torch.tensor([0, -1])

    scores = compute_per_box_structure(image, pred_boxes, gt_boxes, best_gt_indices)

    assert scores.shape == (2,)
    assert 0.80 <= float(scores[0].item()) <= 1.0
    assert float(scores[1].item()) == 0.0


def test_compute_per_box_structure_is_lower_for_bad_box():
    image = torch.zeros((3, 64, 64), dtype=torch.float32)
    image[:, 16:48, 18:46] = 1.0
    pred_boxes = torch.tensor([[18.0, 16.0, 46.0, 48.0], [0.0, 0.0, 28.0, 28.0]])
    gt_boxes = torch.tensor([[18.0, 16.0, 46.0, 48.0]])
    best_gt_indices = torch.tensor([0, 0])

    scores = compute_per_box_structure(image, pred_boxes, gt_boxes, best_gt_indices)

    assert scores[0] > scores[1]
```

- [ ] **Step 2: Run failing test**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round25_mplseg_verifier.py -v
```

Expected: fails because `compute_per_box_structure` is not defined.

- [ ] **Step 3: Implement `compute_per_box_structure`**

Modify imports in `spectral_detection_posttrain/spectral/rlvr_reward.py`:

```python
from spectral_detection_posttrain.spectral.fft_features import (
    compute_amplitude_profile,
    compute_structure_similarity,
)
```

Add this function after `compute_per_box_ramp`:

```python
def compute_per_box_structure(
    image: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    best_gt_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute phase/edge structure similarity for each predicted box.

    Unmatched boxes receive 0.0 because there is no verifiable GT structure.
    """
    values: list[float] = []
    gt_roi_cache: dict[int, torch.Tensor] = {}

    for pred_box, gt_idx in zip(pred_boxes, best_gt_indices):
        idx = int(gt_idx.item())
        if idx < 0 or len(gt_boxes) == 0:
            values.append(0.0)
            continue
        pred_roi = crop_and_resize_roi(image, pred_box)
        if idx not in gt_roi_cache:
            gt_roi_cache[idx] = crop_and_resize_roi(image, gt_boxes[idx])
        score = compute_structure_similarity(pred_roi, gt_roi_cache[idx])
        values.append(float(score.item()))

    return torch.tensor(values, dtype=torch.float32)
```

- [ ] **Step 4: Run structure tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round25_mplseg_verifier.py tests/test_phase_structure_features.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add spectral_detection_posttrain/spectral/rlvr_reward.py tests/test_round25_mplseg_verifier.py
git commit -m "feat: compute structure verifier per box"
```

---

## Task 4: Wire Structure Scores Into The RLVR Verifier

**Files:**
- Modify: `spectral_detection_posttrain/rlvr/detection_verifier.py`
- Modify: `tests/test_round25_mplseg_verifier.py`

- [ ] **Step 1: Add signal activation tests**

Append to `tests/test_round25_mplseg_verifier.py`:

```python
from spectral_detection_posttrain.rlvr.detection_verifier import (
    DetectionVerifierConfig,
    build_rewarded_roi_actions,
    compute_box_rewards,
    shuffle_tp_values,
)


def test_amp_structure_reward_uses_both_components():
    ious = torch.tensor([0.6])
    class_correct = torch.tensor([1.0])
    scores = torch.tensor([0.7])
    matched = torch.tensor([True])
    s_amp = torch.tensor([0.5])
    s_struct = torch.tensor([0.8])

    cfg = DetectionVerifierConfig(
        signal="amp_structure",
        w_iou=1.0,
        w_cls=0.2,
        w_amp=0.1,
        w_struct=0.3,
        w_hconf_fp=0.5,
    )

    reward = compute_box_rewards(cfg, ious, class_correct, scores, matched, s_amp=s_amp, s_struct=s_struct)

    assert float(reward.item()) == torch.tensor(0.6 + 0.2 + 0.05 + 0.24).item()


def test_structure_signal_ignores_amp_component():
    ious = torch.tensor([0.6])
    class_correct = torch.tensor([1.0])
    scores = torch.tensor([0.7])
    matched = torch.tensor([True])
    s_amp = torch.tensor([1.0])
    s_struct = torch.tensor([0.5])

    cfg = DetectionVerifierConfig(signal="structure", w_amp=0.9, w_struct=0.2)
    reward = compute_box_rewards(cfg, ious, class_correct, scores, matched, s_amp=s_amp, s_struct=s_struct)

    assert float(reward.item()) == torch.tensor(0.6 + 0.2 + 0.1).item()


def test_build_rewarded_actions_filters_structure_with_same_mask_as_boxes():
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        "labels": torch.tensor([1, 1]),
        "scores": torch.tensor([0.95, 0.10]),
    }
    target = {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])}
    s_struct = torch.tensor([0.8, 0.2])
    cfg = DetectionVerifierConfig(signal="structure", w_struct=0.3)

    actions = build_rewarded_roi_actions(
        prediction,
        target,
        num_classes=2,
        reward_score_threshold=0.2,
        verifier_cfg=cfg,
        s_struct=s_struct,
    )

    assert actions["structure_values"][0].item() == torch.tensor(0.8).item()


def test_shuffle_tp_values_keeps_unmatched_zero():
    values = torch.tensor([0.2, 0.8, 0.5, 0.0])
    matched = torch.tensor([True, True, True, False])

    shuffled = shuffle_tp_values(values, matched, seed=7)

    assert torch.all(shuffled[~matched] == 0.0)
    assert sorted(shuffled[matched].tolist()) == sorted(values[matched].tolist())
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round25_mplseg_verifier.py -v
```

Expected: fails because `s_struct` and `shuffle_tp_values` are not wired.

- [ ] **Step 3: Generalize shuffling**

In `spectral_detection_posttrain/rlvr/detection_verifier.py`, replace `shuffle_tp_ramp` with this generic function and keep the old name as a compatibility alias:

```python
def shuffle_tp_values(values: torch.Tensor, matched: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    out = values.clone()
    tp_idx = torch.where(matched)[0]
    if tp_idx.numel() < 2:
        out[~matched] = 0.0
        return out
    generator = torch.Generator(device=values.device)
    if seed is not None:
        generator.manual_seed(seed)
    perm = tp_idx[torch.randperm(tp_idx.numel(), generator=generator, device=values.device)]
    out[tp_idx] = values[perm]
    out[~matched] = 0.0
    return out


def shuffle_tp_ramp(values: torch.Tensor, matched: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    return shuffle_tp_values(values, matched, seed=seed)
```

- [ ] **Step 4: Add signal helpers**

Add these helpers above `compute_box_rewards`:

```python
AMP_SIGNALS = {"ramp", "shuffled_ramp", "amp_structure", "shuffled_amp_structure"}
STRUCTURE_SIGNALS = {"structure", "shuffled_structure", "amp_structure", "shuffled_amp_structure"}


def signal_uses_amp(signal: str) -> bool:
    return signal in AMP_SIGNALS


def signal_uses_structure(signal: str) -> bool:
    return signal in STRUCTURE_SIGNALS
```

- [ ] **Step 5: Update `compute_box_rewards`**

Replace the weight activation logic in `compute_box_rewards` with:

```python
    amp_weight = cfg.w_amp if signal_uses_amp(cfg.signal) else 0.0
    struct_weight = cfg.w_struct if signal_uses_structure(cfg.signal) else 0.0
```

Keep the final reward line:

```python
    reward = cfg.w_iou * ious + cfg.w_cls * class_correct + amp_weight * amp + struct_weight * struct
```

- [ ] **Step 6: Update `build_rewarded_roi_actions` signature and masking**

Change the signature:

```python
def build_rewarded_roi_actions(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    num_classes: int,
    max_candidates: int = 40,
    reward_score_threshold: float = 0.2,
    verifier_cfg: DetectionVerifierConfig | None = None,
    s_amp: torch.Tensor | None = None,
    s_struct: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
```

After the `s_amp` keep-mask block, add:

```python
    if s_struct is not None:
        s_struct = s_struct[keep]
```

After the top-k `s_amp` ordering block, add:

```python
        if s_struct is not None:
            s_struct = s_struct[order]
```

After `amp` is created, add:

```python
    struct = torch.zeros_like(best_iou) if s_struct is None else s_struct.to(best_iou.device)
```

Replace the shuffle block with:

```python
    if cfg.signal == "shuffled_ramp":
        amp = shuffle_tp_values(amp, matched)
    if cfg.signal == "shuffled_structure":
        struct = shuffle_tp_values(struct, matched)
    if cfg.signal == "shuffled_amp_structure":
        amp = shuffle_tp_values(amp, matched)
        struct = shuffle_tp_values(struct, matched)
    amp = amp * matched.float()
    struct = struct * matched.float()
```

Update reward call:

```python
    rewards = compute_box_rewards(cfg, best_iou, class_correct, scores, matched, s_amp=amp, s_struct=struct)
```

Add to the returned dict:

```python
        "structure_values": struct.float(),
```

- [ ] **Step 7: Run verifier tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_rlvr_verifier.py tests/test_round25_mplseg_verifier.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```powershell
git add spectral_detection_posttrain/rlvr/detection_verifier.py tests/test_round25_mplseg_verifier.py
git commit -m "feat: wire decoupled spectral verifier rewards"
```

---

## Task 5: Add Structure Reward To RLVR Post-Training

**Files:**
- Modify: `spectral_detection_posttrain/train/posttrain_rlvr.py`
- Modify: `tests/test_rlvr.py`

- [ ] **Step 1: Update stale normalization test**

In `tests/test_rlvr.py`, replace `TestRampNormalization.test_zscore_normalization` with:

```python
    def test_percentile_clamp_normalization(self):
        stats = {"p05": 0.998, "p95": 0.999}
        raw = torch.tensor([0.9985, 0.997, 0.9995], dtype=torch.float32)
        norm = normalize_ramp(raw, stats)
        assert norm[0].item() == pytest.approx(0.5, abs=1e-4)
        assert norm[1].item() == pytest.approx(0.0, abs=1e-4)
        assert norm[2].item() == pytest.approx(1.0, abs=1e-4)
```

- [ ] **Step 2: Add CLI parse test**

Append to `tests/test_round25_mplseg_verifier.py`:

```python
def test_rlvr_parse_args_accepts_amp_structure_signal():
    from spectral_detection_posttrain.train.posttrain_rlvr import parse_args

    args = parse_args([
        "--config", "spectral_detection_posttrain/configs/smoke.yaml",
        "--checkpoint", "runs/baseline/checkpoint.pt",
        "--run-name", "round25_smoke",
        "--signal", "amp_structure",
        "--unfreeze", "cls",
        "--optimizer", "adamw",
        "--reward-lambda", "0.05",
        "--struct-weight", "0.2",
        "--policy-loss-weight", "0.0003",
        "--baseline-kl-weight", "10",
        "--det-loss-weight", "0",
        "--epochs", "1",
    ])

    assert args.signal == "amp_structure"
    assert args.struct_weight == 0.2
```

- [ ] **Step 3: Run failing tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_rlvr.py tests/test_round25_mplseg_verifier.py -v
```

Expected: parse test fails because `--struct-weight` and new signal choices are not accepted.

- [ ] **Step 4: Update imports**

In `spectral_detection_posttrain/train/posttrain_rlvr.py`, modify imports:

```python
from spectral_detection_posttrain.rlvr.detection_verifier import (
    DetectionVerifierConfig,
    build_rewarded_roi_actions,
    signal_uses_amp,
    signal_uses_structure,
)
from spectral_detection_posttrain.spectral.rlvr_reward import (
    compute_per_box_ramp,
    compute_per_box_structure,
    normalize_ramp,
)
```

- [ ] **Step 5: Update CLI arguments**

Replace the `--signal` choices with:

```python
    parser.add_argument(
        "--signal",
        required=True,
        choices=[
            "none",
            "ramp",
            "shuffled_ramp",
            "structure",
            "shuffled_structure",
            "amp_structure",
            "shuffled_amp_structure",
        ],
    )
```

Add after `--reward-lambda`:

```python
    parser.add_argument("--struct-weight", type=float, default=0.0)
```

- [ ] **Step 6: Update verifier config**

Replace:

```python
        w_iou=1.0, w_cls=0.2, w_amp=args.reward_lambda, w_struct=0.0,
```

with:

```python
        w_iou=1.0, w_cls=0.2, w_amp=args.reward_lambda, w_struct=args.struct_weight,
```

- [ ] **Step 7: Load amplitude stats only for amplitude signals**

Replace:

```python
    if args.signal in ("ramp", "shuffled_ramp") and args.r_amp_stats:
```

with:

```python
    if signal_uses_amp(args.signal) and args.r_amp_stats:
```

- [ ] **Step 8: Compute `s_struct_list` beside `s_amp_list`**

Replace:

```python
            s_amp_list: list[torch.Tensor] = []
```

with:

```python
            s_amp_list: list[torch.Tensor] = []
            s_struct_list: list[torch.Tensor] = []
```

Inside the rollout loop, after `best_gt_indices` is filled, replace the amplitude condition:

```python
                if args.signal in ("ramp", "shuffled_ramp") and r_amp_stats is not None and len(pred_boxes) > 0:
```

with:

```python
                if signal_uses_amp(args.signal) and r_amp_stats is not None and len(pred_boxes) > 0:
```

After the amplitude append block, add:

```python
                if signal_uses_structure(args.signal) and len(pred_boxes) > 0:
                    s_struct_list.append(compute_per_box_structure(image, pred_boxes, gt_boxes, best_gt_indices))
                else:
                    s_struct_list.append(torch.zeros(len(pred_boxes)))
```

- [ ] **Step 9: Pass `s_struct` into action building**

Replace the `actions = [` block with:

```python
            actions = [
                build_rewarded_roi_actions(
                    pred,
                    tgt,
                    num_classes=int(config["model"]["num_classes"]),
                    verifier_cfg=verifier_cfg,
                    max_candidates=args.max_candidates,
                    reward_score_threshold=args.reward_score_threshold,
                    s_amp=s_amp,
                    s_struct=s_struct,
                )
                for pred, tgt, s_amp, s_struct in zip(predictions, targets, s_amp_list, s_struct_list)
            ]
```

- [ ] **Step 10: Log reward components**

After `person_rate` is computed, add:

```python
            amp_mean = float(torch.cat([a["amp_values"] for a in actions], dim=0).mean().item()) if actions else 0.0
            struct_mean = float(torch.cat([a["structure_values"] for a in actions], dim=0).mean().item()) if actions else 0.0
```

Add to `progress.set_postfix(...)`:

```python
                amp=amp_mean,
                struct=struct_mean,
```

Add to the epoch `row`:

```python
            "amp_mean": amp_mean,
            "structure_mean": struct_mean,
```

Add to the final `result` dict:

```python
              "struct_weight": args.struct_weight,
```

- [ ] **Step 11: Run post-train tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_rlvr.py tests/test_rlvr_verifier.py tests/test_round25_mplseg_verifier.py -v
```

Expected: all tests pass.

- [ ] **Step 12: Commit**

```powershell
git add spectral_detection_posttrain/train/posttrain_rlvr.py tests/test_rlvr.py tests/test_round25_mplseg_verifier.py
git commit -m "feat: add structure verifier to rlvr training"
```

---

## Task 6: Add Round 2.5 NNI Matrix And Result Rows

**Files:**
- Modify: `spectral_detection_posttrain/nni_rlvr_trial.py`
- Create: `nni_configs/rlvr_round25_search_space.json`
- Create: `nni_configs/rlvr_round25_config.yml`
- Create: `run_nni_rlvr_round25.bat`
- Modify: `tests/test_round25_mplseg_verifier.py`

- [ ] **Step 1: Add objective and config tests**

Append to `tests/test_round25_mplseg_verifier.py`:

```python
def test_round25_objective_keeps_stability_constraints():
    from spectral_detection_posttrain.nni_rlvr_trial import compute_round25_objective

    baseline = {
        "clean": {"ap50": 0.87, "ap75": 0.64, "precision": 0.65, "recall": 0.89, "ece": 0.05, "num_predictions": 124},
        "object_edge_checkerboard": {"ap50": 0.87, "ap75": 0.58, "precision": 0.65, "recall": 0.89, "ece": 0.04, "num_predictions": 124},
        "object_inside_checkerboard": {"ap50": 0.84, "ap75": 0.52, "precision": 0.63, "recall": 0.86, "ece": 0.05, "num_predictions": 130},
        "near_object_checkerboard": {"ap50": 0.86, "ap75": 0.56, "precision": 0.64, "recall": 0.88, "ece": 0.05, "num_predictions": 128},
    }
    metrics = {
        "clean": {"ap50": 0.872, "ap75": 0.66, "precision": 0.66, "recall": 0.89, "ece": 0.04, "num_predictions": 123},
        "object_edge_checkerboard": {"ap50": 0.871, "ap75": 0.60, "precision": 0.66, "recall": 0.89, "ece": 0.035, "num_predictions": 123},
        "object_inside_checkerboard": {"ap50": 0.845, "ap75": 0.54, "precision": 0.64, "recall": 0.86, "ece": 0.045, "num_predictions": 128},
        "near_object_checkerboard": {"ap50": 0.865, "ap75": 0.58, "precision": 0.65, "recall": 0.88, "ece": 0.045, "num_predictions": 125},
    }

    objective = compute_round25_objective(metrics, baseline)

    assert objective["constraint_failed"] == ""
    assert objective["default"] > 0


def test_round25_objective_rejects_ap50_collapse():
    from spectral_detection_posttrain.nni_rlvr_trial import compute_round25_objective

    baseline = {
        "clean": {"ap50": 0.87, "ap75": 0.64, "precision": 0.65, "recall": 0.89, "ece": 0.05, "num_predictions": 124},
        "object_edge_checkerboard": {"ap50": 0.87, "ap75": 0.58, "precision": 0.65, "recall": 0.89, "ece": 0.04, "num_predictions": 124},
        "object_inside_checkerboard": {"ap50": 0.84, "ap75": 0.52, "precision": 0.63, "recall": 0.86, "ece": 0.05, "num_predictions": 130},
        "near_object_checkerboard": {"ap50": 0.86, "ap75": 0.56, "precision": 0.64, "recall": 0.88, "ece": 0.05, "num_predictions": 128},
    }
    metrics = {
        "clean": {"ap50": 0.79, "ap75": 0.66, "precision": 0.66, "recall": 0.89, "ece": 0.04, "num_predictions": 123},
        "object_edge_checkerboard": baseline["object_edge_checkerboard"],
        "object_inside_checkerboard": baseline["object_inside_checkerboard"],
        "near_object_checkerboard": baseline["near_object_checkerboard"],
    }

    objective = compute_round25_objective(metrics, baseline)

    assert objective["default"] == -1.0
    assert objective["constraint_failed"] == "clean_ap50"
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round25_mplseg_verifier.py -v
```

Expected: fails because `compute_round25_objective` is not defined.

- [ ] **Step 3: Add NNI signal helpers**

In `spectral_detection_posttrain/nni_rlvr_trial.py`, import helpers:

```python
from spectral_detection_posttrain.rlvr.detection_verifier import signal_uses_amp, signal_uses_structure
```

In `_run_rlvr`, read `struct_weight`:

```python
    struct_weight = float(params.get("struct_weight", 0.0))
```

Add to the `command` list after `--reward-lambda`:

```python
        "--struct-weight", str(struct_weight),
```

Replace:

```python
    if signal in ("ramp", "shuffled_ramp") and r_amp_stats_path is not None:
```

with:

```python
    if signal_uses_amp(signal) and r_amp_stats_path is not None:
```

In `main`, replace:

```python
    if signal in ("ramp", "shuffled_ramp"):
```

with:

```python
    if signal_uses_amp(signal):
```

- [ ] **Step 4: Evaluate all Round 2.5 scenes**

In `main`, replace the final eval scene list with:

```python
    for mode_key, patch_mode, patch_type in [
        ("clean", "none", "random"),
        ("object_edge_checkerboard", "object_edge", "checkerboard"),
        ("object_inside_checkerboard", "object_inside", "checkerboard"),
        ("near_object_checkerboard", "near_object", "checkerboard"),
    ]:
```

- [ ] **Step 5: Add Round 2.5 objective**

Add below `compute_round23_objective`:

```python
def compute_round25_objective(metrics: dict, baseline: dict) -> dict:
    required = ["clean", "object_edge_checkerboard", "object_inside_checkerboard", "near_object_checkerboard"]
    for scene in required:
        if not metrics.get(scene):
            return {"default": -1.0, "constraint_failed": f"missing_{scene}"}

    clean = metrics["clean"]
    base_clean = baseline["clean"]

    checks = [
        ("clean_ap50", clean.get("ap50", 0.0) >= base_clean["ap50"] - 0.03),
        ("clean_recall", clean.get("recall", 0.0) >= base_clean["recall"] - 0.04),
        ("clean_num_predictions", clean.get("num_predictions", 10**9) <= base_clean["num_predictions"] * 1.20),
    ]
    for scene in required[1:]:
        current = metrics[scene]
        base = baseline[scene]
        checks.extend([
            (f"{scene}_ap50", current.get("ap50", 0.0) >= base["ap50"] - 0.06),
            (f"{scene}_recall", current.get("recall", 0.0) >= base["recall"] - 0.05),
            (f"{scene}_num_predictions", current.get("num_predictions", 10**9) <= base["num_predictions"] * 1.25),
        ])

    for name, ok in checks:
        if not ok:
            return {"default": -1.0, "constraint_failed": name}

    score = 0.0
    for scene in required:
        current = metrics[scene]
        base = baseline[scene]
        ap75_gain = current.get("ap75", 0.0) - base.get("ap75", 0.0)
        ap50_gain = current.get("ap50", 0.0) - base.get("ap50", 0.0)
        ece_gain = base.get("ece", 0.0) - current.get("ece", 0.0)
        fp_gain = base.get("high_conf_fp_count", 0.0) - current.get("high_conf_fp_count", 0.0)
        score += current.get("ap50", 0.0) + 0.8 * current.get("ap75", 0.0)
        score += 2.0 * ap75_gain + 0.5 * ap50_gain + 0.2 * ece_gain + 0.02 * fp_gain

    return {"default": float(score), "constraint_failed": ""}
```

- [ ] **Step 6: Select Round 2.5 objective**

Replace:

```python
    if "round23" in args.run_prefix:
        objective = compute_round23_objective(metrics, baseline)
    else:
        objective = compute_round22_objective(metrics, baseline)
```

with:

```python
    if "round25" in args.run_prefix:
        objective = compute_round25_objective(metrics, baseline)
    elif "round23" in args.run_prefix:
        objective = compute_round23_objective(metrics, baseline)
    else:
        objective = compute_round22_objective(metrics, baseline)
```

- [ ] **Step 7: Make `eval_status` precise**

Replace:

```python
    eval_status = "ok" if len(metrics) >= 2 else "failed"
```

with:

```python
    expected_eval_count = 4 if "round25" in args.run_prefix else 2
    eval_status = "ok" if len(metrics) >= expected_eval_count else "failed"
```

- [ ] **Step 8: Add result fields for structure and patch scenes**

Extend `REQUIRED_ROUND23_RESULT_FIELDS` with:

```python
    "struct_weight",
    "inside_ap50", "inside_ap75", "inside_precision", "inside_recall",
    "inside_num_predictions", "inside_high_conf_fp_count", "inside_ece",
    "near_ap50", "near_ap75", "near_precision", "near_recall",
    "near_num_predictions", "near_high_conf_fp_count", "near_ece",
```

In `build_round23_result_row`, add:

```python
        "struct_weight": float(params.get("struct_weight", 0.0)),
        "inside_ap50": _metric(metrics, "object_inside_checkerboard", "ap50"),
        "inside_ap75": _metric(metrics, "object_inside_checkerboard", "ap75"),
        "inside_precision": _metric(metrics, "object_inside_checkerboard", "precision"),
        "inside_recall": _metric(metrics, "object_inside_checkerboard", "recall"),
        "inside_num_predictions": _metric(metrics, "object_inside_checkerboard", "num_predictions"),
        "inside_high_conf_fp_count": _metric(metrics, "object_inside_checkerboard", "high_conf_fp_count"),
        "inside_ece": _metric(metrics, "object_inside_checkerboard", "ece"),
        "near_ap50": _metric(metrics, "near_object_checkerboard", "ap50"),
        "near_ap75": _metric(metrics, "near_object_checkerboard", "ap75"),
        "near_precision": _metric(metrics, "near_object_checkerboard", "precision"),
        "near_recall": _metric(metrics, "near_object_checkerboard", "recall"),
        "near_num_predictions": _metric(metrics, "near_object_checkerboard", "num_predictions"),
        "near_high_conf_fp_count": _metric(metrics, "near_object_checkerboard", "high_conf_fp_count"),
        "near_ece": _metric(metrics, "near_object_checkerboard", "ece"),
```

- [ ] **Step 9: Create search space**

Create `nni_configs/rlvr_round25_search_space.json`:

```json
{
  "preset": {
    "_type": "choice",
    "_value": [
      {
        "name": "null_no_update",
        "signal": "none",
        "reward_lambda": 0.0,
        "struct_weight": 0.0,
        "policy_loss_weight": 0.0,
        "det_loss_weight": 0.0,
        "baseline_kl_weight": 0.0,
        "box_loss_weight": 0.0,
        "unfreeze": "cls",
        "optimizer": "adamw",
        "temperature": 1.0,
        "max_candidates": 40,
        "reward_score_threshold": 0.2,
        "rollout_source": "baseline",
        "policy_objective": "signed"
      },
      {
        "name": "signed_iou_0003_kl10",
        "signal": "none",
        "reward_lambda": 0.0,
        "struct_weight": 0.0,
        "policy_loss_weight": 0.0003,
        "det_loss_weight": 0.0,
        "baseline_kl_weight": 10.0,
        "box_loss_weight": 0.0,
        "unfreeze": "cls",
        "optimizer": "adamw",
        "temperature": 1.0,
        "max_candidates": 40,
        "reward_score_threshold": 0.2,
        "rollout_source": "baseline",
        "policy_objective": "signed"
      },
      {
        "name": "signed_amp_0003_kl10",
        "signal": "ramp",
        "reward_lambda": 0.1,
        "struct_weight": 0.0,
        "policy_loss_weight": 0.0003,
        "det_loss_weight": 0.0,
        "baseline_kl_weight": 10.0,
        "box_loss_weight": 0.0,
        "unfreeze": "cls",
        "optimizer": "adamw",
        "temperature": 1.0,
        "max_candidates": 40,
        "reward_score_threshold": 0.2,
        "rollout_source": "baseline",
        "policy_objective": "signed"
      },
      {
        "name": "signed_structure_0003_kl10",
        "signal": "structure",
        "reward_lambda": 0.0,
        "struct_weight": 0.2,
        "policy_loss_weight": 0.0003,
        "det_loss_weight": 0.0,
        "baseline_kl_weight": 10.0,
        "box_loss_weight": 0.0,
        "unfreeze": "cls",
        "optimizer": "adamw",
        "temperature": 1.0,
        "max_candidates": 40,
        "reward_score_threshold": 0.2,
        "rollout_source": "baseline",
        "policy_objective": "signed"
      },
      {
        "name": "signed_amp_structure_005_02_0003_kl10",
        "signal": "amp_structure",
        "reward_lambda": 0.05,
        "struct_weight": 0.2,
        "policy_loss_weight": 0.0003,
        "det_loss_weight": 0.0,
        "baseline_kl_weight": 10.0,
        "box_loss_weight": 0.0,
        "unfreeze": "cls",
        "optimizer": "adamw",
        "temperature": 1.0,
        "max_candidates": 40,
        "reward_score_threshold": 0.2,
        "rollout_source": "baseline",
        "policy_objective": "signed"
      },
      {
        "name": "signed_amp_structure_010_03_0003_kl10",
        "signal": "amp_structure",
        "reward_lambda": 0.1,
        "struct_weight": 0.3,
        "policy_loss_weight": 0.0003,
        "det_loss_weight": 0.0,
        "baseline_kl_weight": 10.0,
        "box_loss_weight": 0.0,
        "unfreeze": "cls",
        "optimizer": "adamw",
        "temperature": 1.0,
        "max_candidates": 40,
        "reward_score_threshold": 0.2,
        "rollout_source": "baseline",
        "policy_objective": "signed"
      },
      {
        "name": "signed_shuffled_structure_0003_kl10",
        "signal": "shuffled_structure",
        "reward_lambda": 0.0,
        "struct_weight": 0.2,
        "policy_loss_weight": 0.0003,
        "det_loss_weight": 0.0,
        "baseline_kl_weight": 10.0,
        "box_loss_weight": 0.0,
        "unfreeze": "cls",
        "optimizer": "adamw",
        "temperature": 1.0,
        "max_candidates": 40,
        "reward_score_threshold": 0.2,
        "rollout_source": "baseline",
        "policy_objective": "signed"
      },
      {
        "name": "signed_shuffled_amp_structure_010_03_0003_kl10",
        "signal": "shuffled_amp_structure",
        "reward_lambda": 0.1,
        "struct_weight": 0.3,
        "policy_loss_weight": 0.0003,
        "det_loss_weight": 0.0,
        "baseline_kl_weight": 10.0,
        "box_loss_weight": 0.0,
        "unfreeze": "cls",
        "optimizer": "adamw",
        "temperature": 1.0,
        "max_candidates": 40,
        "reward_score_threshold": 0.2,
        "rollout_source": "baseline",
        "policy_objective": "signed"
      }
    ]
  }
}
```

- [ ] **Step 10: Create NNI config**

Create `nni_configs/rlvr_round25_config.yml`:

```yaml
experimentName: rlvr_round25_mplseg_decoupled
trialCommand: E:/anaconda/01/envs/RLimage/python.exe -m spectral_detection_posttrain.nni_rlvr_trial --config spectral_detection_posttrain/configs/mvp.yaml --run-prefix nni_rlvr_round25 --rlvr-epochs 3 --early-stopping-patience 2
trialCodeDirectory: ..
searchSpaceFile: rlvr_round25_search_space.json
tuner:
  name: GridSearch
trainingService:
  platform: local
trialConcurrency: 1
maxTrialNumber: 8
```

- [ ] **Step 11: Create run script**

Create `run_nni_rlvr_round25.bat`:

```bat
@echo off
cd /d E:\CLIproject\RLimage
E:\anaconda\01\envs\RLimage\nni.exe experiment create --config nni_configs\rlvr_round25_config.yml --port 8095
```

- [ ] **Step 12: Run tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_round25_mplseg_verifier.py tests/test_round23_readable_results.py -v
```

Expected: all tests pass.

- [ ] **Step 13: Commit**

```powershell
git add spectral_detection_posttrain/nni_rlvr_trial.py nni_configs/rlvr_round25_search_space.json nni_configs/rlvr_round25_config.yml run_nni_rlvr_round25.bat tests/test_round25_mplseg_verifier.py
git commit -m "feat: add round25 mplseg rlvr matrix"
```

---

## Task 7: Add Result Summarizer

**Files:**
- Create: `spectral_detection_posttrain/analysis/summarize_round25_results.py`
- Create: `docs/rlvr_round25_mplseg_decoupled_report.md`

- [ ] **Step 1: Create summarizer script**

Create `spectral_detection_posttrain/analysis/summarize_round25_results.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path


SCENES = [
    ("clean", "clean"),
    ("edge", "object_edge_checkerboard"),
    ("inside", "object_inside_checkerboard"),
    ("near", "near_object_checkerboard"),
]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _by_name(rows: list[dict]) -> dict[str, dict]:
    return {str(row.get("name", "")): row for row in rows}


def _delta(row: dict, base: dict, key: str) -> float | None:
    if row.get(key) is None or base.get(key) is None:
        return None
    return float(row[key]) - float(base[key])


def _format_delta(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:+.6f}"


def summarize(rows: list[dict]) -> str:
    named = _by_name(rows)
    baseline = named.get("null_no_update") or named.get("signed_iou_0003_kl10")
    iou = named.get("signed_iou_0003_kl10")
    amp = named.get("signed_amp_0003_kl10")
    struct = named.get("signed_structure_0003_kl10")
    amp_struct = named.get("signed_amp_structure_010_03_0003_kl10")
    shuffled = named.get("signed_shuffled_amp_structure_010_03_0003_kl10")

    lines = ["# Round 2.5 MPLSeg-Decoupled RLVR Report", ""]
    lines.append("## Trial Table")
    lines.append("")
    lines.append("| name | default | failed | clean AP50 | clean AP75 | edge AP75 | inside AP75 | near AP75 | clean ECE |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda item: str(item.get("name", ""))):
        lines.append(
            "| {name} | {default:.6f} | {failed} | {clean_ap50:.6f} | {clean_ap75:.6f} | "
            "{edge_ap75:.6f} | {inside_ap75:.6f} | {near_ap75:.6f} | {clean_ece:.6f} |".format(
                name=row.get("name", ""),
                default=float(row.get("default", -1.0)),
                failed=row.get("constraint_failed", ""),
                clean_ap50=float(row.get("clean_ap50") or 0.0),
                clean_ap75=float(row.get("clean_ap75") or 0.0),
                edge_ap75=float(row.get("edge_ap75") or 0.0),
                inside_ap75=float(row.get("inside_ap75") or 0.0),
                near_ap75=float(row.get("near_ap75") or 0.0),
                clean_ece=float(row.get("clean_ece") or 0.0),
            )
        )

    lines.extend(["", "## Pairwise Interpretation", ""])
    comparisons = [
        ("Amp vs IoU", amp, iou),
        ("Struct vs IoU", struct, iou),
        ("Amp+Struct vs IoU", amp_struct, iou),
        ("Amp+Struct vs Shuffled", amp_struct, shuffled),
    ]
    for label, left, right in comparisons:
        if not left or not right:
            lines.append(f"- {label}: missing row")
            continue
        clean_ap50 = _delta(left, right, "clean_ap50")
        clean_ap75 = _delta(left, right, "clean_ap75")
        edge_ap75 = _delta(left, right, "edge_ap75")
        inside_ap75 = _delta(left, right, "inside_ap75")
        near_ap75 = _delta(left, right, "near_ap75")
        clean_ece = _delta(left, right, "clean_ece")
        lines.append(
            f"- {label}: clean AP50 {_format_delta(clean_ap50)}, "
            f"clean AP75 {_format_delta(clean_ap75)}, edge AP75 {_format_delta(edge_ap75)}, "
            f"inside AP75 {_format_delta(inside_ap75)}, near AP75 {_format_delta(near_ap75)}, "
            f"clean ECE {_format_delta(clean_ece)}"
        )

    lines.extend(["", "## Decision Rule", ""])
    lines.append("- Strong positive: Amp+Struct beats IoU-only and shuffled on AP75 in at least two patch-position scenes while clean AP50 drop is within 0.03.")
    lines.append("- Weak positive: Struct-only beats IoU-only on edge/near AP75 while Amp-only remains neutral.")
    lines.append("- Negative: Amp+Struct does not beat shuffled, or it gains AP75 only by losing clean AP50/recall outside constraints.")
    lines.append("- If negative, keep the stable RLVR shell and replace hand-built structure reward with a learned verifier target.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="runs/nni_rlvr_round25/nni_rlvr_results.jsonl")
    parser.add_argument("--output", default="docs/rlvr_round25_mplseg_decoupled_report.md")
    args = parser.parse_args()

    rows = _load_jsonl(Path(args.results))
    report = summarize(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create initial report scaffold with concrete run commands**

Create `docs/rlvr_round25_mplseg_decoupled_report.md`:

````markdown
# Round 2.5 MPLSeg-Decoupled RLVR Report

This report is generated by:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.analysis.summarize_round25_results --results runs/nni_rlvr_round25/nni_rlvr_results.jsonl --output docs/rlvr_round25_mplseg_decoupled_report.md
```

Expected result file:

```text
runs/nni_rlvr_round25/nni_rlvr_results.jsonl
```
````

- [ ] **Step 3: Run import check**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m py_compile spectral_detection_posttrain/analysis/summarize_round25_results.py
```

Expected: exits with code 0.

- [ ] **Step 4: Commit**

```powershell
git add spectral_detection_posttrain/analysis/summarize_round25_results.py docs/rlvr_round25_mplseg_decoupled_report.md
git commit -m "docs: add round25 result summarizer"
```

---

## Task 8: Smoke Test Round 2.5 Without Launching Full Matrix

**Files:**
- Uses existing code and configs.

- [ ] **Step 1: Run selected unit tests**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m pytest tests/test_phase_structure_features.py tests/test_round25_mplseg_verifier.py tests/test_rlvr_verifier.py tests/test_rlvr_policy_objective.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Run no-update smoke trial**

Create `tmp_round25_null.json` in the workspace with:

```json
{
  "preset": {
    "name": "null_no_update",
    "signal": "none",
    "reward_lambda": 0.0,
    "struct_weight": 0.0,
    "policy_loss_weight": 0.0,
    "det_loss_weight": 0.0,
    "baseline_kl_weight": 0.0,
    "box_loss_weight": 0.0,
    "unfreeze": "cls",
    "optimizer": "adamw",
    "temperature": 1.0,
    "max_candidates": 20,
    "reward_score_threshold": 0.2,
    "rollout_source": "baseline",
    "policy_objective": "signed"
  }
}
```

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.nni_rlvr_trial --config spectral_detection_posttrain/configs/mvp.yaml --run-prefix nni_rlvr_round25_smoke --params-file tmp_round25_null.json --limit-train 4 --limit-val 4 --rlvr-epochs 1 --early-stopping-patience 1
```

Expected:

```text
runs/nni_rlvr_round25_smoke/nni_rlvr_results.jsonl exists
last_trial_result.json has name = null_no_update
eval_status = ok
initial_roi_kl is near zero in the run's initial_sanity.json
```

- [ ] **Step 3: Run one structure smoke trial**

Create `tmp_round25_structure.json`:

```json
{
  "preset": {
    "name": "signed_structure_smoke",
    "signal": "structure",
    "reward_lambda": 0.0,
    "struct_weight": 0.2,
    "policy_loss_weight": 0.0003,
    "det_loss_weight": 0.0,
    "baseline_kl_weight": 10.0,
    "box_loss_weight": 0.0,
    "unfreeze": "cls",
    "optimizer": "adamw",
    "temperature": 1.0,
    "max_candidates": 20,
    "reward_score_threshold": 0.2,
    "rollout_source": "baseline",
    "policy_objective": "signed"
  }
}
```

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.nni_rlvr_trial --config spectral_detection_posttrain/configs/mvp.yaml --run-prefix nni_rlvr_round25_structure_smoke --params-file tmp_round25_structure.json --limit-train 4 --limit-val 4 --rlvr-epochs 1 --early-stopping-patience 1
```

Expected:

```text
run completes
metrics_train.jsonl contains structure_mean
last_trial_result.json has signal = structure
no AP50 collapse constraint failure is required for smoke, because limit-val=4 is noisy
```

- [ ] **Step 4: Remove temporary JSON files**

Run:

```powershell
Remove-Item -LiteralPath tmp_round25_null.json, tmp_round25_structure.json
```

- [ ] **Step 5: Commit smoke-readiness fixes if any**

If any code changes were needed after smoke:

```powershell
git add spectral_detection_posttrain tests nni_configs run_nni_rlvr_round25.bat docs
git commit -m "fix: make round25 smoke trial pass"
```

If no code changes were needed, do not create an empty commit.

---

## Task 9: Run Full Round 2.5 Matrix

**Files:**
- Uses `nni_configs/rlvr_round25_config.yml`
- Produces `runs/nni_rlvr_round25/nni_rlvr_results.jsonl`
- Produces `docs/rlvr_round25_mplseg_decoupled_report.md`

- [ ] **Step 1: Launch NNI**

Run:

```powershell
.\run_nni_rlvr_round25.bat
```

Expected:

```text
NNI starts on port 8095
maxTrialNumber = 8
trialConcurrency = 1
```

- [ ] **Step 2: Wait for all trials**

Use NNI UI or terminal logs until all 8 presets complete. The output file must contain 8 JSONL rows:

```text
runs/nni_rlvr_round25/nni_rlvr_results.jsonl
```

- [ ] **Step 3: Generate report**

Run:

```powershell
E:\anaconda\01\envs\RLimage\python.exe -m spectral_detection_posttrain.analysis.summarize_round25_results --results runs/nni_rlvr_round25/nni_rlvr_results.jsonl --output docs/rlvr_round25_mplseg_decoupled_report.md
```

Expected:

```text
docs/rlvr_round25_mplseg_decoupled_report.md contains Trial Table and Pairwise Interpretation
```

- [ ] **Step 4: Verify sanity gates**

Check each RLVR trial directory:

```powershell
Get-ChildItem -Path runs\nni_rlvr_round25 -Filter initial_sanity.json -Recurse | ForEach-Object { Get-Content $_.FullName }
```

Expected for every non-baseline RLVR trial:

```text
initial_roi_kl <= 0.0001
initial_logit_max_abs_diff <= 0.001
```

- [ ] **Step 5: Commit report**

```powershell
git add docs/rlvr_round25_mplseg_decoupled_report.md
git commit -m "docs: report round25 mplseg rlvr results"
```

---

## Success Criteria

Round 2.5 is successful if all are true:

```text
1. All unit tests pass.
2. No-update trial matches baseline behavior.
3. All 8 Round 2.5 result rows are present and readable.
4. No accepted trial violates clean AP50, recall, or prediction-count constraints.
5. Amp+Struct beats IoU-only and shuffled Amp+Struct on AP75 in at least two of:
   clean, object_edge_checkerboard, object_inside_checkerboard, near_object_checkerboard.
6. Clean AP50 drop stays within 0.03 of baseline.
```

If only `structure` improves edge/near AP75 but `amp_structure` does not, keep the structure branch and reduce amplitude weight in Round 2.6.

If `amp_structure` does not beat shuffled controls, do not claim MPLSeg-style verifier improvement. The correct conclusion is:

```text
The stable RLVR shell works, but the hand-built amplitude/phase-structure verifier is not yet a causal reward signal on Penn-Fudan.
```

---

## Interpretation Matrix

| Outcome | Meaning | Next Action |
|---|---|---|
| `Amp+Struct > IoU` and `Amp+Struct > Shuffled` | Decoupled spectral verifier has usable signal | Run 5-epoch confirmation and expand to VOC person subset |
| `Struct > IoU`, `Amp` neutral | Phase/structure is the useful verifier component | Keep structure branch, lower/remove amp |
| `Amp > IoU`, `Struct` neutral | Round 2.4 was underpowered for amplitude | Search `reward_lambda` with same KL shell |
| `Shuffled >= real` | Verifier values are not causally aligned | Replace hand-built scores with learned verifier target |
| AP50 stable but ECE/FP improves only | Verifier is calibration-biased | Reframe as RLVR calibration reward, add fixed-recall FP metrics |
| AP50/Recall collapses | Training shell regressed | Stop and inspect freeze-state, KL, candidate count, and prediction explosion |

---

## Self-Review Checklist

- Spec coverage: MPLSeg magnitude/phase decoupling is represented by separate `S_amp` and `S_struct` branches.
- Result controls: IoU-only, amplitude-only, structure-only, amplitude+structure, and shuffled controls are included.
- Stability: Round 2.3 rules are preserved; `det_loss_weight=0` remains fixed.
- Evaluation: clean, object-edge, object-inside, and near-object patch scenes are included.
- Causality: plan requires beating shuffled controls before making a frequency/structure verifier claim.
- Placeholders: no implementation step relies on undefined future work.
