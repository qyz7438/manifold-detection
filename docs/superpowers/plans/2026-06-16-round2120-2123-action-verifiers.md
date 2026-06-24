# Round 2.120-2.123 Action Verifiers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the next four 2.x experiment entrypoints for RLVR/DPO with FFT and high-dimensional manifold verifiers.

**Architecture:** Build one shared ROI-delta action layer that samples box actions, decodes them, scores them with verifier functions, and exposes RLVR rewards or DPO pair preferences. Four thin `scripts/round2120...round2123...` entrypoints configure the verifier/objective combination and delegate to the shared runner.

**Tech Stack:** PyTorch, TorchVision Faster R-CNN, existing Penn-Fudan loaders, canonical runner metadata/checkpoint validation, pytest.

---

### Task 1: Shared ROI-Delta Action Data Model

**Files:**
- Create: `spectral_detection_posttrain/rlvr/action_verifier.py`
- Test: `tests/test_action_verifier.py`

- [ ] **Step 1: Write failing tests**

```python
import torch

from spectral_detection_posttrain.rlvr.action_verifier import (
    ActionVerifierConfig,
    build_action_batch,
    decode_box_actions,
)


def test_decode_box_actions_preserves_shape_and_changes_box():
    proposals = torch.tensor([[10.0, 20.0, 30.0, 60.0]])
    deltas = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])

    boxes = decode_box_actions(proposals, deltas, image_size=(100, 100))

    assert boxes.shape == (1, 2, 4)
    assert torch.allclose(boxes[0, 0], proposals[0])
    assert boxes[0, 1, 0] > proposals[0, 0]


def test_build_action_batch_returns_log_probs_and_decoded_boxes():
    proposals = torch.tensor([[10.0, 20.0, 30.0, 60.0], [0.0, 0.0, 10.0, 10.0]])
    mu = torch.zeros((2, 4))
    cfg = ActionVerifierConfig(num_samples=3, sigma=0.1, seed=123)

    batch = build_action_batch(proposals, mu, image_size=(100, 100), cfg=cfg)

    assert batch.proposals.shape == (2, 4)
    assert batch.deltas.shape == (2, 3, 4)
    assert batch.decoded_boxes.shape == (2, 3, 4)
    assert batch.log_probs.shape == (2, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_action_verifier.py -q`

- [ ] **Step 3: Implement data model and sampler**

Implement:
- `ActionVerifierConfig`
- `ActionBatch`
- `decode_box_actions`
- `gaussian_log_prob`
- `build_action_batch`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_action_verifier.py -q`

### Task 2: FFT and Manifold Scorers

**Files:**
- Modify: `spectral_detection_posttrain/rlvr/action_verifier.py`
- Test: `tests/test_action_verifier.py`

- [ ] **Step 1: Write failing tests**

```python
from spectral_detection_posttrain.rlvr.action_verifier import (
    compute_fft_action_quality,
    compute_manifold_action_quality,
)


def test_fft_action_quality_is_action_dependent():
    image = torch.zeros((3, 64, 64))
    image[:, 20:44, 20:44] = 1.0
    boxes = torch.tensor([[[20.0, 20.0, 44.0, 44.0], [0.0, 0.0, 12.0, 12.0]]])

    quality = compute_fft_action_quality(image, boxes, crop_size=32)

    assert quality.shape == (1, 2)
    assert quality[0, 0] > quality[0, 1]


def test_manifold_action_quality_prefers_reference_like_features():
    features = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])
    reference = torch.tensor([[0.0, 0.0], [0.2, 0.1]])

    quality = compute_manifold_action_quality(features, reference)

    assert quality.shape == (1, 2)
    assert quality[0, 0] > quality[0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_action_verifier.py -q`

- [ ] **Step 3: Implement scorers**

Implement:
- image crop and resize helper
- `compute_fft_action_quality`
- `compute_manifold_action_quality`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_action_verifier.py -q`

### Task 3: RLVR Reward and DPO Pair Construction

**Files:**
- Modify: `spectral_detection_posttrain/rlvr/action_verifier.py`
- Test: `tests/test_action_verifier.py`

- [ ] **Step 1: Write failing tests**

```python
from spectral_detection_posttrain.rlvr.action_verifier import (
    build_dpo_pairs,
    build_rlvr_rewards,
)


def test_rlvr_rewards_gate_positive_reward_to_matched_actions():
    iou = torch.tensor([[0.9, 0.2]])
    verifier = torch.tensor([[0.5, 1.0]])
    matched = iou >= 0.5

    rewards = build_rlvr_rewards(iou, verifier, matched, verifier_weight=0.5)

    assert rewards[0, 0] > 0.9
    assert rewards[0, 1] <= 0.0


def test_dpo_pairs_skip_ties_by_margin():
    quality = torch.tensor([[0.8, 0.7], [0.5, 0.49]])
    pairs = build_dpo_pairs(quality, margin=0.05)

    assert pairs.valid.tolist() == [True, False]
    assert pairs.chosen_indices.tolist() == [0, 0]
    assert pairs.rejected_indices.tolist() == [1, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_action_verifier.py -q`

- [ ] **Step 3: Implement reward/pair helpers**

Implement:
- `build_rlvr_rewards`
- `normalize_group_advantage`
- `DpoPairs`
- `build_dpo_pairs`
- `dpo_loss_from_log_probs`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_action_verifier.py -q`

### Task 4: Shared Experiment Runner and Four Entrypoints

**Files:**
- Create: `spectral_detection_posttrain/train/action_verifier_posttrain.py`
- Create: `scripts/round2120_rlvr_fft.py`
- Create: `scripts/round2121_rlvr_manifold.py`
- Create: `scripts/round2122_dpo_fft.py`
- Create: `scripts/round2123_dpo_manifold.py`
- Test: `tests/test_round2120_2123_scripts.py`

- [ ] **Step 1: Write failing script tests**

```python
import importlib


def test_round2120_to_2123_scripts_import():
    for module in [
        "scripts.round2120_rlvr_fft",
        "scripts.round2121_rlvr_manifold",
        "scripts.round2122_dpo_fft",
        "scripts.round2123_dpo_manifold",
    ]:
        imported = importlib.import_module(module)
        assert hasattr(imported, "main")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_round2120_2123_scripts.py -q`

- [ ] **Step 3: Implement shared runner and entrypoints**

Implement:
- argument parser with `--config`, `--checkpoint`, `--run-name`, `--epochs`, `--limit-train`, `--limit-val`, `--objective`, `--verifier`
- `main_for_experiment(objective, verifier, default_run_name)`
- thin scripts that call the shared runner

- [ ] **Step 4: Run import and compile tests**

Run:
`python -m pytest tests/test_round2120_2123_scripts.py tests/test_action_verifier.py -q`
`python -m py_compile scripts/round2120_rlvr_fft.py scripts/round2121_rlvr_manifold.py scripts/round2122_dpo_fft.py scripts/round2123_dpo_manifold.py`

### Task 5: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run focused tests**

Run:
`python -m pytest tests/test_action_verifier.py tests/test_round2120_2123_scripts.py tests/test_rlvr_verifier.py tests/test_roi_policy_loss.py -q`

- [ ] **Step 2: Smoke run no-update/import path**

Run one script with `--epochs 0 --limit-train 1 --limit-val 1` after the baseline checkpoint is available.

- [ ] **Step 3: Report**

Summarize:
- new files
- four version mappings
- tests run
- baseline training status
