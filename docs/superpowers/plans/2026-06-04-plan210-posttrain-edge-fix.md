# Plan 2.10: Post-Training + RPN Edge-Mix Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Round 2.9's three implementation bugs — checkpoint loading, edge_mix flag propagation, and eval-only mode — then re-run post-training sanity and RPN edge-mix comparison correctly.

**Architecture:** Add `--checkpoint` arg and `--edge-mix` flag to `round28_train_eval.py`. Fix `eval_only` mode to load checkpoint before eval. Re-run 4 groups (B1/B2 + G8/G9).

**Tech Stack:** Python, PyTorch, existing `spectral_detection_posttrain` package.

---

## Why Plan 2.10 Exists

Round 2.9 had three bugs invalidating Gap 2 and Part B:

1. **B1/B2 never loaded G1 checkpoint.** G1_CKPT was defined but `--checkpoint` arg didn't exist. B1 ran random init eval (AP50=0.05). B2 was just another fresh training run.
2. **G8/G9 edge_mix was never passed.** `edge_mix` variable assigned but not added to `subprocess.run` command. Both ran clean training; AP75 diff is noise.
3. **eval_only mode doesn't load a checkpoint.** `--epochs 0` uses current model state, not a saved file.

## File Map

- Modify: `scripts/round28_train_eval.py` — add `--checkpoint`, `--edge-mix`, fix `eval_only`
- Modify: `scripts/round29_run_matrix.py` — pass new args correctly

---

## Task 1: Add Checkpoint Loading And Edge-Mix

**Files:**
- Modify: `scripts/round28_train_eval.py`

- [ ] **Step 1: Add new args**

Add after `--seed` in the parser:

```python
parser.add_argument("--checkpoint", default=None, help="Load this checkpoint before training.")
parser.add_argument("--edge-mix", action="store_true", default=False)
```

- [ ] **Step 2: Add checkpoint loading + edge-mix + fix eval_only**

After `model = build_detector(config).to(device)`:

```python
if args.checkpoint:
    from spectral_detection_posttrain.utils.io import load_checkpoint
    load_checkpoint(model, args.checkpoint, device)

if args.epochs == 0:
    if not args.checkpoint:
        raise ValueError("--epochs 0 requires --checkpoint")
    _eval_model(model, val_loader, device, run_dir)
    return
```

In the training loop, before `loss_dict = model(images, targets)`:

```python
if args.edge_mix:
    import random
    from spectral_detection_posttrain.datasets.patch_transform import add_detection_patch
    for i in range(len(images)):
        if random.random() < 0.5:
            images[i] = add_detection_patch(
                images[i].cpu(), targets[i], placement="edge",
                patch_type="checkerboard", patch_size=48,
            ).to(device)
```

- [ ] **Step 3: Verify — smoke test B1**

```powershell
cd E:/CLIproject/RLimage; $env:PYTHONPATH = "E:/CLIproject/RLimage"
E:\anaconda\01\envs\RLimage\python.exe scripts/round28_train_eval.py --run-name round210_smoke_b1 --checkpoint runs/round29_g1_baseline_full/checkpoint_last.pth --afm-type none --trainable-mode full --epochs 0
```

Expected: AP50 ≈ G1 AP50 (0.82+).

- [ ] **Step 4: Commit**

```bash
git add scripts/round28_train_eval.py
git commit -m "feat: add --checkpoint --edge-mix to train/eval"
```

---

## Task 2: Fix Matrix Runner

**Files:**
- Modify: `scripts/round29_run_matrix.py`

- [ ] **Step 1: Rewrite to pass new args**

```python
from __future__ import annotations

import subprocess
import sys

PYTHON = sys.executable
SCRIPT = "scripts/round28_train_eval.py"
G1_CKPT = "runs/round29_g1_baseline_full/checkpoint_last.pth"

GROUPS = [
    ("round210_b1_ckpt_eval", "none", "current", "full", 0, False, G1_CKPT),
    ("round210_b2_posttrain", "none", "current", "box_head_only", 1, False, G1_CKPT),
    ("round210_g8_rpn_clean", "none", "current", "rpn_box_head", 1, False, None),
    ("round210_g9_rpn_mixed", "none", "current", "rpn_box_head", 1, True, None),
]

for run_name, afm_type, residual_mode, trainable_mode, epochs, edge_mix, checkpoint in GROUPS:
    cmd = [PYTHON, SCRIPT, "--run-name", run_name, "--afm-type", afm_type,
           "--afm-residual-mode", residual_mode, "--trainable-mode", trainable_mode,
           "--epochs", str(epochs), "--seed", "42"]
    if edge_mix: cmd.append("--edge-mix")
    if checkpoint: cmd.extend(["--checkpoint", checkpoint])
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
```

- [ ] **Step 2: Run 4 groups**

```powershell
cd E:/CLIproject/RLimage; $env:PYTHONPATH = "E:/CLIproject/RLimage"
E:\anaconda\01\envs\RLimage\python.exe scripts/round29_run_matrix.py
```

Expected: 4 eval_metrics.json files. B1 AP50 ≈ 0.82. B2 stable. G9 loss pattern differs from G8.

- [ ] **Step 3: Commit**

```bash
git add scripts/round29_run_matrix.py
git commit -m "fix: pass --checkpoint and --edge-mix to train/eval"
```

---

## Success Criteria

```text
1. B1 AP50 >= 0.80 (G1 checkpoint loads correctly).
2. B2 AP50 >= B1 AP50 - 5% (post-training does not collapse).
3. G9 uses real 50% edge-mix (training loss differs from G8).
```

---

## Plan 位置

`E:/CLIproject/RLimage/docs/superpowers/plans/2026-06-04-plan210-posttrain-edge-fix.md`
