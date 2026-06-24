# Plan 2.25: Phase Contribution Ablation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Determine whether the magnitude gate (mp) or phase residual (pa) is the critical AFM component.

**Architecture:** 3 variants (mag_only, phase_only, both) × 3 seeds × PF+MobV3 × 3ep full fine-tune.

- `MagOnlyAFMBlock`: magnitude gate active, phase pass-through
- `PhaseOnlyAFMBlock`: magnitude pass-through, phase residual active
- Both: standard MPLSegAFMBlock (control)

**Tech Stack:** Python, PyTorch, `micro_afm.py` (new block types), `round28_train_eval.py`, `round225_runner.py`.

**Runner:** `scripts/round225_runner.py`
