# Plan 3.4: VOC 20-Class Full Validation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Verify A/C post-training on multi-class (20-class) VOC detection across backbones.

**Architecture:** 2 backbones (MobV3, ResNet50) × 3 configs (baseline, A, C) × 3 seeds × VOC2012 full × 3ep baseline + 2ep post-training. Phase 1 trains baseline+mid06 (seed42 only), Phase 2 applies A/C post-training (3 seeds).

**Tech Stack:** Python, PyTorch, `voc_detection.py`, `round34_runner.py`.

**Runner:** `scripts/round34_runner.py`

**Note:** ResNet50 may require afm_channels adjustment if the 256→1024 bug from Plan 2.19 persists. Monitor Phase 1 mid06 training for crashes.
