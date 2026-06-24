# Plan 2.22: Data Volume Sweep

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Determine if post-training benefit depends on training data volume.

**Architecture:** 4 volume levels (30, 60, 90, 136) × 2 configs (baseline + A post-training) × seed42 × PF+MobV3 × 3ep. Baseline trains from scratch; A starts from mid06_5ep checkpoint.

**Tech Stack:** Python, PyTorch, `round28_train_eval.py`, `round222_runner.py`.

**Runner:** `scripts/round222_runner.py`
