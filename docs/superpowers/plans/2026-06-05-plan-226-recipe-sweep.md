# Plan 2.26: Post-Training Recipe Sweep

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Find the optimal post-training recipe (Approach A/C/AC, epoch count) for maximum AP improvement over baseline.

**Architecture:** 6 recipes × 3 seeds × PF+MobV3. All start from mid06_5ep checkpoint.

- A_2ep / A_5ep: weak gate (0.1), AFM-only, 2 or 5 epochs
- C_2ep / C_5ep: feature constraint (MSE 0.05), AFM-only, 2 or 5 epochs
- AC_2ep / AC_5ep: weak gate + feature constraint, 2 or 5 epochs

**Tech Stack:** Python, PyTorch, `round226_runner.py` (custom training with hooks).

**Runner:** `scripts/round226_runner.py`
