# Plan 3.7: Detection Final Report

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Aggregate all detection experiment results and compute A/C post-training win rate statistics.

**Architecture:** Read all eval_metrics.json from 2.16, 2.18, 2.19, 2.20, 2.21, 2.22, 2.25, 2.26, 3.4 runs. Compute A/C-vs-baseline win rate by (backbone × dataset) cross-tab.

**Metrics:** AP50 improvement, AP75 improvement, win rate (% of seeds where A/C > baseline).

**Output:** Summary table and markdown report.
