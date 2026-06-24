# Plans 4.1-4.5: Penn-Fudan Segmentation Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate MPLSeg-style AFM on pixel-level segmentation tasks (Penn-Fudan pedestrian masks).

**Architecture:** FCN-ResNet50 with AFM inserted before classifier (on "out" feature map, 512 channels). Supports baseline, mid06 full fine-tune, and A/C post-training.

**Tech Stack:** Python, PyTorch, torchvision FCN-ResNet50, `round4x_seg_runner.py`.

**Runner:** `scripts/round4x_seg_runner.py` (single script for all 4.x plans):
- 4.1: Baseline only (3 seeds × 3ep)
- 4.2: mid06 AFM (3 seeds × 3ep) 
- 4.5: A/C post-training (3 seeds × 2ep, from mid06_s42 checkpoint)

**Design Note:** AFM is inserted on the FCN classifier's "out" feature (512ch, H/8×W/8). This is the feature map before the final 1×1 conv classifier — analogous to the box_head position in detection.
