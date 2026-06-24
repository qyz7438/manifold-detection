from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from spectral_detection_posttrain.spectral.rlvr_reward import (
    compute_group_reward,
    normalize_ramp,
)
from spectral_detection_posttrain.train.rollout import (
    ROLLOUT_CONFIGS,
    rollout_diversity_check,
)


class TestRolloutConfigs:
    def test_configs_have_three_groups(self):
        assert len(ROLLOUT_CONFIGS) == 3

    def test_configs_ascending_score_threshold(self):
        thresholds = [cfg["score_threshold"] for cfg in ROLLOUT_CONFIGS]
        assert thresholds[0] < thresholds[-1], "score_threshold should increase across configs"

    def test_configs_vary(self):
        configs = ROLLOUT_CONFIGS
        combos = {(cfg["score_threshold"], cfg["nms_threshold"]) for cfg in configs}
        assert len(combos) >= 2, "rollout configs must produce different detection results"


class TestRolloutDiversityCheck:
    def test_identical_rollouts_no_diversity(self):
        pred = [{"boxes": torch.zeros((3, 4)), "labels": torch.zeros(3), "scores": torch.ones(3)}]
        rollouts = [pred, pred, pred]
        result = rollout_diversity_check(rollouts)
        assert not result["has_diversity"]

    def test_different_rollouts_detected(self):
        many = [{"boxes": torch.zeros((10, 4)), "labels": torch.zeros(10), "scores": torch.ones(10)}]
        few = [{"boxes": torch.zeros((2, 4)), "labels": torch.zeros(2), "scores": torch.ones(2)}]
        rollouts = [many, few, few]
        result = rollout_diversity_check(rollouts)
        assert result["has_diversity"]


class TestRampNormalization:
    def test_zscore_normalization(self):
        stats = {"mean": 0.9, "std": 0.05}
        raw = torch.tensor([0.9, 0.95, 0.85], dtype=torch.float32)
        norm = normalize_ramp(raw, stats)
        assert norm[0].item() == pytest.approx(0.0, abs=1e-4)
        assert norm[1].item() == pytest.approx(1.0, abs=1e-4)
        assert norm[2].item() == pytest.approx(-1.0, abs=1e-4)

    def test_all_same_values_zero_std_capped(self):
        stats = {"mean": 0.99, "std": 0.0}
        raw = torch.tensor([0.99, 0.99, 0.99], dtype=torch.float32)
        norm = normalize_ramp(raw, stats)
        assert not torch.isnan(norm).any()


class TestGroupReward:
    def test_all_tp_no_fp_gives_positive_reward(self):
        per_box = torch.tensor([0.8, 0.9, 0.7], dtype=torch.float32)
        is_tp = torch.tensor([True, True, True])
        scores = torch.tensor([0.85, 0.92, 0.75], dtype=torch.float32)
        matched_gt = torch.tensor([0, 1, 2], dtype=torch.long)
        reward = compute_group_reward(per_box, is_tp, scores, total_gt=3, matched_gt_indices=matched_gt)
        assert reward > 0.0

    def test_no_tp_gives_low_reward(self):
        per_box = torch.zeros(3, dtype=torch.float32)
        is_tp = torch.tensor([False, False, False])
        scores = torch.tensor([0.85, 0.92, 0.75], dtype=torch.float32)
        matched_gt = torch.full((3,), -1, dtype=torch.long)
        reward = compute_group_reward(per_box, is_tp, scores, total_gt=3, matched_gt_indices=matched_gt)
        assert reward == 0.0 or reward <= 0.1

    def test_reward_never_negative(self):
        per_box = torch.zeros(5, dtype=torch.float32)
        is_tp = torch.zeros(5, dtype=torch.bool)
        scores = torch.ones(5, dtype=torch.float32) * 0.9
        matched_gt = torch.full((5,), -1, dtype=torch.long)
        reward = compute_group_reward(per_box, is_tp, scores, total_gt=5, matched_gt_indices=matched_gt)
        assert reward >= 0.0

    def test_fp_penalty_reduces_reward(self):
        per_box = torch.tensor([0.9, 0.9, 0.1], dtype=torch.float32)
        is_tp = torch.tensor([True, True, False])
        scores = torch.tensor([0.85, 0.88, 0.92], dtype=torch.float32)
        matched_gt = torch.tensor([0, 1, -1], dtype=torch.long)
        reward_with_fp = compute_group_reward(
            per_box, is_tp, scores, total_gt=2, matched_gt_indices=matched_gt, alpha=0.5, beta=0.0,
        )
        tp_only = compute_group_reward(
            per_box[:2], torch.tensor([True, True]), scores[:2],
            total_gt=2, matched_gt_indices=matched_gt[:2], alpha=0.0, beta=0.0,
        )
        assert reward_with_fp < tp_only

    def test_high_alpha_penalizes_fp_more(self):
        per_box = torch.tensor([0.9, 0.1], dtype=torch.float32)
        is_tp = torch.tensor([True, False])
        scores = torch.tensor([0.7, 0.9], dtype=torch.float32)
        matched_gt = torch.tensor([0, -1], dtype=torch.long)

        r_lo = compute_group_reward(
            per_box, is_tp, scores, total_gt=1, matched_gt_indices=matched_gt, alpha=0.1, beta=0.0,
        )
        r_hi = compute_group_reward(
            per_box, is_tp, scores, total_gt=1, matched_gt_indices=matched_gt, alpha=0.9, beta=0.0,
        )
        assert r_hi <= r_lo


def test_rlvr_training_args_accept_verifier_policy_mode():
    from spectral_detection_posttrain.train.posttrain_rlvr import parse_args

    args = parse_args([
        "--config", "spectral_detection_posttrain/configs/smoke.yaml",
        "--checkpoint", "runs/baseline/checkpoint.pt",
        "--run-name", "smoke",
        "--signal", "none",
        "--unfreeze", "box",
        "--optimizer", "adamw",
        "--reward-lambda", "0.0",
        "--alpha", "0.1",
        "--beta", "0.05",
        "--epochs", "1",
        "--policy-loss-weight", "0.3",
    ])

    assert args.policy_loss_weight == 0.3
    assert args.signal == "none"
