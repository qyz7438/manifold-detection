import pytest
import torch

from spectral_detection_posttrain.rlvr.detection_verifier import (
    DetectionVerifierConfig,
    build_reward_component_summary,
    compute_box_rewards,
    signal_uses_amp,
    signal_uses_structure,
)


def test_shuffled_amp_alias_uses_amp_not_structure():
    assert signal_uses_amp("shuffled_amp")
    assert not signal_uses_structure("shuffled_amp")


def test_amp_only_reward_ignores_structure_value():
    cfg = DetectionVerifierConfig(signal="ramp", w_iou=1.0, w_cls=0.2, w_amp=0.1, w_struct=0.9)
    ious = torch.tensor([0.7])
    class_correct = torch.tensor([1.0])
    scores = torch.tensor([0.8])
    matched = torch.tensor([True])
    s_amp = torch.tensor([0.5])
    s_struct = torch.tensor([1.0])
    reward = compute_box_rewards(cfg, ious, class_correct, scores, matched, s_amp=s_amp, s_struct=s_struct)
    assert reward.item() == pytest.approx(0.7 + 0.2 + 0.05, abs=1e-6)


def test_reward_component_summary_reports_means_and_counts():
    actions = [
        {"amp_values": torch.tensor([0.2, 0.8]), "structure_values": torch.tensor([0.1, 0.3]),
         "rewards": torch.tensor([0.5, 1.0]), "matched": torch.tensor([True, False])},
        {"amp_values": torch.tensor([0.4]), "structure_values": torch.tensor([0.7]),
         "rewards": torch.tensor([0.2]), "matched": torch.tensor([True])},
    ]
    summary = build_reward_component_summary(actions)
    assert summary["amp_mean"] == pytest.approx((0.2 + 0.8 + 0.4) / 3.0)
    assert summary["structure_mean"] == pytest.approx((0.1 + 0.3 + 0.7) / 3.0)
    assert summary["reward_mean"] == pytest.approx((0.5 + 1.0 + 0.2) / 3.0)
    assert summary["candidate_count"] == 3
    assert summary["matched_count"] == 2


def test_round30_posttrain_args_accept_seed_and_shuffled_amp():
    from spectral_detection_posttrain.train.posttrain_rlvr import parse_args
    args = parse_args([
        "--config", "spectral_detection_posttrain/configs/mvp.yaml",
        "--checkpoint", "runs/baseline/checkpoint_last.pth",
        "--run-name", "round30_smoke", "--signal", "shuffled_amp",
        "--unfreeze", "cls", "--optimizer", "adamw",
        "--reward-lambda", "0.1", "--struct-weight", "0.0",
        "--policy-loss-weight", "0.0003", "--baseline-kl-weight", "10.0",
        "--det-loss-weight", "0.0", "--seed", "43", "--epochs", "1",
    ])
    assert args.signal == "shuffled_amp"
    assert args.seed == 43
