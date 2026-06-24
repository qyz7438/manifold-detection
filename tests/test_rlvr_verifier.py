import pytest
import torch

from spectral_detection_posttrain.rlvr.detection_verifier import (
    DetectionVerifierConfig,
    build_rewarded_roi_actions,
    compute_box_rewards,
    normalize_group_advantages,
    shuffle_tp_ramp,
)


def test_compute_box_rewards_rewards_tp_and_penalizes_high_conf_fp():
    cfg = DetectionVerifierConfig(signal="ramp", w_iou=1.0, w_cls=0.2, w_amp=0.1, w_hconf_fp=0.5)
    ious = torch.tensor([0.8, 0.0])
    class_correct = torch.tensor([1.0, 0.0])
    scores = torch.tensor([0.9, 0.95])
    matched = torch.tensor([True, False])
    s_amp = torch.tensor([0.7, 0.0])

    rewards = compute_box_rewards(cfg, ious, class_correct, scores, matched, s_amp=s_amp)

    assert rewards.shape == (2,)
    assert rewards[0] > 0.9
    assert rewards[1] < 0.0


def test_compute_box_rewards_does_not_reward_unmatched_boxes():
    cfg = DetectionVerifierConfig(
        signal="ramp",
        w_iou=1.0,
        w_cls=0.2,
        w_amp=0.5,
        w_hconf_fp=0.5,
        high_conf_threshold=0.8,
    )
    ious = torch.tensor([0.49, 0.0])
    class_correct = torch.tensor([0.0, 0.0])
    scores = torch.tensor([0.2, 0.95])
    matched = torch.tensor([False, False])
    s_amp = torch.tensor([1.0, 1.0])

    rewards = compute_box_rewards(cfg, ious, class_correct, scores, matched, s_amp=s_amp)

    assert rewards[0].item() == pytest.approx(0.0)
    assert rewards[1].item() < 0.0


def test_shuffle_tp_ramp_preserves_fp_zero_and_changes_tp_order():
    values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.0])
    matched = torch.tensor([True, True, True, True, True, False])
    shuffled = shuffle_tp_ramp(values, matched, seed=42)

    assert torch.equal(shuffled[~matched], torch.tensor([0.0]))
    assert sorted(shuffled[matched].tolist()) == sorted(values[matched].tolist())
    assert not torch.equal(shuffled[matched], values[matched])


def test_normalize_group_advantages_has_nonzero_weights_for_all_boxes():
    rewards = torch.tensor([0.2, 0.5, -0.1])
    weights = normalize_group_advantages(rewards, temperature=1.0)

    assert weights.shape == rewards.shape
    assert torch.all(weights > 0)
    assert weights[1] > weights[0] > weights[2]


def test_build_rewarded_roi_actions_marks_fp_as_background_and_adds_missed_gt():
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]),
        "labels": torch.tensor([1, 1]),
        "scores": torch.tensor([0.9, 0.95]),
    }
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        "labels": torch.tensor([1, 1]),
    }
    actions = build_rewarded_roi_actions(prediction, target, num_classes=2, max_candidates=8)

    assert actions["boxes"].shape[0] == actions["labels"].shape[0]
    assert 0 in actions["labels"].tolist()
    assert actions["labels"].tolist().count(1) >= 1
    assert torch.all(actions["weights"] > 0)
    assert "advantages" in actions
    assert "policy_labels" in actions


def test_temperature_changes_weights():
    rewards = torch.tensor([0.2, 0.5, -0.1])
    w_hot = normalize_group_advantages(rewards, temperature=0.2)
    w_cold = normalize_group_advantages(rewards, temperature=2.0)
    assert w_hot.max() > w_cold.max()


def test_shuffle_tp_ramp_only_shuffles_matched_and_fp_stays_zero():
    values = torch.tensor([0.8, 0.6, 0.9, 0.0, 0.0])
    matched = torch.tensor([True, True, True, False, False])
    shuffled = shuffle_tp_ramp(values, matched, seed=123)
    assert torch.all(shuffled[~matched] == 0.0)
    tp_shuffled = shuffled[matched]
    tp_orig = values[matched]
    assert sorted(tp_shuffled.tolist()) == sorted(tp_orig.tolist())


def test_rewarded_actions_include_predicted_policy_labels_and_signed_advantages():
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]),
        "labels": torch.tensor([1, 1]),
        "scores": torch.tensor([0.9, 0.9]),
    }
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([1]),
    }
    actions = build_rewarded_roi_actions(prediction, target, num_classes=2, max_candidates=8)

    assert "policy_labels" in actions
    assert "advantages" in actions
    assert "matched" in actions
    assert actions["policy_labels"].tolist()[:2] == [1, 1]
    assert actions["advantages"][0] > actions["advantages"][1]
    assert actions["matched"].tolist()[:2] == [True, False]


def test_reward_score_threshold_filters_amp_with_same_mask():
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        "labels": torch.tensor([1, 1]),
        "scores": torch.tensor([0.95, 0.10]),
    }
    target = {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])}
    s_amp = torch.tensor([0.7, 0.2])

    actions = build_rewarded_roi_actions(
        prediction, target, num_classes=2, reward_score_threshold=0.2, s_amp=s_amp,
    )

    assert actions["boxes"].shape[0] >= 1
    assert actions["amp_values"][0].item() == pytest.approx(0.7, abs=1e-4)


def test_rewarded_actions_returns_empty_actions_when_no_predictions_pass_threshold():
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.1]),
    }
    target = {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])}

    actions = build_rewarded_roi_actions(prediction, target, num_classes=2, reward_score_threshold=0.2)

    assert actions["boxes"].shape == (0, 4)
    assert actions["matched_gt_boxes"].shape == (0, 4)
    for key in [
        "labels",
        "policy_labels",
        "weights",
        "advantages",
        "rewards",
        "matched",
        "scores",
        "amp_values",
        "structure_values",
    ]:
        assert actions[key].numel() == 0


def test_rewarded_actions_handles_images_without_ground_truth():
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [12.0, 12.0, 20.0, 20.0]]),
        "labels": torch.tensor([1, 1]),
        "scores": torch.tensor([0.9, 0.2]),
    }
    target = {"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.long)}

    actions = build_rewarded_roi_actions(prediction, target, num_classes=2, reward_score_threshold=0.2)

    assert actions["boxes"].shape == (2, 4)
    assert actions["matched_gt_boxes"].shape == (2, 4)
    assert actions["labels"].tolist() == [0, 0]
    assert actions["matched"].tolist() == [False, False]
    assert actions["rewards"][0].item() < 0.0
    assert actions["rewards"][1].item() == pytest.approx(0.0)


def test_percentile_clamp_normalization_produces_range_0_to_1():
    from spectral_detection_posttrain.spectral.rlvr_reward import normalize_ramp
    stats = {"p05": 0.998, "p95": 0.999}
    raw = torch.tensor([0.9985, 0.997, 0.9995], dtype=torch.float32)
    norm = normalize_ramp(raw, stats)
    assert float(norm.min().item()) >= 0.0
    assert float(norm.max().item()) <= 1.0
