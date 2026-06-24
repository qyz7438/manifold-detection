import pytest
import torch
from torch import nn

from spectral_detection_posttrain.train import action_verifier_posttrain as posttrain


class _FakeBoxPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_score = nn.Linear(2, 2)
        self.bbox_pred = nn.Linear(2, 8)


class _FakeRoiHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.box_head = nn.Linear(2, 2)
        self.box_predictor = _FakeBoxPredictor()


class _FakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.rpn = nn.Linear(2, 2)
        self.roi_heads = _FakeRoiHeads()


def test_action_verifier_defaults_are_safe_for_dpo_iou_oracle():
    args = posttrain.parse_args(
        [
            "--config",
            "cfg.yaml",
            "--checkpoint",
            "ckpt.pth",
            "--run-name",
            "safe",
            "--objective",
            "dpo",
            "--verifier",
            "iou_oracle",
        ]
    )

    assert args.policy_loss_weight == pytest.approx(1e-4)
    assert args.baseline_kl_weight == pytest.approx(1.0)
    assert args.det_loss_weight == pytest.approx(1.0)
    assert args.max_pred_multiplier == pytest.approx(2.0)
    assert args.max_fp_rate == pytest.approx(0.6)
    assert not args.bbox_adapter
    assert args.bbox_adapter_hidden_dim == 128
    assert args.bbox_adapter_scale == pytest.approx(1.0)
    assert args.bbox_adapter_delta_weight == pytest.approx(0.0)


def test_action_verifier_parses_bbox_adapter_options():
    args = posttrain.parse_args(
        [
            "--config",
            "cfg.yaml",
            "--checkpoint",
            "ckpt.pth",
            "--run-name",
            "adapter",
            "--objective",
            "dpo",
            "--verifier",
            "iou_oracle",
            "--bbox-adapter",
            "--bbox-adapter-hidden-dim",
            "32",
            "--bbox-adapter-scale",
            "0.5",
            "--bbox-adapter-delta-weight",
            "0.01",
        ]
    )

    assert args.bbox_adapter
    assert args.bbox_adapter_hidden_dim == 32
    assert args.bbox_adapter_scale == pytest.approx(0.5)
    assert args.bbox_adapter_delta_weight == pytest.approx(0.01)


def test_action_verifier_parses_cls_confidence_adapter_options():
    args = posttrain.parse_args(
        [
            "--config",
            "cfg.yaml",
            "--checkpoint",
            "ckpt.pth",
            "--run-name",
            "cls_adapter",
            "--objective",
            "dpo",
            "--verifier",
            "iou_oracle",
            "--bbox-adapter",
            "--cls-adapter",
            "--cls-adapter-scale",
            "0.25",
            "--cls-confidence-loss-weight",
            "0.1",
            "--cls-confidence-score-max",
            "0.6",
            "--cls-confidence-iou-min",
            "0.8",
        ]
    )

    assert args.cls_adapter
    assert args.cls_adapter_scale == pytest.approx(0.25)
    assert args.cls_confidence_loss_weight == pytest.approx(0.1)
    assert args.cls_confidence_score_max == pytest.approx(0.6)
    assert args.cls_confidence_iou_min == pytest.approx(0.8)


def test_freeze_bbox_pred_only_keeps_box_head_and_classifier_frozen():
    model = _FakeDetector()

    trainable = posttrain.freeze_bbox_pred_only(model)

    assert trainable == ["roi_heads.box_predictor.bbox_pred.weight", "roi_heads.box_predictor.bbox_pred.bias"]
    for name, parameter in model.named_parameters():
        if "roi_heads.box_predictor.bbox_pred" in name:
            assert parameter.requires_grad, name
        else:
            assert not parameter.requires_grad, name


def test_compute_weighted_objective_scales_policy_loss_before_adding_keep_losses():
    policy_loss = torch.tensor(10.0)
    kl_loss = torch.tensor(2.0)
    det_loss = torch.tensor(3.0)

    objective = posttrain.compute_weighted_objective(
        policy_loss,
        kl_loss,
        det_loss,
        policy_loss_weight=1e-4,
        baseline_kl_weight=1.0,
        det_loss_weight=1.0,
    )

    assert objective.item() == pytest.approx(5.001)


def test_confidence_correction_loss_targets_low_score_high_iou_positive_class():
    logits = torch.tensor([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0]], requires_grad=True)
    scores = torch.tensor([0.1, 0.9, 0.1])
    best_iou = torch.tensor([0.8, 0.9, 0.4])

    loss = posttrain.confidence_correction_loss(
        logits,
        scores,
        best_iou,
        score_max=0.7,
        iou_min=0.75,
    )

    expected = torch.nn.functional.cross_entropy(logits[:1], torch.ones(1, dtype=torch.long))
    assert loss.item() == pytest.approx(expected.item())


def test_safety_guard_blocks_prediction_explosion_and_high_fp_rate():
    baseline_metrics = {"num_predictions": 100, "false_positive_rate": 0.2}

    pred_guard = posttrain.evaluate_safety_guard(
        {"num_predictions": 201, "false_positive_rate": 0.2},
        baseline_metrics,
        max_pred_multiplier=2.0,
        max_fp_rate=0.6,
    )
    assert pred_guard.triggered
    assert pred_guard.reason == "prediction_count_exceeded"

    fp_guard = posttrain.evaluate_safety_guard(
        {"num_predictions": 120, "false_positive_rate": 0.61},
        baseline_metrics,
        max_pred_multiplier=2.0,
        max_fp_rate=0.6,
    )
    assert fp_guard.triggered
    assert fp_guard.reason == "false_positive_rate_exceeded"

    safe_guard = posttrain.evaluate_safety_guard(
        {"num_predictions": 200, "false_positive_rate": 0.6},
        baseline_metrics,
        max_pred_multiplier=2.0,
        max_fp_rate=0.6,
    )
    assert not safe_guard.triggered
    assert safe_guard.reason == ""


def test_best_checkpoint_is_never_saved_when_safety_guard_triggers():
    guard = posttrain.SafetyGuardResult(triggered=True, reason="false_positive_rate_exceeded")

    assert not posttrain.should_save_best_checkpoint({"ap75": 0.9}, best_ap75=0.5, safety_guard=guard)

    safe = posttrain.SafetyGuardResult(triggered=False)
    assert posttrain.should_save_best_checkpoint({"ap75": 0.9}, best_ap75=0.5, safety_guard=safe)
    assert not posttrain.should_save_best_checkpoint({"ap75": 0.4}, best_ap75=0.5, safety_guard=safe)


def test_action_verifier_builds_eval_loader_with_eval_batch_size(monkeypatch):
    calls = []

    def fake_build_loaders(config, *, limit_train=None, limit_val=None, batch_size=None):
        calls.append(
            {
                "limit_train": limit_train,
                "limit_val": limit_val,
                "batch_size": batch_size,
            }
        )
        return f"train_bs_{batch_size}", f"val_bs_{batch_size}"

    monkeypatch.setattr(posttrain, "build_penn_fudan_loaders", fake_build_loaders)
    config = {
        "posttrain": {"batch_size": 2},
        "eval": {"batch_size": 4},
    }

    train_loader, val_loader = posttrain.build_action_verifier_loaders(
        config,
        limit_train=7,
        limit_val=11,
    )

    assert train_loader == "train_bs_2"
    assert val_loader == "val_bs_4"
    assert calls == [
        {"limit_train": 7, "limit_val": 1, "batch_size": 2},
        {"limit_train": 1, "limit_val": 11, "batch_size": 4},
    ]
