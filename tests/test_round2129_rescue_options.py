import pytest
import sys
import torch

from scripts.round2129_nwpu_posttrain_smoke import (
    build_chain_rescue_candidate_mask,
    blocked_nms_crossing_rescue_loss,
    chain_rescue_ranking_loss,
    class_margin_rescue_loss,
    configure_trainable_parts,
    evaluate_clean_detector,
    local_pre_nms_dpo_loss,
    mine_pre_nms_local_dpo_pairs,
    match_pre_nms_decoded_boxes_to_targets,
    find_same_gt_worse_duplicate_pairs,
    find_same_gt_worse_duplicate_proposal_pairs,
    nms_aware_rescue_ranking_loss,
    parse_args,
    pre_nms_score_rescue_loss,
    select_top_rpn_proposals,
    same_gt_duplicate_ranking_loss,
)
from tests.test_bbox_adapter import _FakeDetector
from spectral_detection_posttrain.models.bbox_adapter import install_residual_bbox_adapter


def test_parse_args_accepts_bbox_localization_loss_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "round2129_nwpu_posttrain_smoke.py",
            "--bbox-localization-loss",
            "ciou",
        ],
    )

    args = parse_args()

    assert args.bbox_localization_loss == "ciou"


def test_clean_eval_temporarily_uses_eval_detector_settings(monkeypatch):
    class RoiHeads:
        score_thresh = 0.001
        detections_per_img = 300

    class FakeModel:
        roi_heads = RoiHeads()

    seen = {}

    def fake_evaluate(model, val_loader, device):
        seen["score_thresh"] = model.roi_heads.score_thresh
        seen["detections_per_img"] = model.roi_heads.detections_per_img
        return {"ap75": 1.0}

    monkeypatch.setattr("scripts.round2129_nwpu_posttrain_smoke.evaluate", fake_evaluate)

    metrics = evaluate_clean_detector(
        FakeModel(),
        val_loader=[],
        device=torch.device("cpu"),
        score_threshold=0.05,
        detections_per_img=100,
    )

    assert metrics == {"ap75": 1.0}
    assert seen == {"score_thresh": 0.05, "detections_per_img": 100}
    assert FakeModel.roi_heads.score_thresh == pytest.approx(0.001)
    assert FakeModel.roi_heads.detections_per_img == 300


def test_parse_args_accepts_bbox_predictor_cls_adapter_trainable_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "round2129_nwpu_posttrain_smoke.py",
            "--trainable-mode",
            "bbox_predictor_cls_adapter",
        ],
    )

    args = parse_args()

    assert args.trainable_mode == "bbox_predictor_cls_adapter"


def test_parse_args_accepts_pre_nms_rescue_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "round2129_nwpu_posttrain_smoke.py",
            "--pre-nms-rescue-loss-weight",
            "0.4",
            "--pre-nms-score-target",
            "0.7",
            "--pre-nms-topk-per-gt",
            "2",
            "--pre-nms-dpo-loss-weight",
            "0.6",
            "--pre-nms-dpo-beta",
            "1.5",
        ],
    )

    args = parse_args()

    assert args.pre_nms_rescue_loss_weight == pytest.approx(0.4)
    assert args.pre_nms_score_target == pytest.approx(0.7)
    assert args.pre_nms_topk_per_gt == 2
    assert args.pre_nms_dpo_loss_weight == pytest.approx(0.6)
    assert args.pre_nms_dpo_beta == pytest.approx(1.5)


def test_parse_args_accepts_verifier_guided_ranking_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "round2129_nwpu_posttrain_smoke.py",
            "--verifier-ranking-loss-weight",
            "0.02",
            "--verifier-ranking-margin",
            "0.3",
            "--verifier-ranking-positive-score-min",
            "0.4",
            "--verifier-ranking-negative-score-max",
            "-0.2",
            "--verifier-ranking-max-pairs",
            "11",
        ],
    )

    args = parse_args()

    assert args.verifier_ranking_loss_weight == pytest.approx(0.02)
    assert args.verifier_ranking_margin == pytest.approx(0.3)
    assert args.verifier_ranking_positive_score_min == pytest.approx(0.4)
    assert args.verifier_ranking_negative_score_max == pytest.approx(-0.2)
    assert args.verifier_ranking_max_pairs == 11


def test_bbox_predictor_cls_adapter_mode_trains_bbox_and_cls_adapter_without_cls_score():
    model = _FakeDetector()
    install_residual_bbox_adapter(
        model,
        hidden_dim=4,
        scale=1.0,
        enable_cls_adapter=True,
        cls_scale=0.25,
    )
    args = type("Args", (), {"trainable_mode": "bbox_predictor_cls_adapter", "rescue_mode": True})()

    trainable = configure_trainable_parts(model, args)

    assert any("bbox_adapter" in name for name in trainable)
    assert any("base_predictor.bbox_pred" in name for name in trainable)
    assert any("cls_adapter" in name for name in trainable)
    assert all("base_predictor.cls_score" not in name for name in trainable)
    for name, parameter in model.named_parameters():
        if (
            "bbox_adapter" in name
            or "cls_adapter" in name
            or "box_predictor.base_predictor.bbox_pred" in name
        ):
            assert parameter.requires_grad, name
        else:
            assert not parameter.requires_grad, name


def test_cls_score_mode_trains_only_base_cls_score():
    model = _FakeDetector()
    install_residual_bbox_adapter(
        model,
        hidden_dim=4,
        scale=1.0,
        enable_cls_adapter=True,
        cls_scale=0.25,
    )
    args = type("Args", (), {"trainable_mode": "cls_score", "rescue_mode": True})()

    trainable = configure_trainable_parts(model, args)

    assert trainable
    assert all("base_predictor.cls_score" in name for name in trainable)
    for name, parameter in model.named_parameters():
        if "box_predictor.base_predictor.cls_score" in name:
            assert parameter.requires_grad, name
        else:
            assert not parameter.requires_grad, name


def test_class_margin_rescue_loss_pushes_target_above_top_non_target():
    logits = torch.tensor(
        [
            [0.0, 0.1, 0.5],
            [0.0, 0.8, 0.1],
        ],
        requires_grad=True,
    )
    best_iou = torch.tensor([0.8, 0.8])
    best_labels = torch.tensor([1, 1])
    candidate_mask = torch.tensor([True, False])

    loss, diagnostics = class_margin_rescue_loss(
        logits,
        best_iou,
        best_labels,
        candidate_mask,
        margin=0.2,
    )

    assert diagnostics["class_margin_count"] == 1
    assert diagnostics["class_margin_active_count"] == 1
    assert loss.item() == pytest.approx((0.5 + 0.2 - 0.1) ** 2)
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad[0, 1]) < 0.0
    assert float(logits.grad[0, 2]) > 0.0
    assert logits.grad[1].abs().sum().item() == pytest.approx(0.0)


def test_class_margin_rescue_loss_can_include_background_as_competitor():
    logits = torch.tensor([[0.8, 0.1, 0.0]], requires_grad=True)
    best_iou = torch.tensor([0.8])
    best_labels = torch.tensor([1])
    candidate_mask = torch.tensor([True])

    loss, diagnostics = class_margin_rescue_loss(
        logits,
        best_iou,
        best_labels,
        candidate_mask,
        margin=0.2,
        include_background=True,
    )

    assert diagnostics["class_margin_count"] == 1
    assert diagnostics["class_margin_active_count"] == 1
    assert loss.item() == pytest.approx((0.8 + 0.2 - 0.1) ** 2)
    loss.backward()
    assert float(logits.grad[0, 0]) > 0.0
    assert float(logits.grad[0, 1]) < 0.0
    assert logits.grad[0, 2].item() == pytest.approx(0.0)


def test_select_top_rpn_proposals_truncates_and_resizes_to_original_image_scale():
    proposals = torch.tensor(
        [
            [0.0, 0.0, 100.0, 200.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
    )

    selected = select_top_rpn_proposals(
        proposals,
        transformed_size=(200, 400),
        original_size=(100, 200),
        max_proposals=1,
    )

    assert selected.shape == (1, 4)
    assert selected.tolist()[0] == pytest.approx([0.0, 0.0, 50.0, 100.0])


def test_match_pre_nms_decoded_boxes_to_targets_uses_class_specific_decoded_boxes():
    proposals = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    box_regression = torch.zeros((1, 3 * 4))
    box_regression[0, 2 * 4 + 0] = 10.0
    gt_boxes = torch.tensor([[10.0, 0.0, 20.0, 10.0]])
    gt_labels = torch.tensor([2])

    best_iou, best_labels, target_boxes, best_gt_indices, decoded_boxes = match_pre_nms_decoded_boxes_to_targets(
        proposals,
        box_regression,
        gt_boxes,
        gt_labels,
        image_size=(32, 32),
        num_classes=3,
    )

    assert best_iou.tolist() == pytest.approx([1.0])
    assert best_labels.tolist() == [2]
    assert best_gt_indices.tolist() == [0]
    assert target_boxes.tolist()[0] == pytest.approx([10.0, 0.0, 20.0, 10.0])
    assert decoded_boxes.tolist()[0] == pytest.approx([10.0, 0.0, 20.0, 10.0])


def test_pre_nms_score_rescue_loss_pushes_only_selected_target_label():
    logits = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        requires_grad=True,
    )
    baseline_logits = logits.detach().clone()
    labels = torch.tensor([1, 1])
    candidate_mask = torch.tensor([True, False])

    loss, diagnostics = pre_nms_score_rescue_loss(
        logits,
        baseline_logits,
        labels,
        candidate_mask,
        score_target=0.7,
        score_threshold=0.05,
    )

    assert diagnostics["pre_nms_rescue_count"] == 1
    assert diagnostics["pre_nms_rescue_active_count"] == 1
    assert diagnostics["pre_nms_rescue_prob_delta_mean"] == pytest.approx(0.0)
    loss.backward()
    assert float(logits.grad[0, 1]) < 0.0
    assert logits.grad[1].abs().sum().item() == pytest.approx(0.0)


def test_mine_pre_nms_local_dpo_pairs_selects_higher_iou_candidate_against_higher_score_duplicate():
    labels = torch.tensor([1, 1, 1, 2])
    best_gt_indices = torch.tensor([0, 0, 0, 0])
    best_iou = torch.tensor([0.82, 0.70, 0.78, 0.60])
    baseline_probs = torch.tensor([0.30, 0.60, 0.20, 0.90])
    candidate_mask = torch.tensor([True, False, True, False])

    chosen, rejected, diagnostics = mine_pre_nms_local_dpo_pairs(
        labels,
        best_gt_indices,
        best_iou,
        baseline_probs,
        candidate_mask,
        min_iou_gap=0.05,
        require_rejected_score_ge_chosen=True,
        max_pairs_per_gt=1,
    )

    assert chosen.tolist() == [0]
    assert rejected.tolist() == [1]
    assert diagnostics["pre_nms_dpo_pair_count"] == 1
    assert diagnostics["pre_nms_dpo_mean_iou_gap"] == pytest.approx(0.12)


def test_local_pre_nms_dpo_loss_pushes_chosen_above_rejected_relative_to_baseline():
    current_logits = torch.tensor(
        [
            [0.0, 0.1, 0.0],
            [0.0, 0.5, 0.0],
        ],
        requires_grad=True,
    )
    baseline_logits = torch.tensor(
        [
            [0.0, 0.1, 0.0],
            [0.0, 0.5, 0.0],
        ]
    )
    labels = torch.tensor([1, 1])
    chosen = torch.tensor([0])
    rejected = torch.tensor([1])

    loss, diagnostics = local_pre_nms_dpo_loss(
        current_logits,
        baseline_logits,
        labels,
        chosen,
        rejected,
        beta=2.0,
    )

    assert diagnostics["pre_nms_dpo_pair_count"] == 1
    assert diagnostics["pre_nms_dpo_loss"] == pytest.approx(0.693147, rel=1e-5)
    loss.backward()
    assert float(current_logits.grad[0, 1]) < 0.0
    assert float(current_logits.grad[1, 1]) > 0.0


def test_build_chain_rescue_candidate_mask_selects_topk_per_unmatched_gt():
    best_iou = torch.tensor([0.90, 0.82, 0.95, 0.40, 0.91])
    best_labels = torch.tensor([1, 1, 2, 1, 1])
    best_gt_indices = torch.tensor([0, 0, 1, 0, 2])
    low_conf_scores = torch.tensor([0.20, 0.10, 0.30, 0.20, 0.40])
    unmatched_gt_mask = torch.tensor([True, True, True, True, True])

    selected = build_chain_rescue_candidate_mask(
        best_iou,
        best_labels,
        best_gt_indices,
        low_conf_scores,
        unmatched_gt_mask,
        low_conf_max=0.5,
        high_iou_min=0.75,
        topk_per_gt=1,
    )

    assert selected.tolist() == [True, False, True, False, True]


def test_chain_rescue_ranking_loss_uses_only_dangerous_negatives():
    logits = torch.tensor(
        [
            [0.0, 0.1, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 0.6, 0.0],
            [0.0, 0.9, 0.0],
        ],
        requires_grad=True,
    )
    best_labels = torch.tensor([1, 1, 1, 1])
    positive_mask = torch.tensor([True, False, False, False])
    dangerous_negative_mask = torch.tensor([False, True, False, False])

    loss, diagnostics = chain_rescue_ranking_loss(
        logits,
        best_labels,
        positive_mask,
        dangerous_negative_mask,
        margin=0.2,
    )

    assert diagnostics["chain_ranking_pair_count"] == 1
    assert diagnostics["chain_ranking_active_count"] == 1
    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    assert float(logits.grad[0, 1]) < 0.0
    assert float(logits.grad[1, 1]) > 0.0
    assert logits.grad[2].abs().sum().item() == pytest.approx(0.0)
    assert logits.grad[3].abs().sum().item() == pytest.approx(0.0)


def test_find_same_gt_worse_duplicate_pairs_selects_lower_iou_final_duplicate():
    candidate_boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [30.0, 30.0, 40.0, 40.0],
        ]
    )
    candidate_labels = torch.tensor([1, 1])
    candidate_gt_indices = torch.tensor([0, 1])
    target = {
        "boxes": torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [30.0, 30.0, 40.0, 40.0],
            ]
        ),
        "labels": torch.tensor([1, 1]),
    }
    final_prediction = {
        "boxes": torch.tensor(
            [
                [1.0, 1.0, 11.0, 11.0],
                [30.0, 30.0, 40.0, 40.0],
            ]
        ),
        "labels": torch.tensor([1, 1]),
        "scores": torch.tensor([0.9, 0.8]),
    }

    candidate_indices, suppressor_boxes, diagnostics = find_same_gt_worse_duplicate_pairs(
        candidate_boxes,
        candidate_labels,
        candidate_gt_indices,
        target,
        final_prediction,
        score_threshold=0.05,
        nms_iou_threshold=0.5,
        min_iou_gap=0.05,
    )

    assert candidate_indices.tolist() == [0]
    assert suppressor_boxes.tolist()[0] == pytest.approx([1.0, 1.0, 11.0, 11.0])
    assert diagnostics["same_gt_duplicate_pair_count"] == 1


def test_same_gt_duplicate_ranking_loss_pushes_candidate_above_suppressor():
    candidate_logits = torch.tensor([[0.0, 0.1, 0.0]], requires_grad=True)
    suppressor_logits = torch.tensor([[0.0, 0.5, 0.0]], requires_grad=True)
    labels = torch.tensor([1])

    loss, diagnostics = same_gt_duplicate_ranking_loss(
        candidate_logits,
        suppressor_logits,
        labels,
        margin=0.2,
    )

    assert diagnostics["same_gt_duplicate_ranking_pair_count"] == 1
    assert diagnostics["same_gt_duplicate_ranking_active_count"] == 1
    assert loss.item() == pytest.approx(0.6)
    loss.backward()
    assert float(candidate_logits.grad[0, 1]) < 0.0
    assert float(suppressor_logits.grad[0, 1]) > 0.0


def test_same_gt_duplicate_ranking_loss_can_detach_suppressor_gradient():
    candidate_logits = torch.tensor([[0.0, 0.1, 0.0]], requires_grad=True)
    suppressor_logits = torch.tensor([[0.0, 0.5, 0.0]], requires_grad=True)
    labels = torch.tensor([1])

    loss, diagnostics = same_gt_duplicate_ranking_loss(
        candidate_logits,
        suppressor_logits,
        labels,
        margin=0.2,
        detach_suppressor=True,
    )

    assert diagnostics["same_gt_duplicate_ranking_pair_count"] == 1
    loss.backward()
    assert float(candidate_logits.grad[0, 1]) < 0.0
    assert suppressor_logits.grad is None or suppressor_logits.grad.abs().sum().item() == pytest.approx(0.0)


def test_find_same_gt_worse_duplicate_proposal_pairs_uses_same_forward_duplicates():
    decoded_boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [30.0, 30.0, 40.0, 40.0],
        ]
    )
    scores = torch.tensor([0.4, 0.8, 0.9])
    labels = torch.tensor([1, 1, 1])
    gt_indices = torch.tensor([0, 0, 1])
    candidate_mask = torch.tensor([True, False, True])
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [30.0, 30.0, 40.0, 40.0]]),
        "labels": torch.tensor([1, 1]),
    }

    positive_indices, negative_indices, diagnostics = find_same_gt_worse_duplicate_proposal_pairs(
        decoded_boxes,
        scores,
        labels,
        gt_indices,
        candidate_mask,
        target,
        nms_iou_threshold=0.5,
        min_iou_gap=0.05,
        require_suppressor_score_ge_candidate=True,
    )

    assert positive_indices.tolist() == [0]
    assert negative_indices.tolist() == [1]
    assert diagnostics["same_gt_proposal_pair_count"] == 1


def test_nms_aware_rescue_ranking_loss_pushes_candidate_above_same_class_suppressor():
    logits = torch.tensor(
        [
            [0.0, 0.1, 0.0],
            [0.0, 0.7, 0.0],
            [0.0, 0.9, 0.0],
            [0.0, 0.0, 0.9],
        ],
        requires_grad=True,
    )
    decoded_boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [30.0, 30.0, 40.0, 40.0],
            [1.0, 1.0, 11.0, 11.0],
        ]
    )
    labels = torch.tensor([1, 1, 1, 2])
    candidate_mask = torch.tensor([True, False, False, False])

    loss, diagnostics = nms_aware_rescue_ranking_loss(
        logits,
        decoded_boxes,
        labels,
        candidate_mask,
        nms_iou_threshold=0.5,
        margin=0.2,
        require_suppressor_score_ge_candidate=True,
    )

    assert diagnostics["nms_aware_pair_count"] == 1
    assert diagnostics["nms_aware_active_count"] == 1
    assert loss.item() == pytest.approx(0.8)
    loss.backward()
    assert float(logits.grad[0, 1]) < 0.0
    assert float(logits.grad[1, 1]) > 0.0
    assert logits.grad[2].abs().sum().item() == pytest.approx(0.0)
    assert logits.grad[3].abs().sum().item() == pytest.approx(0.0)


def test_nms_aware_rescue_ranking_loss_can_detach_suppressor_gradient():
    logits = torch.tensor(
        [
            [0.0, 0.1, 0.0],
            [0.0, 0.7, 0.0],
        ],
        requires_grad=True,
    )
    decoded_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
    labels = torch.tensor([1, 1])
    candidate_mask = torch.tensor([True, False])

    loss, diagnostics = nms_aware_rescue_ranking_loss(
        logits,
        decoded_boxes,
        labels,
        candidate_mask,
        nms_iou_threshold=0.5,
        margin=0.2,
        require_suppressor_score_ge_candidate=True,
        ranking_mode="detached_suppressor",
    )

    assert diagnostics["nms_aware_pair_count"] == 1
    loss.backward()
    assert float(logits.grad[0, 1]) < 0.0
    assert logits.grad[1].abs().sum().item() == pytest.approx(0.0)


def test_blocked_nms_crossing_rescue_loss_combines_score_and_max_suppressor_gaps():
    logits = torch.tensor(
        [
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        requires_grad=True,
    )
    decoded_boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [1.0, 1.0, 11.0, 11.0],
            [1.0, 1.0, 11.0, 11.0],
        ]
    )
    labels = torch.tensor([1, 1, 1, 2])
    best_iou = torch.tensor([0.90, 0.80, 0.70, 0.80])
    candidate_mask = torch.tensor([True, False, False, False])

    loss, diagnostics = blocked_nms_crossing_rescue_loss(
        logits,
        decoded_boxes,
        labels,
        best_iou,
        candidate_mask,
        score_threshold=0.2,
        score_epsilon=0.05,
        nms_iou_threshold=0.5,
        base_margin=0.05,
        iou_margin_scale=0.5,
        max_margin=0.3,
        rank_weight=1.0,
        crossing_weight=1.0,
        require_suppressor_score_ge_candidate=True,
    )

    target_prob = torch.softmax(logits.detach(), dim=1)[0, 1].item()
    max_suppressor_prob = torch.softmax(logits.detach(), dim=1)[:3, 1].max().item()
    expected_cross = 0.25 - target_prob
    expected_rank = max_suppressor_prob + 0.15 - target_prob
    assert diagnostics["blocked_nms_pair_count"] == 1
    assert diagnostics["blocked_nms_active_rank_count"] == 1
    assert diagnostics["blocked_nms_active_crossing_count"] == 1
    assert loss.item() == pytest.approx(expected_cross + expected_rank)
    loss.backward()
    assert float(logits.grad[0, 1]) < 0.0
    assert float(logits.grad[2, 1]) > 0.0
    assert logits.grad[3].abs().sum().item() == pytest.approx(0.0)


def test_blocked_nms_crossing_rescue_loss_delta_mode_rewards_relative_candidate_lift():
    logits = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        requires_grad=True,
    )
    baseline_logits = torch.tensor(
        [
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    decoded_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
    labels = torch.tensor([1, 1])
    best_iou = torch.tensor([0.90, 0.80])
    candidate_mask = torch.tensor([True, False])

    loss, diagnostics = blocked_nms_crossing_rescue_loss(
        logits,
        decoded_boxes,
        labels,
        best_iou,
        candidate_mask,
        score_threshold=0.05,
        score_epsilon=0.02,
        nms_iou_threshold=0.5,
        base_margin=0.05,
        iou_margin_scale=0.0,
        max_margin=0.3,
        rank_weight=1.0,
        crossing_weight=0.0,
        require_suppressor_score_ge_candidate=True,
        ranking_mode="delta",
        baseline_logits=baseline_logits,
    )

    assert diagnostics["blocked_nms_pair_count"] == 1
    assert diagnostics["blocked_nms_candidate_delta_mean"] == pytest.approx(0.09198347479104996)
    assert diagnostics["blocked_nms_suppressor_delta_mean"] == pytest.approx(0.2427835762500763)
    assert diagnostics["blocked_nms_relative_delta_mean"] < 0.0
    loss.backward()
    assert float(logits.grad[0, 1]) < 0.0
    assert float(logits.grad[1, 1]) > 0.0
