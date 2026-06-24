import pytest
import torch
import torch.nn.functional as F

from spectral_detection_posttrain.rlvr.confidence_rescue import (
    BestCheckpointConfig,
    ConfidenceRescueConfig,
    ManifoldGateConfig,
    aligned_box_iou_loss,
    bbox_localization_rescue_loss,
    build_pairwise_rescue_ranking_loss,
    build_verifier_guided_ranking_loss,
    calibrate_classwise_thresholds,
    build_manifold_gate_reference,
    build_confidence_rescue_targets,
    combine_verifier_scores,
    confidence_rescue_increment_loss,
    confidence_rescue_loss,
    confidence_threshold_crossing_loss,
    evaluate_verifier_offline,
    manifold_soft_rescue_weights,
    match_boxes_to_target_boxes,
    select_best_checkpoint_update,
    score_shift_budget_loss,
    score_manifold_gate,
    summarize_confidence_rescue_effect,
    summarize_verifier_gate,
    match_boxes_to_targets,
    summarize_confidence_iou_regions,
)


def test_confidence_iou_regions_identify_rescue_and_suppression_cases():
    scores = torch.tensor([0.2, 0.9, 0.2, 0.9])
    best_iou = torch.tensor([0.8, 0.2, 0.2, 0.8])
    cfg = ConfidenceRescueConfig(
        low_conf_max=0.5,
        high_conf_min=0.7,
        high_iou_min=0.75,
        low_iou_max=0.3,
    )

    summary = summarize_confidence_iou_regions(scores, best_iou, cfg)

    assert summary["low_conf_high_iou_count"] == 1
    assert summary["high_conf_low_iou_count"] == 1
    assert summary["low_conf_low_iou_count"] == 1
    assert summary["high_conf_high_iou_count"] == 1


def test_rescue_targets_boost_low_conf_high_iou_and_suppress_high_conf_low_iou():
    scores = torch.tensor([0.2, 0.9, 0.2, 0.9])
    best_iou = torch.tensor([0.8, 0.2, 0.2, 0.8])
    best_labels = torch.tensor([3, 4, 5, 6])
    cfg = ConfidenceRescueConfig(
        low_conf_max=0.5,
        high_conf_min=0.7,
        high_iou_min=0.75,
        low_iou_max=0.3,
        positive_weight=1.0,
        negative_weight=0.25,
        include_low_conf_negatives=False,
    )

    targets = build_confidence_rescue_targets(scores, best_iou, best_labels, cfg)

    assert targets.target_labels.tolist() == [3, 0, 0, 0]
    assert targets.weights.tolist() == pytest.approx([1.0, 0.25, 0.0, 0.0])
    assert targets.positive_mask.tolist() == [True, False, False, False]
    assert targets.negative_mask.tolist() == [False, True, False, False]


def test_rescue_targets_can_gate_positive_cases_by_verifier_evidence():
    scores = torch.tensor([0.2, 0.2, 0.9])
    best_iou = torch.tensor([0.8, 0.8, 0.2])
    best_labels = torch.tensor([1, 2, 3])
    verifier_scores = torch.tensor([0.9, -0.4, 0.1])
    cfg = ConfidenceRescueConfig(
        low_conf_max=0.5,
        high_conf_min=0.7,
        high_iou_min=0.75,
        low_iou_max=0.3,
        verifier_positive_min=0.0,
    )

    targets = build_confidence_rescue_targets(
        scores,
        best_iou,
        best_labels,
        cfg,
        verifier_scores=verifier_scores,
    )

    assert targets.positive_mask.tolist() == [True, False, False]
    assert targets.target_labels.tolist() == [1, 0, 0]
    assert targets.weights.tolist() == pytest.approx([1.0, 0.0, 0.25])


def test_rescue_targets_use_roi_label_confidence_for_positive_selection_when_provided():
    rollout_scores = torch.tensor([0.01, 0.01, 0.9])
    roi_label_scores = torch.tensor([0.90, 0.20, 0.90])
    best_iou = torch.tensor([0.8, 0.8, 0.2])
    best_labels = torch.tensor([1, 1, 1])
    cfg = ConfidenceRescueConfig(
        low_conf_max=0.5,
        high_conf_min=0.7,
        high_iou_min=0.75,
        low_iou_max=0.3,
    )

    targets = build_confidence_rescue_targets(
        rollout_scores,
        best_iou,
        best_labels,
        cfg,
        low_conf_scores=roi_label_scores,
    )

    assert targets.positive_mask.tolist() == [False, True, False]
    assert targets.negative_mask.tolist() == [False, False, True]
    assert targets.weights.tolist() == pytest.approx([0.0, 1.0, 0.25])


def test_confidence_rescue_loss_uses_weighted_cross_entropy_for_selected_cases():
    logits = torch.tensor(
        [
            [0.0, 0.0, 2.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        requires_grad=True,
    )
    scores = torch.tensor([0.2, 0.9, 0.2])
    best_iou = torch.tensor([0.8, 0.2, 0.2])
    best_labels = torch.tensor([2, 1, 1])
    cfg = ConfidenceRescueConfig(
        low_conf_max=0.5,
        high_conf_min=0.7,
        high_iou_min=0.75,
        low_iou_max=0.3,
        positive_weight=1.0,
        negative_weight=0.25,
        include_low_conf_negatives=False,
    )

    loss, diagnostics = confidence_rescue_loss(logits, scores, best_iou, best_labels, cfg)

    raw = F.cross_entropy(logits[:2], torch.tensor([2, 0]), reduction="none")
    expected = (raw * torch.tensor([1.0, 0.25])).sum() / 1.25
    assert loss.item() == pytest.approx(expected.item())
    assert diagnostics["rescue_positive_count"] == 1
    assert diagnostics["rescue_negative_count"] == 1
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_match_boxes_to_targets_returns_best_iou_and_gt_label():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
            [50.0, 50.0, 60.0, 60.0],
        ]
    )
    gt_boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 32.0, 32.0],
        ]
    )
    gt_labels = torch.tensor([3, 7])

    best_iou, best_labels = match_boxes_to_targets(boxes, gt_boxes, gt_labels)

    assert best_iou.tolist() == pytest.approx([1.0, 100.0 / 144.0, 0.0])
    assert best_labels.tolist() == [3, 7, 0]


def test_combine_verifier_scores_standardizes_against_reference_stats():
    fft = torch.tensor([2.0, 4.0])
    manifold = torch.tensor([10.0, 4.0])
    reference = {
        "fft_mean": 2.0,
        "fft_std": 2.0,
        "manifold_mean": 6.0,
        "manifold_std": 2.0,
    }

    combined = combine_verifier_scores(
        fft,
        manifold,
        reference,
        fft_weight=0.25,
        manifold_weight=0.75,
    )

    assert combined.tolist() == pytest.approx([1.5, -0.5])


def test_summarize_verifier_gate_reports_coverage_and_false_rescue_risk():
    scores = torch.tensor([0.2, 0.2, 0.2, 0.9])
    best_iou = torch.tensor([0.8, 0.2, 0.8, 0.2])
    verifier = torch.tensor([0.5, 0.4, -0.5, 1.0])
    cfg = ConfidenceRescueConfig(
        low_conf_max=0.5,
        high_conf_min=0.7,
        high_iou_min=0.75,
        low_iou_max=0.3,
    )

    summary = summarize_verifier_gate(scores, best_iou, verifier, cfg, threshold=0.0)

    assert summary["gate_low_conf_high_iou_count"] == 1
    assert summary["gate_low_conf_low_iou_count"] == 1
    assert summary["gate_low_conf_high_iou_recall"] == pytest.approx(0.5)
    assert summary["gate_low_conf_precision"] == pytest.approx(0.5)
    assert summary["gate_low_conf_false_rescue_rate"] == pytest.approx(0.5)


def test_summarize_confidence_rescue_effect_reports_lchi_probability_lift_and_threshold_crossing():
    baseline_probs = torch.tensor([0.04, 0.20, 0.60, 0.03])
    current_probs = torch.tensor([0.07, 0.55, 0.65, 0.02])
    lchi_mask = torch.tensor([True, True, False, True])
    verifier_positive_mask = torch.tensor([True, False, False, True])

    summary = summarize_confidence_rescue_effect(
        baseline_probs,
        current_probs,
        lchi_mask,
        verifier_positive_mask=verifier_positive_mask,
        score_threshold=0.05,
        low_conf_max=0.5,
    )

    assert summary["lchi_conf_count"] == 3
    assert summary["lchi_conf_baseline_prob_sum"] == pytest.approx(0.27)
    assert summary["lchi_conf_current_prob_sum"] == pytest.approx(0.64)
    assert summary["lchi_conf_delta_sum"] == pytest.approx(0.37)
    assert summary["lchi_conf_delta_mean"] == pytest.approx(0.37 / 3.0)
    assert summary["lchi_conf_cross_score_threshold_count"] == 1
    assert summary["lchi_conf_cross_score_threshold_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["lchi_conf_cross_low_conf_max_count"] == 1
    assert summary["lchi_conf_cross_low_conf_max_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["verifier_positive_lchi_conf_count"] == 2
    assert summary["verifier_positive_lchi_conf_delta_sum"] == pytest.approx(0.02)
    assert summary["verifier_positive_lchi_conf_delta_mean"] == pytest.approx(0.01)
    assert summary["verifier_positive_lchi_conf_cross_score_threshold_count"] == 1
    assert summary["verifier_positive_lchi_conf_cross_low_conf_max_count"] == 0


def test_confidence_threshold_crossing_loss_targets_only_verifier_positive_lchi():
    logits = torch.tensor(
        [
            [0.0, -3.0],
            [0.0, -3.0],
            [0.0, -3.0],
            [0.0, 3.0],
        ],
        requires_grad=True,
    )
    baseline_logits = torch.tensor(
        [
            [0.0, -4.0],
            [0.0, -4.0],
            [0.0, -4.0],
            [0.0, 3.0],
        ]
    )
    scores = torch.tensor([0.2, 0.2, 0.2, 0.9])
    best_iou = torch.tensor([0.8, 0.8, 0.2, 0.8])
    best_labels = torch.tensor([1, 1, 1, 1])
    verifier_scores = torch.tensor([0.5, -0.5, 0.5, 0.5])
    cfg = ConfidenceRescueConfig(
        low_conf_max=0.5,
        high_conf_min=0.7,
        high_iou_min=0.75,
        low_iou_max=0.3,
        verifier_positive_min=0.0,
    )

    loss, diagnostics = confidence_threshold_crossing_loss(
        logits,
        baseline_logits,
        scores,
        best_iou,
        best_labels,
        cfg,
        verifier_scores=verifier_scores,
        score_threshold=0.05,
        margin=0.02,
    )

    assert diagnostics["confidence_crossing_count"] == 1
    assert diagnostics["confidence_crossing_active_count"] == 1
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad[0, 1]) < 0.0
    assert logits.grad[1].abs().sum().item() == pytest.approx(0.0)
    assert logits.grad[2].abs().sum().item() == pytest.approx(0.0)
    assert logits.grad[3].abs().sum().item() == pytest.approx(0.0)


def test_confidence_threshold_crossing_loss_can_use_final_score_as_baseline_source():
    logits = torch.tensor([[0.0, -4.0]], requires_grad=True)
    baseline_logits = torch.tensor([[0.0, 3.0]])
    final_scores = torch.tensor([0.01])
    best_iou = torch.tensor([0.8])
    best_labels = torch.tensor([1])
    cfg = ConfidenceRescueConfig(
        low_conf_max=0.5,
        high_conf_min=0.7,
        high_iou_min=0.75,
        low_iou_max=0.3,
    )

    loss, diagnostics = confidence_threshold_crossing_loss(
        logits,
        baseline_logits,
        final_scores,
        best_iou,
        best_labels,
        cfg,
        low_conf_scores=final_scores,
        crossing_baseline_scores=final_scores,
        score_threshold=0.05,
        margin=0.02,
    )

    assert diagnostics["confidence_crossing_count"] == 1
    assert loss.item() > 0.0


def test_confidence_rescue_targets_can_filter_positive_candidates():
    scores = torch.tensor([0.1, 0.1, 0.1])
    best_iou = torch.tensor([0.8, 0.8, 0.2])
    best_labels = torch.tensor([1, 1, 1])
    positive_mask = torch.tensor([False, True, True])
    cfg = ConfidenceRescueConfig(
        low_conf_max=0.5,
        high_conf_min=0.7,
        high_iou_min=0.75,
        low_iou_max=0.3,
    )

    targets = build_confidence_rescue_targets(
        scores,
        best_iou,
        best_labels,
        cfg,
        positive_candidate_mask=positive_mask,
    )

    assert targets.positive_mask.tolist() == [False, True, False]
    assert targets.weights.tolist() == pytest.approx([0.0, 1.0, 0.0])


def test_classwise_tp_fp_density_ratio_prefers_class_matched_tp_region():
    features = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.2, 5.0],
            [2.5, 2.5],
            [2.7, 2.5],
        ]
    )
    labels = torch.tensor([1, 1, 2, 2, 1, 1])
    is_positive = torch.tensor([True, True, True, True, False, False])
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
        ]
    )
    ref = build_manifold_gate_reference(features, labels, is_positive, boxes, image_size=(100, 100), num_classes=3)
    cfg = ManifoldGateConfig(mode="density_ratio", k=2, fp_weight=1.0)

    query = torch.tensor([[0.05, 0.0], [2.6, 2.5], [5.1, 5.0]])
    query_labels = torch.tensor([1, 1, 2])
    query_boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
        ]
    )

    scores = score_manifold_gate(ref, query, query_labels, query_boxes, image_size=(100, 100), cfg=cfg)

    assert scores[0] > scores[1]
    assert scores[2] > scores[1]


def test_scale_bucket_calibration_can_use_different_thresholds_per_bucket():
    features = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [2.0, 2.0],
            [2.2, 2.0],
        ]
    )
    labels = torch.tensor([1, 1, 1, 1])
    is_positive = torch.tensor([True, False, True, False])
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 60.0, 60.0],
            [0.0, 0.0, 60.0, 60.0],
        ]
    )

    ref = build_manifold_gate_reference(features, labels, is_positive, boxes, image_size=(100, 100), num_classes=2)

    assert (1, 0) in ref.thresholds
    assert (1, 2) in ref.thresholds
    assert ref.thresholds[(1, 0)] != ref.thresholds[(1, 2)]


def test_margin_mode_requires_target_class_to_beat_next_class():
    features = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [1.0, 0.0],
            [1.2, 0.0],
        ]
    )
    labels = torch.tensor([1, 1, 2, 2])
    is_positive = torch.tensor([True, True, True, True])
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]]).repeat(4, 1)
    ref = build_manifold_gate_reference(features, labels, is_positive, boxes, image_size=(100, 100), num_classes=3)
    cfg = ManifoldGateConfig(mode="margin", k=2)

    query = torch.tensor([[0.1, 0.0], [0.55, 0.0]])
    query_labels = torch.tensor([1, 1])
    query_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]]).repeat(2, 1)
    scores = score_manifold_gate(ref, query, query_labels, query_boxes, image_size=(100, 100), cfg=cfg)

    assert scores[0] > scores[1]


def test_hard_negative_bank_penalizes_known_false_rescue_region():
    features = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.4, 0.0],
            [0.5, 0.0],
        ]
    )
    labels = torch.tensor([1, 1, 1, 1])
    is_positive = torch.tensor([True, True, False, False])
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]]).repeat(4, 1)
    ref = build_manifold_gate_reference(features, labels, is_positive, boxes, image_size=(100, 100), num_classes=2)
    cfg = ManifoldGateConfig(mode="density_ratio", k=2, fp_weight=0.0, hard_negative_weight=2.0)

    query = torch.tensor([[0.05, 0.0], [0.45, 0.0]])
    query_labels = torch.tensor([1, 1])
    query_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]]).repeat(2, 1)
    scores = score_manifold_gate(ref, query, query_labels, query_boxes, image_size=(100, 100), cfg=cfg)

    assert scores[0] > scores[1]


def test_bucket_threshold_calibration_makes_scores_bucket_relative():
    features = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [3.0, 0.0],
            [3.2, 0.0],
        ]
    )
    labels = torch.tensor([1, 1, 1, 1])
    is_positive = torch.tensor([True, False, True, False])
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 60.0, 60.0],
            [0.0, 0.0, 60.0, 60.0],
        ]
    )
    ref = build_manifold_gate_reference(features, labels, is_positive, boxes, image_size=(100, 100), num_classes=2)
    raw_cfg = ManifoldGateConfig(mode="density_ratio", k=1, fp_weight=0.0, use_bucket_thresholds=False)
    calibrated_cfg = ManifoldGateConfig(mode="density_ratio", k=1, fp_weight=0.0, use_bucket_thresholds=True)

    raw = score_manifold_gate(ref, features[[0, 2]], labels[[0, 2]], boxes[[0, 2]], image_size=(100, 100), cfg=raw_cfg)
    calibrated = score_manifold_gate(
        ref,
        features[[0, 2]],
        labels[[0, 2]],
        boxes[[0, 2]],
        image_size=(100, 100),
        cfg=calibrated_cfg,
    )

    assert not torch.allclose(raw, calibrated)
    assert calibrated.shape == raw.shape


def test_classwise_threshold_calibration_maximizes_recall_at_precision_floor():
    verifier_scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.95, 0.7, 0.3])
    labels = torch.tensor([1, 1, 1, 1, 1, 1, 2, 2, 2])
    scores = torch.full_like(verifier_scores, 0.2)
    best_iou = torch.tensor([0.8, 0.8, 0.2, 0.8, 0.2, 0.2, 0.8, 0.2, 0.8])
    cfg = ConfidenceRescueConfig(
        low_conf_max=0.5,
        high_iou_min=0.75,
        low_iou_max=0.3,
    )

    thresholds, diagnostics = calibrate_classwise_thresholds(
        verifier_scores,
        labels,
        scores,
        best_iou,
        cfg,
        min_precision=0.7,
        min_positives=1,
    )

    assert thresholds[1] == pytest.approx(0.6)
    assert diagnostics[1]["precision"] == pytest.approx(0.75)
    assert diagnostics[1]["recall"] == pytest.approx(1.0)
    assert thresholds[2] == pytest.approx(0.95)
    assert diagnostics[2]["precision"] == pytest.approx(1.0)


def test_classwise_threshold_calibration_falls_back_when_precision_floor_unreachable():
    verifier_scores = torch.tensor([0.9, 0.8, 0.7])
    labels = torch.tensor([1, 1, 1])
    scores = torch.full_like(verifier_scores, 0.2)
    best_iou = torch.tensor([0.2, 0.8, 0.2])
    cfg = ConfidenceRescueConfig(low_conf_max=0.5, high_iou_min=0.75, low_iou_max=0.3)

    thresholds, diagnostics = calibrate_classwise_thresholds(
        verifier_scores,
        labels,
        scores,
        best_iou,
        cfg,
        min_precision=0.95,
        fallback_threshold=1.1,
        min_positives=2,
    )

    assert thresholds[1] == pytest.approx(1.1)
    assert diagnostics[1]["selected"] == 0


def test_classwise_threshold_calibration_respects_min_threshold_floor():
    verifier_scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    labels = torch.tensor([1, 1, 1, 1])
    scores = torch.full_like(verifier_scores, 0.2)
    best_iou = torch.tensor([0.8, 0.8, 0.8, 0.2])
    cfg = ConfidenceRescueConfig(low_conf_max=0.5, high_iou_min=0.75, low_iou_max=0.3)

    thresholds, diagnostics = calibrate_classwise_thresholds(
        verifier_scores,
        labels,
        scores,
        best_iou,
        cfg,
        min_precision=0.7,
        min_threshold=0.75,
    )

    assert thresholds[1] == pytest.approx(0.8)
    assert diagnostics[1]["recall"] == pytest.approx(2 / 3)


def test_verifier_high_low_conf_low_iou_cases_become_hard_negatives():
    scores = torch.tensor([0.2, 0.2, 0.2, 0.9])
    best_iou = torch.tensor([0.8, 0.2, 0.2, 0.2])
    labels = torch.tensor([1, 1, 1, 1])
    verifier_scores = torch.tensor([0.7, 0.8, -0.2, 0.9])
    cfg = ConfidenceRescueConfig(
        low_conf_max=0.5,
        high_conf_min=0.7,
        high_iou_min=0.75,
        low_iou_max=0.3,
        verifier_positive_min=0.0,
        verifier_hard_negative_min=0.0,
    )

    targets = build_confidence_rescue_targets(
        scores,
        best_iou,
        labels,
        cfg,
        verifier_scores=verifier_scores,
    )

    assert targets.positive_mask.tolist() == [True, False, False, False]
    assert targets.negative_mask.tolist() == [False, True, False, True]
    assert targets.weights.tolist() == pytest.approx([1.0, 0.25, 0.0, 0.25])


def test_offline_verifier_reports_auc_and_recall_at_precision_targets():
    verifier_scores = torch.tensor([0.9, 0.8, 0.7, 0.2, 0.1])
    scores = torch.full_like(verifier_scores, 0.2)
    best_iou = torch.tensor([0.8, 0.8, 0.2, 0.2, 0.2])
    labels = torch.tensor([1, 1, 1, 1, 1])
    cfg = ConfidenceRescueConfig(low_conf_max=0.5, high_iou_min=0.75, low_iou_max=0.3)

    report = evaluate_verifier_offline(
        verifier_scores,
        labels,
        scores,
        best_iou,
        cfg,
        threshold=0.5,
        precision_targets=(0.7, 0.9),
    )

    assert report["candidate_count"] == 5
    assert report["positive_count"] == 2
    assert report["negative_count"] == 3
    assert report["auc"] == pytest.approx(1.0)
    assert report["precision_at_threshold"] == pytest.approx(2 / 3)
    assert report["recall_at_threshold"] == pytest.approx(1.0)
    assert report["false_rescue_rate_at_threshold"] == pytest.approx(1 / 3)
    assert report["recall_at_precision_0.7"] == pytest.approx(1.0)
    assert report["recall_at_precision_0.9"] == pytest.approx(1.0)


def test_pairwise_rescue_ranking_loss_orders_lchi_above_low_iou_cases():
    logits = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 0.5],
            [0.0, 1.5],
        ],
        requires_grad=True,
    )
    scores = torch.tensor([0.2, 0.2, 0.2, 0.9])
    best_iou = torch.tensor([0.8, 0.8, 0.2, 0.2])
    labels = torch.tensor([1, 1, 1, 1])
    cfg = ConfidenceRescueConfig(low_conf_max=0.5, high_conf_min=0.7, high_iou_min=0.75, low_iou_max=0.3)

    loss, diag = build_pairwise_rescue_ranking_loss(logits, scores, best_iou, labels, cfg, margin=0.1)

    def sp(value: float) -> torch.Tensor:
        return F.softplus(torch.tensor(value))

    expected = (
        sp(0.1 - (1.0 - 0.5))
        + sp(0.1 - (2.0 - 0.5))
        + sp(0.1 - (1.0 - 1.5))
        + sp(0.1 - (2.0 - 1.5))
    ) / 4
    assert loss.item() == pytest.approx(expected.item())
    assert diag["pairwise_rescue_pair_count"] == 4
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_pairwise_dangerous_mode_ignores_ordinary_low_conf_low_iou_negatives():
    logits = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 0.5],
            [0.0, 1.5],
            [0.0, 1.2],
        ],
        requires_grad=True,
    )
    scores = torch.tensor([0.2, 0.2, 0.2, 0.9, 0.2])
    best_iou = torch.tensor([0.8, 0.8, 0.2, 0.2, 0.2])
    labels = torch.tensor([1, 1, 1, 1, 1])
    cfg = ConfidenceRescueConfig(low_conf_max=0.5, high_conf_min=0.7, high_iou_min=0.75, low_iou_max=0.3)

    _, all_diag = build_pairwise_rescue_ranking_loss(
        logits,
        scores,
        best_iou,
        labels,
        cfg,
        margin=0.1,
        negative_mode="all_low_iou",
    )
    loss, dangerous_diag = build_pairwise_rescue_ranking_loss(
        logits,
        scores,
        best_iou,
        labels,
        cfg,
        margin=0.1,
        negative_mode="dangerous",
    )

    assert all_diag["pairwise_rescue_pair_count"] == 6
    assert dangerous_diag["pairwise_rescue_pair_count"] == 2
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_verifier_guided_ranking_loss_orders_high_fft_high_iou_above_low_fft_low_iou():
    logits = torch.tensor(
        [
            [0.0, 0.4],
            [0.0, 1.1],
            [0.0, 1.4],
            [0.0, 0.9],
            [0.0, 2.0],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([1, 1, 1, 1, 1])
    best_iou = torch.tensor([0.8, 0.76, 0.2, 0.25, 0.85])
    verifier_scores = torch.tensor([1.0, 0.8, -0.4, -0.8, -0.2], requires_grad=True)

    loss, diag = build_verifier_guided_ranking_loss(
        logits,
        labels,
        best_iou,
        verifier_scores,
        positive_iou_min=0.75,
        negative_iou_max=0.3,
        positive_score_min=0.5,
        negative_score_max=0.0,
        margin=0.2,
        max_pairs=3,
    )

    assert diag["verifier_ranking_pair_count"] == 3
    assert diag["verifier_ranking_positive_count"] == 2
    assert diag["verifier_ranking_negative_count"] == 2
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0
    assert verifier_scores.grad is None


def test_increment_rescue_targets_baseline_plus_delta_instead_of_hard_label():
    def binary_logits(prob: float) -> torch.Tensor:
        return torch.tensor([0.0, torch.logit(torch.tensor(prob)).item()])

    logits = binary_logits(0.45).unsqueeze(0).requires_grad_(True)
    baseline_logits = binary_logits(0.40).unsqueeze(0)
    scores = torch.tensor([0.2])
    best_iou = torch.tensor([0.8])
    labels = torch.tensor([1])
    cfg = ConfidenceRescueConfig(low_conf_max=0.5, high_iou_min=0.75)

    loss, diag = confidence_rescue_increment_loss(
        logits,
        baseline_logits,
        scores,
        best_iou,
        labels,
        cfg,
        target_delta=0.10,
        target_cap=0.45,
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert diag["rescue_increment_target_mean"] == pytest.approx(0.45, abs=1e-6)
    loss.backward()
    assert logits.grad is not None


def test_increment_rescue_target_never_caps_below_baseline_probability():
    def binary_logits(prob: float) -> torch.Tensor:
        return torch.tensor([0.0, torch.logit(torch.tensor(prob)).item()])

    logits = binary_logits(0.90).unsqueeze(0).requires_grad_(True)
    baseline_logits = binary_logits(0.90).unsqueeze(0)
    scores = torch.tensor([0.2])
    best_iou = torch.tensor([0.8])
    labels = torch.tensor([1])
    cfg = ConfidenceRescueConfig(low_conf_max=0.5, high_iou_min=0.75)

    loss, diag = confidence_rescue_increment_loss(
        logits,
        baseline_logits,
        scores,
        best_iou,
        labels,
        cfg,
        target_delta=0.10,
        target_cap=0.60,
    )

    assert diag["rescue_increment_target_mean"] == pytest.approx(0.90, abs=1e-6)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    loss.backward()
    assert logits.grad is not None


def test_score_shift_budget_penalizes_non_rescue_probability_increase_only():
    def binary_logits(prob: float) -> torch.Tensor:
        return torch.tensor([0.0, torch.logit(torch.tensor(prob)).item()])

    logits = torch.stack([binary_logits(0.90), binary_logits(0.70)]).requires_grad_(True)
    baseline_logits = torch.stack([binary_logits(0.10), binary_logits(0.60)])
    scores = torch.tensor([0.2, 0.9])
    best_iou = torch.tensor([0.8, 0.2])
    labels = torch.tensor([1, 1])
    cfg = ConfidenceRescueConfig(low_conf_max=0.5, high_conf_min=0.7, high_iou_min=0.75, low_iou_max=0.3)

    loss, diag = score_shift_budget_loss(
        logits,
        baseline_logits,
        scores,
        best_iou,
        labels,
        cfg,
        delta=0.05,
    )

    assert loss.item() == pytest.approx(0.05, abs=1e-6)
    assert diag["score_budget_count"] == 1
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad[0].abs().sum()) == pytest.approx(0.0, abs=1e-7)
    assert float(logits.grad[1].abs().sum()) > 0.0


def test_sigmoid_verifier_weight_keeps_threshold_borderline_cases_softly_weighted():
    scores = torch.tensor([0.2, 0.2])
    best_iou = torch.tensor([0.8, 0.8])
    labels = torch.tensor([1, 1])
    verifier_scores = torch.tensor([1.0, -1.0])
    cfg = ConfidenceRescueConfig(
        low_conf_max=0.5,
        high_iou_min=0.75,
        verifier_positive_min=0.0,
        verifier_weight_mode="sigmoid",
        verifier_weight_temperature=1.0,
    )

    targets = build_confidence_rescue_targets(
        scores,
        best_iou,
        labels,
        cfg,
        verifier_scores=verifier_scores,
    )

    assert targets.positive_mask.tolist() == [True, True]
    assert targets.weights.tolist() == pytest.approx(torch.sigmoid(verifier_scores).tolist())


def test_manifold_soft_rescue_weights_are_detached_sigmoid_lchi_weights():
    scores = torch.tensor([0.2, 0.2, 0.9])
    best_iou = torch.tensor([0.8, 0.8, 0.8])
    labels = torch.tensor([1, 1, 1])
    verifier_scores = torch.tensor([1.0, -1.0, 1.0], requires_grad=True)
    thresholds = torch.tensor([0.0, 0.0, 0.0])
    cfg = ConfidenceRescueConfig(low_conf_max=0.5, high_iou_min=0.75)

    weights, mask = manifold_soft_rescue_weights(
        scores,
        best_iou,
        labels,
        verifier_scores,
        cfg,
        thresholds=thresholds,
        temperature=1.0,
    )

    assert mask.tolist() == [True, True, False]
    assert weights.tolist() == pytest.approx([torch.sigmoid(torch.tensor(1.0)).item(), torch.sigmoid(torch.tensor(-1.0)).item(), 0.0])
    assert not weights.requires_grad


def test_bbox_localization_rescue_loss_weights_only_lchi_boxes_and_backprops_to_decoded_box():
    decoded_boxes = torch.tensor(
        [
            [0.0, 0.0, 8.0, 8.0],
            [0.0, 0.0, 20.0, 20.0],
            [40.0, 40.0, 50.0, 50.0],
        ],
        requires_grad=True,
    )
    target_boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 20.0, 20.0],
            [40.0, 40.0, 50.0, 50.0],
        ]
    )
    scores = torch.tensor([0.2, 0.9, 0.2])
    best_iou = torch.tensor([0.8, 0.9, 0.2])
    labels = torch.tensor([1, 1, 1])
    rescue_weights = torch.tensor([0.5, 1.0, 1.0])
    cfg = ConfidenceRescueConfig(low_conf_max=0.5, high_conf_min=0.7, high_iou_min=0.75, low_iou_max=0.3)

    loss, diag = bbox_localization_rescue_loss(
        decoded_boxes,
        target_boxes,
        scores,
        best_iou,
        labels,
        cfg,
        rescue_weights=rescue_weights,
    )

    assert diag["bbox_rescue_count"] == 1
    assert diag["bbox_rescue_weight_sum"] == pytest.approx(0.5)
    assert loss.item() > 0.0
    loss.backward()
    assert decoded_boxes.grad is not None
    assert float(decoded_boxes.grad[0].abs().sum()) > 0.0
    assert float(decoded_boxes.grad[1].abs().sum()) == pytest.approx(0.0, abs=1e-7)
    assert float(decoded_boxes.grad[2].abs().sum()) == pytest.approx(0.0, abs=1e-7)


def test_aligned_box_iou_loss_supports_smooth_l1_giou_diou_and_ciou():
    decoded_boxes = torch.tensor(
        [
            [0.0, 0.0, 8.0, 8.0],
            [1.0, 1.0, 9.0, 9.0],
            [0.0, 0.0, 10.0, 10.0],
        ],
        requires_grad=True,
    )
    target_boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
        ]
    )

    for mode in ("smooth_l1", "giou", "diou", "ciou"):
        decoded_boxes.grad = None
        losses = aligned_box_iou_loss(decoded_boxes, target_boxes, mode=mode)
        assert losses.shape == (3,)
        assert torch.all(losses >= 0)
        assert losses[2].item() == pytest.approx(0.0, abs=1e-6)
        losses[:2].mean().backward(retain_graph=True)
        assert decoded_boxes.grad is not None
        assert float(decoded_boxes.grad[:2].abs().sum()) > 0.0


def test_bbox_localization_rescue_loss_reports_selected_loss_mode():
    decoded_boxes = torch.tensor([[0.0, 0.0, 8.0, 8.0]], requires_grad=True)
    target_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    scores = torch.tensor([0.2])
    best_iou = torch.tensor([0.8])
    labels = torch.tensor([1])
    cfg = ConfidenceRescueConfig(low_conf_max=0.5, high_iou_min=0.75)

    loss, diag = bbox_localization_rescue_loss(
        decoded_boxes,
        target_boxes,
        scores,
        best_iou,
        labels,
        cfg,
        loss_mode="ciou",
    )

    assert diag["bbox_rescue_loss_mode"] == "ciou"
    assert loss.item() > 0.0


def test_match_boxes_to_target_boxes_returns_best_gt_box_and_zero_fallback():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    gt_labels = torch.tensor([3])

    best_iou, labels, target_boxes = match_boxes_to_target_boxes(boxes, gt_boxes, gt_labels)

    assert best_iou.tolist() == pytest.approx([1.0, 0.0])
    assert labels.tolist() == [3, 0]
    assert torch.allclose(target_boxes, torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 0.0, 0.0]]))


def test_safety_guard_rejects_metric_update_when_fp_or_prediction_count_regresses():
    baseline = {"ap75": 0.30, "num_predictions": 100, "false_positive_rate": 0.20, "ece": 0.05}
    current = {"ap75": 0.35, "num_predictions": 230, "false_positive_rate": 0.21, "ece": 0.06}
    cfg = BestCheckpointConfig(
        selection_metric="ap75",
        max_prediction_ratio=2.0,
        max_fp_rate_delta=0.05,
        max_ece_delta=0.05,
    )

    decision = select_best_checkpoint_update(current, baseline, None, cfg)

    assert decision["should_update_best"] is False
    assert decision["safe_to_save_best"] is False
    assert "prediction_ratio" in decision["failed_guards"]


def test_safety_guard_accepts_safe_metric_improvement_and_rejects_worse_best():
    baseline = {"ap75": 0.30, "num_predictions": 100, "false_positive_rate": 0.20, "ece": 0.05}
    safe = {"ap75": 0.34, "num_predictions": 110, "false_positive_rate": 0.21, "ece": 0.055}
    better_best = {"ap75": 0.36}
    cfg = BestCheckpointConfig(selection_metric="ap75")

    accepted = select_best_checkpoint_update(safe, baseline, None, cfg)
    rejected = select_best_checkpoint_update(safe, baseline, better_best, cfg)

    assert accepted["should_update_best"] is True
    assert accepted["safe_to_save_best"] is True
    assert rejected["should_update_best"] is False
    assert rejected["metric_improved"] is False


def test_manifold_reference_feature_source_projection_changes_geometry():
    features = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
        ]
    )
    labels = torch.tensor([1, 1, 1])
    is_positive = torch.tensor([True, True, False])
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]]).repeat(3, 1)

    full_ref = build_manifold_gate_reference(
        features,
        labels,
        is_positive,
        boxes,
        image_size=(100, 100),
        num_classes=2,
        feature_projection="identity",
    )
    sliced_ref = build_manifold_gate_reference(
        features,
        labels,
        is_positive,
        boxes,
        image_size=(100, 100),
        num_classes=2,
        feature_projection="first_half",
    )

    assert full_ref.global_tp.shape[1] == 3
    assert sliced_ref.global_tp.shape[1] == 1
