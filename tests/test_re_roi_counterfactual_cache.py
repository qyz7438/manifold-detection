from __future__ import annotations

import pytest
import torch

from scripts.experiments.re_roi_counterfactual.action_family import ACTION_FAMILY, action_family_hash, get_action_family
from scripts.experiments.re_roi_counterfactual.teacher import (
    Q_TEACHER_WEIGHTS,
    TeacherResult,
    compute_q_teacher,
    compute_teacher_result,
    fit_q_teacher_stats,
    standardize_q_teacher,
    _count_tp_fp,
    _localization_quality,
    _score_margin,
)
from scripts.experiments.re_roi_counterfactual.cache_builder import (
    _resize_target_to_image_size,
)
from scripts.experiments.re_roi_counterfactual.build_cache import (
    _resolve_requested_device,
    _validate_split_name,
    sha256_text_lf,
)


def test_action_family_is_frozen_and_has_identity() -> None:
    family = get_action_family()
    families = [spec.family for spec in family]
    assert "identity_permutation" in families
    assert len(families) == len(set(families))
    assert len(families) == 10
    non_identity = [spec for spec in family if spec.family != "identity_permutation"]
    assert len(non_identity) == 9
    assert action_family_hash() == "re_roi_action_family_v2_9_non_identity_002_step"


def test_cache_cli_resolves_string_device_through_config_contract() -> None:
    assert _resolve_requested_device("cpu") == torch.device("cpu")


def test_cache_builder_blocks_outer_heldout_before_confirmation() -> None:
    assert _validate_split_name("fit") == "fit"
    assert _validate_split_name("tune") == "tune"
    assert _validate_split_name("calibration") == "calibration"
    with pytest.raises(ValueError, match="outer_heldout"):
        _validate_split_name("outer_heldout")


def test_split_hash_is_stable_across_lf_and_crlf(tmp_path) -> None:
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(b'{\n  "value": 1\n}\n')
    crlf.write_bytes(b'{\r\n  "value": 1\r\n}\r\n')

    assert sha256_text_lf(lf) == sha256_text_lf(crlf)


def test_identity_action_preserves_boxes_and_scores() -> None:
    family = get_action_family()
    identity = next(spec for spec in family if spec.family == "identity_permutation")
    boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0], [5.0, 5.0, 15.0, 15.0]])
    scores = torch.tensor([0.5, 0.3])
    labels = torch.tensor([1, 2])
    new_boxes, new_scores = identity.apply(boxes[0:1], scores[0:1], labels[0:1])
    assert torch.allclose(new_boxes, boxes[0:1])
    assert torch.allclose(new_scores, scores[0:1])


def test_translate_action_moves_box() -> None:
    family = get_action_family()
    left = next(spec for spec in family if spec.family == "translate_left")
    boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    scores = torch.tensor([0.5])
    labels = torch.tensor([1])
    new_boxes, new_scores = left.apply(boxes, scores, labels)
    width = 10.0
    assert torch.allclose(new_boxes, torch.tensor([[10.0 - 0.02 * width, 10.0, 20.0 - 0.02 * width, 20.0]]))


def test_scale_action_changes_area() -> None:
    family = get_action_family()
    scale_up = next(spec for spec in family if spec.family == "scale_up")
    boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    scores = torch.tensor([0.5])
    labels = torch.tensor([1])
    new_boxes, _ = scale_up.apply(boxes, scores, labels)
    assert new_boxes.shape == boxes.shape
    assert new_boxes[0, 0] < boxes[0, 0]
    assert new_boxes[0, 2] > boxes[0, 2]
    old_area = (boxes[0, 2] - boxes[0, 0]) * (boxes[0, 3] - boxes[0, 1])
    new_area = (new_boxes[0, 2] - new_boxes[0, 0]) * (new_boxes[0, 3] - new_boxes[0, 1])
    assert torch.log(new_area / old_area).item() == pytest.approx(0.02, abs=1e-6)


def test_drop_action_returns_empty() -> None:
    family = get_action_family()
    drop = next(spec for spec in family if spec.family == "drop")
    boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    scores = torch.tensor([0.5])
    labels = torch.tensor([1])
    new_boxes, new_scores = drop.apply(boxes, scores, labels)
    assert new_boxes.shape[0] == 0
    assert new_scores.shape[0] == 0


def test_q_teacher_scalarization_matches_protocol() -> None:
    result = TeacherResult(
        native_changed=False,
        delta_tp75=1,
        delta_fp75=2,
        delta_fp50=3,
        delta_duplicate=1,
        delta_score_margin=0.1,
        delta_localization_quality=0.2,
        action_energy=0.02,
    )
    q = compute_q_teacher(result)
    expected = (
        1.00 * 1
        - 0.25 * 2
        - 0.10 * 3
        - 0.50 * 1
        + 0.10 * 0.1
        + 0.10 * 0.2
        - 1.00 * 0.02
    )
    assert abs(q - expected) < 1e-6


def test_q_teacher_stats_standardization() -> None:
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = fit_q_teacher_stats(values)
    standardized = standardize_q_teacher(values, stats)
    assert abs(float(standardized.median()) - 0.0) < 1e-5


def test_teacher_result_detects_native_changed() -> None:
    baseline = {
        "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
        "scores": torch.tensor([0.5]),
        "labels": torch.tensor([1]),
    }
    counterfactual = {
        "boxes": torch.tensor([[11.0, 10.0, 21.0, 20.0]]),
        "scores": torch.tensor([0.5]),
        "labels": torch.tensor([1]),
    }
    gt = {"boxes": torch.tensor([[12.0, 12.0, 22.0, 22.0]]), "labels": torch.tensor([1])}
    result = compute_teacher_result(baseline, counterfactual, gt, action_energy=0.02)
    assert result.native_changed is True


def test_teacher_result_identity_has_zero_deltas() -> None:
    baseline = {
        "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
        "scores": torch.tensor([0.5]),
        "labels": torch.tensor([1]),
    }
    gt = {"boxes": torch.tensor([[12.0, 12.0, 22.0, 22.0]]), "labels": torch.tensor([1])}
    result = compute_teacher_result(baseline, baseline, gt, action_energy=0.0)
    assert result.native_changed is False
    assert result.delta_tp75 == 0
    assert result.delta_fp75 == 0
    assert result.delta_duplicate == 0


def test_target_boxes_are_scaled_into_transformed_detector_coordinates() -> None:
    target = {
        "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
        "labels": torch.tensor([1]),
    }

    transformed = _resize_target_to_image_size(target, (100, 200), (200, 300))

    assert torch.equal(transformed["labels"], target["labels"])
    assert torch.allclose(
        transformed["boxes"],
        torch.tensor([[15.0, 40.0, 45.0, 80.0]]),
    )
    assert transformed is not target


def test_tp_fp_matching_is_score_ordered_and_uses_each_gt_once() -> None:
    gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [2.0, 0.0, 12.0, 10.0]])
    gt_labels = torch.tensor([1, 1])
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0]])
    scores = torch.tensor([0.9, 0.8])
    labels = torch.tensor([1, 1])

    tp, fp = _count_tp_fp(
        boxes,
        scores,
        labels,
        gt_boxes,
        gt_labels,
        iou_threshold=0.5,
        score_threshold=0.05,
    )

    assert (tp, fp) == (2, 0)


def test_quality_terms_do_not_credit_duplicate_predictions_to_one_gt() -> None:
    gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0]])
    gt_labels = torch.tensor([1, 1])
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0]])
    scores = torch.tensor([0.9, 0.8])
    labels = torch.tensor([1, 1])

    localization = _localization_quality(boxes, scores, labels, gt_boxes, gt_labels)
    margin = _score_margin(boxes, scores, labels, gt_boxes, gt_labels)

    assert localization == pytest.approx(1.0)
    assert margin == pytest.approx(0.9)
