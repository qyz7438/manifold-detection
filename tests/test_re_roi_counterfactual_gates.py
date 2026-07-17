"""Tests for the re-ROI counterfactual locked gates."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from scripts.experiments.re_roi_counterfactual.action_family import get_action_family
from scripts.experiments.re_roi_counterfactual.gates import (
    check_calibration,
    check_identity,
    check_re_roi_gain,
    check_static_baseline,
    check_support,
    run_all_gates,
)
from scripts.experiments.re_roi_counterfactual.ranker import ReROIRankerDataset, evaluate_ranker, make_model_for_arm, train_ranker


def _synthetic_split(
    *,
    num_images: int,
    num_detections: int = 4,
    feature_dim: int = 4,
    top_k: int = 2,
    seed: int,
) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    family_specs = get_action_family()
    records: list[dict[str, Any]] = []
    for image_id in range(num_images):
        h_pre_base = torch.randn(num_detections, feature_dim)
        boxes = torch.rand(num_detections, 4) * 200.0
        boxes[:, 2:] += boxes[:, :2] + 5.0
        scores = torch.rand(num_detections)
        labels = torch.randint(1, 5, (num_detections,))
        actions: list[dict[str, Any]] = []
        for candidate_index in range(min(top_k, num_detections)):
            for family_index, spec in enumerate(family_specs):
                h_post = h_pre_base[candidate_index].clone()
                if spec.family == "identity_permutation":
                    q_teacher = 0.0
                else:
                    # Residual will be proportional to candidate_index after prior removal.
                    h_post[1] = float(candidate_index)
                    q_teacher = float(family_index * 10 + candidate_index)
                actions.append(
                    {
                        "candidate_index": candidate_index,
                        "family": spec.family,
                        "h_pre": h_pre_base[candidate_index].clone(),
                        "h_post": h_post,
                        "teacher": {
                            "native_changed": spec.family != "identity_permutation",
                            "delta_tp75": 0,
                            "delta_fp75": 0,
                            "delta_fp50": 0,
                            "delta_duplicate": 0,
                            "delta_score_margin": 0.0,
                            "delta_localization_quality": 0.0,
                            "action_energy": spec.energy,
                        },
                        "q_teacher": q_teacher,
                    }
                )
        records.append(
            {
                "image_id": image_id,
                "image_size": (320, 320),
                "baseline": {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                    "proposal_boxes": boxes.clone(),
                    "roi_features": h_pre_base,
                },
                "fpn_keys": ["0"],
                "actions": actions,
            }
        )
    return records


def test_support_gate_passes_with_enough_records() -> None:
    tune = _synthetic_split(num_images=80, seed=1)
    cal = _synthetic_split(num_images=50, seed=2)
    result = check_support(tune, cal)
    assert result.passed
    assert result.name == "support"


def test_identity_gate_passes_for_zero_identity() -> None:
    tune = _synthetic_split(num_images=10, seed=3)
    dataset = ReROIRankerDataset(tune, arm="C")
    model = make_model_for_arm("C", feature_dim=4, max_label=dataset.max_label, identity_idx=dataset.identity_idx)
    result = check_identity(dataset, model)
    assert result.passed


def test_calibration_positive_rate_uses_raw_noop_relative_utility() -> None:
    records = _synthetic_split(num_images=4, top_k=1, seed=13)
    for record in records:
        for row in record["actions"]:
            row["q_teacher"] = (
                0.0
                if row["family"] == "identity_permutation"
                else -0.1
                if row["family"] == "score_up"
                else -10.0
            )
    dataset = ReROIRankerDataset(records, arm="C")
    preferred_family = dataset.family_to_index["score_up"]

    class PreferNegativeRawAction(torch.nn.Module):
        def forward(self, batch):
            return (batch["family_idx"] == preferred_family).float() * 10.0

    result = check_calibration(
        PreferNegativeRawAction(),
        dataset,
        min_positive_rate=0.5,
        min_coverage=0.0,
    )

    assert result.passed is False
    assert result.detail["positive_rate"] == 0.0
    assert result.value <= 1e-7


def test_re_roi_gain_gate_detects_post_action_signal() -> None:
    fit = _synthetic_split(num_images=30, seed=4)
    tune = _synthetic_split(num_images=20, seed=5)
    dataset_fit_b = ReROIRankerDataset(fit, arm="B")
    prior = dataset_fit_b.prior
    q_stats = dataset_fit_b.q_stats
    max_label = dataset_fit_b.max_label
    identity_idx = dataset_fit_b.identity_idx

    fit_c = ReROIRankerDataset(fit, arm="C", prior=prior, q_stats=q_stats, max_label=max_label)
    tune_b = ReROIRankerDataset(tune, arm="B", prior=prior, q_stats=q_stats, max_label=max_label)
    tune_c = ReROIRankerDataset(tune, arm="C", prior=prior, q_stats=q_stats, max_label=max_label)

    model_b = make_model_for_arm("B", feature_dim=4, max_label=max_label, identity_idx=identity_idx)
    model_c = make_model_for_arm("C", feature_dim=4, max_label=max_label, identity_idx=identity_idx)
    train_ranker(model_b, fit_c, epochs=15, lr=1e-2, lambda_rank=0.1, batch_size=16, seed=11)
    train_ranker(model_c, fit_c, epochs=15, lr=1e-2, lambda_rank=0.1, batch_size=16, seed=12)

    result = check_re_roi_gain(model_b, model_c, tune_c)
    assert result.passed, result.detail


def test_run_all_gates_runs_and_reports_keys() -> None:
    fit = _synthetic_split(num_images=80, seed=6)
    tune = _synthetic_split(num_images=60, seed=7)
    cal = _synthetic_split(num_images=40, seed=8)
    results = run_all_gates(
        fit,
        tune,
        cal,
        feature_dim=4,
        epochs=8,
        lr=1e-2,
        lambda_rank=0.1,
        batch_size=16,
        seed=9,
    )
    assert set(results.keys()) == {
        "support",
        "identity",
        "re_roi_gain",
        "bundle_integrity",
        "static_baseline",
        "calibration",
        "generalization",
    }
    # Support and identity are structural and must pass on clean synthetic data.
    assert results["support"].passed
    assert results["identity"].passed
