"""Tests for the re-ROI counterfactual ranker."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from scripts.experiments.re_roi_counterfactual.action_family import get_action_family
from scripts.experiments.re_roi_counterfactual.ranker import (
    DeepSetResidualRanker,
    ReROIRankerDataset,
    ZeroResidualModel,
    evaluate_ranker,
    fit_family_prior,
    make_model_for_arm,
    paired_bootstrap_lcb,
    ranker_loss,
    train_ranker,
)
from scripts.experiments.re_roi_counterfactual.teacher import fit_q_teacher_stats, standardize_q_teacher


def _synthetic_records(
    *,
    num_images: int = 40,
    num_detections: int = 4,
    feature_dim: int = 4,
    top_k: int = 2,
    seed: int = 7,
) -> list[dict[str, Any]]:
    """Generate synthetic cache records with a candidate-index signal in h_post."""
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
                if spec.family == "identity_permutation":
                    q_teacher = 0.0
                    h_post = h_pre_base[candidate_index].clone()
                else:
                    # Signal: residual will be driven by candidate_index via h_post[1].
                    h_post = h_pre_base[candidate_index].clone()
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


def test_dataset_collate_shapes() -> None:
    records = _synthetic_records(num_images=3)
    prior = fit_family_prior(records)
    q_stats = fit_q_teacher_stats(torch.tensor([r["q_teacher"] for rec in records for r in rec["actions"]]))
    dataset = ReROIRankerDataset(records, arm="C", prior=prior, q_stats=q_stats)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=ReROIRankerDataset.collate_fn
    )
    batch = next(iter(loader))
    assert batch["h_pre"].shape == (4, 4)
    assert batch["h_post"].shape == (4, 4)
    assert batch["baseline_roi_features"].shape[0] == 4
    assert batch["mask"].shape == batch["baseline_roi_features"].shape[:2]
    assert batch["residual"].shape == (4,)


def test_arm_d_shuffles_post_features() -> None:
    records = _synthetic_records(num_images=1)
    prior = fit_family_prior(records)
    q_stats = fit_q_teacher_stats(torch.tensor([r["q_teacher"] for rec in records for r in rec["actions"]]))
    dataset_c = ReROIRankerDataset(records, arm="C", prior=prior, q_stats=q_stats)
    dataset_d = ReROIRankerDataset(records, arm="D", prior=prior, q_stats=q_stats)
    # Same row order, but D h_post differs from C for non-identity rows.
    diff = 0
    for row_c, row_d in zip(dataset_c.rows, dataset_d.rows):
        if row_c.family == "identity_permutation":
            continue
        if not torch.allclose(row_c.h_post, row_d.h_post):
            diff += 1
    assert diff > 0


def test_model_forward_and_identity_zero() -> None:
    records = _synthetic_records(num_images=2)
    prior = fit_family_prior(records)
    q_stats = fit_q_teacher_stats(torch.tensor([r["q_teacher"] for rec in records for r in rec["actions"]]))
    dataset = ReROIRankerDataset(records, arm="C", prior=prior, q_stats=q_stats)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, collate_fn=ReROIRankerDataset.collate_fn
    )
    batch = next(iter(loader))
    model = DeepSetResidualRanker(feature_dim=4, max_label=5, identity_idx=dataset.identity_idx)
    out = model(batch)
    assert out.shape == (8,)
    identity_mask = batch["family_idx"] == dataset.identity_idx
    assert torch.allclose(out[identity_mask], torch.zeros_like(out[identity_mask]))


def test_training_reduces_loss() -> None:
    records = _synthetic_records(num_images=10)
    prior = fit_family_prior(records)
    q_stats = fit_q_teacher_stats(torch.tensor([r["q_teacher"] for rec in records for r in rec["actions"]]))
    dataset = ReROIRankerDataset(records, arm="C", prior=prior, q_stats=q_stats)
    model = make_model_for_arm("C", feature_dim=4, max_label=dataset.max_label, identity_idx=dataset.identity_idx)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, collate_fn=ReROIRankerDataset.collate_fn
    )
    batch = next(iter(loader))
    with torch.no_grad():
        initial_pred = model(batch)
    initial_loss = ranker_loss(
        initial_pred,
        batch["residual"],
        batch["family_idx"],
        batch["image_id"],
        dataset.identity_idx,
        lambda_rank=0.1,
    )

    train_ranker(model, dataset, epochs=30, lr=1e-2, lambda_rank=0.1, batch_size=8, seed=42)

    with torch.no_grad():
        final_pred = model(batch)
    final_loss = ranker_loss(
        final_pred,
        batch["residual"],
        batch["family_idx"],
        batch["image_id"],
        dataset.identity_idx,
        lambda_rank=0.1,
    )
    assert final_loss.item() < initial_loss.item()


def test_evaluation_returns_metrics() -> None:
    records = _synthetic_records(num_images=10)
    prior = fit_family_prior(records)
    q_stats = fit_q_teacher_stats(torch.tensor([r["q_teacher"] for rec in records for r in rec["actions"]]))
    dataset = ReROIRankerDataset(records, arm="C", prior=prior, q_stats=q_stats)
    model = make_model_for_arm("C", feature_dim=4, max_label=dataset.max_label, identity_idx=dataset.identity_idx)
    metrics = evaluate_ranker(model, dataset, batch_size=8)
    assert 0.0 <= metrics.mae
    assert metrics.relative_mae_gain <= 1.0
    assert 0.0 <= metrics.pairwise_accuracy <= 1.0
    assert 0.0 <= metrics.sign_auroc <= 1.0
    assert metrics.num_rows == len(dataset)


def test_zero_residual_model_predictions_are_zero() -> None:
    records = _synthetic_records(num_images=5)
    prior = fit_family_prior(records)
    q_stats = fit_q_teacher_stats(torch.tensor([r["q_teacher"] for rec in records for r in rec["actions"]]))
    dataset = ReROIRankerDataset(records, arm="A", prior=prior, q_stats=q_stats)
    metrics = evaluate_ranker(ZeroResidualModel(), dataset, batch_size=8)
    assert metrics.mae == pytest.approx(metrics.relative_mae_gain * metrics.mae + metrics.mae, abs=1e-6)


def test_paired_bootstrap_lcb_on_identical_metrics() -> None:
    per_image = {i: {0: 0.6, 1: 0.7} for i in range(20)}
    lcb = paired_bootstrap_lcb(per_image, per_image, n_bootstrap=1000, seed=1)
    assert -0.05 <= lcb <= 0.05
