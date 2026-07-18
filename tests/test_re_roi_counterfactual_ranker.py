"""Tests for the re-ROI counterfactual ranker."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from scripts.experiments.re_roi_counterfactual.action_family import get_action_family
from scripts.experiments.re_roi_counterfactual.ranker import (
    _fit_q_teacher_stats_from_records,
    DeepSetResidualRanker,
    ReROIRankerDataset,
    ZeroResidualModel,
    action_selection_controls,
    build_grouped_batches,
    evaluate_reconstructed_ranker,
    evaluate_ranker,
    evaluate_utility_shuffle,
    fit_family_prior,
    make_model_for_arm,
    paired_bootstrap_interval,
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


def test_dataset_collate_handles_mixed_detection_counts_and_masks_padding() -> None:
    records = _synthetic_records(num_images=2, num_detections=4, top_k=1)
    for key in ("boxes", "scores", "labels", "proposal_boxes", "roi_features"):
        records[1]["baseline"][key] = records[1]["baseline"][key][:2]
    dataset = ReROIRankerDataset(records, arm="C")
    first = next(index for index, row in enumerate(dataset.rows) if row.image_id == 0 and row.family_idx != dataset.identity_idx)
    second = next(index for index, row in enumerate(dataset.rows) if row.image_id == 1 and row.family_idx != dataset.identity_idx)
    batch = ReROIRankerDataset.collate_fn([dataset[first], dataset[second]])
    model = make_model_for_arm(
        "C", feature_dim=4, max_label=dataset.max_label, identity_idx=dataset.identity_idx
    ).eval()

    assert batch["baseline_scores"].shape == (2, 4)
    assert batch["baseline_labels"].shape == (2, 4)
    assert batch["acted_mask"].shape == (2, 4)
    assert batch["mask"][1].tolist() == [True, True, False, False]
    with torch.no_grad():
        reference = model(batch)
    changed = dict(batch)
    changed["baseline_roi_features"] = batch["baseline_roi_features"].clone()
    changed["baseline_roi_features"][1, 2:] = 10_000.0
    with torch.no_grad():
        padded_changed = model(changed)
    assert torch.allclose(reference, padded_changed, atol=1e-6)


def test_arm_d_shuffles_post_features() -> None:
    records = _synthetic_records(num_images=1)
    for record in records:
        for family_index, row in enumerate(record["actions"]):
            row["h_post"] = row["h_post"].clone()
            row["h_post"][0] = float(family_index)
    prior = fit_family_prior(records)
    q_stats = fit_q_teacher_stats(torch.tensor([r["q_teacher"] for rec in records for r in rec["actions"]]))
    dataset_c = ReROIRankerDataset(records, arm="C", prior=prior, q_stats=q_stats)
    dataset_d = ReROIRankerDataset(records, arm="D", prior=prior, q_stats=q_stats)
    # Same row order, and every D row receives another action's h_post.
    for row_c, row_d in zip(dataset_c.rows, dataset_d.rows):
        assert not torch.allclose(row_c.h_post, row_d.h_post)


def test_internal_q_stats_exclude_identity_rows() -> None:
    records = _synthetic_records(num_images=2, top_k=1)
    non_identity_values = []
    for record in records:
        for row in record["actions"]:
            if row["family"] == "identity_permutation":
                row["q_teacher"] = -10_000.0
            else:
                non_identity_values.append(row["q_teacher"])

    actual = _fit_q_teacher_stats_from_records(records)
    expected = fit_q_teacher_stats(torch.tensor(non_identity_values))

    assert torch.equal(actual["median"], expected["median"])
    assert torch.equal(actual["iqr"], expected["iqr"])


def test_ranker_loss_excludes_identity_regression_rows() -> None:
    pred = torch.tensor([1000.0, 0.0])
    target = torch.tensor([0.0, 1.0])
    family_idx = torch.tensor([0, 1])
    image_id = torch.tensor([1, 1])

    loss = ranker_loss(
        pred,
        target,
        family_idx,
        image_id,
        identity_idx=0,
        lambda_rank=0.0,
    )
    expected = torch.nn.functional.smooth_l1_loss(
        pred[1:], target[1:], beta=0.05
    )

    assert loss == pytest.approx(expected)


def test_grouped_batches_keep_image_family_rows_together_and_exclude_identity() -> None:
    records = _synthetic_records(num_images=3, top_k=2)
    dataset = ReROIRankerDataset(records, arm="C")

    batches = build_grouped_batches(dataset, batch_size=8, seed=42, epoch=1)
    flattened = [index for batch in batches for index in batch]
    expected = [
        index
        for index, row in enumerate(dataset.rows)
        if row.family_idx != dataset.identity_idx
    ]

    assert sorted(flattened) == expected
    for image_id in range(3):
        for family_idx in range(1, len(dataset.family_to_index)):
            group = {
                index
                for index, row in enumerate(dataset.rows)
                if row.image_id == image_id and row.family_idx == family_idx
            }
            assert any(group <= set(batch) for batch in batches)


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


def test_sparse_star_pair_term_is_permutation_invariant_and_edge_sensitive() -> None:
    records = _synthetic_records(num_images=1, num_detections=4, top_k=1)
    dataset = ReROIRankerDataset(records, arm="C")
    batch = ReROIRankerDataset.collate_fn([dataset[1]])
    model = DeepSetResidualRanker(
        feature_dim=4, max_label=dataset.max_label, identity_idx=dataset.identity_idx
    ).eval()

    with torch.no_grad():
        reference = model(batch)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = dict(batch)
    for key in (
        "baseline_roi_features",
        "baseline_boxes",
        "baseline_scores",
        "baseline_labels",
        "mask",
        "acted_mask",
    ):
        permuted[key] = batch[key][:, permutation]
    with torch.no_grad():
        reordered = model(permuted)
    assert torch.allclose(reference, reordered, atol=1e-6)

    changed = dict(batch)
    changed["baseline_boxes"] = batch["baseline_boxes"].clone()
    changed["baseline_boxes"][:, 1] += torch.tensor([20.0, 0.0, 20.0, 0.0])
    with torch.no_grad():
        edge_changed = model(changed)
    assert not torch.allclose(reference, edge_changed)
    assert hasattr(model, "v_net")


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
    assert metrics.num_rows == sum(
        row.family_idx != dataset.identity_idx for row in dataset.rows
    )


def test_reconstructed_and_fixed_control_metrics_are_reported() -> None:
    records = _synthetic_records(num_images=10)
    dataset = ReROIRankerDataset(records, arm="C")
    model = ZeroResidualModel()

    reconstructed = evaluate_reconstructed_ranker(model, dataset, batch_size=8)
    shuffled = evaluate_utility_shuffle(dataset, seed=42)
    controls = action_selection_controls(dataset, seed=42)

    assert reconstructed.num_rows == 180
    assert set(shuffled) == {"residual", "reconstructed_total"}
    assert set(controls) == {"oracle_top1", "matched_rate_random", "always_noop"}
    assert controls["always_noop"]["mean_raw_utility"] == 0.0
    assert controls["matched_rate_random"]["action_image_rate"] == pytest.approx(
        controls["oracle_top1"]["action_image_rate"]
    )


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


def test_paired_bootstrap_interval_reports_common_support() -> None:
    left = {i: {1: 0.8, 2: 0.6} for i in range(20)}
    right = {i: {1: 0.5, 2: 0.4} for i in range(10, 30)}

    result = paired_bootstrap_interval(left, right, n_bootstrap=1000, seed=42)

    assert result["common_image_count"] == 10
    assert result["point_delta"] == pytest.approx(0.25)
    assert result["lower"] <= result["point_delta"] <= result["upper"]
