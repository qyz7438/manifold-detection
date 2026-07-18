"""Locked gates for the re-ROI counterfactual evidence protocol.

Each gate returns a structured result.  The top-level ``run_all_gates`` helper
trains the controlled B/C/D arms on the fit split and evaluates them on tune
and calibration splits.  No gate thresholds may be relaxed after results are
read.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any

import torch

from scripts.experiments.re_roi_counterfactual.ranker import (
    ReROIRankerDataset,
    ZeroResidualModel,
    action_selection_controls,
    evaluate_reconstructed_ranker,
    evaluate_ranker,
    evaluate_utility_shuffle_metrics,
    make_model_for_arm,
    paired_bootstrap_interval,
    paired_bootstrap_lcb,
    train_ranker,
)
from scripts.experiments.re_roi_counterfactual.action_family import get_action_family


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    value: float
    threshold: str
    detail: dict[str, Any]


@dataclass
class ProtocolEvaluation:
    gates: dict[str, GateResult]
    metrics: dict[str, dict[str, dict[str, dict[str, Any]]]]
    controls: dict[str, Any]
    models: dict[str, torch.nn.Module]
    protocol_state: dict[str, Any]


def _identity_family() -> str:
    return "identity_permutation"


def check_support(
    tune_records: list[dict[str, Any]],
    calibration_records: list[dict[str, Any]],
    *,
    fit_records: list[dict[str, Any]] | None = None,
    min_tune_images: int = 60,
    min_calibration_images: int = 40,
    min_family_tune_images: int = 24,
    min_non_identity_tune_rows: int = 500,
) -> GateResult:
    """Check that fit/tune/calibration splits have enough support."""
    identity = _identity_family()
    expected_families = {
        spec.family for spec in get_action_family() if spec.family != identity
    }
    tune_images = {record["image_id"] for record in tune_records}
    calibration_images = {record["image_id"] for record in calibration_records}

    non_identity_rows = 0
    family_images: dict[str, set[int]] = {}
    for record in tune_records:
        for row in record["actions"]:
            if row["family"] == identity:
                continue
            non_identity_rows += 1
            family_images.setdefault(row["family"], set()).add(record["image_id"])

    passed = True
    detail: dict[str, Any] = {
        "tune_images": len(tune_images),
        "calibration_images": len(calibration_images),
        "non_identity_tune_rows": non_identity_rows,
        "family_tune_images": {f: len(imgs) for f, imgs in family_images.items()},
    }
    if fit_records is not None:
        fit_families = {
            row["family"]
            for record in fit_records
            for row in record["actions"]
            if row["family"] != identity
        }
        detail["fit_families"] = sorted(fit_families)
        if fit_families != expected_families:
            passed = False

    if len(tune_images) < min_tune_images:
        passed = False
    if len(calibration_images) < min_calibration_images:
        passed = False
    if non_identity_rows < min_non_identity_tune_rows:
        passed = False
    if set(family_images) != expected_families:
        passed = False
    if any(len(imgs) < min_family_tune_images for imgs in family_images.values()):
        passed = False

    value = float(
        min(
            len(tune_images) / max(min_tune_images, 1),
            len(calibration_images) / max(min_calibration_images, 1),
            non_identity_rows / max(min_non_identity_tune_rows, 1),
            min((len(imgs) / max(min_family_tune_images, 1) for imgs in family_images.values()), default=1.0),
        )
    )

    return GateResult(
        name="support",
        passed=passed,
        value=value,
        threshold=f"tune>={min_tune_images}, cal>={min_calibration_images}, rows>={min_non_identity_tune_rows}, families>=9, family_images>={min_family_tune_images}",
        detail=detail,
    )


def check_identity(
    tune_dataset: ReROIRankerDataset,
    model: torch.nn.Module,
    *,
    batch_size: int = 32,
    tol: float = 1e-7,
    device: torch.device | None = None,
) -> GateResult:
    """Identity rows must have zero teacher residual and zero model prediction."""
    if device is None:
        device = torch.device("cpu")

    identity_idx = tune_dataset.identity_idx
    teacher_residuals = [
        row.residual for row in tune_dataset.rows if row.family_idx == identity_idx
    ]
    teacher_max = max((abs(r) for r in teacher_residuals), default=0.0)

    model.to(device)
    model.eval()
    from scripts.experiments.re_roi_counterfactual.ranker import _gather_predictions

    out = _gather_predictions(model, tune_dataset, batch_size, device)
    identity_mask = out["family_idx"] == identity_idx
    pred_identity = out["pred"][identity_mask]
    model_max = float(torch.abs(pred_identity).max().item()) if pred_identity.numel() else 0.0

    passed = teacher_max <= tol and model_max <= tol
    return GateResult(
        name="identity",
        passed=passed,
        value=max(teacher_max, model_max),
        threshold=f"<= {tol}",
        detail={"teacher_identity_max_abs": teacher_max, "model_identity_max_abs": model_max},
    )


def check_re_roi_gain(
    model_b: torch.nn.Module,
    model_c: torch.nn.Module,
    tune_dataset: ReROIRankerDataset,
    *,
    min_lift: float = 0.03,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> GateResult:
    """Arm C must beat Arm B on image-equal residual pairwise accuracy."""
    metrics_b = evaluate_ranker(model_b, tune_dataset, batch_size=batch_size, device=device)
    metrics_c = evaluate_ranker(model_c, tune_dataset, batch_size=batch_size, device=device)
    lcb = paired_bootstrap_lcb(metrics_c.per_image_pairwise, metrics_b.per_image_pairwise)
    diff = metrics_c.pairwise_accuracy - metrics_b.pairwise_accuracy
    passed = diff >= min_lift and lcb > 0.0
    return GateResult(
        name="re_roi_gain",
        passed=passed,
        value=diff,
        threshold=f"diff >= {min_lift} and LCB > 0",
        detail={"accuracy_c": metrics_c.pairwise_accuracy, "accuracy_b": metrics_b.pairwise_accuracy, "lcb": lcb},
    )


def check_bundle_integrity(
    model_c: torch.nn.Module,
    model_d: torch.nn.Module,
    tune_dataset_c: ReROIRankerDataset,
    tune_dataset_d: ReROIRankerDataset,
    *,
    min_lift: float = 0.03,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> GateResult:
    """Arm C must beat the bundle-shuffled Arm D."""
    metrics_c = evaluate_ranker(model_c, tune_dataset_c, batch_size=batch_size, device=device)
    metrics_d = evaluate_ranker(model_d, tune_dataset_d, batch_size=batch_size, device=device)
    lcb = paired_bootstrap_lcb(metrics_c.per_image_pairwise, metrics_d.per_image_pairwise)
    diff = metrics_c.pairwise_accuracy - metrics_d.pairwise_accuracy
    passed = diff >= min_lift and lcb > 0.0
    return GateResult(
        name="bundle_integrity",
        passed=passed,
        value=diff,
        threshold=f"diff >= {min_lift} and LCB > 0",
        detail={"accuracy_c": metrics_c.pairwise_accuracy, "accuracy_d": metrics_d.pairwise_accuracy, "lcb": lcb},
    )


def check_static_baseline(
    model_c: torch.nn.Module,
    tune_dataset: ReROIRankerDataset,
    *,
    min_pairwise_lift: float = 0.03,
    min_relative_mae_gain: float = 0.10,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> GateResult:
    """Arm C must beat the family-prior (zero-residual) baseline."""
    metrics_c = evaluate_ranker(model_c, tune_dataset, batch_size=batch_size, device=device)
    metrics_zero = evaluate_ranker(ZeroResidualModel(), tune_dataset, batch_size=batch_size, device=device)
    pairwise_diff = metrics_c.pairwise_accuracy - metrics_zero.pairwise_accuracy
    mae_gain = metrics_c.relative_mae_gain - metrics_zero.relative_mae_gain
    passed = pairwise_diff >= min_pairwise_lift and mae_gain >= min_relative_mae_gain
    return GateResult(
        name="static_baseline",
        passed=passed,
        value=pairwise_diff,
        threshold=f"pairwise_diff >= {min_pairwise_lift}, relative_mae_gain >= {min_relative_mae_gain}",
        detail={
            "accuracy_c": metrics_c.pairwise_accuracy,
            "accuracy_zero": metrics_zero.pairwise_accuracy,
            "relative_mae_gain_c": metrics_c.relative_mae_gain,
            "relative_mae_gain_zero": metrics_zero.relative_mae_gain,
        },
    )


def check_calibration(
    model: torch.nn.Module,
    calibration_dataset: ReROIRankerDataset,
    *,
    alpha: float = 0.10,
    min_positive_rate: float = 0.90,
    min_coverage: float = 0.90,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> GateResult:
    """Grouped split-conformal LCB top-1 action has positive oracle utility."""
    from scripts.experiments.re_roi_counterfactual.ranker import _gather_predictions

    if device is None:
        device = torch.device("cpu")
    model.to(device)
    model.eval()
    out = _gather_predictions(model, calibration_dataset, batch_size, device)

    identity_idx = calibration_dataset.identity_idx
    non_identity = out["family_idx"] != identity_idx
    pred = out["pred"][non_identity]
    target = out["target"][non_identity]
    family_idx = out["family_idx"][non_identity]
    prior_by_index = torch.zeros(len(calibration_dataset.family_to_index), dtype=torch.float32)
    for family, index in calibration_dataset.family_to_index.items():
        prior_by_index[index] = float(calibration_dataset.prior.get(family, 0.0))
    prior = prior_by_index[family_idx]
    total_pred = pred + prior
    total_target = target + prior
    q_raw = torch.tensor(
        [row.q_teacher_raw for row in calibration_dataset.rows if row.family_idx != identity_idx],
        dtype=torch.float32,
    )
    image_id = out["image_id"][non_identity]

    # Grouped one-sided scores protect the selected action in each image.
    unique_images = torch.unique(image_id)
    group_scores: list[float] = []
    for img in unique_images:
        mask = image_id == img
        group_scores.append(float((total_pred[mask] - total_target[mask]).max().item()))
    if not group_scores:
        return GateResult("calibration", False, 0.0, f"coverage>={min_coverage}, positive>={min_positive_rate}", {})
    scores_tensor = torch.tensor(group_scores, dtype=torch.float32)
    correction = _conformal_upper_quantile(scores_tensor, alpha=alpha)

    positive_count = 0
    coverage_count = 0
    total = 0
    for img in unique_images:
        mask = image_id == img
        lcbs = total_pred[mask] - correction
        best_local = int(torch.argmax(lcbs).item())
        selected_lcb = float(lcbs[best_local].item())
        selected_q = float(q_raw[mask][best_local].item())
        selected_target = float(total_target[mask][best_local].item())
        total += 1
        if selected_q > 0.0:
            positive_count += 1
        if selected_target >= selected_lcb - 1e-8:
            coverage_count += 1

    positive_rate = positive_count / total if total else 0.0
    coverage = coverage_count / total if total else 0.0
    passed = positive_rate >= min_positive_rate and coverage >= min_coverage
    return GateResult(
        name="calibration",
        passed=passed,
        value=positive_rate,
        threshold=f"positive_rate>={min_positive_rate}, coverage>={min_coverage}",
        detail={"positive_rate": positive_rate, "coverage": coverage, "num_images": total, "correction": correction},
    )


def _conformal_upper_quantile(scores: torch.Tensor, *, alpha: float) -> float:
    """Finite-sample one-sided conformal order statistic."""
    if scores.numel() == 0:
        raise ValueError("conformal scores must be non-empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    ordered = torch.sort(scores.float().flatten()).values
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * (1.0 - alpha)))
    return float(ordered[rank - 1].item())


def check_generalization(
    model_c: torch.nn.Module,
    fit_dataset: ReROIRankerDataset,
    tune_dataset: ReROIRankerDataset,
    *,
    max_pairwise_gap: float = 0.10,
    max_mae_gain_gap: float = 0.15,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> GateResult:
    """Fit-to-tune gap on residual pairwise accuracy and relative MAE gain."""
    metrics_fit = evaluate_ranker(model_c, fit_dataset, batch_size=batch_size, device=device)
    metrics_tune = evaluate_ranker(model_c, tune_dataset, batch_size=batch_size, device=device)
    pairwise_gap = metrics_fit.pairwise_accuracy - metrics_tune.pairwise_accuracy
    mae_gain_gap = metrics_fit.relative_mae_gain - metrics_tune.relative_mae_gain
    passed = pairwise_gap <= max_pairwise_gap and mae_gain_gap <= max_mae_gain_gap
    return GateResult(
        name="generalization",
        passed=passed,
        value=pairwise_gap,
        threshold=f"pairwise_gap <= {max_pairwise_gap}, mae_gain_gap <= {max_mae_gain_gap}",
        detail={
            "pairwise_fit": metrics_fit.pairwise_accuracy,
            "pairwise_tune": metrics_tune.pairwise_accuracy,
            "mae_gain_fit": metrics_fit.relative_mae_gain,
            "mae_gain_tune": metrics_tune.relative_mae_gain,
        },
    )


def run_protocol_evaluation(
    fit_records: list[dict[str, Any]],
    tune_records: list[dict[str, Any]],
    calibration_records: list[dict[str, Any]],
    *,
    feature_dim: int | None = None,
    hidden_dim: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    lambda_rank: float = 0.1,
    batch_size: int = 32,
    device: torch.device | None = None,
    seed: int = 42,
) -> ProtocolEvaluation:
    """Train controlled arms and return gates, metrics, controls, and models."""
    if device is None:
        device = torch.device("cpu")

    support = check_support(
        tune_records, calibration_records, fit_records=fit_records
    )
    if not support.passed:
        return ProtocolEvaluation(
            gates={"support": support},
            metrics={},
            controls={},
            models={},
            protocol_state={},
        )

    if feature_dim is None and fit_records:
        feature_dim = int(fit_records[0]["baseline"]["roi_features"].shape[-1])
    if feature_dim is None:
        raise ValueError("feature_dim must be provided when fit_records is empty")

    # Fit prior and Q stats on fit split; share with tune/calibration.
    fit_dataset_b = ReROIRankerDataset(fit_records, arm="B")
    max_label = fit_dataset_b.max_label
    num_families = len(fit_dataset_b.family_to_index)
    identity_idx = fit_dataset_b.identity_idx
    prior = fit_dataset_b.prior
    q_stats = fit_dataset_b.q_stats

    tune_dataset_b = ReROIRankerDataset(tune_records, arm="B", prior=prior, q_stats=q_stats, max_label=max_label)
    tune_dataset_c = ReROIRankerDataset(tune_records, arm="C", prior=prior, q_stats=q_stats, max_label=max_label)
    tune_dataset_d = ReROIRankerDataset(tune_records, arm="D", prior=prior, q_stats=q_stats, max_label=max_label)
    fit_dataset_c = ReROIRankerDataset(fit_records, arm="C", prior=prior, q_stats=q_stats, max_label=max_label)
    fit_dataset_d = ReROIRankerDataset(fit_records, arm="D", prior=prior, q_stats=q_stats, max_label=max_label)
    calibration_dataset_c = ReROIRankerDataset(
        calibration_records, arm="C", prior=prior, q_stats=q_stats, max_label=max_label
    )

    models: dict[str, torch.nn.Module] = {}
    for arm in ("B", "C", "D"):
        torch.manual_seed(seed)
        models[arm] = make_model_for_arm(
            arm,
            feature_dim,
            max_label,
            num_families,
            identity_idx,
            hidden_dim=hidden_dim,
        )

    train_kwargs = {
        "epochs": epochs,
        "lr": lr,
        "lambda_rank": lambda_rank,
        "batch_size": batch_size,
        "device": device,
        "seed": seed,
    }
    models["B"] = train_ranker(models["B"], fit_dataset_b, **train_kwargs)
    models["C"] = train_ranker(models["C"], fit_dataset_c, **train_kwargs)
    models["D"] = train_ranker(models["D"], fit_dataset_d, **train_kwargs)

    gates = {
        "support": support,
        "identity": check_identity(
            tune_dataset_c, models["C"], batch_size=batch_size, device=device
        ),
        "re_roi_gain": check_re_roi_gain(
            models["B"], models["C"], tune_dataset_c, batch_size=batch_size, device=device
        ),
        "bundle_integrity": check_bundle_integrity(
            models["C"],
            models["D"],
            tune_dataset_c,
            tune_dataset_d,
            batch_size=batch_size,
            device=device,
        ),
        "static_baseline": check_static_baseline(
            models["C"], tune_dataset_c, batch_size=batch_size, device=device
        ),
        "calibration": check_calibration(
            models["C"], calibration_dataset_c, batch_size=batch_size, device=device
        ),
        "generalization": check_generalization(
            models["C"], fit_dataset_c, tune_dataset_c, batch_size=batch_size, device=device
        ),
    }

    split_datasets = {
        "fit": {"A": fit_dataset_c, "B": fit_dataset_b, "C": fit_dataset_c, "D": fit_dataset_d},
        "tune": {"A": tune_dataset_c, "B": tune_dataset_b, "C": tune_dataset_c, "D": tune_dataset_d},
    }
    metrics: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    metric_objects: dict[str, dict[str, dict[str, Any]]] = {}
    for split_name, datasets in split_datasets.items():
        metrics[split_name] = {}
        metric_objects[split_name] = {}
        for arm, dataset in datasets.items():
            model = ZeroResidualModel() if arm == "A" else models[arm]
            residual_metrics = evaluate_ranker(
                model, dataset, batch_size=batch_size, device=device
            )
            total_metrics = evaluate_reconstructed_ranker(
                model, dataset, batch_size=batch_size, device=device
            )
            metric_objects[split_name][arm] = {
                "residual": residual_metrics,
                "reconstructed_total": total_metrics,
            }
            metrics[split_name][arm] = {
                "residual": residual_metrics.to_dict(),
                "reconstructed_total": total_metrics.to_dict(),
            }

    shuffled_metrics = evaluate_utility_shuffle_metrics(tune_dataset_c, seed=seed)
    c_pairwise = metric_objects["tune"]["C"]["residual"].per_image_pairwise
    comparisons = {
        "C_vs_A_family_prior": paired_bootstrap_interval(
            c_pairwise,
            metric_objects["tune"]["A"]["residual"].per_image_pairwise,
        ),
        "C_vs_B_static_roi": paired_bootstrap_interval(
            c_pairwise,
            metric_objects["tune"]["B"]["residual"].per_image_pairwise,
        ),
        "C_vs_D_bundle_shuffle": paired_bootstrap_interval(
            c_pairwise,
            metric_objects["tune"]["D"]["residual"].per_image_pairwise,
        ),
        "C_vs_utility_shuffle": paired_bootstrap_interval(
            c_pairwise, shuffled_metrics["residual"].per_image_pairwise
        ),
    }
    controls = {
        "utility_shuffle": {
            name: value.to_dict() for name, value in shuffled_metrics.items()
        },
        "paired_residual_pairwise_intervals": comparisons,
        **action_selection_controls(tune_dataset_c, seed=seed),
    }
    protocol_state = {
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "max_label": max_label,
        "num_families": num_families,
        "identity_idx": identity_idx,
        "family_to_index": fit_dataset_b.family_to_index,
        "family_prior": prior,
        "q_teacher_stats": {
            key: float(value.item()) for key, value in q_stats.items()
        },
    }
    return ProtocolEvaluation(
        gates=gates,
        metrics=metrics,
        controls=controls,
        models=models,
        protocol_state=protocol_state,
    )


def run_all_gates(
    fit_records: list[dict[str, Any]],
    tune_records: list[dict[str, Any]],
    calibration_records: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, GateResult]:
    """Compatibility wrapper returning only the locked gate results."""
    return run_protocol_evaluation(
        fit_records, tune_records, calibration_records, **kwargs
    ).gates
