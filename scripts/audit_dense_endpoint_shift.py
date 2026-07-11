"""Diagnose detector-train to detector-validation endpoint distribution shift."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.dense_endpoint.shift_audit.001.json"
)
CONFIG_SHA256 = "daeb0ee3229e5554ce034d7293ed3bb4fa5ee4cbfb830e0dcbe2c1a88e9b2d69"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict[str, Any]:
    if sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise ValueError("canonical dense endpoint shift-audit config SHA256 mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("version_id") != "det.energy.dense_endpoint.shift_audit.001":
        raise ValueError("dense endpoint shift-audit version mismatch")
    if config.get("sources", {}).get("new_inference_forbidden") is not True:
        raise ValueError("shift audit must forbid new inference")
    if config.get("sources", {}).get("training_forbidden") is not True:
        raise ValueError("shift audit must forbid training")
    return config


def empirical_ks(left: torch.Tensor, right: torch.Tensor) -> float:
    x = torch.as_tensor(left, dtype=torch.float64).flatten()
    y = torch.as_tensor(right, dtype=torch.float64).flatten()
    if x.numel() == 0 or y.numel() == 0:
        raise ValueError("KS samples must be non-empty")
    points = torch.unique(torch.cat((x, y))).sort().values
    cdf_x = (x[:, None] <= points[None, :]).double().mean(dim=0)
    cdf_y = (y[:, None] <= points[None, :]).double().mean(dim=0)
    return float((cdf_x - cdf_y).abs().max().item())


def _load_cache(path: Path) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "dense_endpoint_absolute_v1" or not isinstance(
        payload.get("records"), list
    ):
        raise ValueError("unsupported dense endpoint cache")
    return payload["records"]


def _scalar_summary(values: torch.Tensor) -> dict[str, float]:
    x = torch.as_tensor(values, dtype=torch.float64).flatten()
    q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75], dtype=x.dtype))
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "q25": float(q[0].item()),
        "median": float(q[1].item()),
        "q75": float(q[2].item()),
    }


def _feature_matrix(records: Sequence[dict[str, Any]]) -> torch.Tensor:
    rows = []
    for record in records:
        node_mean = record["node_features"].float().mean(dim=0)
        pair = record["pair_features"].float()
        pair_mean = pair.mean(dim=0) if pair.numel() else torch.zeros(5)
        rows.append(torch.cat((node_mean, pair_mean)))
    return torch.stack(rows).double()


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    x = torch.as_tensor(left, dtype=torch.float64).flatten()
    y = torch.as_tensor(right, dtype=torch.float64).flatten()
    x = x - x.mean()
    y = y - y.mean()
    denominator = float((x.square().sum().sqrt() * y.square().sum().sqrt()).item())
    return None if denominator == 0 else float((x * y).sum().item() / denominator)


def _quantile_calibration(
    prediction: torch.Tensor, target: torch.Tensor, bins: int
) -> list[dict[str, float | int]]:
    order = torch.argsort(target)
    chunks = torch.tensor_split(order, int(bins))
    output = []
    for index, chunk in enumerate(chunks):
        output.append(
            {
                "bin": index,
                "count": int(chunk.numel()),
                "target_mean": float(target[chunk].mean().item()),
                "prediction_mean": float(prediction[chunk].mean().item()),
                "residual_mean": float((prediction[chunk] - target[chunk]).mean().item()),
            }
        )
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "nwpu_dense_endpoint_shift_audit_s42" / "eval_metrics.json",
    )
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    from spectral_detection_posttrain.methods.energy_transport.dense_endpoint import (
        DenseSetEnergyEndpoint,
        RobustTeacherStats,
    )
    import scripts.train_dense_endpoint_geometry_control as geometry
    import scripts.train_nwpu_dense_endpoint_absolute as absolute

    config = load_config()
    paths = {
        "cleanval": ROOT / config["sources"]["cleanval_result"],
        "fit_tune": ROOT / config["sources"]["fit_tune_cache"],
        "outer_train": ROOT / config["sources"]["outer_train_cache"],
        "validation": ROOT / config["sources"]["validation_cache"],
        "checkpoint": ROOT / config["sources"]["full_checkpoint"],
    }
    hashes = {
        "cleanval": config["sources"]["cleanval_sha256"],
        "fit_tune": config["sources"]["fit_tune_cache_sha256"],
        "outer_train": config["sources"]["outer_train_cache_sha256"],
        "validation": config["sources"]["validation_cache_sha256"],
        "checkpoint": config["sources"]["full_checkpoint_sha256"],
    }
    for name, path in paths.items():
        if sha256_file(path) != hashes[name]:
            raise ValueError(f"shift audit {name} SHA256 mismatch")
    source = json.loads(paths["cleanval"].read_text(encoding="utf-8"))
    if source.get("scientific_status") != config["sources"]["required_status"]:
        raise ValueError("cleanval decision mismatch")
    train_records = [*_load_cache(paths["fit_tune"]), *_load_cache(paths["outer_train"])]
    validation_records = _load_cache(paths["validation"])
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu")
    teacher_stats = RobustTeacherStats(
        median=checkpoint["teacher_median"].float(), iqr=checkpoint["teacher_iqr"].float()
    )
    feature_stats = checkpoint["feature_stats"]
    model = DenseSetEnergyEndpoint(node_dim=22, pair_dim=5, hidden_dim=32)
    model.load_state_dict(checkpoint["model"])
    train_targets = absolute._targets(train_records, teacher_stats)
    validation_targets = absolute._targets(validation_records, teacher_stats)
    train_prediction = geometry._predict(
        model, train_records, "local_full", feature_stats, absolute.load_config(), torch.device("cpu")
    )
    validation_prediction = geometry._predict(
        model,
        validation_records,
        "local_full",
        feature_stats,
        absolute.load_config(),
        torch.device("cpu"),
    )
    fit_mean = float(source["teacher_stats"]["train_target_mean"])
    train_metrics = absolute.regression_metrics(train_prediction, train_targets, fit_mean=fit_mean)
    validation_metrics = absolute.regression_metrics(
        validation_prediction, validation_targets, fit_mean=fit_mean
    )
    residual = validation_prediction - validation_targets
    residual_mean = float(residual.mean().item())
    centered_prediction = validation_prediction - residual_mean
    centered_metrics = absolute.regression_metrics(
        centered_prediction, validation_targets, fit_mean=fit_mean
    )
    centered_mae_gain = (
        float(validation_metrics["mae"]) - float(centered_metrics["mae"])
    ) / float(validation_metrics["mae"])

    train_teacher = torch.stack([record["teacher_values"] for record in train_records]).double()
    validation_teacher = torch.stack(
        [record["teacher_values"] for record in validation_records]
    ).double()
    coordinate_shift = {}
    for index, name in enumerate(config["analysis"]["teacher_coordinates"]):
        train_values = train_teacher[:, index]
        validation_values = validation_teacher[:, index]
        train_summary = _scalar_summary(train_values)
        validation_summary = _scalar_summary(validation_values)
        coordinate_shift[name] = {
            "train": train_summary,
            "validation": validation_summary,
            "mean_shift_in_train_std": (
                validation_summary["mean"] - train_summary["mean"]
            )
            / max(train_summary["std"], 1e-12),
            "empirical_ks": empirical_ks(train_values, validation_values),
        }

    train_features = _feature_matrix(train_records)
    validation_features = _feature_matrix(validation_records)
    mean = train_features.mean(dim=0)
    centered = train_features - mean
    covariance = centered.T @ centered / max(train_features.shape[0] - 1, 1)
    ridge = float(config["analysis"]["covariance_ridge_fraction"]) * covariance.diag().mean()
    precision = torch.linalg.pinv(covariance + ridge * torch.eye(covariance.shape[0], dtype=covariance.dtype))

    def distances(values: torch.Tensor) -> torch.Tensor:
        delta = values - mean
        return torch.einsum("ni,ij,nj->n", delta, precision, delta).clamp_min(0).sqrt()

    train_distance = distances(train_features)
    validation_distance = distances(validation_features)
    threshold = float(
        torch.quantile(
            train_distance,
            torch.tensor(float(config["analysis"]["ood_reference_quantile"]), dtype=torch.float64),
        ).item()
    )
    validation_ood_fraction = float(validation_distance.gt(threshold).double().mean().item())
    feature_shift = []
    for index in range(train_features.shape[1]):
        train_summary = _scalar_summary(train_features[:, index])
        validation_summary = _scalar_summary(validation_features[:, index])
        feature_shift.append(
            {
                "index": index,
                "mean_shift_in_train_std": (
                    validation_summary["mean"] - train_summary["mean"]
                )
                / max(train_summary["std"], 1e-12),
                "empirical_ks": empirical_ks(
                    train_features[:, index], validation_features[:, index]
                ),
            }
        )
    prediction_counts = torch.tensor(
        [math.log1p(int(record["prediction_count"])) for record in validation_records]
    )
    ground_truth_counts = torch.tensor(
        [math.log1p(int(record["ground_truth_count"])) for record in validation_records]
    )
    calibration_shift = (
        float(validation_metrics["pairwise_accuracy"]) >= 0.80
        and abs(residual_mean)
        >= float(config["analysis"]["calibration_shift_min_abs_residual_mean"])
        and centered_mae_gain
        >= float(config["analysis"]["calibration_shift_min_centered_mae_relative_gain"])
    )
    feature_ood = validation_ood_fraction >= float(
        config["analysis"]["feature_ood_min_validation_fraction"]
    )
    if calibration_shift and feature_ood:
        status = "calibration_shift_with_feature_ood"
    elif calibration_shift:
        status = "calibration_shift_without_feature_ood"
    elif feature_ood:
        status = "feature_ood_without_simple_calibration_shift"
    else:
        status = "unresolved_generalization_gap"
    git_dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    if args.require_clean_git and git_dirty:
        raise RuntimeError("--require-clean-git requested but repository is dirty")
    result = {
        "completed": True,
        "scientific_status": status,
        "version_id": config["version_id"],
        "experiment_scope": "post_validation_cache_only_shift_diagnosis",
        "config_sha256": CONFIG_SHA256,
        "source_cleanval_sha256": config["sources"]["cleanval_sha256"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": git_dirty,
        "new_inference_run": False,
        "training_run": False,
        "original_endpoint_status_unchanged": "detector_unseen_absolute_endpoint_frozen",
        "metrics": {
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_oracle_mean_centered": centered_metrics,
        },
        "calibration": {
            "validation_residual_mean": residual_mean,
            "validation_residual_std": float(residual.std(unbiased=False).item()),
            "oracle_centered_mae_relative_gain": centered_mae_gain,
            "residual_vs_log_prediction_count_pearson": _pearson(residual, prediction_counts),
            "residual_vs_log_ground_truth_count_pearson": _pearson(
                residual, ground_truth_counts
            ),
            "target_quantile_curve": _quantile_calibration(
                validation_prediction,
                validation_targets,
                int(config["analysis"]["target_quantile_bins"]),
            ),
        },
        "teacher_coordinate_shift": coordinate_shift,
        "feature_ood": {
            "train_distance_95pct": threshold,
            "validation_fraction_above_train_95pct": validation_ood_fraction,
            "train_distance": _scalar_summary(train_distance),
            "validation_distance": _scalar_summary(validation_distance),
            "per_dimension_shift": feature_shift,
        },
        "diagnosis": {
            "calibration_shift": calibration_shift,
            "feature_ood": feature_ood,
        },
        "decision_rule": config["decision_rule"],
        "claim_boundary": config["claim_boundary"],
        "validated_claim": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    result = run(_parse_args())
    print(
        json.dumps(
            {
                "status": result["scientific_status"],
                "calibration": result["calibration"],
                "diagnosis": result["diagnosis"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
