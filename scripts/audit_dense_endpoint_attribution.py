"""Attribute the frozen dense endpoint using fit/tune artifacts only."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    / "det.energy.dense_endpoint.attribution.001.json"
)
CONFIG_SHA256 = "ed5bec95c5caa700317a104843e62321b8671ccf5ad48fad13c126ca2dc4432f"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict[str, Any]:
    if sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise ValueError("canonical dense endpoint attribution config SHA256 mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("version_id") != "det.energy.dense_endpoint.attribution.001":
        raise ValueError("dense endpoint attribution version mismatch")
    if config.get("source", {}).get("outer_cache_forbidden") is not True:
        raise ValueError("outer cache must remain forbidden")
    if config.get("source", {}).get("new_data_forbidden") is not True:
        raise ValueError("new data must remain forbidden")
    return config


def apply_node_ablation(
    standardized_features: torch.Tensor, ablation: str, config: dict[str, Any]
) -> torch.Tensor:
    if standardized_features.ndim != 2 or standardized_features.shape[1] != 22:
        raise ValueError("standardized node features must have shape (N, 22)")
    if ablation == "none" or ablation == "pair_branch_disabled":
        return standardized_features
    mapping = {
        "score_zero": "score",
        "box_geometry_zero": "box_geometry",
        "conflict_geometry_zero": "conflict_geometry",
        "all_geometry_zero": "all_geometry",
        "class_code_zero": "class_code",
    }
    if ablation not in mapping:
        raise ValueError(f"unknown attribution ablation {ablation!r}")
    output = standardized_features.clone()
    output[:, config["feature_groups"][mapping[ablation]]] = 0.0
    return output


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    x = torch.as_tensor(left, dtype=torch.float64).flatten()
    y = torch.as_tensor(right, dtype=torch.float64).flatten()
    if x.shape != y.shape or x.numel() < 2:
        raise ValueError("Pearson inputs must be aligned")
    x = x - x.mean()
    y = y - y.mean()
    denominator = float((x.square().sum().sqrt() * y.square().sum().sqrt()).item())
    return None if denominator == 0.0 else float((x * y).sum().item() / denominator)


@torch.inference_mode()
def _predict(
    model: torch.nn.Module,
    records: Sequence[dict[str, Any]],
    ablation: str,
    feature_stats: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    quality = []
    unary = []
    pair = []
    for record in records:
        nodes = (record["node_features"].float() - feature_stats["node_mean"]) / feature_stats[
            "node_scale"
        ]
        pairs = (record["pair_features"].float() - feature_stats["pair_mean"]) / feature_stats[
            "pair_scale"
        ]
        nodes = apply_node_ablation(nodes, ablation, config)
        output = model(nodes, pairs)
        current_quality = output.quality
        if ablation == "pair_branch_disabled":
            current_quality = model.identity_bias + output.unary_contribution
        quality.append(current_quality.cpu())
        unary.append(output.unary_contribution.cpu())
        pair.append(output.pair_contribution.cpu())
    return torch.stack(quality), torch.stack(unary), torch.stack(pair)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "nwpu_dense_endpoint_attribution_s42_fit_tune" / "eval_metrics.json",
    )
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    from spectral_detection_posttrain.methods.energy_transport.dense_endpoint import (
        DenseSetEnergyEndpoint,
        RobustTeacherStats,
    )
    import scripts.train_nwpu_dense_endpoint_absolute as absolute

    config = load_config()
    source_result = (ROOT / config["source"]["result"]).resolve()
    fit_tune_cache = (ROOT / config["source"]["fit_tune_cache"]).resolve()
    checkpoint_path = (ROOT / config["source"]["full_checkpoint"]).resolve()
    for name, path, expected in (
        ("result", source_result, config["source"]["result_sha256"]),
        ("fit_tune_cache", fit_tune_cache, config["source"]["fit_tune_cache_sha256"]),
        ("full_checkpoint", checkpoint_path, config["source"]["full_checkpoint_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"attribution source {name} SHA256 mismatch")
    source = json.loads(source_result.read_text(encoding="utf-8"))
    if (
        source.get("completed") is not True
        or source.get("scientific_status") != config["source"]["required_status"]
        or source.get("detector_validation_read") is not False
        or source.get("outer_read_count") != 1
    ):
        raise ValueError("absolute endpoint result contract mismatch")
    cache = torch.load(fit_tune_cache, map_location="cpu")
    if cache.get("format") != "dense_endpoint_absolute_v1":
        raise ValueError("unsupported fit/tune cache format")
    records = cache.get("records")
    fit_count = int(config["source"]["inner_fit_count"])
    tune_count = int(config["source"]["inner_tune_count"])
    if not isinstance(records, list) or len(records) != fit_count + tune_count:
        raise ValueError("fit/tune cache support mismatch")
    fit_records = records[:fit_count]
    tune_records = records[fit_count:]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    feature_stats = checkpoint["feature_stats"]
    teacher_stats = RobustTeacherStats(
        median=checkpoint["teacher_median"].float(), iqr=checkpoint["teacher_iqr"].float()
    )
    model = DenseSetEnergyEndpoint(node_dim=22, pair_dim=5, hidden_dim=32)
    model.load_state_dict(checkpoint["model"])
    fit_targets = absolute._targets(fit_records, teacher_stats)
    tune_targets = absolute._targets(tune_records, teacher_stats)
    fit_mean = float(source["teacher_stats"]["fit_target_mean"])

    predictions = {}
    metrics = {}
    contributions = {}
    for ablation in ["none", *config["ablations"]]:
        prediction, unary, pair = _predict(model, tune_records, ablation, feature_stats, config)
        predictions[ablation] = prediction
        metrics[ablation] = absolute.regression_metrics(prediction, tune_targets, fit_mean=fit_mean)
        contributions[ablation] = {
            "unary_mean": float(unary.mean().item()),
            "unary_std": float(unary.std(unbiased=False).item()),
            "pair_mean": float(pair.mean().item()),
            "pair_std": float(pair.std(unbiased=False).item()),
        }

    fit_node_means = torch.stack(
        [record["node_features"].float().mean(dim=0) for record in fit_records]
    )
    feature_correlations = {
        str(index): _pearson(fit_node_means[:, index], fit_targets) for index in range(22)
    }
    baseline = metrics["none"]
    geometry = metrics["all_geometry_zero"]
    pair_disabled = metrics["pair_branch_disabled"]
    geometry_pairwise_drop = float(baseline["pairwise_accuracy"]) - float(
        geometry["pairwise_accuracy"]
    )
    geometry_mae_increase = float(geometry["mae"]) - float(baseline["mae"])
    pair_pairwise_drop = float(baseline["pairwise_accuracy"]) - float(
        pair_disabled["pairwise_accuracy"]
    )
    pair_mae_increase = float(pair_disabled["mae"]) - float(baseline["mae"])
    gates = {
        "tune_support": int(baseline["count"]) == int(config["gates"]["tune_count"]),
        "geometry_pairwise": geometry_pairwise_drop
        >= float(config["gates"]["geometry_pairwise_drop_min"]),
        "geometry_mae": geometry_mae_increase
        >= float(config["gates"]["geometry_mae_increase_min"]),
        "pair_pairwise": pair_pairwise_drop >= float(config["gates"]["pair_pairwise_drop_min"]),
        "pair_mae": pair_mae_increase >= float(config["gates"]["pair_mae_increase_min"]),
    }
    geometry_supported = all(gates.values())
    git_dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    if args.require_clean_git and git_dirty:
        raise RuntimeError("--require-clean-git requested but repository is dirty")
    result = {
        "completed": True,
        "scientific_status": (
            "fresh_geometry_control_design_warranted"
            if geometry_supported
            else "geometry_endpoint_frozen_score_set_statistics_only"
        ),
        "version_id": config["version_id"],
        "experiment_scope": "fit_tune_artifact_only_zero_training_attribution",
        "config_sha256": CONFIG_SHA256,
        "source_result_sha256": config["source"]["result_sha256"],
        "fit_tune_cache_sha256": config["source"]["fit_tune_cache_sha256"],
        "full_checkpoint_sha256": config["source"]["full_checkpoint_sha256"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": git_dirty,
        "new_data_read": False,
        "outer_cache_read": False,
        "outer_metrics_used_for_decision": False,
        "fit_feature_target_pearson": feature_correlations,
        "tune_metrics": metrics,
        "tune_contributions": contributions,
        "summary": {
            "geometry_pairwise_drop": geometry_pairwise_drop,
            "geometry_mae_increase": geometry_mae_increase,
            "pair_pairwise_drop": pair_pairwise_drop,
            "pair_mae_increase": pair_mae_increase,
        },
        "gates": {"all_passed": geometry_supported, "gates": gates},
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
            {"status": result["scientific_status"], "summary": result["summary"], "gates": result["gates"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
