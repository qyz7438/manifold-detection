"""Train the fresh-resplit strong geometry control for the dense endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as torch_f


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.dense_endpoint.geometry_control.001.json"
)
CONFIG_SHA256 = "612dfd2be6cc57767a4234267a5ed3f078d84048129905bc2f86256b3e0b04b4"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_ids_hash(image_ids: Sequence[int]) -> str:
    encoded = json.dumps(sorted(int(value) for value in image_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def load_config() -> dict[str, Any]:
    if sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise ValueError("canonical dense geometry-control config SHA256 mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("version_id") != "det.energy.dense_endpoint.geometry_control.001":
        raise ValueError("dense geometry-control version mismatch")
    source = config.get("sources", {})
    if source.get("old_inner_tune_deserialized_but_discarded_before_analysis") is not True:
        raise ValueError("old tune discard boundary drift")
    if source.get("old_outer_forbidden") is not True or source.get("new_data_forbidden") is not True:
        raise ValueError("geometry control must forbid old outer and new data")
    return config


def _non_identity_permutation(count: int, seed: int) -> torch.Tensor:
    if count < 2:
        return torch.arange(count)
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(count, generator=generator)
    identity = torch.arange(count)
    return permutation.roll(1) if torch.equal(permutation, identity) else permutation


def resplit_image_ids(
    image_ids: Sequence[int], config: dict[str, Any]
) -> tuple[list[int], list[int]]:
    seed = int(config["resplit"]["seed"])
    ordered = sorted(
        (int(value) for value in image_ids),
        key=lambda image_id: hashlib.sha256(
            f"dense-geometry-control:{seed}:{image_id}".encode("ascii")
        ).digest(),
    )
    fit_count = int(config["resplit"]["fit_count"])
    return sorted(ordered[:fit_count]), sorted(ordered[fit_count:])


def apply_geometry_control(
    features: torch.Tensor, *, seed: int, config: dict[str, Any]
) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] != 22:
        raise ValueError("node features must have shape (N, 22)")
    output = features.clone()
    columns = config["controls"]["strong_geometry_columns"]
    permutation = _non_identity_permutation(features.shape[0], seed).to(features.device)
    output[:, columns] = features[permutation][:, columns]
    return output


def _fit_feature_stats(records: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    nodes = torch.cat([record["node_features"].float() for record in records], dim=0)
    pairs = torch.cat([record["pair_features"].float() for record in records], dim=0)
    return {
        "node_mean": nodes.mean(dim=0),
        "node_scale": nodes.std(dim=0, unbiased=False).clamp_min(1e-4),
        "pair_mean": pairs.mean(dim=0),
        "pair_scale": pairs.std(dim=0, unbiased=False).clamp_min(1e-4),
    }


def _features(
    record: dict[str, Any],
    arm: str,
    feature_stats: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    nodes = record["node_features"].float()
    pairs = record["pair_features"].float()
    if arm == "strong_geometry_shuffle":
        nodes = apply_geometry_control(
            nodes,
            seed=int(config["controls"]["geometry_shuffle_seed"]) + int(record["image_id"]),
            config=config,
        )
        pairs = record["topology_pair_features"].float()
    nodes = (nodes - feature_stats["node_mean"]) / feature_stats["node_scale"]
    pairs = (pairs - feature_stats["pair_mean"]) / feature_stats["pair_scale"]
    if arm == "score_class_only":
        keep = set(int(value) for value in config["controls"]["score_class_columns"])
        drop = [index for index in range(22) if index not in keep]
        nodes[:, drop] = 0.0
        pairs = pairs.new_zeros((0, 5))
    return nodes.to(device), pairs.to(device)


def _new_model(config: dict[str, Any], device: torch.device):
    from spectral_detection_posttrain.methods.energy_transport.dense_endpoint import DenseSetEnergyEndpoint

    return DenseSetEnergyEndpoint(
        node_dim=int(config["model"]["node_dim"]),
        pair_dim=int(config["model"]["pair_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
    ).to(device)


def _train(
    records: Sequence[dict[str, Any]],
    targets: torch.Tensor,
    arm: str,
    feature_stats: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, float | int]]:
    seed = int(config["controls"]["same_initialization_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _new_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    accumulation = int(config["training"]["grad_accum_images"])
    losses = []
    model.train()
    for epoch in range(int(config["training"]["epochs"])):
        order = list(range(len(records)))
        random.Random(seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for position, index in enumerate(order, start=1):
            chunk_start = ((position - 1) // accumulation) * accumulation
            chunk_size = min(accumulation, len(order) - chunk_start)
            nodes, pairs = _features(records[index], arm, feature_stats, config, device)
            prediction = model(nodes, pairs).quality
            loss = torch_f.smooth_l1_loss(
                prediction,
                targets[index].to(device),
                beta=float(config["training"]["smooth_l1_beta"]),
            )
            (loss / chunk_size).backward()
            total += float(loss.detach().item())
            if position % accumulation == 0 or position == len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        losses.append(total / len(records))
    return model, {
        "epochs": int(config["training"]["epochs"]),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
    }


@torch.inference_mode()
def _predict(
    model: torch.nn.Module,
    records: Sequence[dict[str, Any]],
    arm: str,
    feature_stats: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    output = []
    for record in records:
        nodes, pairs = _features(record, arm, feature_stats, config, device)
        output.append(model(nodes, pairs).quality.detach().cpu())
    return torch.stack(output).float()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "runs" / "nwpu_dense_endpoint_geometry_control_s42_resplit",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    from spectral_detection_posttrain.methods.energy_transport.dense_endpoint import RobustTeacherStats
    import scripts.train_nwpu_dense_endpoint_absolute as absolute

    config = load_config()
    attribution_path = (ROOT / config["sources"]["attribution_result"]).resolve()
    cache_path = (ROOT / config["sources"]["fit_tune_cache"]).resolve()
    if sha256_file(attribution_path) != config["sources"]["attribution_sha256"]:
        raise ValueError("attribution result SHA256 mismatch")
    if sha256_file(cache_path) != config["sources"]["fit_tune_cache_sha256"]:
        raise ValueError("fit/tune cache SHA256 mismatch")
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    if attribution.get("scientific_status") != config["sources"]["required_status"]:
        raise ValueError("attribution decision mismatch")
    payload = torch.load(cache_path, map_location="cpu")
    if payload.get("format") != "dense_endpoint_absolute_v1":
        raise ValueError("unsupported source cache")
    all_records = payload.get("records")
    source_count = int(config["sources"]["source_inner_fit_count"])
    if not isinstance(all_records, list) or len(all_records) <= source_count:
        raise ValueError("combined source cache support mismatch")
    # The combined file is deserialized, then old tune rows are discarded here.
    source_records = all_records[:source_count]
    old_tune_discarded_count = len(all_records) - source_count
    source_ids = [int(record["image_id"]) for record in source_records]
    if image_ids_hash(source_ids) != config["sources"]["source_inner_fit_sha256"]:
        raise ValueError("source inner-fit image hash mismatch")
    fit_ids, holdout_ids = resplit_image_ids(source_ids, config)
    if image_ids_hash(fit_ids) != config["resplit"]["fit_sha256"]:
        raise ValueError("geometry-control fit hash mismatch")
    if image_ids_hash(holdout_ids) != config["resplit"]["holdout_sha256"]:
        raise ValueError("geometry-control holdout hash mismatch")
    by_id = {int(record["image_id"]): record for record in source_records}
    fit_records = [by_id[image_id] for image_id in fit_ids]
    holdout_records = [by_id[image_id] for image_id in holdout_ids]

    teacher_stats = RobustTeacherStats.fit(
        torch.stack([record["teacher_values"] for record in fit_records])
    )
    feature_stats = _fit_feature_stats(fit_records)
    fit_targets = absolute._targets(fit_records, teacher_stats)
    holdout_targets = absolute._targets(holdout_records, teacher_stats)
    fit_mean = float(fit_targets.mean().item())
    device = torch.device(args.device)
    models = {}
    training = {}
    evaluation = {}
    checkpoint_hashes = {}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    for arm in config["arms"]:
        models[arm], training[arm] = _train(
            fit_records, fit_targets, arm, feature_stats, config, device
        )
        fit_prediction = _predict(
            models[arm], fit_records, arm, feature_stats, config, device
        )
        holdout_prediction = _predict(
            models[arm], holdout_records, arm, feature_stats, config, device
        )
        evaluation[arm] = {
            "fit": absolute.regression_metrics(fit_prediction, fit_targets, fit_mean=fit_mean),
            "holdout": absolute.regression_metrics(
                holdout_prediction, holdout_targets, fit_mean=fit_mean
            ),
        }
        checkpoint = args.run_dir / arm / "endpoint.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": models[arm].state_dict(),
                "arm": arm,
                "config_sha256": CONFIG_SHA256,
                "feature_stats": feature_stats,
                "teacher_median": teacher_stats.median,
                "teacher_iqr": teacher_stats.iqr,
            },
            checkpoint,
        )
        checkpoint_hashes[arm] = sha256_file(checkpoint)
    full = evaluation["local_full"]["holdout"]
    controls = ("strong_geometry_shuffle", "score_class_only")
    pairwise_gains = {
        arm: float(full["pairwise_accuracy"])
        - float(evaluation[arm]["holdout"]["pairwise_accuracy"])
        for arm in controls
    }
    mae_gains = {
        arm: float(evaluation[arm]["holdout"]["mae"]) - float(full["mae"])
        for arm in controls
    }
    best_constant = min(float(full["zero_baseline_mae"]), float(full["fit_mean_baseline_mae"]))
    constant_gain = (best_constant - float(full["mae"])) / best_constant
    fit_gap = float(evaluation["local_full"]["fit"]["pairwise_accuracy"]) - float(
        full["pairwise_accuracy"]
    )
    gates = {
        "holdout_support": int(full["count"]) == int(config["gates"]["holdout_count"]),
        "full_pairwise": float(full["pairwise_accuracy"])
        >= float(config["gates"]["full_pairwise_min"]),
        "full_auroc": full["auroc_positive"] is not None
        and float(full["auroc_positive"]) >= float(config["gates"]["full_auroc_min"]),
        "constant_baseline": constant_gain
        >= float(config["gates"]["constant_mae_relative_gain_min"]),
        "pairwise_controls": all(
            value >= float(config["gates"]["pairwise_gain_over_each_control_min"])
            for value in pairwise_gains.values()
        ),
        "mae_controls": all(
            value >= float(config["gates"]["mae_gain_over_each_control_min"])
            for value in mae_gains.values()
        ),
        "fit_holdout_gap": fit_gap
        <= float(config["gates"]["fit_to_holdout_pairwise_gap_max"]),
    }
    all_passed = all(gates.values())
    git_dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    if args.require_clean_git and git_dirty:
        raise RuntimeError("--require-clean-git requested but repository is dirty")
    result = {
        "completed": True,
        "scientific_status": (
            "geometry_endpoint_signal_supported_train_only_resplit"
            if all_passed
            else "geometry_conditioned_endpoint_frozen"
        ),
        "version_id": config["version_id"],
        "experiment_scope": "researcher_adaptive_old_inner_fit_resplit",
        "config_sha256": CONFIG_SHA256,
        "source_attribution_sha256": config["sources"]["attribution_sha256"],
        "source_cache_sha256": config["sources"]["fit_tune_cache_sha256"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": git_dirty,
        "old_tune_records_deserialized_then_discarded": old_tune_discarded_count,
        "old_tune_used_for_statistics_training_evaluation_or_gates": False,
        "old_outer_cache_read": False,
        "new_data_read": False,
        "detector_inference_run": False,
        "resplit": {
            "fit_count": len(fit_records),
            "fit_sha256": config["resplit"]["fit_sha256"],
            "holdout_count": len(holdout_records),
            "holdout_sha256": config["resplit"]["holdout_sha256"],
        },
        "teacher_stats": {
            "median": teacher_stats.median.tolist(),
            "iqr": teacher_stats.iqr.tolist(),
            "fit_target_mean": fit_mean,
        },
        "training": training,
        "checkpoint_hashes": checkpoint_hashes,
        "evaluation": evaluation,
        "summary": {
            "full_holdout": full,
            "pairwise_gains_over_controls": pairwise_gains,
            "mae_gains_over_controls": mae_gains,
            "constant_mae_relative_gain": constant_gain,
            "fit_to_holdout_pairwise_gap": fit_gap,
        },
        "gates": {"all_passed": all_passed, "gates": gates},
        "decision_rule": config["decision_rule"],
        "claim_boundary": config["claim_boundary"],
        "validated_claim": False,
    }
    (args.run_dir / "eval_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
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
