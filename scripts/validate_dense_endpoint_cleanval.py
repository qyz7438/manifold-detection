"""Validate the fixed dense endpoint on detector-unseen NWPU validation images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.dense_endpoint.cleanval.001.json"
)
CONFIG_SHA256 = "82f56e7772042904c6da3dbb8559e630a351a35ccba5582421aeb1d0d8f8d05e"


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
        raise ValueError("canonical dense endpoint cleanval config SHA256 mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("version_id") != "det.energy.dense_endpoint.cleanval.001":
        raise ValueError("dense endpoint cleanval version mismatch")
    if config.get("dataset", {}).get("validation_read_once_after_all_models_frozen") is not True:
        raise ValueError("validation read-once boundary drift")
    if config.get("training", {}).get("all_hyperparameters_frozen") is not True:
        raise ValueError("cleanval hyperparameters must be frozen")
    return config


def combine_train_records(
    first: Sequence[dict[str, Any]], second: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    combined = [*first, *second]
    image_ids = [int(record["image_id"]) for record in combined]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("duplicate image IDs across train caches")
    return combined


def _load_cache(path: Path) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "dense_endpoint_absolute_v1" or not isinstance(
        payload.get("records"), list
    ):
        raise ValueError("unsupported dense endpoint train cache")
    return payload["records"]


def _validation_ids(
    data_root: Path, annotation: Path, config: dict[str, Any]
) -> list[int]:
    from spectral_detection_posttrain.datasets.nwpu_vhr10 import nwpu_positive_image_ids

    ids = nwpu_positive_image_ids(data_root, annotation)
    rng = np.random.RandomState(int(config["training"]["seed"]))
    rng.shuffle(ids)
    split = int(len(ids) * 0.7)
    return sorted(int(value) for value in ids[split:])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "runs" / "nwpu_dense_endpoint_cleanval_s42",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    from spectral_detection_posttrain.methods.energy_transport.dense_endpoint import RobustTeacherStats
    from spectral_detection_posttrain.models import build_detector
    from spectral_detection_posttrain.utils.io import load_checkpoint
    from spectral_detection_posttrain.utils.seed import set_seed
    import scripts.train_dense_endpoint_geometry_control as geometry
    import scripts.train_nwpu_dense_endpoint_absolute as absolute

    config = load_config()
    geometry_result = (ROOT / config["sources"]["geometry_control_result"]).resolve()
    fit_tune_cache = (ROOT / config["sources"]["fit_tune_cache"]).resolve()
    outer_train_cache = (ROOT / config["sources"]["outer_train_cache"]).resolve()
    for name, path, expected in (
        ("geometry_control", geometry_result, config["sources"]["geometry_control_sha256"]),
        ("fit_tune_cache", fit_tune_cache, config["sources"]["fit_tune_cache_sha256"]),
        ("outer_train_cache", outer_train_cache, config["sources"]["outer_train_cache_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"cleanval source {name} SHA256 mismatch")
    source = json.loads(geometry_result.read_text(encoding="utf-8"))
    if source.get("scientific_status") != config["sources"]["required_status"]:
        raise ValueError("strong geometry control decision mismatch")
    train_records = combine_train_records(
        _load_cache(fit_tune_cache), _load_cache(outer_train_cache)
    )
    train_ids = [int(record["image_id"]) for record in train_records]
    if (
        len(train_records) != int(config["dataset"]["train_count"])
        or image_ids_hash(train_ids) != config["dataset"]["train_sha256"]
    ):
        raise ValueError("full train cache manifest mismatch")

    git_dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    if args.require_clean_git and git_dirty:
        raise RuntimeError("--require-clean-git requested but repository is dirty")
    device = torch.device(args.device)
    set_seed(int(config["training"]["seed"]))
    teacher_stats = RobustTeacherStats.fit(
        torch.stack([record["teacher_values"] for record in train_records])
    )
    feature_stats = geometry._fit_feature_stats(train_records)
    train_targets = absolute._targets(train_records, teacher_stats)
    fit_mean = float(train_targets.mean().item())
    models = {}
    training = {}
    for arm in config["arms"]:
        if arm == "teacher_shuffle":
            permutation = geometry._non_identity_permutation(
                len(train_records), int(config["controls"]["teacher_shuffle_seed"])
            )
            models[arm], training[arm] = geometry._train(
                train_records,
                train_targets[permutation],
                "local_full",
                feature_stats,
                config,
                device,
            )
        else:
            models[arm], training[arm] = geometry._train(
                train_records, train_targets, arm, feature_stats, config, device
            )

    # The validation IDs, images, and teacher targets are first materialized here,
    # after every model and control has frozen parameters.
    checkpoint = (args.checkpoint or ROOT / config["detector"]["checkpoint"]).resolve()
    annotation = (args.annotation or ROOT / "data" / "NWPU_VHR10_coco.json").resolve()
    data_root = (args.data_root or ROOT / "data" / "NWPU VHR-10 dataset").resolve()
    if sha256_file(checkpoint) != config["detector"]["checkpoint_sha256"]:
        raise ValueError("detector checkpoint SHA256 mismatch")
    if sha256_file(annotation) != config["dataset"]["annotation_sha256"]:
        raise ValueError("annotation SHA256 mismatch")
    validation_ids = _validation_ids(data_root, annotation, config)
    if (
        len(validation_ids) != int(config["dataset"]["validation_count"])
        or image_ids_hash(validation_ids) != config["dataset"]["validation_sha256"]
    ):
        raise ValueError("detector validation manifest mismatch")
    detector = build_detector(absolute._detector_config(config, data_root, annotation)).to(device)
    load_checkpoint(detector, checkpoint, device)
    detector.roi_heads.score_thresh = float(config["detector"]["score_threshold"])
    detector.roi_heads.nms_thresh = float(config["detector"]["nms_threshold"])
    detector.roi_heads.detections_per_img = int(config["detector"]["detections_per_image"])
    detector.eval()
    validation_records = absolute._build_records(
        detector,
        validation_ids,
        data_root,
        annotation,
        config,
        device,
        split_name="detector_validation_read_once",
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    validation_cache = args.run_dir / "detector_validation_read_once_cache.pt"
    validation_cache_sha = absolute._write_cache(
        validation_cache,
        validation_records,
        {
            "config_sha256": CONFIG_SHA256,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_dirty": git_dirty,
            "split": "detector_validation",
            "read_count": 1,
            "models_frozen_before_read": True,
        },
    )
    validation_targets = absolute._targets(validation_records, teacher_stats)
    evaluation = {}
    checkpoint_hashes = {}
    for arm, model in models.items():
        feature_arm = "local_full" if arm == "teacher_shuffle" else arm
        train_prediction = geometry._predict(
            model, train_records, feature_arm, feature_stats, config, device
        )
        validation_prediction = geometry._predict(
            model, validation_records, feature_arm, feature_stats, config, device
        )
        evaluation[arm] = {
            "train": absolute.regression_metrics(
                train_prediction, train_targets, fit_mean=fit_mean
            ),
            "detector_validation": absolute.regression_metrics(
                validation_prediction, validation_targets, fit_mean=fit_mean
            ),
        }
        output = args.run_dir / arm / "endpoint.pth"
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "arm": arm,
                "config_sha256": CONFIG_SHA256,
                "feature_stats": feature_stats,
                "teacher_median": teacher_stats.median,
                "teacher_iqr": teacher_stats.iqr,
            },
            output,
        )
        checkpoint_hashes[arm] = sha256_file(output)
    full = evaluation["local_full"]["detector_validation"]
    control_arms = ("strong_geometry_shuffle", "score_class_only", "teacher_shuffle")
    pairwise_gains = {
        arm: float(full["pairwise_accuracy"])
        - float(evaluation[arm]["detector_validation"]["pairwise_accuracy"])
        for arm in control_arms
    }
    mae_gains = {
        arm: float(evaluation[arm]["detector_validation"]["mae"]) - float(full["mae"])
        for arm in control_arms
    }
    best_constant = min(float(full["zero_baseline_mae"]), float(full["fit_mean_baseline_mae"]))
    constant_gain = (best_constant - float(full["mae"])) / best_constant
    train_gap = float(evaluation["local_full"]["train"]["pairwise_accuracy"]) - float(
        full["pairwise_accuracy"]
    )
    gates = {
        "validation_support": int(full["count"]) == int(config["gates"]["validation_count"]),
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
        "train_validation_gap": train_gap
        <= float(config["gates"]["train_to_validation_pairwise_gap_max"]),
    }
    all_passed = all(gates.values())
    result = {
        "completed": True,
        "scientific_status": (
            "detector_unseen_absolute_endpoint_validated"
            if all_passed
            else "detector_unseen_absolute_endpoint_frozen"
        ),
        "version_id": config["version_id"],
        "experiment_scope": "nwpu_detector_unseen_absolute_endpoint_validation",
        "config_sha256": CONFIG_SHA256,
        "source_geometry_control_sha256": config["sources"]["geometry_control_sha256"],
        "train_cache_hashes": {
            "fit_tune": config["sources"]["fit_tune_cache_sha256"],
            "outer_train": config["sources"]["outer_train_cache_sha256"],
        },
        "validation_cache_sha256": validation_cache_sha,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": git_dirty,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "validation_read_count": 1,
        "models_frozen_before_validation_read": True,
        "detector_parameters_trained": False,
        "actions_evaluated": False,
        "teacher_stats": {
            "median": teacher_stats.median.tolist(),
            "iqr": teacher_stats.iqr.tolist(),
            "train_target_mean": fit_mean,
        },
        "training": training,
        "checkpoint_hashes": checkpoint_hashes,
        "evaluation": evaluation,
        "summary": {
            "full_validation": full,
            "pairwise_gains_over_controls": pairwise_gains,
            "mae_gains_over_controls": mae_gains,
            "constant_mae_relative_gain": constant_gain,
            "train_to_validation_pairwise_gap": train_gap,
        },
        "gates": {"all_passed": all_passed, "gates": gates},
        "decision_rule": config["decision_rule"],
        "claim_boundary": config["claim_boundary"],
        "validated_claim": all_passed,
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
