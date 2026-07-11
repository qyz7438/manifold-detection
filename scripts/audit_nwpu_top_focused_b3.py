"""Zero-training candidate-local and top-focused audit of frozen B3 heads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
V1_SCRIPT = ROOT / "scripts" / "probe_nwpu_joint_delta_u.py"
V2_SCRIPT = ROOT / "scripts" / "probe_nwpu_joint_delta_u_v2.py"
LOCKED_CONFIG = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.top_focused_audit.001.json"
)
LOCKED_CONFIG_SHA256 = "f08fcf6b25db1dae15ca97d6289b52ea214050f44018ef2da22c0374487fe2fe"
DEFAULT_RUN_DIR = ROOT / "runs" / "nwpu_top_focused_audit_s42_reused_outer"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_config(path: str | Path = LOCKED_CONFIG) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if resolved != LOCKED_CONFIG.resolve():
        raise ValueError(f"config must be the locked file: {LOCKED_CONFIG}")
    if sha256_file(resolved) != LOCKED_CONFIG_SHA256:
        raise ValueError("locked top-focused audit config hash mismatch")
    return json.loads(resolved.read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(LOCKED_CONFIG))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--device")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def _load_script(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _manifest(records: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    image_ids = sorted(int(record[0]["image_id"]) for record in records)
    encoded = json.dumps(image_ids, separators=(",", ":")).encode("ascii")
    return {
        "count": len(image_ids),
        "image_ids_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _feature_map_from_checkpoint(payload: dict[str, Any], device: torch.device) -> Any:
    from spectral_detection_posttrain.methods.energy_transport.spatial_counterfactual import (
        FrozenSpatialFeatureMap,
    )

    values = payload["feature_map"]
    return FrozenSpatialFeatureMap(
        channel_mean=torch.tensor(values["channel_mean"], device=device),
        channel_scale=torch.tensor(values["channel_scale"], device=device),
        channel_components=torch.tensor(values["channel_components"], device=device),
        source_mean=torch.tensor(values["source_mean"], device=device),
        source_scale=torch.tensor(values["source_scale"], device=device),
        source_components=torch.tensor(values["source_components"], device=device),
        feature_mean=torch.tensor(values["feature_mean"], device=device),
        feature_scale=torch.tensor(values["feature_scale"], device=device),
        block_count=int(values["block_count"]),
    )


def _models_from_checkpoint(
    payload: dict[str, Any], key: str, device: torch.device
) -> dict[str, Any]:
    from spectral_detection_posttrain.methods.energy_transport.linear_identifiability import (
        RidgeRegressor,
    )

    return {
        str(arm): RidgeRegressor(
            weights=torch.tensor(values["weights"], device=device),
            bias=torch.tensor(values["bias"], device=device),
            l2=float(values["l2"]),
        )
        for arm, values in payload[key].items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_locked_config(args.config)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.require_clean_git and _git_value("status", "--porcelain"):
        raise RuntimeError("--require-clean-git requested but repository is dirty")
    source = config["source_b3"]
    source_config_path = (ROOT / source["config_path"]).resolve()
    source_metrics_path = (ROOT / source["metrics_path"]).resolve()
    checkpoint_path = (ROOT / source["checkpoint_path"]).resolve()
    for path, expected, label in (
        (source_config_path, source["config_sha256"], "B3 config"),
        (source_metrics_path, source["metrics_sha256"], "B3 metrics"),
        (checkpoint_path, source["checkpoint_sha256"], "B3 checkpoint"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{label} hash mismatch")
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    if (
        source_metrics["inputs"]["code_commit"] != source["code_commit"]
        or source_metrics["inputs"]["config_sha256"] != source["config_sha256"]
        or source_metrics["checkpoint"]["sha256"] != source["checkpoint_sha256"]
    ):
        raise ValueError("B3 artifact provenance mismatch")

    v1 = _load_script(V1_SCRIPT, "top_focused_v1")
    v2 = _load_script(V2_SCRIPT, "top_focused_v2")
    source_cache = (ROOT / source_config["source_train_cache"]["path"]).resolve()
    source_metadata = (ROOT / source_config["source_train_cache"]["metadata"]).resolve()
    v2._verify_file(
        source_cache, source_config["source_train_cache"]["sha256"], "source cache"
    )
    v2._verify_file(
        source_metadata,
        source_config["source_train_cache"]["metadata_sha256"],
        "source metadata",
    )
    records = v1.read_probe_cache(source_cache)
    _, outer_heldout = v2.split_train_records(
        records,
        calibration_fraction=float(source_config["split"]["outer_heldout_fraction"]),
        seed=int(source_config["split"]["outer_seed"]),
    )
    if _manifest(outer_heldout) != source_config["split"]["outer_heldout_manifest"]:
        raise ValueError("outer heldout manifest mismatch")

    from spectral_detection_posttrain.methods.energy_transport.decomposed_actionability import (
        sign_classification_metrics,
    )
    from spectral_detection_posttrain.methods.energy_transport.linear_identifiability import (
        predict_ridge_regression,
        within_image_shuffle_order,
    )
    from spectral_detection_posttrain.methods.energy_transport.spatial_counterfactual import (
        extract_spatial_candidate_rows,
        move_spatial_candidate_rows,
        spatial_counterfactual_blocks,
        transform_spatial_features,
    )
    from spectral_detection_posttrain.methods.energy_transport.top_focused_audit import (
        evaluate_top_focused_gates,
        median_sign_metrics,
        top_focused_rank_metrics,
    )
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

    set_seed(int(source_config["dataset"]["seed"]))
    device = resolve_device(
        {"device": args.device or ("cuda" if torch.cuda.is_available() else "cpu")}
    )
    v1.validate_deterministic_cuda_environment(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if (
        checkpoint["version_id"] != source_metrics["version_id"]
        or checkpoint["config_sha256"] != source["config_sha256"]
    ):
        raise ValueError("B3 checkpoint metadata mismatch")
    feature_map = _feature_map_from_checkpoint(checkpoint, device)
    sign_models = _models_from_checkpoint(checkpoint, "sign_models", device)
    rank_models = _models_from_checkpoint(checkpoint, "rank_models", device)
    if set(sign_models) != set(config["arms"]) or set(rank_models) != set(config["arms"]):
        raise ValueError("B3 checkpoint arm set mismatch")

    outer_rows = move_spatial_candidate_rows(
        extract_spatial_candidate_rows(outer_heldout, num_classes=11), device
    )
    blocks = spatial_counterfactual_blocks(
        outer_rows.spatial_features, outer_rows.action_features
    )
    full_features = transform_spatial_features(outer_rows, blocks, feature_map)
    feature_order, feature_changed_fraction = within_image_shuffle_order(
        outer_rows.image_ids,
        seed=int(source_config["controls"]["feature_shuffle_seed"]) + 2,
    )
    if feature_changed_fraction < float(
        source_config["controls"]["min_changed_row_fraction"]
    ):
        raise ValueError("frozen B3 outer feature control coverage below minimum")
    arm_features = {
        "local_full": full_features,
        "within_image_feature_shuffle": full_features[feature_order.to(device)],
        "within_image_label_shuffle": full_features,
    }
    arms: dict[str, Any] = {}
    metric_config = config["metrics"]
    for arm_value in config["arms"]:
        arm = str(arm_value)
        features = arm_features[arm]
        sign_scores = predict_ridge_regression(sign_models[arm], features)
        rank_scores = predict_ridge_regression(rank_models[arm], features)
        arms[arm] = {
            "median_sign": median_sign_metrics(
                sign_scores, outer_rows.target, outer_rows.image_ids
            ),
            "original_sign_diagnostic": sign_classification_metrics(
                sign_scores,
                outer_rows.target,
                outer_rows.image_ids,
                target_epsilon=float(metric_config["target_epsilon"]),
            ),
            "top_rank": top_focused_rank_metrics(
                rank_scores,
                outer_rows.target,
                outer_rows.image_ids,
                target_epsilon=float(metric_config["target_epsilon"]),
                min_target_gap=float(metric_config["min_target_gap"]),
                lcb_z=float(metric_config["lcb_z"]),
            ),
        }
    audit = {
        "support": {
            "candidate_count": int(outer_rows.target.numel()),
            "image_count": int(torch.unique(outer_rows.image_ids).numel()),
            "manifest_image_count": len(outer_heldout),
        },
        "arms": arms,
    }
    gates = evaluate_top_focused_gates(audit, config)
    payload = {
        "completed": True,
        "scientific_status": "audit_passed" if gates["all_passed"] else "audit_failed",
        "version_id": config["version_id"],
        "claim_boundary": config["probe"]["claim_boundary"],
        "inputs": {
            "code_commit": _git_value("rev-parse", "HEAD"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "config_sha256": LOCKED_CONFIG_SHA256,
            "source_b3_metrics_sha256": source["metrics_sha256"],
            "source_b3_checkpoint_sha256": source["checkpoint_sha256"],
            "outer_heldout_manifest": _manifest(outer_heldout),
            "device": str(device),
        },
        "protocol": {
            "performs_new_training": False,
            "researcher_adaptive": True,
            "outer_split_seen_in_prior_analyses": True,
            "feature_control_changed_fraction": feature_changed_fraction,
        },
        **audit,
        "gates": gates,
    }
    result_path = run_dir / "eval_metrics.json"
    result_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    payload["output"] = str(result_path)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    payload = run(parse_args(argv))
    print(
        json.dumps(
            {
                "completed": payload["completed"],
                "scientific_status": payload["scientific_status"],
                "gates": payload["gates"],
                "output": payload["output"],
            },
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
