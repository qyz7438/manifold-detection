"""Nested train-cache probe for action-aligned 7x7 ROI structure."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
V1_SCRIPT = ROOT / "scripts" / "probe_nwpu_joint_delta_u.py"
V2_SCRIPT = ROOT / "scripts" / "probe_nwpu_joint_delta_u_v2.py"
LINEAR_SCRIPT = ROOT / "scripts" / "probe_nwpu_linear_identifiability.py"
LOCKED_CONFIG = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.spatial_counterfactual.001.json"
)
LOCKED_CONFIG_SHA256 = "71e1f3589ee05f7674246b46b24536313574c6fb6c0e087eaad2f19069ce1429"
DEFAULT_RUN_DIR = ROOT / "runs" / "nwpu_spatial_counterfactual_s42_nested"


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
        raise ValueError("locked spatial counterfactual config hash mismatch")
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


def _json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_locked_config(args.config)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.require_clean_git and _git_value("status", "--porcelain"):
        raise RuntimeError("--require-clean-git requested but repository is dirty")
    v1 = _load_script(V1_SCRIPT, "spatial_cf_v1")
    v2 = _load_script(V2_SCRIPT, "spatial_cf_v2")
    linear = _load_script(LINEAR_SCRIPT, "spatial_cf_linear")
    source_cache = (ROOT / config["source_train_cache"]["path"]).resolve()
    source_metadata = (ROOT / config["source_train_cache"]["metadata"]).resolve()
    v2._verify_file(
        source_cache, config["source_train_cache"]["sha256"], "source train cache"
    )
    v2._verify_file(
        source_metadata,
        config["source_train_cache"]["metadata_sha256"],
        "source train cache metadata",
    )
    metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    if metadata.get("pairing") != config["source_train_cache"]["pairing"]:
        raise ValueError("source train cache pairing mismatch")
    records = v1.read_probe_cache(source_cache)
    expected_manifest = {
        "count": int(config["dataset"]["train_images"]),
        "image_ids_sha256": config["dataset"]["train_image_ids_sha256"],
    }
    if _manifest(records) != expected_manifest:
        raise ValueError("source train cache manifest mismatch")
    outer_fit, outer_heldout = v2.split_train_records(
        records,
        calibration_fraction=float(config["split"]["outer_heldout_fraction"]),
        seed=int(config["split"]["outer_seed"]),
    )
    if _manifest(outer_fit) != config["split"]["outer_fit_manifest"]:
        raise ValueError("outer fit manifest mismatch")
    if _manifest(outer_heldout) != config["split"]["outer_heldout_manifest"]:
        raise ValueError("outer heldout manifest mismatch")
    inner_fit, inner_tune = v2.split_train_records(
        outer_fit,
        calibration_fraction=float(config["split"]["inner_tune_fraction"]),
        seed=int(config["split"]["inner_seed"]),
    )

    from spectral_detection_posttrain.methods.energy_transport import (
        calibrated_selection_metrics,
        constant_utility_baselines,
    )
    from spectral_detection_posttrain.methods.energy_transport.linear_identifiability import (
        predict_ridge_regression,
        within_image_shuffle_order,
    )
    from spectral_detection_posttrain.methods.energy_transport.spatial_counterfactual import (
        evaluate_spatial_counterfactual_gates,
        extract_spatial_candidate_rows,
        fit_spatial_feature_map,
        move_spatial_candidate_rows,
        spatial_counterfactual_blocks,
        spatial_layout_shuffle,
        transform_spatial_features,
    )
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

    set_seed(int(config["dataset"]["seed"]))
    device = resolve_device(
        {"device": args.device or ("cuda" if torch.cuda.is_available() else "cpu")}
    )
    v1.validate_deterministic_cuda_environment(device)
    torch.use_deterministic_algorithms(True, warn_only=True)
    fit_rows = move_spatial_candidate_rows(
        extract_spatial_candidate_rows(inner_fit, num_classes=11), device
    )
    tune_rows = move_spatial_candidate_rows(
        extract_spatial_candidate_rows(inner_tune, num_classes=11), device
    )

    structured_fit_blocks = spatial_counterfactual_blocks(
        fit_rows.spatial_features, fit_rows.action_features
    )
    structured_tune_blocks = spatial_counterfactual_blocks(
        tune_rows.spatial_features, tune_rows.action_features
    )
    structured_map = fit_spatial_feature_map(
        fit_rows,
        structured_fit_blocks,
        channel_pca_dim=int(config["features"]["channel_pca_dim"]),
        final_pca_dim=int(config["features"]["structured_pca_dim"]),
    )
    global_fit_blocks = fit_rows.spatial_features.mean(dim=(-2, -1))[:, None, :]
    global_tune_blocks = tune_rows.spatial_features.mean(dim=(-2, -1))[:, None, :]
    global_map = fit_spatial_feature_map(
        fit_rows,
        global_fit_blocks,
        channel_pca_dim=int(config["features"]["channel_pca_dim"]),
        final_pca_dim=int(config["features"]["global_pca_dim"]),
    )
    shuffled_fit_spatial, spatial_fit_fraction = spatial_layout_shuffle(
        fit_rows.spatial_features,
        seed=int(config["controls"]["spatial_shuffle_seed"]),
    )
    shuffled_tune_spatial, spatial_tune_fraction = spatial_layout_shuffle(
        tune_rows.spatial_features,
        seed=int(config["controls"]["spatial_shuffle_seed"]) + 1,
    )
    layout_fit_blocks = spatial_counterfactual_blocks(
        shuffled_fit_spatial, fit_rows.action_features
    )
    layout_tune_blocks = spatial_counterfactual_blocks(
        shuffled_tune_spatial, tune_rows.action_features
    )
    layout_map = fit_spatial_feature_map(
        fit_rows,
        layout_fit_blocks,
        channel_pca_dim=int(config["features"]["channel_pca_dim"]),
        final_pca_dim=int(config["features"]["structured_pca_dim"]),
    )
    action_fit_order, _ = within_image_shuffle_order(
        fit_rows.image_ids, seed=int(config["controls"]["action_shuffle_seed"])
    )
    action_tune_order, _ = within_image_shuffle_order(
        tune_rows.image_ids, seed=int(config["controls"]["action_shuffle_seed"]) + 1
    )
    fit_alignment_delta = fit_rows.action_features[
        action_fit_order.to(device), :4
    ]
    tune_alignment_delta = tune_rows.action_features[
        action_tune_order.to(device), :4
    ]
    action_fit_fraction = float(
        fit_alignment_delta.ne(fit_rows.action_features[:, :4])
        .any(dim=1)
        .float()
        .mean()
        .item()
    )
    action_tune_fraction = float(
        tune_alignment_delta.ne(tune_rows.action_features[:, :4])
        .any(dim=1)
        .float()
        .mean()
        .item()
    )
    action_fit_blocks = spatial_counterfactual_blocks(
        fit_rows.spatial_features,
        fit_rows.action_features,
        alignment_delta=fit_alignment_delta,
    )
    action_tune_blocks = spatial_counterfactual_blocks(
        tune_rows.spatial_features,
        tune_rows.action_features,
        alignment_delta=tune_alignment_delta,
    )
    action_map = fit_spatial_feature_map(
        fit_rows,
        action_fit_blocks,
        channel_pca_dim=int(config["features"]["channel_pca_dim"]),
        final_pca_dim=int(config["features"]["structured_pca_dim"]),
    )
    control_fractions = {
        "spatial_inner_fit": spatial_fit_fraction,
        "spatial_inner_tune": spatial_tune_fraction,
        "action_inner_fit": action_fit_fraction,
        "action_inner_tune": action_tune_fraction,
    }
    if min(control_fractions.values()) < float(
        config["controls"]["min_changed_fraction"]
    ):
        raise ValueError("spatial/action control coverage below locked minimum")

    arm_features = {
        "structured_full": (
            transform_spatial_features(fit_rows, structured_fit_blocks, structured_map),
            transform_spatial_features(tune_rows, structured_tune_blocks, structured_map),
        ),
        "global_pooled": (
            transform_spatial_features(fit_rows, global_fit_blocks, global_map),
            transform_spatial_features(tune_rows, global_tune_blocks, global_map),
        ),
        "spatial_layout_shuffle": (
            transform_spatial_features(fit_rows, layout_fit_blocks, layout_map),
            transform_spatial_features(tune_rows, layout_tune_blocks, layout_map),
        ),
        "delta_alignment_shuffle": (
            transform_spatial_features(fit_rows, action_fit_blocks, action_map),
            transform_spatial_features(tune_rows, action_tune_blocks, action_map),
        ),
    }
    models: dict[str, Any] = {}
    selection: dict[str, Any] = {}
    for arm_value in config["arms"]:
        arm = str(arm_value)
        models[arm], selection[arm] = linear._fit_arm(
            v1=v1,
            config=config,
            fit_features=arm_features[arm][0],
            tune_features=arm_features[arm][1],
            fit_target=fit_rows.target,
            fit_evaluation_target=fit_rows.target,
            tune_selection_target=tune_rows.target,
            tune_evaluation_target=tune_rows.target,
            fit_image_ids=fit_rows.image_ids,
            tune_image_ids=tune_rows.image_ids,
        )

    outer_rows = move_spatial_candidate_rows(
        extract_spatial_candidate_rows(outer_heldout, num_classes=11), device
    )
    structured_outer_blocks = spatial_counterfactual_blocks(
        outer_rows.spatial_features, outer_rows.action_features
    )
    global_outer_blocks = outer_rows.spatial_features.mean(dim=(-2, -1))[:, None, :]
    shuffled_outer_spatial, spatial_outer_fraction = spatial_layout_shuffle(
        outer_rows.spatial_features,
        seed=int(config["controls"]["spatial_shuffle_seed"]) + 2,
    )
    layout_outer_blocks = spatial_counterfactual_blocks(
        shuffled_outer_spatial, outer_rows.action_features
    )
    action_outer_order, _ = within_image_shuffle_order(
        outer_rows.image_ids, seed=int(config["controls"]["action_shuffle_seed"]) + 2
    )
    outer_alignment_delta = outer_rows.action_features[
        action_outer_order.to(device), :4
    ]
    action_outer_fraction = float(
        outer_alignment_delta.ne(outer_rows.action_features[:, :4])
        .any(dim=1)
        .float()
        .mean()
        .item()
    )
    action_outer_blocks = spatial_counterfactual_blocks(
        outer_rows.spatial_features,
        outer_rows.action_features,
        alignment_delta=outer_alignment_delta,
    )
    control_fractions.update(
        {
            "spatial_outer": spatial_outer_fraction,
            "action_outer": action_outer_fraction,
        }
    )
    outer_features = {
        "structured_full": transform_spatial_features(
            outer_rows, structured_outer_blocks, structured_map
        ),
        "global_pooled": transform_spatial_features(
            outer_rows, global_outer_blocks, global_map
        ),
        "spatial_layout_shuffle": transform_spatial_features(
            outer_rows, layout_outer_blocks, layout_map
        ),
        "delta_alignment_shuffle": transform_spatial_features(
            outer_rows, action_outer_blocks, action_map
        ),
    }
    heldout_arms: dict[str, Any] = {}
    for arm_value in config["arms"]:
        arm = str(arm_value)
        prediction = predict_ridge_regression(models[arm], outer_features[arm])
        threshold = selection[arm]["tune_calibration"]["threshold"]
        frozen_threshold = float("inf") if threshold is None else float(threshold)
        heldout_arms[arm] = {
            "raw": linear._raw_metrics(
                v1, prediction, outer_rows.target, outer_rows.image_ids, config
            ),
            "frozen_threshold": calibrated_selection_metrics(
                prediction,
                outer_rows.target,
                outer_rows.image_ids,
                threshold=frozen_threshold,
                target_epsilon=float(config["calibration"]["target_epsilon"]),
                lcb_z=float(config["calibration"]["lcb_z"]),
            ),
        }
    heldout = {
        "support": {
            "candidate_count": int(outer_rows.target.numel()),
            "image_count": int(torch.unique(outer_rows.image_ids).numel()),
            "manifest_image_count": len(outer_heldout),
        },
        "baselines": constant_utility_baselines(
            outer_rows.target,
            outer_rows.image_ids,
            target_epsilon=float(config["calibration"]["target_epsilon"]),
        ),
        "arms": heldout_arms,
    }
    gates = evaluate_spatial_counterfactual_gates(
        {"heldout": heldout, "selection": selection}, config
    )
    checkpoint_path = run_dir / "selected_spatial_models.pt"
    torch.save(
        {
            "version_id": config["version_id"],
            "config_sha256": LOCKED_CONFIG_SHA256,
            "feature_maps": {
                "structured_full": _json_value(asdict(structured_map)),
                "global_pooled": _json_value(asdict(global_map)),
                "spatial_layout_shuffle": _json_value(asdict(layout_map)),
                "delta_alignment_shuffle": _json_value(asdict(action_map)),
            },
            "models": {arm: _json_value(asdict(model)) for arm, model in models.items()},
        },
        checkpoint_path,
    )
    git_status = _git_value("status", "--porcelain")
    payload = {
        "completed": True,
        "scientific_status": "probe_passed" if gates["all_passed"] else "probe_failed",
        "version_id": config["version_id"],
        "claim_boundary": config["probe"]["claim_boundary"],
        "inputs": {
            "code_commit": _git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_status),
            "config_sha256": LOCKED_CONFIG_SHA256,
            "source_train_cache_sha256": config["source_train_cache"]["sha256"],
            "inner_fit_manifest": _manifest(inner_fit),
            "inner_tune_manifest": _manifest(inner_tune),
            "outer_heldout_manifest": _manifest(outer_heldout),
            "device": str(device),
        },
        "protocol": {
            "researcher_adaptive": True,
            "selection_scope": "inner_tune_only",
            "outer_evaluated_after_selection": True,
            "control_changed_fractions": control_fractions,
        },
        "selection": selection,
        "heldout": heldout,
        "gates": gates,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
    }
    result_path = run_dir / "eval_metrics.json"
    result_path.write_text(
        json.dumps(_json_value(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
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
