"""Nested train-cache-only linear identifiability probe for NWPU Delta-U."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
V1_SCRIPT = ROOT / "scripts" / "probe_nwpu_joint_delta_u.py"
V2_SCRIPT = ROOT / "scripts" / "probe_nwpu_joint_delta_u_v2.py"
LOCKED_CONFIG = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.linear_identifiability.001.json"
)
LOCKED_CONFIG_SHA256 = "2de392baf1effd2b9fce7cf9463fef1e844a62b97bce76b8ee0928d49719686a"
DEFAULT_RUN_DIR = ROOT / "runs" / "nwpu_linear_identifiability_s42_nested"


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
        raise ValueError("locked linear identifiability config hash mismatch")
    return json.loads(resolved.read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(LOCKED_CONFIG))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--device")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def select_ridge_candidate(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("ridge selection requires candidates")

    def rank(candidate: dict[str, Any]) -> tuple[float, ...]:
        feasible = not bool(candidate["used_identity_fallback"])
        tune_raw = candidate["tune_raw"]
        lcb = float(candidate["tune_calibration"]["metrics"]["mean_delta_u_lcb"])
        return (
            float(feasible),
            lcb if feasible else float("-inf"),
            min(
                float(tune_raw["pairwise_accuracy"]),
                float(tune_raw["pairwise_accuracy_candidate_weighted"]),
            ),
            -float(tune_raw["mae"]),
            float(candidate["l2"]),
        )

    return max(candidates, key=rank)


def _load_script(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _calibration_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    calibration = config["calibration"]
    return {
        "target_epsilon": float(calibration["target_epsilon"]),
        "min_selected_actions": int(calibration["min_selected_actions"]),
        "max_action_image_rate": float(calibration["max_action_image_rate"]),
        "min_positive_precision_lift": float(
            calibration["min_positive_precision_lift"]
        ),
        "min_mean_delta_u_lcb": float(calibration["min_mean_delta_u_lcb"]),
        "lcb_z": float(calibration["lcb_z"]),
    }


def _calibration_payload(calibrated: Any) -> dict[str, Any]:
    threshold = float(calibrated.threshold)
    return {
        "threshold": threshold if math.isfinite(threshold) else None,
        "used_identity_fallback": bool(calibrated.used_identity_fallback),
        "candidates_evaluated": int(calibrated.candidates_evaluated),
        "metrics": _json_value(calibrated.metrics),
    }


def _raw_metrics(
    v1: Any,
    predicted: torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return v1.probe_metrics(
        predicted,
        target,
        image_ids,
        action_budget=1,
        target_epsilon=float(config["calibration"]["target_epsilon"]),
    )


def _fit_arm(
    *,
    v1: Any,
    config: Mapping[str, Any],
    fit_features: torch.Tensor,
    tune_features: torch.Tensor,
    fit_target: torch.Tensor,
    fit_evaluation_target: torch.Tensor,
    tune_selection_target: torch.Tensor,
    tune_evaluation_target: torch.Tensor,
    fit_image_ids: torch.Tensor,
    tune_image_ids: torch.Tensor,
) -> tuple[Any, dict[str, Any]]:
    from spectral_detection_posttrain.methods.energy_transport import (
        calibrate_conservative_threshold,
    )
    from spectral_detection_posttrain.methods.energy_transport.linear_identifiability import (
        fit_ridge_regression,
        image_balanced_weights,
        predict_ridge_regression,
    )

    fit_weights = image_balanced_weights(fit_image_ids)
    candidates: list[dict[str, Any]] = []
    for l2_value in config["ridge"]["lambdas"]:
        model = fit_ridge_regression(
            fit_features,
            fit_target,
            sample_weights=fit_weights,
            l2=float(l2_value),
        )
        fit_prediction = predict_ridge_regression(model, fit_features)
        tune_prediction = predict_ridge_regression(model, tune_features)
        selection_raw = _raw_metrics(
            v1,
            tune_prediction,
            tune_selection_target,
            tune_image_ids,
            config,
        )
        calibrated = calibrate_conservative_threshold(
            tune_prediction,
            tune_selection_target,
            tune_image_ids,
            **_calibration_kwargs(config),
        )
        candidates.append(
            {
                "l2": float(l2_value),
                "model": model,
                "used_identity_fallback": bool(calibrated.used_identity_fallback),
                "fit_raw": _raw_metrics(
                    v1,
                    fit_prediction,
                    fit_evaluation_target,
                    fit_image_ids,
                    config,
                ),
                "tune_raw": selection_raw,
                "tune_raw_real_target": _raw_metrics(
                    v1,
                    tune_prediction,
                    tune_evaluation_target,
                    tune_image_ids,
                    config,
                ),
                "tune_calibration": _calibration_payload(calibrated),
            }
        )
    selected = select_ridge_candidate(candidates)
    summary = {
        "selected_l2": selected["l2"],
        "used_identity_fallback": selected["used_identity_fallback"],
        "fit_raw": selected["fit_raw"],
        "tune_raw": selected["tune_raw_real_target"],
        "tune_selection_raw": selected["tune_raw"],
        "tune_calibration": selected["tune_calibration"],
        "sweep": [
            {key: value for key, value in candidate.items() if key != "model"}
            for candidate in candidates
        ],
    }
    return selected["model"], summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_locked_config(args.config)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.require_clean_git and _git_value("status", "--porcelain"):
        raise RuntimeError("--require-clean-git requested but repository is dirty")

    v1 = _load_script(V1_SCRIPT, "linear_identifiability_v1")
    v2 = _load_script(V2_SCRIPT, "linear_identifiability_v2")
    source_cache = (ROOT / config["source_train_cache"]["path"]).resolve()
    source_metadata_path = (ROOT / config["source_train_cache"]["metadata"]).resolve()
    source_v2_config = (ROOT / config["source_v2_config"]["path"]).resolve()
    v2._verify_file(
        source_cache, config["source_train_cache"]["sha256"], "source train cache"
    )
    v2._verify_file(
        source_metadata_path,
        config["source_train_cache"]["metadata_sha256"],
        "source train cache metadata",
    )
    v2._verify_file(
        source_v2_config,
        config["source_v2_config"]["sha256"],
        "source v2 config",
    )
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if source_metadata.get("pairing") != config["source_train_cache"]["pairing"]:
        raise ValueError("source cache pairing mismatch")
    expected_train_manifest = {
        "count": int(config["dataset"]["train_images"]),
        "image_ids_sha256": config["dataset"]["train_image_ids_sha256"],
    }
    if source_metadata.get("train_manifest") != expected_train_manifest:
        raise ValueError("source cache train manifest mismatch")
    records = v1.read_probe_cache(source_cache)
    if _manifest(records) != expected_train_manifest:
        raise ValueError("source cache records do not match locked train manifest")

    outer_fit_records, outer_heldout_records = v2.split_train_records(
        records,
        calibration_fraction=float(config["split"]["outer_heldout_fraction"]),
        seed=int(config["split"]["outer_seed"]),
    )
    if _manifest(outer_fit_records) != config["split"]["outer_fit_manifest"]:
        raise ValueError("outer fit manifest mismatch")
    if _manifest(outer_heldout_records) != config["split"]["outer_heldout_manifest"]:
        raise ValueError("outer heldout manifest mismatch")
    inner_fit_records, inner_tune_records = v2.split_train_records(
        outer_fit_records,
        calibration_fraction=float(config["split"]["inner_tune_fraction"]),
        seed=int(config["split"]["inner_seed"]),
    )

    from spectral_detection_posttrain.methods.energy_transport import (
        calibrated_selection_metrics,
        constant_utility_baselines,
    )
    from spectral_detection_posttrain.methods.energy_transport.linear_identifiability import (
        evaluate_identifiability_gates,
        extract_local_candidate_rows,
        fit_local_feature_map,
        move_local_candidate_rows,
        predict_ridge_regression,
        transform_local_candidate_rows,
        within_image_shuffle_order,
    )
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

    set_seed(int(config["dataset"]["seed"]))
    device = resolve_device(
        {"device": args.device or ("cuda" if torch.cuda.is_available() else "cpu")}
    )
    v1.validate_deterministic_cuda_environment(device)
    torch.use_deterministic_algorithms(True, warn_only=True)
    inner_fit_rows = move_local_candidate_rows(
        extract_local_candidate_rows(inner_fit_records, num_classes=11), device
    )
    inner_tune_rows = move_local_candidate_rows(
        extract_local_candidate_rows(inner_tune_records, num_classes=11), device
    )
    feature_map = fit_local_feature_map(
        inner_fit_rows, pca_dim=int(config["features"]["roi_pca_dim"])
    )
    full_fit_features = transform_local_candidate_rows(inner_fit_rows, feature_map)
    full_tune_features = transform_local_candidate_rows(inner_tune_rows, feature_map)

    feature_fit_order, feature_fit_fraction = within_image_shuffle_order(
        inner_fit_rows.image_ids,
        seed=int(config["controls"]["feature_shuffle_seed"]),
    )
    feature_tune_order, feature_tune_fraction = within_image_shuffle_order(
        inner_tune_rows.image_ids,
        seed=int(config["controls"]["feature_shuffle_seed"]) + 1,
    )
    label_fit_order, label_fit_fraction = within_image_shuffle_order(
        inner_fit_rows.image_ids,
        seed=int(config["controls"]["label_shuffle_seed"]),
    )
    label_tune_order, label_tune_fraction = within_image_shuffle_order(
        inner_tune_rows.image_ids,
        seed=int(config["controls"]["label_shuffle_seed"]) + 1,
    )
    control_fractions = {
        "feature_inner_fit": feature_fit_fraction,
        "feature_inner_tune": feature_tune_fraction,
        "label_inner_fit": label_fit_fraction,
        "label_inner_tune": label_tune_fraction,
    }
    if min(control_fractions.values()) < float(
        config["controls"]["min_changed_row_fraction"]
    ):
        raise ValueError("within-image control coverage is below locked minimum")
    feature_fit_order = feature_fit_order.to(device)
    feature_tune_order = feature_tune_order.to(device)
    label_fit_order = label_fit_order.to(device)
    label_tune_order = label_tune_order.to(device)

    arm_inputs = {
        "local_full": {
            "fit_features": full_fit_features,
            "tune_features": full_tune_features,
            "fit_target": inner_fit_rows.target,
            "tune_selection_target": inner_tune_rows.target,
        },
        "within_image_feature_shuffle": {
            "fit_features": full_fit_features[feature_fit_order],
            "tune_features": full_tune_features[feature_tune_order],
            "fit_target": inner_fit_rows.target,
            "tune_selection_target": inner_tune_rows.target,
        },
        "within_image_label_shuffle": {
            "fit_features": full_fit_features,
            "tune_features": full_tune_features,
            "fit_target": inner_fit_rows.target[label_fit_order],
            "tune_selection_target": inner_tune_rows.target[label_tune_order],
        },
    }
    models: dict[str, Any] = {}
    selection: dict[str, Any] = {}
    for arm_value in config["arms"]:
        arm = str(arm_value)
        models[arm], selection[arm] = _fit_arm(
            v1=v1,
            config=config,
            fit_features=arm_inputs[arm]["fit_features"],
            tune_features=arm_inputs[arm]["tune_features"],
            fit_target=arm_inputs[arm]["fit_target"],
            fit_evaluation_target=inner_fit_rows.target,
            tune_selection_target=arm_inputs[arm]["tune_selection_target"],
            tune_evaluation_target=inner_tune_rows.target,
            fit_image_ids=inner_fit_rows.image_ids,
            tune_image_ids=inner_tune_rows.image_ids,
        )

    outer_rows = move_local_candidate_rows(
        extract_local_candidate_rows(outer_heldout_records, num_classes=11), device
    )
    full_outer_features = transform_local_candidate_rows(outer_rows, feature_map)
    feature_outer_order, feature_outer_fraction = within_image_shuffle_order(
        outer_rows.image_ids,
        seed=int(config["controls"]["feature_shuffle_seed"]) + 2,
    )
    feature_outer_order = feature_outer_order.to(device)
    control_fractions["feature_outer_heldout"] = feature_outer_fraction
    if feature_outer_fraction < float(config["controls"]["min_changed_row_fraction"]):
        raise ValueError("outer heldout feature control coverage is below locked minimum")
    outer_features = {
        "local_full": full_outer_features,
        "within_image_feature_shuffle": full_outer_features[feature_outer_order],
        "within_image_label_shuffle": full_outer_features,
    }
    heldout_arms: dict[str, Any] = {}
    for arm_value in config["arms"]:
        arm = str(arm_value)
        prediction = predict_ridge_regression(models[arm], outer_features[arm])
        threshold = selection[arm]["tune_calibration"]["threshold"]
        frozen_threshold = float("inf") if threshold is None else float(threshold)
        heldout_arms[arm] = {
            "raw": _raw_metrics(
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
            "unconditional_top1": calibrated_selection_metrics(
                prediction,
                outer_rows.target,
                outer_rows.image_ids,
                threshold=float("-inf"),
                target_epsilon=float(config["calibration"]["target_epsilon"]),
                lcb_z=float(config["calibration"]["lcb_z"]),
            ),
        }

    heldout = {
        "heldout_evaluated_once": True,
        "researcher_seen_split": True,
        "support": {
            "candidate_count": int(outer_rows.target.numel()),
            "image_count": int(torch.unique(outer_rows.image_ids).numel()),
            "manifest_image_count": int(len(outer_heldout_records)),
        },
        "baselines": constant_utility_baselines(
            outer_rows.target,
            outer_rows.image_ids,
            target_epsilon=float(config["calibration"]["target_epsilon"]),
        ),
        "arms": heldout_arms,
    }
    gate_payload = {"heldout": heldout, "selection": selection}
    gates = evaluate_identifiability_gates(gate_payload, config)

    checkpoint_path = run_dir / "selected_linear_models.pt"
    torch.save(
        {
            "version_id": config["version_id"],
            "config_sha256": LOCKED_CONFIG_SHA256,
            "feature_map": _json_value(asdict(feature_map)),
            "models": {
                arm: _json_value(asdict(model)) for arm, model in models.items()
            },
            "selection": {
                arm: {
                    "selected_l2": result["selected_l2"],
                    "threshold": result["tune_calibration"]["threshold"],
                }
                for arm, result in selection.items()
            },
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
            "config": str(Path(args.config).resolve()),
            "config_sha256": LOCKED_CONFIG_SHA256,
            "source_train_cache": str(source_cache),
            "source_train_cache_sha256": config["source_train_cache"]["sha256"],
            "outer_fit_manifest": _manifest(outer_fit_records),
            "inner_fit_manifest": _manifest(inner_fit_records),
            "inner_tune_manifest": _manifest(inner_tune_records),
            "outer_heldout_manifest": _manifest(outer_heldout_records),
            "device": str(device),
            "torch_version": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "protocol": {
            "model_selection_scope": "inner_tune_only",
            "heldout_evaluated_once": True,
            "heldout_used_for_model_or_threshold_selection": False,
            "per_image_regression_weighting": True,
            "control_changed_row_fractions": control_fractions,
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
