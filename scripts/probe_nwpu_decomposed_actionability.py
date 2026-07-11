"""Nested train-cache probe for decomposed sign, rank, and abstention."""

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
    / "det.energy.decomposed_actionability.001.json"
)
LOCKED_CONFIG_SHA256 = "c059ef07b8a8edc8e4f8854b203672d340b68f09260966b275b05aaa2fdf51b8"
DEFAULT_RUN_DIR = ROOT / "runs" / "nwpu_decomposed_actionability_s42_nested"


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
        raise ValueError("locked decomposed actionability config hash mismatch")
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sign_target(target: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.where(target > float(epsilon), 1.0, -1.0)


def _rank_metrics(
    v1: Any,
    prediction: torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return v1.probe_metrics(
        prediction,
        target,
        image_ids,
        action_budget=1,
        target_epsilon=float(config["targets"]["target_epsilon"]),
    )


def fit_sign_head(
    *,
    config: Mapping[str, Any],
    fit_features: torch.Tensor,
    tune_features: torch.Tensor,
    fit_selection_target: torch.Tensor,
    tune_selection_target: torch.Tensor,
    fit_real_target: torch.Tensor,
    tune_real_target: torch.Tensor,
    fit_image_ids: torch.Tensor,
    tune_image_ids: torch.Tensor,
) -> tuple[Any, dict[str, Any]]:
    from spectral_detection_posttrain.methods.energy_transport.decomposed_actionability import (
        class_image_balanced_weights,
        sign_classification_metrics,
    )
    from spectral_detection_posttrain.methods.energy_transport.linear_identifiability import (
        fit_ridge_regression,
        predict_ridge_regression,
    )

    epsilon = float(config["targets"]["target_epsilon"])
    fit_sign_target = _sign_target(fit_selection_target, epsilon)
    fit_weights = class_image_balanced_weights(fit_sign_target, fit_image_ids)
    candidates: list[dict[str, Any]] = []
    for l2_value in config["ridge"]["sign_lambdas"]:
        model = fit_ridge_regression(
            fit_features,
            fit_sign_target,
            sample_weights=fit_weights,
            l2=float(l2_value),
        )
        fit_scores = predict_ridge_regression(model, fit_features)
        tune_scores = predict_ridge_regression(model, tune_features)
        selection_metrics = sign_classification_metrics(
            tune_scores,
            tune_selection_target,
            tune_image_ids,
            target_epsilon=epsilon,
        )
        candidates.append(
            {
                "l2": float(l2_value),
                "model": model,
                "fit_real": sign_classification_metrics(
                    fit_scores,
                    fit_real_target,
                    fit_image_ids,
                    target_epsilon=epsilon,
                ),
                "tune_real": sign_classification_metrics(
                    tune_scores,
                    tune_real_target,
                    tune_image_ids,
                    target_epsilon=epsilon,
                ),
                "tune_selection": selection_metrics,
            }
        )
    selected = max(
        candidates,
        key=lambda item: (
            min(
                float(item["tune_selection"]["auroc_equal_image"]),
                float(item["tune_selection"]["auroc_candidate_weighted"]),
            ),
            float(item["tune_selection"]["balanced_accuracy"]),
            float(item["l2"]),
        ),
    )
    return selected["model"], {
        "selected_l2": selected["l2"],
        "fit_sign": selected["fit_real"],
        "tune_sign": selected["tune_real"],
        "tune_selection_sign": selected["tune_selection"],
        "sign_sweep": [
            {key: value for key, value in candidate.items() if key != "model"}
            for candidate in candidates
        ],
    }


def fit_rank_head(
    *,
    v1: Any,
    config: Mapping[str, Any],
    fit_features: torch.Tensor,
    tune_features: torch.Tensor,
    fit_selection_target: torch.Tensor,
    tune_selection_target: torch.Tensor,
    fit_real_target: torch.Tensor,
    tune_real_target: torch.Tensor,
    fit_image_ids: torch.Tensor,
    tune_image_ids: torch.Tensor,
) -> tuple[Any, dict[str, Any]]:
    from spectral_detection_posttrain.methods.energy_transport.decomposed_actionability import (
        within_image_pairwise_rows,
    )
    from spectral_detection_posttrain.methods.energy_transport.linear_identifiability import (
        fit_ridge_regression,
        image_balanced_weights,
        predict_ridge_regression,
    )

    pair_rows = within_image_pairwise_rows(
        fit_features,
        fit_selection_target,
        fit_image_ids,
        min_target_gap=float(config["targets"]["rank_min_target_gap"]),
    )
    pair_weights = image_balanced_weights(pair_rows.image_ids)
    candidates: list[dict[str, Any]] = []
    for l2_value in config["ridge"]["rank_lambdas"]:
        model = fit_ridge_regression(
            pair_rows.features,
            pair_rows.target,
            sample_weights=pair_weights,
            l2=float(l2_value),
        )
        fit_scores = predict_ridge_regression(model, fit_features)
        tune_scores = predict_ridge_regression(model, tune_features)
        selection_metrics = _rank_metrics(
            v1, tune_scores, tune_selection_target, tune_image_ids, config
        )
        candidates.append(
            {
                "l2": float(l2_value),
                "model": model,
                "fit_real": _rank_metrics(
                    v1, fit_scores, fit_real_target, fit_image_ids, config
                ),
                "tune_real": _rank_metrics(
                    v1, tune_scores, tune_real_target, tune_image_ids, config
                ),
                "tune_selection": selection_metrics,
            }
        )
    selected = max(
        candidates,
        key=lambda item: (
            min(
                float(item["tune_selection"]["pairwise_accuracy"]),
                float(
                    item["tune_selection"]["pairwise_accuracy_candidate_weighted"]
                ),
            ),
            float(item["l2"]),
        ),
    )
    return selected["model"], {
        "selected_l2": selected["l2"],
        "fit_rank": selected["fit_real"],
        "tune_rank": selected["tune_real"],
        "tune_selection_rank": selected["tune_selection"],
        "fit_pair_count": int(pair_rows.target.numel()),
        "rank_sweep": [
            {key: value for key, value in candidate.items() if key != "model"}
            for candidate in candidates
        ],
    }


def _fit_arm(
    *,
    v1: Any,
    config: Mapping[str, Any],
    fit_features: torch.Tensor,
    tune_features: torch.Tensor,
    fit_selection_target: torch.Tensor,
    tune_selection_target: torch.Tensor,
    fit_real_target: torch.Tensor,
    tune_real_target: torch.Tensor,
    fit_image_ids: torch.Tensor,
    tune_image_ids: torch.Tensor,
) -> tuple[Any, Any, float, dict[str, Any]]:
    from spectral_detection_posttrain.methods.energy_transport.decomposed_actionability import (
        calibrate_ranked_abstention,
    )
    from spectral_detection_posttrain.methods.energy_transport.linear_identifiability import (
        predict_ridge_regression,
    )

    sign_model, sign_summary = fit_sign_head(
        config=config,
        fit_features=fit_features,
        tune_features=tune_features,
        fit_selection_target=fit_selection_target,
        tune_selection_target=tune_selection_target,
        fit_real_target=fit_real_target,
        tune_real_target=tune_real_target,
        fit_image_ids=fit_image_ids,
        tune_image_ids=tune_image_ids,
    )
    rank_model, rank_summary = fit_rank_head(
        v1=v1,
        config=config,
        fit_features=fit_features,
        tune_features=tune_features,
        fit_selection_target=fit_selection_target,
        tune_selection_target=tune_selection_target,
        fit_real_target=fit_real_target,
        tune_real_target=tune_real_target,
        fit_image_ids=fit_image_ids,
        tune_image_ids=tune_image_ids,
    )
    tune_sign_scores = predict_ridge_regression(sign_model, tune_features)
    tune_rank_scores = predict_ridge_regression(rank_model, tune_features)
    calibration = calibrate_ranked_abstention(
        tune_rank_scores,
        tune_sign_scores,
        tune_selection_target,
        tune_image_ids,
        target_epsilon=float(config["targets"]["target_epsilon"]),
        min_selected_actions=int(config["calibration"]["min_selected_actions"]),
        max_action_image_rate=float(config["calibration"]["max_action_image_rate"]),
        min_positive_precision_lift=float(
            config["calibration"]["min_positive_precision_lift"]
        ),
        min_mean_delta_u_lcb=float(
            config["calibration"]["min_mean_delta_u_lcb"]
        ),
        lcb_z=float(config["calibration"]["lcb_z"]),
    )
    threshold = float(calibration["sign_threshold"])
    summary = {
        **sign_summary,
        **rank_summary,
        "used_identity_fallback": bool(calibration["used_identity_fallback"]),
        "tune_abstention_calibration": calibration,
    }
    return sign_model, rank_model, threshold, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_locked_config(args.config)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.require_clean_git and _git_value("status", "--porcelain"):
        raise RuntimeError("--require-clean-git requested but repository is dirty")
    v1 = _load_script(V1_SCRIPT, "decomposed_v1")
    v2 = _load_script(V2_SCRIPT, "decomposed_v2")
    source_cache = (ROOT / config["source_train_cache"]["path"]).resolve()
    source_metadata = (ROOT / config["source_train_cache"]["metadata"]).resolve()
    v2._verify_file(source_cache, config["source_train_cache"]["sha256"], "source cache")
    v2._verify_file(
        source_metadata,
        config["source_train_cache"]["metadata_sha256"],
        "source metadata",
    )
    metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    if metadata.get("pairing") != config["source_train_cache"]["pairing"]:
        raise ValueError("source cache pairing mismatch")
    records = v1.read_probe_cache(source_cache)
    expected_manifest = {
        "count": int(config["dataset"]["train_images"]),
        "image_ids_sha256": config["dataset"]["train_image_ids_sha256"],
    }
    if _manifest(records) != expected_manifest:
        raise ValueError("source cache train manifest mismatch")
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
        constant_utility_baselines,
    )
    from spectral_detection_posttrain.methods.energy_transport.decomposed_actionability import (
        evaluate_decomposed_actionability_gates,
        ranked_abstention_metrics,
        sign_classification_metrics,
        within_image_pairwise_rows,
    )
    from spectral_detection_posttrain.methods.energy_transport.linear_identifiability import (
        predict_ridge_regression,
        within_image_shuffle_order,
    )
    from spectral_detection_posttrain.methods.energy_transport.spatial_counterfactual import (
        extract_spatial_candidate_rows,
        fit_spatial_feature_map,
        move_spatial_candidate_rows,
        spatial_counterfactual_blocks,
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
    fit_blocks = spatial_counterfactual_blocks(
        fit_rows.spatial_features, fit_rows.action_features
    )
    tune_blocks = spatial_counterfactual_blocks(
        tune_rows.spatial_features, tune_rows.action_features
    )
    feature_map = fit_spatial_feature_map(
        fit_rows,
        fit_blocks,
        channel_pca_dim=int(config["features"]["channel_pca_dim"]),
        final_pca_dim=int(config["features"]["final_pca_dim"]),
    )
    fit_features = transform_spatial_features(fit_rows, fit_blocks, feature_map)
    tune_features = transform_spatial_features(tune_rows, tune_blocks, feature_map)
    if fit_features.shape[1] != int(config["features"]["final_feature_dim"]):
        raise ValueError("locked final feature dimension mismatch")

    fit_feature_order, fit_feature_fraction = within_image_shuffle_order(
        fit_rows.image_ids, seed=int(config["controls"]["feature_shuffle_seed"])
    )
    tune_feature_order, tune_feature_fraction = within_image_shuffle_order(
        tune_rows.image_ids, seed=int(config["controls"]["feature_shuffle_seed"]) + 1
    )
    fit_label_order, fit_label_fraction = within_image_shuffle_order(
        fit_rows.image_ids, seed=int(config["controls"]["label_shuffle_seed"])
    )
    tune_label_order, tune_label_fraction = within_image_shuffle_order(
        tune_rows.image_ids, seed=int(config["controls"]["label_shuffle_seed"]) + 1
    )
    fit_feature_order = fit_feature_order.to(device)
    tune_feature_order = tune_feature_order.to(device)
    fit_label_order = fit_label_order.to(device)
    tune_label_order = tune_label_order.to(device)
    control_fractions = {
        "feature_inner_fit": fit_feature_fraction,
        "feature_inner_tune": tune_feature_fraction,
        "label_inner_fit": fit_label_fraction,
        "label_inner_tune": tune_label_fraction,
    }
    if min(control_fractions.values()) < float(
        config["controls"]["min_changed_row_fraction"]
    ):
        raise ValueError("B3 shuffle control coverage below locked minimum")

    arm_inputs = {
        "local_full": (
            fit_features,
            tune_features,
            fit_rows.target,
            tune_rows.target,
        ),
        "within_image_feature_shuffle": (
            fit_features[fit_feature_order],
            tune_features[tune_feature_order],
            fit_rows.target,
            tune_rows.target,
        ),
        "within_image_label_shuffle": (
            fit_features,
            tune_features,
            fit_rows.target[fit_label_order],
            tune_rows.target[tune_label_order],
        ),
    }
    sign_models: dict[str, Any] = {}
    rank_models: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    selection: dict[str, Any] = {}
    for arm_value in config["arms"]:
        arm = str(arm_value)
        arm_fit, arm_tune, fit_selection_target, tune_selection_target = arm_inputs[arm]
        (
            sign_models[arm],
            rank_models[arm],
            thresholds[arm],
            selection[arm],
        ) = _fit_arm(
            v1=v1,
            config=config,
            fit_features=arm_fit,
            tune_features=arm_tune,
            fit_selection_target=fit_selection_target,
            tune_selection_target=tune_selection_target,
            fit_real_target=fit_rows.target,
            tune_real_target=tune_rows.target,
            fit_image_ids=fit_rows.image_ids,
            tune_image_ids=tune_rows.image_ids,
        )

    outer_rows = move_spatial_candidate_rows(
        extract_spatial_candidate_rows(outer_heldout, num_classes=11), device
    )
    outer_blocks = spatial_counterfactual_blocks(
        outer_rows.spatial_features, outer_rows.action_features
    )
    outer_features = transform_spatial_features(outer_rows, outer_blocks, feature_map)
    outer_feature_order, outer_feature_fraction = within_image_shuffle_order(
        outer_rows.image_ids, seed=int(config["controls"]["feature_shuffle_seed"]) + 2
    )
    outer_label_order, outer_label_fraction = within_image_shuffle_order(
        outer_rows.image_ids, seed=int(config["controls"]["label_shuffle_seed"]) + 2
    )
    control_fractions.update(
        {
            "feature_outer": outer_feature_fraction,
            "label_outer": outer_label_fraction,
        }
    )
    outer_arm_features = {
        "local_full": outer_features,
        "within_image_feature_shuffle": outer_features[
            outer_feature_order.to(device)
        ],
        "within_image_label_shuffle": outer_features,
    }
    heldout_arms: dict[str, Any] = {}
    epsilon = float(config["targets"]["target_epsilon"])
    for arm_value in config["arms"]:
        arm = str(arm_value)
        features = outer_arm_features[arm]
        sign_scores = predict_ridge_regression(sign_models[arm], features)
        rank_scores = predict_ridge_regression(rank_models[arm], features)
        heldout_arms[arm] = {
            "sign": sign_classification_metrics(
                sign_scores,
                outer_rows.target,
                outer_rows.image_ids,
                target_epsilon=epsilon,
            ),
            "rank": _rank_metrics(
                v1, rank_scores, outer_rows.target, outer_rows.image_ids, config
            ),
            "abstention": ranked_abstention_metrics(
                rank_scores,
                sign_scores,
                outer_rows.target,
                outer_rows.image_ids,
                sign_threshold=thresholds[arm],
                target_epsilon=epsilon,
                lcb_z=float(config["calibration"]["lcb_z"]),
            ),
        }
    outer_pairs = within_image_pairwise_rows(
        outer_features,
        outer_rows.target,
        outer_rows.image_ids,
        min_target_gap=float(config["targets"]["rank_min_target_gap"]),
    )
    heldout = {
        "support": {
            "candidate_count": int(outer_rows.target.numel()),
            "image_count": int(torch.unique(outer_rows.image_ids).numel()),
            "manifest_image_count": len(outer_heldout),
            "eligible_pair_count": int(outer_pairs.target.numel()),
        },
        "baselines": constant_utility_baselines(
            outer_rows.target, outer_rows.image_ids, target_epsilon=epsilon
        ),
        "arms": heldout_arms,
    }
    gates = evaluate_decomposed_actionability_gates(
        {"heldout": heldout, "selection": selection}, config
    )
    checkpoint_path = run_dir / "selected_decomposed_models.pt"
    torch.save(
        {
            "version_id": config["version_id"],
            "config_sha256": LOCKED_CONFIG_SHA256,
            "feature_map": _json_value(asdict(feature_map)),
            "sign_models": {
                arm: _json_value(asdict(model)) for arm, model in sign_models.items()
            },
            "rank_models": {
                arm: _json_value(asdict(model)) for arm, model in rank_models.items()
            },
            "sign_thresholds": thresholds,
        },
        checkpoint_path,
    )
    payload = {
        "completed": True,
        "scientific_status": "probe_passed" if gates["all_passed"] else "probe_failed",
        "version_id": config["version_id"],
        "claim_boundary": config["probe"]["claim_boundary"],
        "inputs": {
            "code_commit": _git_value("rev-parse", "HEAD"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "config_sha256": LOCKED_CONFIG_SHA256,
            "source_train_cache_sha256": config["source_train_cache"]["sha256"],
            "inner_fit_manifest": _manifest(inner_fit),
            "inner_tune_manifest": _manifest(inner_tune),
            "outer_heldout_manifest": _manifest(outer_heldout),
            "device": str(device),
        },
        "protocol": {
            "researcher_adaptive": True,
            "outer_split_seen_in_prior_analyses": True,
            "selection_scope": "inner_tune_only",
            "outer_evaluated_after_freeze": True,
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
