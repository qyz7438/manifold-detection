"""Nested direct listwise candidate-or-no-op diagnostic on the NWPU train cache."""

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
    / "det.energy.listwise_noop.001.json"
)
LOCKED_CONFIG_SHA256 = "3c29eaa334ef88a7ae0e9b04936ac4259c96d021055d46d0daf93c734495718a"
DEFAULT_RUN_DIR = ROOT / "runs" / "nwpu_listwise_noop_s42_reused_outer"


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
        raise ValueError("locked listwise no-op config hash mismatch")
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
    return {"count": len(image_ids), "image_ids_sha256": hashlib.sha256(encoded).hexdigest()}


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


def _feature_map_from_checkpoint(payload: dict[str, Any], device: torch.device) -> Any:
    from spectral_detection_posttrain.methods.energy_transport.spatial_counterfactual import (
        FrozenSpatialFeatureMap,
    )

    values = payload["feature_map"]
    tensors = {
        key: torch.tensor(value, device=device)
        for key, value in values.items()
        if key != "block_count"
    }
    return FrozenSpatialFeatureMap(**tensors, block_count=int(values["block_count"]))


def _fit_arm(
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
) -> tuple[Any, float, dict[str, Any]]:
    from spectral_detection_posttrain.methods.energy_transport.listwise_noop import (
        calibrate_noop_margin,
        fit_listwise_model,
        listwise_decision_metrics,
        predict_listwise_scores,
    )

    epsilon = float(config["targets"]["target_epsilon"])
    calibration_config = config["calibration"]
    candidates: list[dict[str, Any]] = []
    for l2_value in config["optimization"]["lambdas"]:
        model = fit_listwise_model(
            fit_features,
            fit_selection_target,
            fit_image_ids,
            target_epsilon=epsilon,
            l2=float(l2_value),
            max_iter=int(config["optimization"]["max_iter"]),
        )
        tune_scores = predict_listwise_scores(model, tune_features)
        calibration = calibrate_noop_margin(
            tune_scores,
            noop_logit=model.noop_logit,
            target=tune_selection_target,
            image_ids=tune_image_ids,
            target_epsilon=epsilon,
            min_selected_actions=int(calibration_config["min_selected_actions"]),
            max_action_image_rate=float(calibration_config["max_action_image_rate"]),
            min_positive_precision_lift=float(
                calibration_config["min_positive_precision_lift"]
            ),
            min_mean_delta_u_lcb=float(
                calibration_config["min_mean_delta_u_lcb"]
            ),
            lcb_z=float(calibration_config["lcb_z"]),
        )
        candidates.append(
            {"l2": float(l2_value), "model": model, "calibration": calibration}
        )
    selected = max(
        candidates,
        key=lambda item: (
            not bool(item["calibration"]["used_identity_fallback"]),
            float(item["calibration"]["metrics"]["mean_delta_u_lcb"]),
            float(item["calibration"]["metrics"]["mean_delta_u"]),
            float(item["l2"]),
        ),
    )
    model = selected["model"]
    margin = float(selected["calibration"]["noop_margin"])
    metric_kwargs = {
        "noop_logit": model.noop_logit,
        "noop_margin": margin,
        "target_epsilon": epsilon,
        "lcb_z": float(calibration_config["lcb_z"]),
    }
    fit_metrics = listwise_decision_metrics(
        predict_listwise_scores(model, fit_features),
        target=fit_real_target,
        image_ids=fit_image_ids,
        **metric_kwargs,
    )
    tune_metrics = listwise_decision_metrics(
        predict_listwise_scores(model, tune_features),
        target=tune_real_target,
        image_ids=tune_image_ids,
        **metric_kwargs,
    )
    return model, margin, {
        "selected_l2": selected["l2"],
        "used_identity_fallback": bool(selected["calibration"]["used_identity_fallback"]),
        "noop_margin": margin,
        "fit_metrics": fit_metrics,
        "tune_metrics": tune_metrics,
        "tune_selection_calibration": selected["calibration"],
        "sweep": [
            {"l2": item["l2"], "calibration": item["calibration"]}
            for item in candidates
        ],
    }


def _rate_matched_random_values(
    target: torch.Tensor, image_ids: torch.Tensor, action_rate: float
) -> list[float]:
    return [
        float(target[image_ids == image_id].mean().item()) * float(action_rate)
        for image_id in torch.unique(image_ids, sorted=True)
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_locked_config(args.config)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.require_clean_git and _git_value("status", "--porcelain"):
        raise RuntimeError("--require-clean-git requested but repository is dirty")
    source = config["source_b3"]
    source_config_path = (ROOT / source["config_path"]).resolve()
    checkpoint_path = (ROOT / source["checkpoint_path"]).resolve()
    if sha256_file(source_config_path) != source["config_sha256"]:
        raise ValueError("source B3 config hash mismatch")
    if sha256_file(checkpoint_path) != source["checkpoint_sha256"]:
        raise ValueError("source B3 checkpoint hash mismatch")
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    v1 = _load_script(V1_SCRIPT, "listwise_v1")
    v2 = _load_script(V2_SCRIPT, "listwise_v2")
    source_cache = (ROOT / source_config["source_train_cache"]["path"]).resolve()
    source_metadata = (ROOT / source_config["source_train_cache"]["metadata"]).resolve()
    v2._verify_file(source_cache, source_config["source_train_cache"]["sha256"], "source cache")
    v2._verify_file(
        source_metadata,
        source_config["source_train_cache"]["metadata_sha256"],
        "source metadata",
    )
    records = v1.read_probe_cache(source_cache)
    outer_fit, outer_heldout = v2.split_train_records(
        records,
        calibration_fraction=float(source_config["split"]["outer_heldout_fraction"]),
        seed=int(source_config["split"]["outer_seed"]),
    )
    inner_fit, inner_tune = v2.split_train_records(
        outer_fit,
        calibration_fraction=float(source_config["split"]["inner_tune_fraction"]),
        seed=int(source_config["split"]["inner_seed"]),
    )
    if _manifest(outer_heldout) != source_config["split"]["outer_heldout_manifest"]:
        raise ValueError("outer heldout manifest mismatch")

    from spectral_detection_posttrain.methods.energy_transport.linear_identifiability import (
        within_image_shuffle_order,
    )
    from spectral_detection_posttrain.methods.energy_transport.listwise_noop import (
        evaluate_listwise_gates,
        listwise_decision_metrics,
        paired_gain_stats,
        predict_listwise_scores,
    )
    from spectral_detection_posttrain.methods.energy_transport.spatial_counterfactual import (
        extract_spatial_candidate_rows,
        move_spatial_candidate_rows,
        spatial_counterfactual_blocks,
        transform_spatial_features,
    )
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

    set_seed(int(source_config["dataset"]["seed"]))
    device = resolve_device(
        {"device": args.device or ("cuda" if torch.cuda.is_available() else "cpu")}
    )
    v1.validate_deterministic_cuda_environment(device)
    torch.use_deterministic_algorithms(True, warn_only=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    feature_map = _feature_map_from_checkpoint(checkpoint, device)

    def rows_and_features(record_split: Any) -> tuple[Any, torch.Tensor]:
        rows = move_spatial_candidate_rows(
            extract_spatial_candidate_rows(record_split, num_classes=11), device
        )
        blocks = spatial_counterfactual_blocks(rows.spatial_features, rows.action_features)
        return rows, transform_spatial_features(rows, blocks, feature_map)

    fit_rows, fit_features = rows_and_features(inner_fit)
    tune_rows, tune_features = rows_and_features(inner_tune)
    feature_seed = int(config["controls"]["feature_shuffle_seed"])
    label_seed = int(config["controls"]["label_shuffle_seed"])
    fit_feature_order, fit_feature_fraction = within_image_shuffle_order(
        fit_rows.image_ids, seed=feature_seed
    )
    tune_feature_order, tune_feature_fraction = within_image_shuffle_order(
        tune_rows.image_ids, seed=feature_seed + 1
    )
    fit_label_order, fit_label_fraction = within_image_shuffle_order(
        fit_rows.image_ids, seed=label_seed
    )
    tune_label_order, tune_label_fraction = within_image_shuffle_order(
        tune_rows.image_ids, seed=label_seed + 1
    )
    changed_fractions = {
        "feature_inner_fit": fit_feature_fraction,
        "feature_inner_tune": tune_feature_fraction,
        "label_inner_fit": fit_label_fraction,
        "label_inner_tune": tune_label_fraction,
    }
    if min(changed_fractions.values()) < float(config["controls"]["min_changed_row_fraction"]):
        raise ValueError("listwise control coverage below minimum")
    arm_inputs = {
        "local_full": (fit_features, tune_features, fit_rows.target, tune_rows.target),
        "within_image_feature_shuffle": (
            fit_features[fit_feature_order.to(device)],
            tune_features[tune_feature_order.to(device)],
            fit_rows.target,
            tune_rows.target,
        ),
        "within_image_label_shuffle": (
            fit_features,
            tune_features,
            fit_rows.target[fit_label_order.to(device)],
            tune_rows.target[tune_label_order.to(device)],
        ),
    }
    models: dict[str, Any] = {}
    margins: dict[str, float] = {}
    selection: dict[str, Any] = {}
    for arm_value in config["arms"]:
        arm = str(arm_value)
        arm_fit, arm_tune, fit_selection_target, tune_selection_target = arm_inputs[arm]
        models[arm], margins[arm], selection[arm] = _fit_arm(
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

    outer_rows, outer_features = rows_and_features(outer_heldout)
    outer_feature_order, outer_feature_fraction = within_image_shuffle_order(
        outer_rows.image_ids, seed=feature_seed + 2
    )
    if outer_feature_fraction < float(config["controls"]["min_changed_row_fraction"]):
        raise ValueError("outer feature control coverage below minimum")
    changed_fractions["feature_outer"] = outer_feature_fraction
    outer_arm_features = {
        "local_full": outer_features,
        "within_image_feature_shuffle": outer_features[outer_feature_order.to(device)],
        "within_image_label_shuffle": outer_features,
    }
    arms: dict[str, Any] = {}
    epsilon = float(config["targets"]["target_epsilon"])
    lcb_z = float(config["calibration"]["lcb_z"])
    for arm_value in config["arms"]:
        arm = str(arm_value)
        scores = predict_listwise_scores(models[arm], outer_arm_features[arm])
        arms[arm] = listwise_decision_metrics(
            scores,
            noop_logit=models[arm].noop_logit,
            noop_margin=margins[arm],
            target=outer_rows.target,
            image_ids=outer_rows.image_ids,
            target_epsilon=epsilon,
            lcb_z=lcb_z,
        )
    full_values = arms["local_full"]["per_image_delta_u"]
    random_values = _rate_matched_random_values(
        outer_rows.target,
        outer_rows.image_ids,
        float(arms["local_full"]["action_image_rate"]),
    )
    paired_gains = {
        "vs_rate_matched_random": paired_gain_stats(full_values, random_values, lcb_z=lcb_z),
        "vs_feature_shuffle": paired_gain_stats(
            full_values,
            arms["within_image_feature_shuffle"]["per_image_delta_u"],
            lcb_z=lcb_z,
        ),
        "vs_label_shuffle": paired_gain_stats(
            full_values,
            arms["within_image_label_shuffle"]["per_image_delta_u"],
            lcb_z=lcb_z,
        ),
    }
    oracle_values = [
        max(0.0, float(outer_rows.target[outer_rows.image_ids == image_id].max().item()))
        for image_id in torch.unique(outer_rows.image_ids, sorted=True)
    ]
    audit = {
        "support": {
            "candidate_count": int(outer_rows.target.numel()),
            "image_count": int(torch.unique(outer_rows.image_ids).numel()),
            "manifest_image_count": len(outer_heldout),
        },
        "arms": arms,
        "paired_gains": paired_gains,
        "baselines": {
            "always_noop_mean_delta_u": 0.0,
            "rate_matched_random_mean_delta_u": sum(random_values) / len(random_values),
            "oracle_top_mean_delta_u": sum(oracle_values) / len(oracle_values),
        },
    }
    gates = evaluate_listwise_gates({**audit, "selection": selection}, config)
    model_path = run_dir / "selected_listwise_models.pt"
    torch.save(
        {
            "version_id": config["version_id"],
            "config_sha256": LOCKED_CONFIG_SHA256,
            "models": {arm: _json_value(asdict(model)) for arm, model in models.items()},
            "noop_margins": margins,
        },
        model_path,
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
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "inner_fit_manifest": _manifest(inner_fit),
            "inner_tune_manifest": _manifest(inner_tune),
            "outer_heldout_manifest": _manifest(outer_heldout),
            "device": str(device),
        },
        "protocol": {
            "researcher_adaptive": True,
            "outer_split_seen_in_prior_analyses": True,
            "selection_scope": "inner_tune_only",
            "control_changed_fractions": changed_fractions,
        },
        "selection": selection,
        **audit,
        "gates": gates,
        "checkpoint": {"path": str(model_path), "sha256": sha256_file(model_path)},
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
    print(json.dumps({"completed": payload["completed"], "scientific_status": payload["scientific_status"], "gates": payload["gates"], "output": payload["output"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
