"""Posthoc fit-split replay for the failed NWPU joint Delta-U probe."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
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
    / "det.energy.joint_probe.002.json"
)
DEFAULT_RUN_DIR = ROOT / "runs" / "nwpu_joint_delta_u_probe_v2_s42_traincal_cleanval180"
FIT_PAIRWISE_LEARNED_MIN = 0.60
FIT_PAIRWISE_NOT_LEARNED_MAX = 0.55
HELDOUT_PAIRWISE_FAILED_MAX = 0.55


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(LOCKED_CONFIG))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def classify_fit_replay(
    *,
    fit_raw: Mapping[str, Any],
    fit_calibration: Mapping[str, Any],
    fit_baselines: Mapping[str, Any],
    calibration_raw: Mapping[str, Any],
    validation_raw: Mapping[str, Any],
) -> dict[str, Any]:
    fit_pairwise = float(fit_raw["pairwise_accuracy"])
    fit_weighted = float(fit_raw["pairwise_accuracy_candidate_weighted"])
    calibration_pairwise = float(calibration_raw["pairwise_accuracy"])
    calibration_weighted = float(
        calibration_raw["pairwise_accuracy_candidate_weighted"]
    )
    validation_pairwise = float(validation_raw["pairwise_accuracy"])
    validation_weighted = float(
        validation_raw["pairwise_accuracy_candidate_weighted"]
    )
    threshold_feasible = not bool(fit_calibration["used_identity_fallback"])
    fit_mae_gain = float(fit_baselines["zero_utility_mae"]) - float(fit_raw["mae"])
    fit_sign_gain = float(fit_raw["sign_accuracy"]) - float(
        fit_baselines["always_negative_sign_accuracy"]
    )
    fit_learned = (
        min(fit_pairwise, fit_weighted) >= FIT_PAIRWISE_LEARNED_MIN
        and threshold_feasible
    )
    heldout_failed = (
        max(
            calibration_pairwise,
            calibration_weighted,
            validation_pairwise,
            validation_weighted,
        )
        <= HELDOUT_PAIRWISE_FAILED_MAX
    )
    fit_not_learned = (
        max(fit_pairwise, fit_weighted) <= FIT_PAIRWISE_NOT_LEARNED_MAX
        and not threshold_feasible
        and fit_mae_gain <= 0.0
    )
    if fit_learned and heldout_failed:
        diagnosis = "fit_signal_learned_but_heldout_failed"
    elif fit_not_learned:
        diagnosis = "fit_signal_not_learned"
    else:
        diagnosis = "mixed_fit_signal"
    return {
        "diagnosis": diagnosis,
        "thresholds": {
            "fit_pairwise_learned_min": FIT_PAIRWISE_LEARNED_MIN,
            "fit_pairwise_not_learned_max": FIT_PAIRWISE_NOT_LEARNED_MAX,
            "heldout_pairwise_failed_max": HELDOUT_PAIRWISE_FAILED_MAX,
        },
        "fit_threshold_feasible": threshold_feasible,
        "fit_pairwise_gain_over_zero": fit_pairwise
        - float(fit_baselines["zero_utility_pairwise_accuracy"]),
        "fit_pairwise_candidate_weighted_gain_over_zero": fit_weighted
        - float(fit_baselines["zero_utility_pairwise_accuracy"]),
        "fit_mae_gain_over_zero": fit_mae_gain,
        "fit_sign_gain_over_always_negative": fit_sign_gain,
        "fit_to_calibration_pairwise_gap": fit_pairwise - calibration_pairwise,
        "fit_to_validation_pairwise_gap": fit_pairwise - validation_pairwise,
        "fit_to_validation_candidate_weighted_gap": fit_weighted
        - validation_weighted,
    }


def unconditional_top1_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    *,
    target_epsilon: float,
    lcb_z: float,
) -> dict[str, float | int]:
    from spectral_detection_posttrain.methods.energy_transport import (
        calibrated_selection_metrics,
    )

    return calibrated_selection_metrics(
        predicted,
        target,
        image_ids,
        threshold=float("-inf"),
        target_epsilon=target_epsilon,
        lcb_z=lcb_z,
    )


def _git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _calibration_payload(calibrated: Any, json_value: Any) -> dict[str, Any]:
    threshold = calibrated.threshold
    return {
        "threshold": None
        if threshold is None or not math.isfinite(float(threshold))
        else float(threshold),
        "used_identity_fallback": bool(calibrated.used_identity_fallback),
        "candidates_evaluated": int(calibrated.candidates_evaluated),
        "metrics": json_value(calibrated.metrics),
        "interpretation": "fit_resubstitution_only_not_a_deployment_threshold",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    v1_spec = __import__("importlib.util").util.spec_from_file_location(
        "fit_replay_v1", V1_SCRIPT
    )
    v2_spec = __import__("importlib.util").util.spec_from_file_location(
        "fit_replay_v2", V2_SCRIPT
    )
    if v1_spec is None or v1_spec.loader is None or v2_spec is None or v2_spec.loader is None:
        raise RuntimeError("failed to load joint Delta-U scripts")
    v1 = __import__("importlib.util").util.module_from_spec(v1_spec)
    v2 = __import__("importlib.util").util.module_from_spec(v2_spec)
    v1_spec.loader.exec_module(v1)
    v2_spec.loader.exec_module(v2)

    config_path = Path(args.config).resolve()
    config = v2.load_locked_config(config_path)
    run_dir = Path(args.run_dir).resolve()
    eval_path = run_dir / "eval_metrics.json"
    if args.require_clean_git and _git_value("status", "--porcelain"):
        raise RuntimeError("--require-clean-git requested but the repository is dirty")
    if not eval_path.is_file():
        raise FileNotFoundError(f"missing completed v002 artifact: {eval_path}")
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    if not evaluation.get("completed") or evaluation.get("version_id") != config["version_id"]:
        raise ValueError("source v002 evaluation is incomplete or has the wrong version")
    recorded_config_hash = evaluation["inputs"]["environment"]["config_file_hash"]
    if v2.sha256_file(config_path) != recorded_config_hash:
        raise ValueError("source evaluation config hash does not match the locked config")

    source_cache = (ROOT / config["source_train_cache"]["path"]).resolve()
    v2._verify_file(
        source_cache,
        config["source_train_cache"]["sha256"],
        "source train cache",
    )
    records = v1.read_probe_cache(source_cache)
    expected_manifest = {
        "count": int(config["dataset"]["train_images"]),
        "image_ids_sha256": config["dataset"]["train_image_ids_sha256"],
    }
    if v2._manifest([record[0]["image_id"] for record in records]) != expected_manifest:
        raise ValueError("source train records do not match the locked manifest")
    fit_records, _ = v2.split_train_records(
        records,
        calibration_fraction=float(config["split"]["calibration_fraction"]),
        seed=int(config["split"]["seed"]),
    )
    fit_manifest = v2._manifest([record[0]["image_id"] for record in fit_records])
    if fit_manifest != evaluation["inputs"]["fit_manifest"]:
        raise ValueError("fit replay manifest differs from the completed v002 run")

    from spectral_detection_posttrain.methods.energy_transport import (
        build_symmetric_box_candidates,
        calibrate_conservative_threshold,
        constant_utility_baselines,
    )
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

    set_seed(int(config["dataset"]["seed"]))
    device = resolve_device(
        {"device": args.device or ("cuda" if torch.cuda.is_available() else "cpu")}
    )
    v1.validate_deterministic_cuda_environment(device)
    torch.use_deterministic_algorithms(True, warn_only=True)
    candidate_deltas = build_symmetric_box_candidates(
        tuple(float(value) for value in config["candidate_pool"]["step_sizes"])
    ).to(device)
    if int(candidate_deltas.shape[0]) != int(config["candidate_pool"]["candidate_count"]):
        raise ValueError("locked candidate grid count mismatch")

    target_reference: torch.Tensor | None = None
    image_reference: torch.Tensor | None = None
    fit_baselines: dict[str, Any] | None = None
    arm_results: dict[str, Any] = {}
    for arm_value in config["arms"]:
        arm = str(arm_value)
        checkpoint = run_dir / arm / "policy_best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing selected arm checkpoint: {checkpoint}")
        model = v2._load_arm_model(
            checkpoint,
            fit_records[0][0],
            candidate_deltas,
            config,
            device,
        )
        predicted, target, image_ids = v2._arm_rows(
            v1, model, fit_records, candidate_deltas, config, device, arm
        )
        if target_reference is None:
            target_reference = target
            image_reference = image_ids
            fit_baselines = constant_utility_baselines(
                target,
                image_ids,
                target_epsilon=float(config["loss"]["target_epsilon"]),
            )
            zero_raw = v1.probe_metrics(
                torch.zeros_like(target),
                target,
                image_ids,
                action_budget=1,
                target_epsilon=float(config["loss"]["target_epsilon"]),
            )
            fit_baselines.update(
                {
                    "zero_utility_sign_accuracy": zero_raw["sign_accuracy"],
                    "zero_utility_pairwise_accuracy_equal_image": zero_raw[
                        "pairwise_accuracy"
                    ],
                    "zero_utility_pairwise_accuracy_candidate_weighted": zero_raw[
                        "pairwise_accuracy_candidate_weighted"
                    ],
                }
            )
        elif not torch.equal(target_reference, target) or not torch.equal(
            image_reference, image_ids
        ):
            raise ValueError("fit replay arms do not share identical utility rows")
        raw = v1.probe_metrics(
            predicted,
            target,
            image_ids,
            action_budget=1,
            target_epsilon=float(config["loss"]["target_epsilon"]),
        )
        calibrated = calibrate_conservative_threshold(
            predicted,
            target,
            image_ids,
            **v2._calibration_kwargs(config),
        )
        calibration_payload = _calibration_payload(calibrated, v2._json_value)
        assert fit_baselines is not None
        arm_results[arm] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": v2.sha256_file(checkpoint),
            "raw": raw,
            "unconditional_top1": unconditional_top1_metrics(
                predicted,
                target,
                image_ids,
                target_epsilon=float(config["loss"]["target_epsilon"]),
                lcb_z=float(config["calibration"]["lcb_z"]),
            ),
            "fit_resubstitution_calibration": calibration_payload,
            "heldout_reference": {
                "calibration_raw": evaluation["calibration"][arm]["raw"],
                "calibration_frozen_threshold": {
                    "threshold": evaluation["calibration"][arm]["threshold"],
                    "used_identity_fallback": evaluation["calibration"][arm][
                        "used_identity_fallback"
                    ],
                    "metrics": evaluation["calibration"][arm]["metrics"],
                },
                "clean_validation_raw": evaluation["validation"]["raw"][arm],
                "clean_validation_frozen_threshold": evaluation["validation"][
                    "calibrated"
                ][arm],
            },
            "gap_diagnosis": classify_fit_replay(
                fit_raw=raw,
                fit_calibration=calibration_payload,
                fit_baselines=fit_baselines,
                calibration_raw=evaluation["calibration"][arm]["raw"],
                validation_raw=evaluation["validation"]["raw"][arm],
            ),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert fit_baselines is not None
    payload = {
        "completed": True,
        "scope": "posthoc_fit_replay",
        "scientific_status": "diagnostic_only",
        "protocol": {
            "no_retraining": True,
            "clean_validation_reused_from_artifact": True,
            "fit_labels_reused_for_diagnostic_evaluation": True,
            "fit_thresholds_are_not_valid_for_deployment": True,
        },
        "claim_boundary": config["probe"]["claim_boundary"],
        "inputs": {
            "analyzer_git_commit": _git_value("rev-parse", "HEAD"),
            "analyzer_git_dirty": bool(_git_value("status", "--porcelain")),
            "source_eval": str(eval_path),
            "source_eval_sha256": v2.sha256_file(eval_path),
            "source_eval_git_commit": evaluation["inputs"]["code_commit"],
            "config": str(config_path),
            "config_sha256": recorded_config_hash,
            "source_train_cache": str(source_cache),
            "source_train_cache_sha256": config["source_train_cache"]["sha256"],
            "fit_manifest": fit_manifest,
        },
        "fit_baselines": fit_baselines,
        "arms": arm_results,
        "primary_diagnosis": arm_results["joint_full"]["gap_diagnosis"]["diagnosis"],
    }
    output = Path(args.output).resolve() if args.output else run_dir / "fit_replay_metrics.json"
    output.write_text(
        json.dumps(v2._json_value(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    payload["output"] = str(output)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    payload = run(parse_args(argv))
    print(
        json.dumps(
            {
                "completed": payload["completed"],
                "primary_diagnosis": payload["primary_diagnosis"],
                "output": payload["output"],
            },
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
