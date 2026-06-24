from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from spectral_detection_posttrain.utils.io import append_jsonl, save_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THREE_FEATURES = ["fft_edge_truncation@64", "phase_edge@64", "phase_abs_high@11"]
FOUR_64_FEATURES = ["fft_edge_truncation@64", "phase_edge@64", "high_edge@64", "high_energy_ratio@64"]
SCENE_FEATURES = ["fft_edge_truncation@64", "phase_edge@64"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NNI wrapper for NWPU raw-iFFT gated post-training.")
    parser.add_argument("--run-prefix", default="nni_raw_ifft_round2154")
    parser.add_argument("--params-json", default=None)
    parser.add_argument("--params-file", default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--limit-train", type=int, default=100000)
    parser.add_argument("--limit-val", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-proposals", type=int, default=60)
    parser.add_argument("--skip-diagnostics", action="store_true")
    return parser.parse_args(argv)


def expand_preset(params: dict[str, Any]) -> dict[str, Any]:
    preset = params.get("preset", params)
    if not isinstance(preset, dict):
        raise ValueError("NNI params must contain a dict preset")
    return dict(preset)


def _get_trial_params(params_json: str | None, params_file: str | None = None) -> dict[str, Any]:
    if params_file:
        return json.loads(Path(params_file).read_text(encoding="utf-8-sig"))
    if params_json:
        try:
            return json.loads(params_json)
        except json.JSONDecodeError:
            return json.loads(params_json.replace('\\"', '"'))
    try:
        import nni

        return dict(nni.get_next_parameter())
    except Exception:
        return {"preset": {"name": "debug_three_bbox01", "feature_set": "three"}}


def _feature_set(name: str) -> list[str]:
    if name == "three":
        return THREE_FEATURES
    if name == "four64":
        return FOUR_64_FEATURES
    if name == "scene":
        return SCENE_FEATURES
    raise ValueError(f"Unknown feature_set: {name}")


def _flag(command: list[str], flag: str, value: Any) -> None:
    command.extend([flag, str(value)])


def build_round2129_command(
    params: dict[str, Any],
    *,
    run_prefix: str,
    epochs: int,
    limit_train: int = 100000,
    limit_val: int = 100000,
    batch_size: int = 2,
    max_proposals: int = 60,
    skip_diagnostics: bool = False,
) -> tuple[list[str], str]:
    name = str(params.get("name", "raw_ifft_trial"))
    run_name = f"{run_prefix}/{name}"
    features = _feature_set(str(params.get("feature_set", "three")))
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "round2129_nwpu_posttrain_smoke.py"),
        "--run-name",
        run_name,
        "--limit-train",
        str(int(limit_train)),
        "--limit-val",
        str(int(limit_val)),
        "--batch-size",
        str(int(batch_size)),
        "--epochs",
        str(int(epochs)),
        "--max-proposals",
        str(int(max_proposals)),
        "--rollout-score-threshold",
        str(float(params.get("rollout_score_threshold", 0.001))),
        "--rollout-detections-per-img",
        str(int(params.get("rollout_detections_per_img", 300))),
        "--rescue-mode",
        "--rescue-verifier-mode",
        str(params.get("verifier_mode", "raw_ifft")),
        "--rescue-verifier-gate",
        "0.0",
        "--rescue-raw-ifft-features",
        *features,
        "--rescue-target-mode",
        "increment",
        "--trainable-mode",
        str(params.get("trainable_mode", "adapter")),
        "--record-grad-diagnostics",
    ]
    numeric_args = {
        "--bbox-rescue-loss-weight": params.get("bbox_rescue_loss_weight", 0.05),
        "--bbox-rescue-weight-temperature": params.get("bbox_temperature", 0.2),
        "--rescue-loss-weight": params.get("rescue_loss_weight", 0.0),
        "--rescue-increment-delta": params.get("rescue_increment_delta", 0.05),
        "--rescue-increment-cap": params.get("rescue_increment_cap", 0.6),
        "--det-loss-weight": params.get("det_loss_weight", 0.1),
        "--policy-loss-weight": params.get("policy_loss_weight", 0.001),
        "--kl-weight": params.get("kl_weight", 1.0),
        "--lr": params.get("lr", 3e-5),
        "--adapter-lr": params.get("adapter_lr", params.get("lr", 3e-5)),
        "--cls-adapter-scale": params.get("cls_adapter_scale", 0.25),
        "--rescue-raw-ifft-target-precision": params.get("target_precision", 0.8),
        "--rescue-raw-ifft-margin-std-frac": params.get("margin_std_frac", 0.0),
    }
    for flag, value in numeric_args.items():
        _flag(command, flag, value)
    if bool(params.get("sigmoid_gate", False)):
        command.extend(["--rescue-verifier-weight-mode", "sigmoid"])
        _flag(command, "--rescue-verifier-weight-temperature", params.get("verifier_temperature", 1.0))
    else:
        command.extend(["--rescue-verifier-weight-mode", "hard"])
    if bool(params.get("score_budget", False)):
        _flag(command, "--score-budget-loss-weight", params.get("score_budget_loss_weight", 0.01))
        _flag(command, "--score-budget-delta", params.get("score_budget_delta", 0.05))
    if "predictor_lr" in params:
        _flag(command, "--predictor-lr", params["predictor_lr"])
    if "cls_score_lr" in params:
        _flag(command, "--cls-score-lr", params["cls_score_lr"])
    if "kl_cls_weight" in params:
        _flag(command, "--kl-cls-weight", params["kl_cls_weight"])
    if "kl_box_weight" in params:
        _flag(command, "--kl-box-weight", params["kl_box_weight"])
    if "confidence_crossing_loss_weight" in params:
        _flag(command, "--confidence-crossing-loss-weight", params["confidence_crossing_loss_weight"])
    if "confidence_crossing_margin" in params:
        _flag(command, "--confidence-crossing-margin", params["confidence_crossing_margin"])
    if "hd_fusion_pca_components" in params:
        _flag(command, "--rescue-hd-fusion-pca-components", params["hd_fusion_pca_components"])
    if "hd_fusion_hd_scorer" in params:
        _flag(command, "--rescue-hd-fusion-hd-scorer", params["hd_fusion_hd_scorer"])
    if "hd_fusion_method" in params:
        _flag(command, "--rescue-hd-fusion-method", params["hd_fusion_method"])
    if "scene_groups" in params:
        command.append("--rescue-raw-ifft-scene-groups")
        command.extend([str(item) for item in params["scene_groups"]])
    if "scene_target_precision" in params:
        _flag(command, "--rescue-raw-ifft-scene-target-precision", params["scene_target_precision"])
    if "scene_min_positives" in params:
        _flag(command, "--rescue-raw-ifft-scene-min-positives", params["scene_min_positives"])
    if skip_diagnostics or bool(params.get("skip_diagnostics", False)):
        command.extend(
            [
                "--skip-offline-verifier-report",
                "--skip-initial-rescue-diagnostics",
                "--skip-epoch-rescue-diagnostics",
                "--skip-final-rescue-diagnostics",
            ]
        )
    return command, run_name


def _run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not current_pythonpath else str(PROJECT_ROOT) + os.pathsep + current_pythonpath
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=env)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_objective(final: dict[str, Any], baseline: dict[str, Any], history: dict[str, Any]) -> dict[str, Any]:
    delta_ap75 = float(final.get("ap75", 0.0)) - float(baseline.get("ap75", 0.0))
    delta_ap50 = float(final.get("ap50", 0.0)) - float(baseline.get("ap50", 0.0))
    delta_ece = float(final.get("ece", 0.0)) - float(baseline.get("ece", 0.0))
    delta_fp_rate = float(final.get("false_positive_rate", 0.0)) - float(baseline.get("false_positive_rate", 0.0))
    pred_ratio = float(final.get("num_predictions", 0.0)) / max(1.0, float(baseline.get("num_predictions", 0.0)))
    rows = list(history.get("history", []))
    last = rows[-1] if rows else {}
    bbox_rescue_count = int(last.get("bbox_rescue_count", 0) or 0)
    bbox_grad = float(last.get("grad_bbox_rescue_bbox_adapter_l2", 0.0) or 0.0)
    lchi_conf_delta_mean = float(last.get("lchi_conf_delta_mean", 0.0) or 0.0)
    lchi_conf_cross_score_threshold_count = int(last.get("lchi_conf_cross_score_threshold_count", 0) or 0)
    verifier_lchi_conf_delta_mean = float(last.get("verifier_positive_lchi_conf_delta_mean", 0.0) or 0.0)
    verifier_lchi_conf_cross_score_threshold_count = int(
        last.get("verifier_positive_lchi_conf_cross_score_threshold_count", 0) or 0
    )
    confidence_crossing_count = int(last.get("confidence_crossing_count", 0) or 0)
    confidence_crossing_active_count = int(last.get("confidence_crossing_active_count", 0) or 0)

    failed = ""
    if delta_fp_rate > 0.03:
        failed = "false_positive_rate"
    elif pred_ratio > 1.2:
        failed = "prediction_count"
    elif delta_ap50 < -0.02:
        failed = "ap50_drop"
    elif delta_ece > 0.03:
        failed = "ece"
    elif bbox_rescue_count <= 0 or bbox_grad <= 0.0:
        failed = "no_bbox_rescue_signal"

    if failed:
        score = -1.0
    else:
        score = (
            delta_ap75
            + 0.25 * delta_ap50
            - 0.1 * max(0.0, delta_ece)
            - 0.2 * max(0.0, delta_fp_rate)
        )
    return {
        "default": float(score),
        "constraint_failed": failed,
        "delta_ap75": float(delta_ap75),
        "delta_ap50": float(delta_ap50),
        "delta_ece": float(delta_ece),
        "delta_fp_rate": float(delta_fp_rate),
        "pred_ratio": float(pred_ratio),
        "bbox_rescue_count": bbox_rescue_count,
        "bbox_rescue_grad": float(bbox_grad),
        "lchi_conf_delta_mean": float(lchi_conf_delta_mean),
        "lchi_conf_cross_score_threshold_count": lchi_conf_cross_score_threshold_count,
        "verifier_positive_lchi_conf_delta_mean": float(verifier_lchi_conf_delta_mean),
        "verifier_positive_lchi_conf_cross_score_threshold_count": verifier_lchi_conf_cross_score_threshold_count,
        "confidence_crossing_count": confidence_crossing_count,
        "confidence_crossing_active_count": confidence_crossing_active_count,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.chdir(PROJECT_ROOT)
    params = expand_preset(_get_trial_params(args.params_json, args.params_file))
    command, run_name = build_round2129_command(
        params,
        run_prefix=args.run_prefix,
        epochs=int(args.epochs),
        limit_train=int(args.limit_train),
        limit_val=int(args.limit_val),
        batch_size=int(args.batch_size),
        max_proposals=int(args.max_proposals),
        skip_diagnostics=bool(args.skip_diagnostics),
    )
    _run(command)
    run_dir = PROJECT_ROOT / "runs" / run_name
    baseline = _load_json(run_dir / "baseline_eval_metrics.json")
    final = _load_json(run_dir / "eval_metrics.json")
    history = _load_json(run_dir / "metrics_train.json")
    objective = build_objective(final, baseline, history)
    row = {"name": params.get("name", ""), "run_name": run_name, **params, **objective}
    save_json(row, run_dir / "nni_objective.json")
    append_jsonl(row, PROJECT_ROOT / "runs" / args.run_prefix / "nni_raw_ifft_results.jsonl")
    try:
        import nni

        nni.report_final_result(objective)
    except Exception:
        pass
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
