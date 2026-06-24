from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from spectral_detection_posttrain.analysis.round30_results import build_round30_result_row
from spectral_detection_posttrain.methods.rlvr.detection_verifier import signal_uses_amp, signal_uses_structure
from spectral_detection_posttrain.utils.io import append_jsonl, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NNI trial for RLVR post-training matrix.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-prefix", default="nni_rlvr")
    parser.add_argument("--params-json", default=None)
    parser.add_argument("--params-file", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--rlvr-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    return parser.parse_args()


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
        return {"signal": "ramp", "unfreeze": "cls", "optimizer": "adamw",
                "reward_lambda": 0.3, "alpha": 0.5, "beta": 0.3}


def _run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT) if not current_pythonpath
        else str(PROJECT_ROOT) + os.pathsep + current_pythonpath
    )
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=env)


def _python() -> str:
    return sys.executable


def _optional_limit_args(limit_train: int | None, limit_val: int | None) -> list[str]:
    args: list[str] = []
    if limit_train is not None:
        args.extend(["--limit-train", str(limit_train)])
    if limit_val is not None:
        args.extend(["--limit-val", str(limit_val)])
    return args


def _ensure_baseline(args: argparse.Namespace) -> Path:
    run_name = f"{args.run_prefix}/baseline"
    checkpoint = Path("runs") / run_name / "checkpoint_last.pth"
    if checkpoint.exists():
        return checkpoint
    _run([
        _python(), "-m", "spectral_detection_posttrain.trainers.detection.train_baseline",
        "--config", args.config, "--run-name", run_name, "--epochs", "1",
        *_optional_limit_args(args.limit_train, args.limit_val),
    ])
    return checkpoint


def _ensure_quality_head(
    args: argparse.Namespace, baseline_ckpt: Path,
) -> tuple[Path, Path, Path, Path]:
    run_name = f"{args.run_prefix}/qh"
    train_candidates = Path("runs") / f"{args.run_prefix}/candidates_train" / "candidates.pt"
    val_candidates = Path("runs") / f"{args.run_prefix}/candidates_val" / "candidates.pt"
    qh_checkpoint = Path("runs") / run_name / "quality_head_best.pth"

    if not train_candidates.exists():
        _run([
            _python(), "-m", "spectral_detection_posttrain.signals.fft.roi_spectral_dataset",
            "--config", args.config, "--checkpoint", str(baseline_ckpt),
            "--split", "train", "--run-name", f"{args.run_prefix}/candidates_train",
            "--output", str(train_candidates),
            *_optional_limit_args(args.limit_train, args.limit_val),
        ])

    if not val_candidates.exists():
        _run([
            _python(), "-m", "spectral_detection_posttrain.signals.fft.roi_spectral_dataset",
            "--config", args.config, "--checkpoint", str(baseline_ckpt),
            "--split", "val", "--run-name", f"{args.run_prefix}/candidates_val",
            "--output", str(val_candidates),
            *_optional_limit_args(args.limit_train, args.limit_val),
        ])

    if not qh_checkpoint.exists():
        _run([
            _python(), "-m", "spectral_detection_posttrain.trainers.detection.train_quality_head",
            "--config", args.config,
            "--train-candidates", str(train_candidates),
            "--val-candidates", str(val_candidates),
            "--run-name", run_name, "--feature-mode", "roi_amp_structure",
            "--epochs", "8", "--early-stopping-patience", "4",
        ])

    return train_candidates, val_candidates, qh_checkpoint


def _precompute_r_amp_stats(args: argparse.Namespace, baseline_ckpt: Path) -> Path:
    stats_path = Path("runs") / args.run_prefix / "r_amp_stats.json"
    if stats_path.exists():
        return stats_path

    import torch
    from torch.utils.data import DataLoader
    from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
    from spectral_detection_posttrain.core.models import build_detector
    from spectral_detection_posttrain.signals.fft.rlvr_reward import compute_r_amp_stats_from_loader
    from spectral_detection_posttrain.utils.config import load_config
    from spectral_detection_posttrain.utils.io import load_checkpoint
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config)

    train_loader, _ = build_penn_fudan_loaders(
        config, limit_train=args.limit_train, limit_val=args.limit_val,
        batch_size=int(config["train"].get("batch_size", 2)),
    )
    model_cfg = dict(config)
    model_cfg["model"] = dict(config["model"])
    model_cfg["model"]["pretrained"] = False
    model = build_detector(model_cfg).to(device)
    load_checkpoint(model, str(baseline_ckpt), device)

    stats = compute_r_amp_stats_from_loader(model, train_loader, device, config)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats_path


def _run_rlvr(
    args: argparse.Namespace, params: dict, baseline_ckpt: Path,
    qh_ckpt: Path | None, r_amp_stats_path: Path | None,
) -> Path:
    signal = str(params.get("signal", "ramp"))
    unfreeze = str(params.get("unfreeze", "box"))
    optimizer = str(params.get("optimizer", "adamw"))
    reward_lambda = float(params.get("reward_lambda", 0.1))
    alpha = float(params.get("alpha", 0.1))
    beta = float(params.get("beta", 0.05))
    policy_loss_weight = float(params.get("policy_loss_weight", 0.3))
    box_loss_weight = float(params.get("box_loss_weight", 0.0))
    temperature = float(params.get("temperature", 1.0))
    max_candidates = int(params.get("max_candidates", 40))
    reward_score_threshold = float(params.get("reward_score_threshold", 0.2))
    struct_weight = float(params.get("struct_weight", 0.0))
    seed = int(params.get("seed", 42))
    det_loss_weight = float(params.get("det_loss_weight", 0.0))
    baseline_kl_weight = float(params.get("baseline_kl_weight", 1.0))
    recovery_loss_weight = float(params.get("recovery_loss_weight", 0.0))
    rollout_source = str(params.get("rollout_source", "baseline"))
    policy_objective = str(params.get("policy_objective", "signed"))

    name_tag = str(params.get("name", signal))
    tag = f"{name_tag}_{unfreeze}_{optimizer}"
    run_name = f"{args.run_prefix}/rlvr_{tag}"
    result_path = Path("runs") / run_name / "rlvr_result.json"

    if result_path.exists():
        return result_path

    command = [
        _python(), "-m", "spectral_detection_posttrain.trainers.detection.posttrain_rlvr",
        "--config", args.config,
        "--baseline", str(baseline_ckpt),
        "--run-name", run_name,
        "--signal", signal,
        "--unfreeze", unfreeze,
        "--optimizer", optimizer,
        "--reward-lambda", str(reward_lambda),
        "--struct-weight", str(struct_weight),
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--policy-loss-weight", str(policy_loss_weight),
        "--box-loss-weight", str(box_loss_weight),
        "--temperature", str(temperature),
        "--max-candidates", str(max_candidates),
        "--reward-score-threshold", str(reward_score_threshold),
        "--det-loss-weight", str(det_loss_weight),
        "--baseline-kl-weight", str(baseline_kl_weight),
        "--recovery-loss-weight", str(recovery_loss_weight),
        "--rollout-source", rollout_source,
        "--policy-objective", policy_objective,
        "--seed", str(seed),
        "--epochs", str(args.rlvr_epochs),
        "--early-stopping-patience", str(args.early_stopping_patience),
    ]
    if signal_uses_amp(signal) and r_amp_stats_path is not None:
        command.extend(["--r-amp-stats", str(r_amp_stats_path)])
    command.extend(_optional_limit_args(args.limit_train, args.limit_val))

    _run(command)
    return result_path


def _evaluate_rlvr(
    args: argparse.Namespace, rlvr_result_path: Path, params: dict,
) -> dict[str, Any]:
    rlvr_result = json.loads(rlvr_result_path.read_text(encoding="utf-8"))
    run_dir = rlvr_result_path.parent
    best_checkpoint = run_dir / "checkpoint_best.pth"
    last_checkpoint = run_dir / "checkpoint_last.pth"
    ckpt = best_checkpoint if best_checkpoint.exists() else last_checkpoint

    eval_configs = [
        ("clean", "none", "random"),
        ("checkerboard", "random", "checkerboard"),
    ]
    all_metrics: dict[str, Any] = {}
    for patch_tag, patch_mode, patch_type in eval_configs:
        eval_run_name = f"{rlvr_result_path.parent.name}_eval_{patch_tag}"
        _run([
            _python(), "-m", "spectral_detection_posttrain.eval.eval_detector",
            "--config", args.config, "--checkpoint", str(ckpt),
            "--run-name", eval_run_name,
            "--patch-mode", patch_mode,
            "--patch-type", patch_type,
        ])
        metrics_path = Path("runs") / eval_run_name / "eval_metrics.json"
        if metrics_path.exists():
            all_metrics[patch_tag] = json.loads(metrics_path.read_text(encoding="utf-8"))

    return all_metrics


def _objective(metrics: dict[str, Any]) -> float:
    def _per_scene(ap50: float, ap75: float, ece: float, fp_rate: float) -> float:
        ece_val = ece if ece is not None else 1.0
        fp_val = fp_rate if fp_rate is not None else 1.0
        return ap50 + ap75 - ece_val - fp_val

    total = 0.0
    for patch_tag in ["clean", "checkerboard"]:
        m = metrics.get(patch_tag, {})
        if not m:
            continue
        ap50 = float(m.get("ap50", 0))
        ap75_val = m.get("ap75")
        ap75 = float(ap75_val) if ap75_val is not None else ap50 * 0.7
        ece_val = m.get("ece")
        ece = float(ece_val) if ece_val is not None else 1.0
        fp_rate = float(m.get("high_conf_fp_rate", 0))
        total += _per_scene(ap50, ap75, ece, fp_rate)
    return total


def _report_to_nni(result: dict[str, Any]) -> None:
    try:
        import nni
        nni.report_final_result(result)
    except Exception:
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def expand_preset(params: dict) -> dict:
    preset = params.get("preset", params)
    if not isinstance(preset, dict):
        raise TypeError("NNI preset must be a dict")
    return dict(preset)


def compute_round2_objective(metrics: dict, baseline: dict) -> dict:
    clean = metrics.get("clean", {})
    edge = metrics.get("object_edge_checkerboard", {})
    base_clean = baseline["clean"]
    base_edge = baseline["object_edge_checkerboard"]
    if clean.get("ap50", 0) < base_clean["ap50"] - 0.07:
        return {"default": -1.0, "constraint_failed": "ap50_clean"}
    if edge.get("ap50", 0) < base_edge["ap50"] - 0.08:
        return {"default": -1.0, "constraint_failed": "ap50_object_edge_checkerboard"}
    if clean.get("recall", 0) < base_clean["recall"] - 0.04:
        return {"default": -1.0, "constraint_failed": "recall_clean"}
    if clean.get("ap75", 0) < base_clean["ap75"] - 0.07:
        return {"default": -1.0, "constraint_failed": "ap75_clean"}
    score = clean["ap50"] + edge["ap50"] + 0.5 * edge["ap75"] - 0.2 * clean["ece"] - 0.2 * edge["ece"]
    return {"default": float(score), "constraint_failed": ""}


def compute_round23_objective(metrics: dict, baseline: dict) -> dict:
    clean = metrics.get("clean", {})
    edge = metrics.get("object_edge_checkerboard", {})
    if not clean or not edge:
        return {"default": -1.0, "constraint_failed": "eval_missing"}
    base_clean = baseline["clean"]
    base_edge = baseline["object_edge_checkerboard"]

    checks = [
        ("clean_ap50", clean.get("ap50", 0) >= base_clean["ap50"] - 0.03),
        ("clean_ap75", clean.get("ap75", 0) >= base_clean["ap75"] - 0.08),
        ("clean_recall", clean.get("recall", 0) >= base_clean["recall"] - 0.04),
        ("clean_precision", clean.get("precision", 0) >= base_clean["precision"] - 0.08),
        ("clean_num_predictions", clean.get("num_predictions", 10**9) <= base_clean["num_predictions"] * 1.20),
        ("edge_ap50", edge.get("ap50", 0) >= base_edge["ap50"] - 0.06),
        ("edge_num_predictions", edge.get("num_predictions", 10**9) <= base_edge["num_predictions"] * 1.25),
    ]
    for name, ok in checks:
        if not ok:
            return {"default": -1.0, "constraint_failed": name}

    score = (
        clean["ap50"] + 0.5 * clean["ap75"]
        + edge["ap50"] + 0.5 * edge["ap75"]
        - 0.2 * clean.get("ece", 0.0)
        - 0.2 * edge.get("ece", 0.0)
    )
    return {"default": float(score), "constraint_failed": ""}


def compute_round25_objective(metrics: dict, baseline: dict) -> dict:
    required = ["clean", "object_edge_checkerboard", "object_inside_checkerboard", "near_object_checkerboard"]
    for scene in required:
        if not metrics.get(scene):
            return {"default": -1.0, "constraint_failed": f"missing_{scene}"}
    clean = metrics["clean"]
    base_clean = baseline["clean"]
    checks = [
        ("clean_ap50", clean.get("ap50", 0.0) >= base_clean["ap50"] - 0.03),
        ("clean_recall", clean.get("recall", 0.0) >= base_clean["recall"] - 0.04),
        ("clean_num_predictions", clean.get("num_predictions", 10**9) <= base_clean["num_predictions"] * 1.20),
    ]
    for scene in required[1:]:
        current = metrics[scene]
        base = baseline.get(scene, baseline["clean"])
        checks.extend([
            (f"{scene}_ap50", current.get("ap50", 0.0) >= base["ap50"] - 0.06),
            (f"{scene}_recall", current.get("recall", 0.0) >= base["recall"] - 0.05),
            (f"{scene}_num_predictions", current.get("num_predictions", 10**9) <= base["num_predictions"] * 1.25),
        ])
    for name, ok in checks:
        if not ok:
            return {"default": -1.0, "constraint_failed": name}
    score = 0.0
    for scene in required:
        current = metrics[scene]
        score += current.get("ap50", 0.0) + 0.5 * current.get("ap75", 0.0)
        score -= 0.2 * current.get("ece", 0.0)
    return {"default": float(score), "constraint_failed": ""}


_REQUIRED_BASE_FIELDS = [
    "name", "default", "constraint_failed", "run_name", "checkpoint", "eval_status",
    "signal", "reward_lambda", "policy_loss_weight", "det_loss_weight",
    "baseline_kl_weight", "box_loss_weight", "struct_weight", "unfreeze", "optimizer",
    "temperature", "max_candidates", "reward_score_threshold", "rollout_source",
    "policy_objective",
    "clean_ap50", "clean_ap75", "clean_precision", "clean_recall",
    "clean_num_predictions", "clean_high_conf_fp_count", "clean_ece",
    "edge_ap50", "edge_ap75", "edge_precision", "edge_recall",
    "edge_num_predictions", "edge_high_conf_fp_count", "edge_ece",
    "inside_ap50", "inside_ap75", "inside_precision", "inside_recall",
    "inside_num_predictions", "inside_high_conf_fp_count", "inside_ece",
    "near_ap50", "near_ap75", "near_precision", "near_recall",
    "near_num_predictions", "near_high_conf_fp_count", "near_ece",
]
REQUIRED_ROUND23_RESULT_FIELDS = _REQUIRED_BASE_FIELDS


def _metric(metrics: dict, scene: str, key: str):
    return metrics.get(scene, {}).get(key)


def build_round23_result_row(
    params: dict, metrics: dict, objective: dict,
    run_name: str, checkpoint: str, eval_status: str,
) -> dict:
    row = {
        "name": params.get("name", ""),
        "default": objective.get("default", -1.0),
        "constraint_failed": objective.get("constraint_failed", ""),
        "run_name": run_name,
        "checkpoint": checkpoint,
        "eval_status": eval_status,
        "signal": params.get("signal", ""),
        "reward_lambda": float(params.get("reward_lambda", 0.0)),
        "policy_loss_weight": float(params.get("policy_loss_weight", 0.0)),
        "det_loss_weight": float(params.get("det_loss_weight", 0.0)),
        "baseline_kl_weight": float(params.get("baseline_kl_weight", 0.0)),
        "box_loss_weight": float(params.get("box_loss_weight", 0.0)),
        "unfreeze": params.get("unfreeze", ""),
        "optimizer": params.get("optimizer", ""),
        "temperature": float(params.get("temperature", 1.0)),
        "max_candidates": int(params.get("max_candidates", 0)),
        "reward_score_threshold": float(params.get("reward_score_threshold", 0.0)),
        "rollout_source": params.get("rollout_source", ""),
        "policy_objective": params.get("policy_objective", ""),
        "struct_weight": float(params.get("struct_weight", 0.0)),
        "clean_ap50": _metric(metrics, "clean", "ap50"),
        "clean_ap75": _metric(metrics, "clean", "ap75"),
        "clean_precision": _metric(metrics, "clean", "precision"),
        "clean_recall": _metric(metrics, "clean", "recall"),
        "clean_num_predictions": _metric(metrics, "clean", "num_predictions"),
        "clean_high_conf_fp_count": _metric(metrics, "clean", "high_conf_fp_count"),
        "clean_ece": _metric(metrics, "clean", "ece"),
        "edge_ap50": _metric(metrics, "object_edge_checkerboard", "ap50"),
        "edge_ap75": _metric(metrics, "object_edge_checkerboard", "ap75"),
        "edge_precision": _metric(metrics, "object_edge_checkerboard", "precision"),
        "edge_recall": _metric(metrics, "object_edge_checkerboard", "recall"),
        "edge_num_predictions": _metric(metrics, "object_edge_checkerboard", "num_predictions"),
        "edge_high_conf_fp_count": _metric(metrics, "object_edge_checkerboard", "high_conf_fp_count"),
        "edge_ece": _metric(metrics, "object_edge_checkerboard", "ece"),
        "inside_ap50": _metric(metrics, "object_inside_checkerboard", "ap50"),
        "inside_ap75": _metric(metrics, "object_inside_checkerboard", "ap75"),
        "inside_precision": _metric(metrics, "object_inside_checkerboard", "precision"),
        "inside_recall": _metric(metrics, "object_inside_checkerboard", "recall"),
        "inside_num_predictions": _metric(metrics, "object_inside_checkerboard", "num_predictions"),
        "inside_high_conf_fp_count": _metric(metrics, "object_inside_checkerboard", "high_conf_fp_count"),
        "inside_ece": _metric(metrics, "object_inside_checkerboard", "ece"),
        "near_ap50": _metric(metrics, "near_object_checkerboard", "ap50"),
        "near_ap75": _metric(metrics, "near_object_checkerboard", "ap75"),
        "near_precision": _metric(metrics, "near_object_checkerboard", "precision"),
        "near_recall": _metric(metrics, "near_object_checkerboard", "recall"),
        "near_num_predictions": _metric(metrics, "near_object_checkerboard", "num_predictions"),
        "near_high_conf_fp_count": _metric(metrics, "near_object_checkerboard", "high_conf_fp_count"),
        "near_ece": _metric(metrics, "near_object_checkerboard", "ece"),
    }
    for field in REQUIRED_ROUND23_RESULT_FIELDS:
        row.setdefault(field, None)
    return row


def validate_expected_presets(expected_names: list[str], rows: list[dict]) -> dict:
    seen = {row.get("name") for row in rows}
    missing = [name for name in expected_names if name not in seen]
    extra = sorted(name for name in seen if name and name not in set(expected_names))
    return {"complete": not missing and not extra, "missing": missing, "extra": extra}


def collect_eval_status(metrics: dict) -> str:
    has_clean = bool(metrics.get("clean"))
    has_edge = bool(metrics.get("object_edge_checkerboard"))
    if has_clean and has_edge:
        return "ok"
    if not has_clean and not has_edge:
        return "missing_clean_and_edge"
    if not has_clean:
        return "missing_clean"
    return "missing_edge"


def compute_round22_objective(metrics: dict, baseline: dict) -> dict:
    clean = metrics.get("clean", {})
    edge = metrics.get("object_edge_checkerboard", {})
    if not clean or not edge:
        return {"default": -1.0, "constraint_failed": "eval_missing"}
    base_clean = baseline["clean"]
    base_edge = baseline["object_edge_checkerboard"]

    if clean.get("ap50", 0) < base_clean["ap50"] - 0.05:
        return {"default": -1.0, "constraint_failed": "ap50_clean"}
    if clean.get("recall", 0) < base_clean["recall"] - 0.05:
        return {"default": -1.0, "constraint_failed": "recall_clean"}
    if clean.get("ap75", 0) < base_clean["ap75"] - 0.10:
        return {"default": -1.0, "constraint_failed": "ap75_clean"}
    if clean.get("num_predictions", 10**9) > base_clean["num_predictions"] * 1.30:
        return {"default": -1.0, "constraint_failed": "num_predictions_clean"}
    if clean.get("precision", 0) < base_clean["precision"] - 0.10:
        return {"default": -1.0, "constraint_failed": "precision_clean"}
    if edge.get("ap50", 0) < base_edge["ap50"] - 0.08:
        return {"default": -1.0, "constraint_failed": "ap50_object_edge_checkerboard"}

    score = (
        clean["ap50"] + 0.5 * clean["ap75"]
        + edge["ap50"] + 0.5 * edge["ap75"]
        - 0.2 * clean.get("ece", 0.0)
        - 0.2 * edge.get("ece", 0.0)
    )
    return {"default": float(score), "constraint_failed": ""}


def _eval_patch(python_exe: str, args: argparse.Namespace, ckpt: Path, patch_mode: str, patch_type: str, run_name: str) -> dict:
    _run([
        python_exe, "-m", "spectral_detection_posttrain.eval.eval_detector",
        "--config", args.config, "--checkpoint", str(ckpt),
        "--run-name", run_name,
        "--patch-mode", patch_mode,
        "--patch-type", patch_type,
    ])
    metrics_path = Path("runs") / run_name / "eval_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return {}


def _ensure_baseline_metrics(args: argparse.Namespace) -> Path:
    metrics_path = Path("runs") / args.run_prefix / "baseline_metrics.json"
    if metrics_path.exists():
        return metrics_path
    baseline_ckpt = _ensure_baseline(args)
    py = _python()
    metrics: dict[str, Any] = {}
    for mode_key, patch_mode, patch_type in [
        ("clean", "none", "random"),
        ("object_edge_checkerboard", "object_edge", "checkerboard"),
        ("object_inside_checkerboard", "object_inside", "checkerboard"),
        ("near_object_checkerboard", "near_object", "checkerboard"),
    ]:
        m = _eval_patch(py, args, baseline_ckpt, patch_mode, patch_type,
                        f"{args.run_prefix}/baseline_eval_{mode_key}")
        if m:
            metrics[mode_key] = {k: m.get(k) for k in ["ap50", "ap75", "precision", "recall", "ece", "high_conf_fp_count", "high_conf_fp_rate", "num_predictions"]}
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics_path


def main() -> None:
    args = parse_args()
    params = _get_trial_params(args.params_json, args.params_file)
    params = expand_preset(params)
    signal = str(params.get("signal", "ramp"))

    print(f"Trial params: {json.dumps(params, indent=2)}", flush=True)

    baseline_metrics_path = _ensure_baseline_metrics(args)
    baseline_ckpt = _ensure_baseline(args)

    r_amp_stats_path: Path | None = None
    if signal_uses_amp(signal):
        r_amp_stats_path = _precompute_r_amp_stats(args, baseline_ckpt)

    rlvr_result_path = _run_rlvr(args, params, baseline_ckpt, None, r_amp_stats_path)
    rlvr_result = json.loads(rlvr_result_path.read_text(encoding="utf-8"))
    ckpt_path = Path(rlvr_result_path).parent / "checkpoint_best.pth"
    if not ckpt_path.exists():
        ckpt_path = Path(rlvr_result_path).parent / "checkpoint_last.pth"

    py = _python()
    metrics = {}
    use_four_scenes = "round30" in args.run_prefix or "round25" in args.run_prefix
    eval_scenes = (
        [("clean", "none", "random"),
         ("object_edge_checkerboard", "object_edge", "checkerboard"),
         ("object_inside_checkerboard", "object_inside", "checkerboard"),
         ("near_object_checkerboard", "near_object", "checkerboard")]
        if use_four_scenes
        else [
            ("clean", "none", "random"),
            ("object_edge_checkerboard", "object_edge", "checkerboard"),
        ]
    )
    for mode_key, patch_mode, patch_type in eval_scenes:
        try:
            m = _eval_patch(py, args, ckpt_path, patch_mode, patch_type,
                            f"{rlvr_result_path.parent.name}_eval_{mode_key}")
            if m:
                metrics[mode_key] = m
        except Exception:
            print(f"eval failed for {mode_key}, continuing", flush=True)

    baseline = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))

    # select objective based on run_prefix
    if "round30" in args.run_prefix:
        objective = compute_round30_objective(metrics, baseline)
    elif "round25" in args.run_prefix:
        objective = compute_round25_objective(metrics, baseline)
    elif "round23" in args.run_prefix:
        objective = compute_round23_objective(metrics, baseline)
    else:
        objective = compute_round22_objective(metrics, baseline)

    expected_eval_count = 4 if use_four_scenes else 2
    eval_status = "ok" if len(metrics) >= expected_eval_count else "failed"

    if "round30" in args.run_prefix:
        result = build_round30_result_row(
            params=params, metrics=metrics, objective=objective,
            run_name=rlvr_result_path.parent.name,
            checkpoint=str(ckpt_path), eval_status=eval_status,
        )
    else:
        result = build_round23_result_row(
            params=params, metrics=metrics, objective=objective,
            run_name=rlvr_result_path.parent.name,
            checkpoint=str(ckpt_path), eval_status=eval_status,
        )

    result_path = Path("runs") / args.run_prefix / "nni_rlvr_results.jsonl"
    append_jsonl(result, result_path)
    save_json(result, Path("runs") / args.run_prefix / "last_trial_result.json")
    _report_to_nni(result)


if __name__ == "__main__":
    main()
