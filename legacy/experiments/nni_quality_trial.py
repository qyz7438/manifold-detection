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

from spectral_detection_posttrain.utils.io import append_jsonl, save_json


FEATURE_MODE_MAP = {
    "ROI-only": "roi",
    "ROI+Amp": "roi_amp",
    "ROI+Amp+Struct": "roi_amp_structure",
    "roi": "roi",
    "roi_amp": "roi_amp",
    "roi_amp_structure": "roi_amp_structure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NNI trial for detector/QH/rerank matrix training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-prefix", default="nni_quality_matrix")
    parser.add_argument("--params-json", default=None, help="Local smoke parameters when not launched by NNI.")
    parser.add_argument("--params-file", default=None, help="Path to local smoke parameter JSON.")
    parser.add_argument("--fixed-recall", type=float, default=0.85)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
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
        return {
            "detector_epochs": 1,
            "quality_head": "ROI+Amp+Struct",
            "qh_epochs": 8,
            "alpha": 0.9,
        }


def _run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not current_pythonpath
        else str(PROJECT_ROOT) + os.pathsep + current_pythonpath
    )
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=env)


def _python() -> str:
    return sys.executable


def _optional_limit_args(limit_train: int | None, limit_val: int | None) -> list[str]:
    args = []
    if limit_train is not None:
        args.extend(["--limit-train", str(limit_train)])
    if limit_val is not None:
        args.extend(["--limit-val", str(limit_val)])
    return args


def _ensure_baseline(args: argparse.Namespace, detector_epochs: int) -> Path:
    run_name = f"{args.run_prefix}/baseline_e{detector_epochs}"
    checkpoint = Path("runs") / run_name / "checkpoint_last.pth"
    if checkpoint.exists():
        return checkpoint
    command = [
        _python(),
        "-m",
        "spectral_detection_posttrain.trainers.detection.train_baseline",
        "--config",
        args.config,
        "--run-name",
        run_name,
        "--epochs",
        str(detector_epochs),
        *_optional_limit_args(args.limit_train, args.limit_val),
    ]
    _run(command)
    return checkpoint


def _ensure_candidates(args: argparse.Namespace, checkpoint: Path, detector_epochs: int, split: str) -> Path:
    run_name = f"{args.run_prefix}/candidates_e{detector_epochs}_{split}"
    output = Path("runs") / run_name / "candidates.pt"
    if output.exists():
        return output
    command = [
        _python(),
        "-m",
        "spectral_detection_posttrain.signals.fft.roi_spectral_dataset",
        "--config",
        args.config,
        "--checkpoint",
        str(checkpoint),
        "--split",
        split,
        "--run-name",
        run_name,
        "--output",
        str(output),
        *_optional_limit_args(args.limit_train, args.limit_val),
    ]
    _run(command)
    return output


def _ensure_quality_head(
    args: argparse.Namespace,
    train_candidates: Path,
    val_candidates: Path,
    detector_epochs: int,
    feature_mode: str,
    qh_epochs: int,
) -> Path:
    run_name = f"{args.run_prefix}/qh_e{detector_epochs}_{feature_mode}_qe{qh_epochs}"
    checkpoint = Path("runs") / run_name / "quality_head_best.pth"
    if checkpoint.exists():
        return checkpoint
    command = [
        _python(),
        "-m",
        "spectral_detection_posttrain.trainers.detection.train_quality_head",
        "--config",
        args.config,
        "--train-candidates",
        str(train_candidates),
        "--val-candidates",
        str(val_candidates),
        "--run-name",
        run_name,
        "--feature-mode",
        feature_mode,
        "--epochs",
        str(qh_epochs),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--early-stopping-min-delta",
        str(args.early_stopping_min_delta),
    ]
    _run(command)
    return checkpoint


def _evaluate(
    args: argparse.Namespace,
    val_candidates: Path,
    qh_checkpoint: Path,
    detector_epochs: int,
    feature_mode: str,
    qh_epochs: int,
    alpha: float,
) -> dict[str, Any]:
    alpha_tag = str(alpha).replace(".", "p")
    run_name = f"{args.run_prefix}/eval_e{detector_epochs}_{feature_mode}_qe{qh_epochs}_a{alpha_tag}"
    metrics_path = Path("runs") / run_name / "eval_rerank_metrics.json"
    if not metrics_path.exists():
        command = [
            _python(),
            "-m",
            "spectral_detection_posttrain.eval.eval_rerank",
            "--config",
            args.config,
            "--candidates",
            str(val_candidates),
            "--quality-checkpoint",
            str(qh_checkpoint),
            "--run-name",
            run_name,
            "--method",
            "learned",
            "--combine",
            "blend",
            "--alpha",
            str(alpha),
        ]
        _run(command)
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _objective(metrics: dict[str, Any], fixed_recall: float) -> float:
    fixed_key = f"precision_at_recall_{str(fixed_recall).replace('.', '_')}"
    precision_fixed = metrics.get(fixed_key)
    if precision_fixed is None:
        precision_fixed = 0.0
    ece = metrics.get("ece")
    if ece is None:
        ece = 1.0
    return float(metrics["ap50"] + precision_fixed - ece - metrics["high_conf_fp_rate"])


def _report_to_nni(result: dict[str, Any]) -> None:
    try:
        import nni

        nni.report_final_result(result)
    except Exception:
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    params = _get_trial_params(args.params_json, args.params_file)
    detector_epochs = int(params["detector_epochs"])
    feature_mode = FEATURE_MODE_MAP[str(params["quality_head"])]
    qh_epochs = int(params["qh_epochs"])
    alpha = float(params["alpha"])

    checkpoint = _ensure_baseline(args, detector_epochs)
    train_candidates = _ensure_candidates(args, checkpoint, detector_epochs, "train")
    val_candidates = _ensure_candidates(args, checkpoint, detector_epochs, "val")
    qh_checkpoint = _ensure_quality_head(args, train_candidates, val_candidates, detector_epochs, feature_mode, qh_epochs)
    metrics = _evaluate(args, val_candidates, qh_checkpoint, detector_epochs, feature_mode, qh_epochs, alpha)

    result = {
        "default": _objective(metrics, args.fixed_recall),
        "detector_epochs": detector_epochs,
        "quality_head": params["quality_head"],
        "feature_mode": feature_mode,
        "qh_epochs": qh_epochs,
        "alpha": alpha,
        "ap50": metrics["ap50"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "high_conf_fp_rate": metrics["high_conf_fp_rate"],
        "high_conf_fp_count": metrics.get("high_conf_fp_count"),
        "ece": metrics.get("ece"),
        "q_spec_auc_tp_vs_fp": metrics.get("q_spec_auc_tp_vs_fp"),
        f"precision_at_recall_{str(args.fixed_recall).replace('.', '_')}": metrics.get(
            f"precision_at_recall_{str(args.fixed_recall).replace('.', '_')}"
        ),
        "eval_run": str(Path("runs") / f"{args.run_prefix}"),
    }
    result_path = Path("runs") / args.run_prefix / "nni_matrix_results.jsonl"
    append_jsonl(result, result_path)
    save_json(result, Path("runs") / args.run_prefix / "last_trial_result.json")
    _report_to_nni(result)


if __name__ == "__main__":
    main()
