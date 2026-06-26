from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_round2213_overnight_rlvr_sweep import (
    BASELINE_AP75,
    PROJECT_ROOT,
    RUNS_DIR,
    command_for_run,
    now_iso,
    read_json,
    write_json,
)


SUMMARY_PATH = RUNS_DIR / "round2218_short_dpo_sweep_summary.json"


def summarize_run(run_name: str) -> dict[str, Any]:
    run_dir = RUNS_DIR / run_name
    baseline = read_json(run_dir / "baseline_eval_metrics.json")
    final = read_json(run_dir / "eval_metrics.json")
    diagnostics = read_json(run_dir / "rescue_diagnostics.json")
    config = read_json(run_dir / "round_config.json")
    history_payload = read_json(run_dir / "metrics_train.json")
    history = history_payload.get("history", []) if isinstance(history_payload, dict) else []
    best = max(history, key=lambda row: float(row.get("ap75", -1.0))) if history else {}
    last = history[-1] if history else {}
    baseline_ap75 = float(baseline.get("ap75", BASELINE_AP75))
    final_ap75 = final.get("ap75")
    best_ap75 = best.get("ap75")
    return {
        "run": run_name,
        "complete": bool(final),
        "epochs_recorded": len(history),
        "baseline_ap75": baseline_ap75,
        "final_ap50": final.get("ap50"),
        "final_ap75": final_ap75,
        "delta_final_ap75": float(final_ap75) - baseline_ap75 if final_ap75 is not None else None,
        "best_epoch": best.get("epoch"),
        "best_ap75": best_ap75,
        "delta_best_ap75": float(best_ap75) - baseline_ap75 if best_ap75 is not None else None,
        "final_predictions": final.get("num_predictions"),
        "final_fp_rate": final.get("false_positive_rate"),
        "final_ece": final.get("ece"),
        "lchi_prob_delta_mean": diagnostics.get("lchi_prob_delta_mean"),
        "verifier_positive_lchi_prob_delta_mean": diagnostics.get("verifier_positive_lchi_prob_delta_mean"),
        "last_pre_nms_dpo_loss": last.get("pre_nms_dpo_loss"),
        "last_pre_nms_dpo_pair_count": last.get("pre_nms_dpo_pair_count"),
        "last_pre_nms_dpo_win_rate": last.get("pre_nms_dpo_win_rate"),
        "last_pre_nms_dpo_preference_margin_mean": last.get("pre_nms_dpo_preference_margin_mean"),
        "last_grad_pre_nms_dpo_total_l2": last.get("grad_pre_nms_dpo_total_l2"),
        "config": {
            "epochs": config.get("epochs"),
            "lr": config.get("lr"),
            "trainable_mode": config.get("trainable_mode"),
            "det_loss_weight": config.get("det_loss_weight"),
            "policy_loss_weight": config.get("policy_loss_weight"),
            "kl_weight": config.get("kl_weight"),
            "rescue_loss_weight": config.get("rescue_loss_weight"),
            "bbox_rescue_loss_weight": config.get("bbox_rescue_loss_weight"),
            "score_budget_loss_weight": config.get("score_budget_loss_weight"),
            "pre_nms_dpo_loss_weight": config.get("pre_nms_dpo_loss_weight"),
            "pre_nms_dpo_max_pairs_per_gt": config.get("pre_nms_dpo_max_pairs_per_gt"),
            "pre_nms_topk_per_gt": config.get("pre_nms_topk_per_gt"),
        },
    }


def load_summary() -> dict[str, Any]:
    if SUMMARY_PATH.exists():
        return read_json(SUMMARY_PATH)
    return {
        "created_at": now_iso(),
        "baseline_ap75": BASELINE_AP75,
        "events": [],
        "runs": {},
        "current_best": {"run": None, "ap75": BASELINE_AP75},
    }


def append_event(summary: dict[str, Any], event: dict[str, Any]) -> None:
    summary.setdefault("events", []).append({"timestamp": now_iso(), **event})
    write_json(SUMMARY_PATH, summary)


def run_complete(run_name: str) -> bool:
    return (RUNS_DIR / run_name / "eval_metrics.json").exists()


def launch_run(
    run_name: str,
    overrides: dict[str, Any],
    reason: str,
    summary: dict[str, Any],
    scan_seconds: int,
    deadline: datetime,
) -> None:
    if run_complete(run_name):
        run_summary = summarize_run(run_name)
        summary.setdefault("runs", {})[run_name] = run_summary
        append_event(summary, {"event": "skip_existing_complete", "run": run_name, "reason": reason})
        return

    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    command = command_for_run(run_name, overrides)
    log_path = run_dir / "supervisor_train.log"
    append_event(summary, {"event": "launch", "run": run_name, "reason": reason, "overrides": overrides})
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"timestamp": now_iso(), "event": "launch", "command": command}, ensure_ascii=False) + "\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        next_scan = datetime.min
        while process.poll() is None:
            if datetime.now() >= next_scan:
                append_event(summary, {"event": "scan", "run": run_name, "pid": process.pid})
                next_scan = datetime.now() + timedelta(seconds=scan_seconds)
            if datetime.now() >= deadline:
                append_event(summary, {"event": "deadline_reached_waiting_for_run", "run": run_name, "pid": process.pid})
                break
            time.sleep(min(10, max(1, scan_seconds)))
        returncode = process.wait()
        log.write(json.dumps({"timestamp": now_iso(), "event": "finished", "returncode": returncode}, ensure_ascii=False) + "\n")

    run_summary = summarize_run(run_name)
    summary.setdefault("runs", {})[run_name] = run_summary
    if run_summary.get("best_ap75") is not None:
        current_best = summary.get("current_best", {})
        if float(run_summary["best_ap75"]) > float(current_best.get("ap75", -1.0)):
            summary["current_best"] = {"run": run_name, "ap75": run_summary["best_ap75"]}
    append_event(summary, {"event": "complete", "run": run_name, "returncode": returncode, "summary": run_summary})


def dpo_common(weight: float) -> dict[str, Any]:
    return {
        "epochs": 5,
        "pre_nms_dpo_loss_weight": weight,
        "pre_nms_dpo_beta": 1.0,
        "pre_nms_dpo_min_iou_gap": 0.05,
        "pre_nms_dpo_max_pairs_per_gt": 2,
        "pre_nms_topk_per_gt": 2,
        "pre_nms_high_iou_min": 0.75,
        "pre_nms_low_conf_max": 0.5,
        "pre_nms_dpo_require_rejected_score_ge_chosen": True,
        "skip_offline_verifier_report": True,
    }


def dpo_only(weight: float) -> dict[str, Any]:
    config = dpo_common(weight)
    config.update(
        {
            "rescue_loss_weight": 0.0,
            "bbox_rescue_loss_weight": 0.0,
            "score_budget_loss_weight": 0.0,
            "rescue_pairwise_loss_weight": 0.0,
            "verifier_ranking_loss_weight": 0.0,
            "pre_nms_rescue_loss_weight": 0.0,
            "rescue_verifier_mode": "none",
        }
    )
    return config


def dpo_plus_rescue(weight: float) -> dict[str, Any]:
    config = dpo_common(weight)
    config.update({"trainable_mode": "predictor"})
    return config


def dpo_cls_score_only(weight: float) -> dict[str, Any]:
    config = dpo_only(weight)
    config.update(
        {
            "trainable_mode": "cls_score",
            "adapter_lr": None,
            "predictor_lr": None,
            "cls_score_lr": 1.0e-4,
        }
    )
    return config


def build_queue() -> list[tuple[str, dict[str, Any], str]]:
    queue: list[tuple[str, dict[str, Any], str]] = []
    weights = [0.01, 0.03, 0.1]
    tags = {0.01: "w001", 0.03: "w003", 0.1: "w01"}
    for weight in weights:
        queue.append(
            (
                f"round2218_pre_nms_dpo_only_{tags[weight]}_5ep",
                dpo_only(weight),
                "pre-NMS DPO on classification logits, no rescue losses",
            )
        )
    for weight in weights:
        queue.append(
            (
                f"round2219_pre_nms_dpo_rescue_{tags[weight]}_5ep",
                dpo_plus_rescue(weight),
                "round2211 rescue recipe plus pre-NMS DPO preference",
            )
        )
    for weight in weights:
        queue.append(
            (
                f"round2220_pre_nms_dpo_cls_score_only_{tags[weight]}_5ep",
                dpo_cls_score_only(weight),
                "strict cls_score-only trainable scope for DPO isolation",
            )
        )
    return queue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round 2218 short clean DPO sweep.")
    parser.add_argument("--hours", type=float, default=9.0)
    parser.add_argument("--scan-seconds", type=int, default=1500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deadline = datetime.now() + timedelta(hours=float(args.hours))
    summary = load_summary()
    queue = build_queue()
    append_event(
        summary,
        {
            "event": "supervisor_start",
            "scan_seconds": int(args.scan_seconds),
            "hours": float(args.hours),
            "deadline": deadline.isoformat(timespec="seconds"),
            "queue": [item[0] for item in queue],
        },
    )
    for run_name, overrides, reason in queue:
        if datetime.now() >= deadline:
            append_event(summary, {"event": "deadline_reached_before_launch", "next_run": run_name})
            break
        launch_run(run_name, overrides, reason, summary, int(args.scan_seconds), deadline)
    append_event(summary, {"event": "supervisor_done", "current_best": summary.get("current_best")})


if __name__ == "__main__":
    main()
