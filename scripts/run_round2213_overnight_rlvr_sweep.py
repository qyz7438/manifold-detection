from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"E:\anaconda\01\envs\RLimage\python.exe")
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "round2129_nwpu_posttrain_smoke.py"
RUNS_DIR = PROJECT_ROOT / "runs"
ROUND2211 = RUNS_DIR / "round2211_relax_det05_lr1e4_clean_hd_fusion_15ep"
ROUND2212 = RUNS_DIR / "round2212_fft_ranking_global_clean_15ep"
SUMMARY_PATH = RUNS_DIR / "round2213_overnight_summary.json"
BASELINE_AP75 = 0.2939084504400764
ROUND2211_AP75 = 0.3026143912147811


ARG_FLAGS = {
    "limit_train": "--limit-train",
    "limit_val": "--limit-val",
    "batch_size": "--batch-size",
    "epochs": "--epochs",
    "max_proposals": "--max-proposals",
    "proposal_source": "--proposal-source",
    "rollout_score_threshold": "--rollout-score-threshold",
    "rollout_detections_per_img": "--rollout-detections-per-img",
    "eval_detections_per_img": "--eval-detections-per-img",
    "score_threshold": "--score-threshold",
    "sigma": "--sigma",
    "lr": "--lr",
    "adapter_lr": "--adapter-lr",
    "predictor_lr": "--predictor-lr",
    "cls_score_lr": "--cls-score-lr",
    "weight_decay": "--weight-decay",
    "trainable_mode": "--trainable-mode",
    "det_loss_weight": "--det-loss-weight",
    "policy_loss_weight": "--policy-loss-weight",
    "kl_weight": "--kl-weight",
    "kl_cls_weight": "--kl-cls-weight",
    "kl_box_weight": "--kl-box-weight",
    "rescue_loss_weight": "--rescue-loss-weight",
    "rescue_low_conf_max": "--rescue-low-conf-max",
    "rescue_high_conf_min": "--rescue-high-conf-min",
    "rescue_high_iou_min": "--rescue-high-iou-min",
    "rescue_low_iou_max": "--rescue-low-iou-max",
    "rescue_positive_weight": "--rescue-positive-weight",
    "rescue_negative_weight": "--rescue-negative-weight",
    "rescue_hard_negative_verifier_gate": "--rescue-hard-negative-verifier-gate",
    "rescue_pairwise_loss_weight": "--rescue-pairwise-loss-weight",
    "rescue_pairwise_margin": "--rescue-pairwise-margin",
    "rescue_pairwise_negative_mode": "--rescue-pairwise-negative-mode",
    "rescue_target_mode": "--rescue-target-mode",
    "rescue_increment_delta": "--rescue-increment-delta",
    "rescue_increment_cap": "--rescue-increment-cap",
    "score_budget_loss_weight": "--score-budget-loss-weight",
    "score_budget_delta": "--score-budget-delta",
    "confidence_crossing_loss_weight": "--confidence-crossing-loss-weight",
    "confidence_crossing_margin": "--confidence-crossing-margin",
    "rescue_low_conf_source": "--rescue-low-conf-source",
    "rescue_positive_filter": "--rescue-positive-filter",
    "class_margin_loss_weight": "--class-margin-loss-weight",
    "class_margin_margin": "--class-margin-margin",
    "bbox_rescue_loss_weight": "--bbox-rescue-loss-weight",
    "bbox_rescue_weight_temperature": "--bbox-rescue-weight-temperature",
    "bbox_localization_loss": "--bbox-localization-loss",
    "chain_topk_per_gt": "--chain-topk-per-gt",
    "chain_bbox_loss_weight": "--chain-bbox-loss-weight",
    "chain_cls_margin_loss_weight": "--chain-cls-margin-loss-weight",
    "chain_cls_margin_margin": "--chain-cls-margin-margin",
    "chain_ranking_loss_weight": "--chain-ranking-loss-weight",
    "chain_ranking_margin": "--chain-ranking-margin",
    "chain_dangerous_negative_min_score": "--chain-dangerous-negative-min-score",
    "verifier_ranking_loss_weight": "--verifier-ranking-loss-weight",
    "verifier_ranking_margin": "--verifier-ranking-margin",
    "verifier_ranking_positive_iou_min": "--verifier-ranking-positive-iou-min",
    "verifier_ranking_negative_iou_max": "--verifier-ranking-negative-iou-max",
    "verifier_ranking_positive_score_min": "--verifier-ranking-positive-score-min",
    "verifier_ranking_negative_score_max": "--verifier-ranking-negative-score-max",
    "verifier_ranking_max_pairs": "--verifier-ranking-max-pairs",
    "nms_aware_ranking_loss_weight": "--nms-aware-ranking-loss-weight",
    "nms_aware_ranking_margin": "--nms-aware-ranking-margin",
    "nms_aware_nms_iou": "--nms-aware-nms-iou",
    "nms_aware_ranking_mode": "--nms-aware-ranking-mode",
    "blocked_nms_loss_weight": "--blocked-nms-loss-weight",
    "blocked_nms_score_epsilon": "--blocked-nms-score-epsilon",
    "blocked_nms_iou": "--blocked-nms-iou",
    "blocked_nms_base_margin": "--blocked-nms-base-margin",
    "blocked_nms_iou_margin_scale": "--blocked-nms-iou-margin-scale",
    "blocked_nms_max_margin": "--blocked-nms-max-margin",
    "blocked_nms_rank_weight": "--blocked-nms-rank-weight",
    "blocked_nms_crossing_weight": "--blocked-nms-crossing-weight",
    "blocked_nms_ranking_mode": "--blocked-nms-ranking-mode",
    "pre_nms_rescue_loss_weight": "--pre-nms-rescue-loss-weight",
    "pre_nms_score_target": "--pre-nms-score-target",
    "pre_nms_low_conf_max": "--pre-nms-low-conf-max",
    "pre_nms_high_iou_min": "--pre-nms-high-iou-min",
    "pre_nms_topk_per_gt": "--pre-nms-topk-per-gt",
    "pre_nms_dpo_loss_weight": "--pre-nms-dpo-loss-weight",
    "pre_nms_dpo_beta": "--pre-nms-dpo-beta",
    "pre_nms_dpo_min_iou_gap": "--pre-nms-dpo-min-iou-gap",
    "pre_nms_dpo_max_pairs_per_gt": "--pre-nms-dpo-max-pairs-per-gt",
    "same_gt_duplicate_ranking_loss_weight": "--same-gt-duplicate-ranking-loss-weight",
    "same_gt_duplicate_ranking_margin": "--same-gt-duplicate-ranking-margin",
    "same_gt_duplicate_pair_source": "--same-gt-duplicate-pair-source",
    "same_gt_duplicate_nms_iou": "--same-gt-duplicate-nms-iou",
    "same_gt_duplicate_min_iou_gap": "--same-gt-duplicate-min-iou-gap",
    "rescue_verifier_mode": "--rescue-verifier-mode",
    "rescue_verifier_gate": "--rescue-verifier-gate",
    "rescue_verifier_weight_mode": "--rescue-verifier-weight-mode",
    "rescue_verifier_weight_temperature": "--rescue-verifier-weight-temperature",
    "rescue_fft_weight": "--rescue-fft-weight",
    "rescue_manifold_weight": "--rescue-manifold-weight",
    "rescue_fft_crop_size": "--rescue-fft-crop-size",
    "rescue_raw_ifft_features": "--rescue-raw-ifft-features",
    "rescue_raw_ifft_target_precision": "--rescue-raw-ifft-target-precision",
    "rescue_raw_ifft_margin_std_frac": "--rescue-raw-ifft-margin-std-frac",
    "rescue_raw_ifft_score_method": "--rescue-raw-ifft-score-method",
    "rescue_raw_ifft_scene_groups": "--rescue-raw-ifft-scene-groups",
    "rescue_raw_ifft_scene_target_precision": "--rescue-raw-ifft-scene-target-precision",
    "rescue_raw_ifft_scene_min_positives": "--rescue-raw-ifft-scene-min-positives",
    "rescue_hd_fusion_pca_components": "--rescue-hd-fusion-pca-components",
    "rescue_hd_fusion_hd_scorer": "--rescue-hd-fusion-hd-scorer",
    "rescue_hd_fusion_method": "--rescue-hd-fusion-method",
    "rescue_manifold_k": "--rescue-manifold-k",
    "rescue_manifold_gate_mode": "--rescue-manifold-gate-mode",
    "rescue_manifold_score_mode": "--rescue-manifold-score-mode",
    "rescue_manifold_fp_weight": "--rescue-manifold-fp-weight",
    "rescue_hard_negative_weight": "--rescue-hard-negative-weight",
    "rescue_margin_weight": "--rescue-margin-weight",
    "rescue_manifold_feature_source": "--rescue-manifold-feature-source",
    "rescue_manifold_feature_projection": "--rescue-manifold-feature-projection",
    "rescue_class_threshold_min_precision": "--rescue-class-threshold-min-precision",
    "rescue_class_threshold_min_positives": "--rescue-class-threshold-min-positives",
    "rescue_reference_refresh_epochs": "--rescue-reference-refresh-epochs",
    "selection_metric": "--selection-metric",
    "selection_min_delta": "--selection-min-delta",
    "safety_max_prediction_ratio": "--safety-max-prediction-ratio",
    "safety_max_prediction_delta": "--safety-max-prediction-delta",
    "safety_max_fp_rate_delta": "--safety-max-fp-rate-delta",
    "safety_max_high_conf_fp_rate_delta": "--safety-max-high-conf-fp-rate-delta",
    "safety_max_ece_delta": "--safety-max-ece-delta",
    "cls_adapter_scale": "--cls-adapter-scale",
}

BOOL_FLAGS = {
    "rescue_mode": "--rescue-mode",
    "record_grad_diagnostics": "--record-grad-diagnostics",
    "rescue_include_low_conf_negatives": "--rescue-include-low-conf-negatives",
    "rescue_use_hard_negative_mining": "--rescue-use-hard-negative-mining",
    "rescue_use_bucket_thresholds": "--rescue-use-bucket-thresholds",
    "rescue_use_class_thresholds": "--rescue-use-class-thresholds",
    "skip_initial_rescue_diagnostics": "--skip-initial-rescue-diagnostics",
    "skip_epoch_rescue_diagnostics": "--skip-epoch-rescue-diagnostics",
    "skip_final_rescue_diagnostics": "--skip-final-rescue-diagnostics",
    "skip_offline_verifier_report": "--skip-offline-verifier-report",
    "selection_lower_is_better": "--selection-lower-is-better",
    "same_gt_duplicate_detach_suppressor": "--same-gt-duplicate-detach-suppressor",
}

OPTIONAL_BOOL_FLAGS = {
    "chain_cls_margin_include_background": (
        "--chain-cls-margin-include-background",
        "--no-chain-cls-margin-include-background",
    ),
    "nms_aware_require_suppressor_score_ge_candidate": (
        "--nms-aware-require-suppressor-score-ge-candidate",
        "--no-nms-aware-require-suppressor-score-ge-candidate",
    ),
    "blocked_nms_require_suppressor_score_ge_candidate": (
        "--blocked-nms-require-suppressor-score-ge-candidate",
        "--no-blocked-nms-require-suppressor-score-ge-candidate",
    ),
    "pre_nms_dpo_require_rejected_score_ge_chosen": (
        "--pre-nms-dpo-require-rejected-score-ge-chosen",
        "--no-pre-nms-dpo-require-rejected-score-ge-chosen",
    ),
    "same_gt_duplicate_require_suppressor_score_ge_candidate": (
        "--same-gt-duplicate-require-suppressor-score-ge-candidate",
        "--no-same-gt-duplicate-require-suppressor-score-ge-candidate",
    ),
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_base_config() -> dict[str, Any]:
    config = read_json(ROUND2211 / "round_config.json")
    if not config:
        raise RuntimeError(f"Missing base config: {ROUND2211 / 'round_config.json'}")
    config["rescue_mode"] = True
    config["record_grad_diagnostics"] = True
    return config


def command_for_run(run_name: str, overrides: dict[str, Any]) -> list[str]:
    config = load_base_config()
    config.update(overrides)
    command = [str(PYTHON), str(TRAIN_SCRIPT), "--run-name", run_name]
    for key, flag in BOOL_FLAGS.items():
        if bool(config.get(key, False)):
            command.append(flag)
    for key, flags in OPTIONAL_BOOL_FLAGS.items():
        if key in config and config[key] is not None:
            command.append(flags[0] if bool(config[key]) else flags[1])
    for key, flag in ARG_FLAGS.items():
        if key not in config or config[key] is None:
            continue
        value = config[key]
        if isinstance(value, list):
            command.append(flag)
            command.extend(str(item) for item in value)
        else:
            command.extend([flag, str(value)])
    return command


def summarize_run(run_name: str) -> dict[str, Any]:
    run_dir = RUNS_DIR / run_name
    baseline = read_json(run_dir / "baseline_eval_metrics.json")
    final = read_json(run_dir / "eval_metrics.json")
    diagnostics = read_json(run_dir / "rescue_diagnostics.json")
    verifier = read_json(run_dir / "verifier_offline_report.json")
    config = read_json(run_dir / "round_config.json")
    history_payload = read_json(run_dir / "metrics_train.json")
    history = history_payload.get("history", []) if isinstance(history_payload, dict) else []
    best = max(history, key=lambda row: float(row.get("ap75", -1.0))) if history else {}
    baseline_ap75 = float(baseline.get("ap75", BASELINE_AP75))
    final_ap75 = final.get("ap75")
    best_ap75 = best.get("ap75")
    last = history[-1] if history else {}
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
        "verifier_auc": verifier.get("auc"),
        "verifier_precision_at_threshold": verifier.get("precision_at_threshold"),
        "verifier_recall_at_threshold": verifier.get("recall_at_threshold"),
        "verifier_selected_at_threshold": verifier.get("selected_at_threshold"),
        "lchi_prob_delta_mean": diagnostics.get("lchi_prob_delta_mean"),
        "verifier_positive_lchi_prob_delta_mean": diagnostics.get("verifier_positive_lchi_prob_delta_mean"),
        "last_verifier_ranking_pair_count": last.get("verifier_ranking_pair_count"),
        "last_verifier_ranking_loss": last.get("verifier_ranking_loss"),
        "config": {
            "epochs": config.get("epochs"),
            "lr": config.get("lr"),
            "kl_weight": config.get("kl_weight"),
            "det_loss_weight": config.get("det_loss_weight"),
            "policy_loss_weight": config.get("policy_loss_weight"),
            "verifier_ranking_loss_weight": config.get("verifier_ranking_loss_weight"),
            "rescue_verifier_weight_mode": config.get("rescue_verifier_weight_mode"),
            "rescue_high_iou_min": config.get("rescue_high_iou_min"),
        },
    }


def process_alive(pid: int) -> bool:
    command = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        f"$p = Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue; if ($p) {{ exit 0 }} else {{ exit 1 }}",
    ]
    return subprocess.run(command, cwd=PROJECT_ROOT).returncode == 0


def load_summary() -> dict[str, Any]:
    if SUMMARY_PATH.exists():
        return read_json(SUMMARY_PATH)
    return {
        "created_at": now_iso(),
        "baseline_ap75": BASELINE_AP75,
        "round2211_ap75": ROUND2211_AP75,
        "events": [],
        "runs": {},
        "current_best": {
            "run": "round2211_relax_det05_lr1e4_clean_hd_fusion_15ep",
            "ap75": ROUND2211_AP75,
        },
    }


def append_event(summary: dict[str, Any], event: dict[str, Any]) -> None:
    event = {"timestamp": now_iso(), **event}
    summary.setdefault("events", []).append(event)
    write_json(SUMMARY_PATH, summary)


def safety_clean(run_summary: dict[str, Any]) -> bool:
    fp = run_summary.get("final_fp_rate")
    pred = run_summary.get("final_predictions")
    ece = run_summary.get("final_ece")
    if fp is not None and float(fp) > 0.50:
        return False
    if pred is not None and int(pred) > 1600:
        return False
    if ece is not None and float(ece) > 0.12:
        return False
    return True


def select_queue(round2212_summary: dict[str, Any]) -> list[tuple[str, dict[str, Any], str]]:
    best_ap75 = float(round2212_summary.get("best_ap75") or -1.0)
    improved = best_ap75 >= ROUND2211_AP75 + 0.001
    safe = safety_clean(round2212_summary)
    if improved and safe:
        return [
            (
                "round2213_fft_ranking_w03_clean_15ep",
                {"verifier_ranking_loss_weight": 0.03},
                "2212 improved safely; test stronger verifier ranking",
            ),
            (
                "round2214_policy003_clean_15ep",
                {"policy_loss_weight": 0.003},
                "conservative policy signal increase",
            ),
            (
                "round2215_policy005_clean_15ep",
                {"policy_loss_weight": 0.005},
                "policy signal increase if 0.003 stays safe",
            ),
            (
                "round2216_softgate_clean_15ep",
                {"rescue_verifier_weight_mode": "sigmoid"},
                "soft verifier weighting after stable improvement",
            ),
        ]
    return [
        (
            "round2213_fft_ranking_w003_clean_15ep",
            {"verifier_ranking_loss_weight": 0.003},
            "2212 did not safely beat 2211; test weaker verifier ranking",
        ),
        (
            "round2214_lr1e4_30ep_clean",
            {"epochs": 30, "verifier_ranking_loss_weight": 0.0},
            "extend round2211 recipe because epoch 15 was still improving",
        ),
        (
            "round2215_policy003_clean_15ep",
            {"policy_loss_weight": 0.003, "verifier_ranking_loss_weight": 0.0},
            "conservative policy signal increase",
        ),
        (
            "round2216_policy005_clean_15ep",
            {"policy_loss_weight": 0.005, "verifier_ranking_loss_weight": 0.0},
            "second policy signal step if safe",
        ),
        (
            "round2217_softgate_clean_15ep",
            {"rescue_verifier_weight_mode": "sigmoid", "verifier_ranking_loss_weight": 0.0},
            "soft verifier weighting as coverage expansion",
        ),
    ]


def run_complete(run_name: str) -> bool:
    return (RUNS_DIR / run_name / "eval_metrics.json").exists()


def monitor_running_process(
    process: subprocess.Popen[Any],
    run_name: str,
    summary: dict[str, Any],
    scan_seconds: int,
    deadline: datetime,
) -> int:
    while process.poll() is None:
        append_event(
            summary,
            {
                "event": "scan",
                "run": run_name,
                "pid": process.pid,
                "returncode": None,
                "deadline": deadline.isoformat(timespec="seconds"),
            },
        )
        if datetime.now() >= deadline:
            append_event(summary, {"event": "deadline_reached_waiting_for_run", "run": run_name, "pid": process.pid})
            return process.wait()
        time.sleep(scan_seconds)
    return int(process.returncode or 0)


def launch_run(run_name: str, overrides: dict[str, Any], reason: str, summary: dict[str, Any], scan_seconds: int, deadline: datetime) -> None:
    run_dir = RUNS_DIR / run_name
    if run_complete(run_name):
        run_summary = summarize_run(run_name)
        summary.setdefault("runs", {})[run_name] = run_summary
        append_event(summary, {"event": "skip_existing_complete", "run": run_name, "reason": reason})
        return
    command = command_for_run(run_name, overrides)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "supervisor_train.log"
    append_event(
        summary,
        {"event": "launch", "run": run_name, "reason": reason, "overrides": overrides, "command": command},
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"timestamp": now_iso(), "event": "launch", "command": command}, ensure_ascii=False) + "\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env={**dict(**__import__("os").environ), "PYTHONPATH": str(PROJECT_ROOT)},
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        returncode = monitor_running_process(process, run_name, summary, scan_seconds, deadline)
        log.write(json.dumps({"timestamp": now_iso(), "event": "finished", "returncode": returncode}, ensure_ascii=False) + "\n")
    run_summary = summarize_run(run_name)
    summary.setdefault("runs", {})[run_name] = run_summary
    if run_summary.get("best_ap75") is not None:
        current_best = summary.get("current_best", {})
        if float(run_summary["best_ap75"]) > float(current_best.get("ap75", -1.0)):
            summary["current_best"] = {"run": run_name, "ap75": run_summary["best_ap75"]}
    append_event(summary, {"event": "complete", "run": run_name, "returncode": returncode, "summary": run_summary})


def wait_for_round2212(summary: dict[str, Any], current_pid: int | None, scan_seconds: int, deadline: datetime) -> dict[str, Any]:
    while not run_complete("round2212_fft_ranking_global_clean_15ep"):
        alive = process_alive(current_pid) if current_pid else None
        epoch_count = len(list(ROUND2212.glob("rescue_diagnostics_epoch_*.json")))
        append_event(
            summary,
            {
                "event": "scan_existing",
                "run": "round2212_fft_ranking_global_clean_15ep",
                "pid": current_pid,
                "pid_alive": alive,
                "epoch_diag_count": epoch_count,
            },
        )
        if alive is False and not run_complete("round2212_fft_ranking_global_clean_15ep"):
            append_event(summary, {"event": "existing_run_missing_final_eval", "run": "round2212_fft_ranking_global_clean_15ep"})
            break
        if datetime.now() >= deadline:
            append_event(summary, {"event": "deadline_reached_waiting_for_existing", "run": "round2212_fft_ranking_global_clean_15ep"})
            return {}
        time.sleep(scan_seconds)
    run_summary = summarize_run("round2212_fft_ranking_global_clean_15ep")
    summary.setdefault("runs", {})["round2212_fft_ranking_global_clean_15ep"] = run_summary
    if run_summary.get("best_ap75") is not None:
        current_best = summary.get("current_best", {})
        if float(run_summary["best_ap75"]) > float(current_best.get("ap75", -1.0)):
            summary["current_best"] = {"run": "round2212_fft_ranking_global_clean_15ep", "ap75": run_summary["best_ap75"]}
    append_event(summary, {"event": "existing_complete", "run": "round2212_fft_ranking_global_clean_15ep", "summary": run_summary})
    return run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round 2213+ supervised overnight NWPU RLVR sweep.")
    parser.add_argument("--hours", type=float, default=9.0)
    parser.add_argument("--scan-seconds", type=int, default=1500)
    parser.add_argument("--current-pid", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_summary()
    deadline = datetime.now() + timedelta(hours=float(args.hours))
    append_event(
        summary,
        {
            "event": "supervisor_start",
            "scan_seconds": int(args.scan_seconds),
            "hours": float(args.hours),
            "current_pid": args.current_pid,
            "deadline": deadline.isoformat(timespec="seconds"),
        },
    )
    round2212_summary = wait_for_round2212(summary, args.current_pid, int(args.scan_seconds), deadline)
    if not round2212_summary:
        append_event(summary, {"event": "stop_without_round2212_summary"})
        return
    queue = select_queue(round2212_summary)
    append_event(summary, {"event": "queue_selected", "queue": [item[0] for item in queue]})
    for run_name, overrides, reason in queue:
        if datetime.now() >= deadline:
            append_event(summary, {"event": "deadline_reached_before_launch", "next_run": run_name})
            break
        launch_run(run_name, overrides, reason, summary, int(args.scan_seconds), deadline)
    append_event(summary, {"event": "supervisor_done", "current_best": summary.get("current_best")})


if __name__ == "__main__":
    main()
