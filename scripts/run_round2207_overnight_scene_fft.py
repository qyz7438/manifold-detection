from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"E:\anaconda\01\envs\RLimage\python.exe")
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "round2129_nwpu_posttrain_smoke.py"
BASELINE_AP75 = 0.2939084504400764


BASE_ARGS: dict[str, Any] = {
    "limit_train": 100000,
    "limit_val": 100000,
    "batch_size": 2,
    "epochs": 5,
    "max_proposals": 100,
    "proposal_source": "rollout",
    "rollout_score_threshold": 0.001,
    "rollout_detections_per_img": 300,
    "score_threshold": 0.05,
    "sigma": 0.1,
    "lr": 3e-5,
    "adapter_lr": 3e-5,
    "predictor_lr": 1e-5,
    "cls_score_lr": 5e-6,
    "trainable_mode": "predictor",
    "det_loss_weight": 0.1,
    "policy_loss_weight": 0.001,
    "kl_weight": 1.0,
    "kl_cls_weight": 1.0,
    "kl_box_weight": 1.0,
    "rescue_loss_weight": 0.05,
    "rescue_target_mode": "increment",
    "rescue_increment_delta": 0.2,
    "rescue_increment_cap": 0.8,
    "score_budget_loss_weight": 0.02,
    "score_budget_delta": 0.05,
    "bbox_rescue_loss_weight": 0.2,
    "bbox_rescue_weight_temperature": 0.2,
    "bbox_localization_loss": "giou",
    "rescue_verifier_mode": "raw_ifft_scene",
    "rescue_verifier_gate": 0.0,
    "rescue_verifier_weight_mode": "hard",
    "rescue_raw_ifft_scene_groups": ["maritime", "vehicle", "compact"],
    "rescue_raw_ifft_scene_target_precision": 0.7,
    "rescue_raw_ifft_scene_min_positives": 2,
    "rescue_raw_ifft_score_method": "train_effect_sum",
    "rescue_raw_ifft_margin_std_frac": 0.0,
    "rescue_raw_ifft_target_precision": 0.8,
    "rescue_raw_ifft_features": ["fft_edge_truncation@64", "phase_edge@64", "phase_abs_high@11"],
    "rescue_hd_fusion_pca_components": 96,
    "rescue_hd_fusion_hd_scorer": "logistic",
    "rescue_hd_fusion_method": "train_effect",
    "safety_max_fp_rate_delta": 0.02,
    "safety_max_high_conf_fp_rate_delta": 0.03,
    "safety_max_ece_delta": 0.03,
    "cls_adapter_scale": 0.25,
}


SEED_TRIALS: list[dict[str, Any]] = [
    {"name": "seed_scene_t07", "reason": "baseline scene-wise FFT recipe"},
    {"name": "seed_scene_t06", "reason": "increase verifier coverage", "rescue_raw_ifft_scene_target_precision": 0.6},
    {"name": "seed_scene_t08", "reason": "increase verifier precision", "rescue_raw_ifft_scene_target_precision": 0.8},
    {"name": "seed_scene_mc", "reason": "remove weak/disabled vehicle group", "rescue_raw_ifft_scene_groups": ["maritime", "compact"]},
    {"name": "seed_scene_maritime", "reason": "exploit strongest ship/harbor signal", "rescue_raw_ifft_scene_groups": ["maritime"]},
    {"name": "seed_scene_vehicle_t06", "reason": "test dedicated vehicle signal", "rescue_raw_ifft_scene_groups": ["vehicle"], "rescue_raw_ifft_scene_target_precision": 0.6},
    {"name": "seed_scene_sigmoid05", "reason": "soft gate against threshold brittleness", "rescue_verifier_weight_mode": "sigmoid", "rescue_verifier_weight_temperature": 0.5},
    {"name": "seed_scene_bbox03", "reason": "stronger AP75 localization signal", "bbox_rescue_loss_weight": 0.3},
    {"name": "seed_scene_bbox01", "reason": "weaker localization to reduce drift", "bbox_rescue_loss_weight": 0.1},
    {"name": "seed_scene_bbox_only", "reason": "remove confidence rescue, keep localization", "rescue_loss_weight": 0.0},
    {"name": "seed_scene_delta_small", "reason": "more conservative confidence calibration", "rescue_increment_delta": 0.1, "rescue_increment_cap": 0.6},
    {"name": "seed_scene_lr2e5", "reason": "lower LR stability", "lr": 2e-5, "adapter_lr": 2e-5},
    {"name": "seed_scene_lr5e5", "reason": "stronger update with safety guards", "lr": 5e-5, "adapter_lr": 5e-5},
    {"name": "seed_scene_adapter", "reason": "adapter-only capacity control", "trainable_mode": "adapter"},
    {
        "name": "seed_raw_bbox_only",
        "reason": "raw iFFT non-scene control",
        "rescue_verifier_mode": "raw_ifft",
        "rescue_loss_weight": 0.0,
        "bbox_rescue_loss_weight": 0.3,
    },
    {
        "name": "seed_hd_fusion",
        "reason": "HD fusion control",
        "rescue_verifier_mode": "raw_ifft_hd_fusion",
    },
]


ARG_FLAGS = {
    "limit_train": "--limit-train",
    "limit_val": "--limit-val",
    "batch_size": "--batch-size",
    "epochs": "--epochs",
    "max_proposals": "--max-proposals",
    "proposal_source": "--proposal-source",
    "rollout_score_threshold": "--rollout-score-threshold",
    "rollout_detections_per_img": "--rollout-detections-per-img",
    "score_threshold": "--score-threshold",
    "sigma": "--sigma",
    "lr": "--lr",
    "adapter_lr": "--adapter-lr",
    "predictor_lr": "--predictor-lr",
    "cls_score_lr": "--cls-score-lr",
    "trainable_mode": "--trainable-mode",
    "det_loss_weight": "--det-loss-weight",
    "policy_loss_weight": "--policy-loss-weight",
    "kl_weight": "--kl-weight",
    "kl_cls_weight": "--kl-cls-weight",
    "kl_box_weight": "--kl-box-weight",
    "rescue_loss_weight": "--rescue-loss-weight",
    "rescue_target_mode": "--rescue-target-mode",
    "rescue_increment_delta": "--rescue-increment-delta",
    "rescue_increment_cap": "--rescue-increment-cap",
    "score_budget_loss_weight": "--score-budget-loss-weight",
    "score_budget_delta": "--score-budget-delta",
    "bbox_rescue_loss_weight": "--bbox-rescue-loss-weight",
    "bbox_rescue_weight_temperature": "--bbox-rescue-weight-temperature",
    "bbox_localization_loss": "--bbox-localization-loss",
    "chain_bbox_loss_weight": "--chain-bbox-loss-weight",
    "rescue_verifier_mode": "--rescue-verifier-mode",
    "rescue_verifier_gate": "--rescue-verifier-gate",
    "rescue_verifier_weight_mode": "--rescue-verifier-weight-mode",
    "rescue_verifier_weight_temperature": "--rescue-verifier-weight-temperature",
    "rescue_raw_ifft_scene_groups": "--rescue-raw-ifft-scene-groups",
    "rescue_raw_ifft_scene_target_precision": "--rescue-raw-ifft-scene-target-precision",
    "rescue_raw_ifft_scene_min_positives": "--rescue-raw-ifft-scene-min-positives",
    "rescue_raw_ifft_score_method": "--rescue-raw-ifft-score-method",
    "rescue_raw_ifft_margin_std_frac": "--rescue-raw-ifft-margin-std-frac",
    "rescue_raw_ifft_target_precision": "--rescue-raw-ifft-target-precision",
    "rescue_raw_ifft_features": "--rescue-raw-ifft-features",
    "rescue_hd_fusion_pca_components": "--rescue-hd-fusion-pca-components",
    "rescue_hd_fusion_hd_scorer": "--rescue-hd-fusion-hd-scorer",
    "rescue_hd_fusion_method": "--rescue-hd-fusion-method",
    "safety_max_fp_rate_delta": "--safety-max-fp-rate-delta",
    "safety_max_high_conf_fp_rate_delta": "--safety-max-high-conf-fp-rate-delta",
    "safety_max_ece_delta": "--safety-max-ece-delta",
    "cls_adapter_scale": "--cls-adapter-scale",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive 9-hour scene-wise FFT optimization runner.")
    parser.add_argument("--run-prefix", default="round2207_adaptive_scene_fft")
    parser.add_argument("--hours", type=float, default=9.0)
    parser.add_argument("--results-dir", type=Path, default=Path("runs/round2207_adaptive_scene_fft"))
    parser.add_argument("--start-index", type=int, default=0)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_trial_key(config: dict[str, Any]) -> str:
    ignored = {"name", "reason"}
    payload = {key: value for key, value in config.items() if key not in ignored}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def command_for_trial(run_name: str, trial: dict[str, Any]) -> list[str]:
    config = dict(BASE_ARGS)
    config.update({key: value for key, value in trial.items() if key not in {"name", "reason"}})
    command = [str(PYTHON), str(TRAIN_SCRIPT), "--run-name", run_name, "--rescue-mode", "--record-grad-diagnostics"]
    for key, flag in ARG_FLAGS.items():
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, list):
            command.append(flag)
            command.extend(str(item) for item in value)
        else:
            command.extend([flag, str(value)])
    return command


def summarize_run(run_dir: Path) -> dict[str, Any]:
    baseline = read_json(run_dir / "baseline_eval_metrics.json")
    final = read_json(run_dir / "eval_metrics.json")
    verifier = read_json(run_dir / "verifier_offline_report.json")
    diagnostics = read_json(run_dir / "rescue_diagnostics.json")
    reference = read_json(run_dir / "rescue_reference_stats.json")
    history_payload = read_json(run_dir / "metrics_train.json")
    history = history_payload.get("history", history_payload if isinstance(history_payload, list) else [])
    best = None
    if history:
        best = max(history, key=lambda row: float(row.get("ap75", -1.0)))
    baseline_ap75 = baseline.get("ap75", BASELINE_AP75)
    groups = []
    for group in reference.get("raw_ifft_scene_groups", []):
        groups.append(
            {
                "name": group.get("name"),
                "enabled": group.get("enabled"),
                "candidate_count": group.get("candidate_count"),
                "positive_count": group.get("positive_count"),
                "negative_count": group.get("negative_count"),
                "threshold": group.get("threshold"),
                "calibration": group.get("calibration"),
            }
        )
    return {
        "baseline_ap75": baseline_ap75,
        "baseline_ap50": baseline.get("ap50"),
        "final_ap50": final.get("ap50"),
        "final_ap75": final.get("ap75"),
        "delta_final_ap75": (
            float(final["ap75"]) - float(baseline_ap75)
            if "ap75" in final and baseline_ap75 is not None
            else None
        ),
        "best_epoch": best.get("epoch") if best else None,
        "best_ap75": best.get("ap75") if best else None,
        "delta_best_ap75": (
            float(best["ap75"]) - float(baseline_ap75)
            if best and "ap75" in best and baseline_ap75 is not None
            else None
        ),
        "final_fp_rate": final.get("false_positive_rate"),
        "final_ece": final.get("ece"),
        "final_predictions": final.get("num_predictions"),
        "verifier_auc": verifier.get("auc"),
        "verifier_precision_at_threshold": verifier.get("precision_at_threshold"),
        "verifier_recall_at_threshold": verifier.get("recall_at_threshold"),
        "verifier_selected_at_threshold": verifier.get("selected_at_threshold"),
        "verifier_positive_lchi_rate": diagnostics.get("verifier_positive_lchi_rate"),
        "verifier_positive_low_conf_precision": diagnostics.get("verifier_positive_low_conf_precision"),
        "lchi_prob_delta_mean": diagnostics.get("lchi_prob_delta_mean"),
        "scene_groups": groups,
    }


def score_summary(summary: dict[str, Any]) -> float:
    delta_best = float(summary.get("delta_best_ap75") or -1.0)
    delta_final = float(summary.get("delta_final_ap75") or -1.0)
    precision = float(summary.get("verifier_positive_low_conf_precision") or 0.0)
    fp_rate = float(summary.get("final_fp_rate") or 1.0)
    ece = float(summary.get("final_ece") or 1.0)
    return delta_best + 0.25 * delta_final + 0.001 * precision - 0.002 * max(0.0, fp_rate - 0.48) - 0.001 * max(0.0, ece - 0.086)


def best_completed(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    completed = [row for row in results if row.get("event") == "complete" and row.get("returncode") == 0]
    if not completed:
        return None
    return max(completed, key=lambda row: score_summary(row.get("summary", {})))


def mutate_from_best(best_row: dict[str, Any], attempt: int) -> list[dict[str, Any]]:
    trial = dict(best_row["trial_config"])
    base_name = str(trial.get("name", "best"))
    mutations = []
    summary = best_row.get("summary", {})
    coverage = float(summary.get("verifier_recall_at_threshold") or 0.0)
    precision = float(summary.get("verifier_precision_at_threshold") or 0.0)
    delta_best = float(summary.get("delta_best_ap75") or 0.0)

    if int(trial.get("epochs", BASE_ARGS["epochs"])) < 15:
        longer = dict(trial)
        longer["name"] = f"exploit_{base_name}_15ep"
        longer["reason"] = f"extend best 5ep trial; delta_best={delta_best:.6f}"
        longer["epochs"] = 15
        mutations.append(longer)

    if int(trial.get("epochs", BASE_ARGS["epochs"])) < 20 and delta_best > 0.002:
        longer20 = dict(trial)
        longer20["name"] = f"exploit_{base_name}_20ep"
        longer20["reason"] = f"extend strong trial to 20ep; delta_best={delta_best:.6f}"
        longer20["epochs"] = 20
        mutations.append(longer20)

    if coverage < 0.45:
        looser = dict(trial)
        looser["name"] = f"mut_{base_name}_looser_gate_{attempt}"
        looser["reason"] = "coverage low, reduce target precision"
        looser["rescue_raw_ifft_scene_target_precision"] = max(
            0.5,
            float(trial.get("rescue_raw_ifft_scene_target_precision", BASE_ARGS["rescue_raw_ifft_scene_target_precision"])) - 0.1,
        )
        mutations.append(looser)

    if precision < 0.7:
        stricter = dict(trial)
        stricter["name"] = f"mut_{base_name}_stricter_gate_{attempt}"
        stricter["reason"] = "precision low, increase target precision"
        stricter["rescue_raw_ifft_scene_target_precision"] = min(
            0.9,
            float(trial.get("rescue_raw_ifft_scene_target_precision", BASE_ARGS["rescue_raw_ifft_scene_target_precision"])) + 0.1,
        )
        mutations.append(stricter)

    for weight in (0.1, 0.2, 0.3):
        if abs(float(trial.get("bbox_rescue_loss_weight", BASE_ARGS["bbox_rescue_loss_weight"])) - weight) > 1e-12:
            candidate = dict(trial)
            candidate["name"] = f"mut_{base_name}_bbox{str(weight).replace('.', 'p')}_{attempt}"
            candidate["reason"] = "scan bbox rescue strength around current best"
            candidate["bbox_rescue_loss_weight"] = weight
            mutations.append(candidate)

    if float(trial.get("rescue_loss_weight", BASE_ARGS["rescue_loss_weight"])) > 0.0:
        no_conf = dict(trial)
        no_conf["name"] = f"mut_{base_name}_bbox_only_{attempt}"
        no_conf["reason"] = "test whether confidence rescue hurts AP75"
        no_conf["rescue_loss_weight"] = 0.0
        mutations.append(no_conf)

    if str(trial.get("rescue_verifier_weight_mode", BASE_ARGS["rescue_verifier_weight_mode"])) == "hard":
        soft = dict(trial)
        soft["name"] = f"mut_{base_name}_soft_gate_{attempt}"
        soft["reason"] = "use sigmoid weights to increase gradient continuity"
        soft["rescue_verifier_weight_mode"] = "sigmoid"
        soft["rescue_verifier_weight_temperature"] = 0.5
        mutations.append(soft)

    for lr in (2e-5, 3e-5, 5e-5):
        if abs(float(trial.get("lr", BASE_ARGS["lr"])) - lr) > 1e-12:
            candidate = dict(trial)
            candidate["name"] = f"mut_{base_name}_lr{lr:.0e}_{attempt}".replace("-", "m")
            candidate["reason"] = "scan LR around current best"
            candidate["lr"] = lr
            candidate["adapter_lr"] = lr
            mutations.append(candidate)

    if str(trial.get("trainable_mode", BASE_ARGS["trainable_mode"])) != "adapter":
        adapter = dict(trial)
        adapter["name"] = f"mut_{base_name}_adapter_{attempt}"
        adapter["reason"] = "adapter-only version of current best"
        adapter["trainable_mode"] = "adapter"
        mutations.append(adapter)

    return mutations


def load_prior_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    start = datetime.now()
    deadline = start + timedelta(hours=float(args.hours))
    status_path = args.results_dir / "adaptive_status.json"
    results_path = args.results_dir / "adaptive_results.jsonl"
    driver_log_path = args.results_dir / "adaptive_driver.log"

    queue = list(SEED_TRIALS)
    seen: set[str] = set()
    results: list[dict[str, Any]] = load_prior_results(results_path)
    for row in results:
        config = row.get("trial_config")
        if isinstance(config, dict):
            seen.add(canonical_trial_key(config))

    env = os.environ.copy()
    pythonpath = str(PROJECT_ROOT)
    env["PYTHONPATH"] = pythonpath if not env.get("PYTHONPATH") else pythonpath + os.pathsep + env["PYTHONPATH"]

    status: dict[str, Any] = {
        "start_time": start.isoformat(timespec="seconds"),
        "deadline_time": deadline.isoformat(timespec="seconds"),
        "hours": float(args.hours),
        "run_prefix": str(args.run_prefix),
        "pid": os.getpid(),
        "state": "running",
        "completed": 0,
        "failed": 0,
        "last_update": now_iso(),
    }
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    trial_index = int(args.start_index)
    mutation_round = 0
    with driver_log_path.open("a", encoding="utf-8") as driver_log:
        driver_log.write(f"[{now_iso()}] START adaptive deadline={deadline.isoformat(timespec='seconds')} pid={os.getpid()}\n")
        while datetime.now() < deadline:
            if not queue:
                best = best_completed(results)
                if best is None:
                    break
                mutation_round += 1
                queue.extend(mutate_from_best(best, mutation_round))
            if not queue:
                break
            trial = queue.pop(0)
            merged_for_key = dict(BASE_ARGS)
            merged_for_key.update({key: value for key, value in trial.items() if key not in {"name", "reason"}})
            trial_key = canonical_trial_key(merged_for_key)
            if trial_key in seen:
                continue
            seen.add(trial_key)
            if datetime.now() >= deadline:
                break

            run_name = f"{args.run_prefix}/{trial_index:03d}_{trial['name']}"
            run_dir = PROJECT_ROOT / "runs" / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            command = command_for_trial(run_name, trial)
            start_trial = datetime.now()
            row = {
                "event": "start",
                "trial_index": trial_index,
                "trial_config": trial,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "start_time": start_trial.isoformat(timespec="seconds"),
                "deadline_time": deadline.isoformat(timespec="seconds"),
                "command": command,
            }
            append_jsonl(results_path, row)
            driver_log.write(
                f"[{now_iso()}] RUN {trial_index:03d} {trial['name']} reason={trial.get('reason', '')}\n"
            )
            driver_log.flush()

            stdout_path = run_dir / "adaptive_stdout.log"
            stderr_path = run_dir / "adaptive_stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=stdout, stderr=stderr)

            elapsed = time.time() - start_trial.timestamp()
            summary = summarize_run(run_dir)
            completed = {
                **row,
                "event": "complete",
                "end_time": now_iso(),
                "elapsed_seconds": elapsed,
                "returncode": int(result.returncode),
                "summary": summary,
            }
            append_jsonl(results_path, completed)
            results.append(completed)
            if result.returncode == 0:
                status["completed"] = int(status["completed"]) + 1
            else:
                status["failed"] = int(status["failed"]) + 1
            best = best_completed(results)
            status["last_update"] = now_iso()
            status["last_trial"] = completed
            status["current_best"] = best
            status["queue_length"] = len(queue)
            status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
            driver_log.write(
                f"[{now_iso()}] DONE {trial_index:03d} rc={result.returncode} "
                f"best_ap75={summary.get('best_ap75')} delta={summary.get('delta_best_ap75')} "
                f"final={summary.get('final_ap75')}\n"
            )
            driver_log.flush()
            trial_index += 1

    status["state"] = "complete"
    status["end_time"] = now_iso()
    status["last_update"] = now_iso()
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
