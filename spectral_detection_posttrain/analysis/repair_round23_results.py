from __future__ import annotations

import argparse
import json
from pathlib import Path

from spectral_detection_posttrain.experiments.nni_rlvr_trial import (
    build_round23_result_row,
    collect_eval_status,
    compute_round23_objective,
)

ROUND23_PRESETS = {
    "null_no_update": "rlvr_null_no_update_cls_adamw",
    "det_only_cls": "rlvr_det_only_cls_cls_adamw",
    "signed_iou_0003_kl10": "rlvr_signed_iou_0003_kl10_cls_adamw",
    "signed_ramp_0003_kl10": "rlvr_signed_ramp_0003_kl10_cls_adamw",
    "signed_shuffled_0003_kl10": "rlvr_signed_shuffled_0003_kl10_cls_adamw",
    "weighted_ce_iou_0003_kl10": "rlvr_weighted_ce_iou_0003_kl10_cls_adamw",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair Round 2.3 RLVR result rows.")
    parser.add_argument("--run-prefix", default="nni_rlvr_round23")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _params_from_result(name: str, result: dict) -> dict:
    return {
        "name": name,
        "signal": result.get("signal", "none"),
        "reward_lambda": result.get("reward_lambda", 0.0),
        "policy_loss_weight": result.get("policy_loss_weight", 0.0),
        "det_loss_weight": result.get("det_loss_weight", 0.0),
        "baseline_kl_weight": result.get("baseline_kl_weight", 0.0),
        "box_loss_weight": result.get("box_loss_weight", 0.0),
        "unfreeze": result.get("unfreeze", "cls"),
        "optimizer": result.get("optimizer", "adamw"),
        "temperature": result.get("temperature", 1.0),
        "max_candidates": result.get("max_candidates", 40),
        "reward_score_threshold": result.get("reward_score_threshold", 0.2),
        "rollout_source": result.get("rollout_source", "baseline"),
        "policy_objective": result.get("policy_objective", "signed"),
    }


def _load_eval_metrics(runs_root: Path, run_dir_name: str) -> dict:
    metrics = {}
    clean_path = runs_root / f"{run_dir_name}_eval_clean" / "eval_metrics.json"
    edge_path = runs_root / f"{run_dir_name}_eval_object_edge_checkerboard" / "eval_metrics.json"
    if clean_path.exists():
        metrics["clean"] = _load_json(clean_path)
    if edge_path.exists():
        metrics["object_edge_checkerboard"] = _load_json(edge_path)
    return metrics


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    round_root = runs_root / args.run_prefix
    baseline = _load_json(round_root / "baseline_metrics.json")
    output = Path(args.output) if args.output else round_root / "nni_rlvr_results_repaired.jsonl"

    rows = []
    for preset_name, run_dir_name in ROUND23_PRESETS.items():
        run_dir = round_root / run_dir_name
        result = _load_json(run_dir / "rlvr_result.json")
        params = _params_from_result(preset_name, result)
        metrics = _load_eval_metrics(runs_root, run_dir_name)
        status = collect_eval_status(metrics)
        if status == "ok":
            objective = compute_round23_objective(metrics, baseline)
        else:
            objective = {"default": -1.0, "constraint_failed": status}
        checkpoint = run_dir / "checkpoint_best.pth"
        if not checkpoint.exists():
            checkpoint = run_dir / "checkpoint_last.pth"
        row = build_round23_result_row(
            params=params, metrics=metrics, objective=objective,
            run_name=run_dir_name,
            checkpoint=str(checkpoint) if checkpoint.exists() else "",
            eval_status=status,
        )
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print({"output": str(output), "rows": len(rows),
           "statuses": {row["name"]: row["eval_status"] for row in rows}})


if __name__ == "__main__":
    main()
