from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FOUR_SCENES = [
    "clean",
    "object_edge_checkerboard",
    "object_inside_checkerboard",
    "near_object_checkerboard",
]

SCENE_PREFIX = {
    "clean": "clean",
    "object_edge_checkerboard": "edge",
    "object_inside_checkerboard": "inside",
    "near_object_checkerboard": "near",
}

METRICS = [
    "ap50", "ap75", "precision", "recall", "ece",
    "high_conf_fp_count", "high_conf_fp_rate", "num_predictions",
]


def scene_metric_key(scene: str, metric: str) -> str:
    return f"{SCENE_PREFIX[scene]}_{metric}"


def build_round30_result_row(
    params: dict[str, Any], metrics: dict[str, dict[str, Any]],
    objective: dict[str, Any], run_name: str, checkpoint: str, eval_status: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": params.get("name", ""),
        "default": float(objective.get("default", -1.0)),
        "constraint_failed": objective.get("constraint_failed", ""),
        "run_name": run_name,
        "checkpoint": checkpoint,
        "eval_status": eval_status,
        "signal": params.get("signal", ""),
        "reward_lambda": float(params.get("reward_lambda", 0.0)),
        "struct_weight": float(params.get("struct_weight", 0.0)),
        "policy_loss_weight": float(params.get("policy_loss_weight", 0.0)),
        "baseline_kl_weight": float(params.get("baseline_kl_weight", 0.0)),
        "temperature": float(params.get("temperature", 1.0)),
        "max_candidates": int(params.get("max_candidates", 0)),
        "reward_score_threshold": float(params.get("reward_score_threshold", 0.0)),
        "seed": int(params.get("seed", 42)),
        "unfreeze": params.get("unfreeze", "cls"),
        "optimizer": params.get("optimizer", "adamw"),
        "rollout_source": params.get("rollout_source", "baseline"),
        "policy_objective": params.get("policy_objective", "signed"),
    }
    for scene in FOUR_SCENES:
        scene_metrics = metrics.get(scene, {})
        for metric in METRICS:
            row[scene_metric_key(scene, metric)] = scene_metrics.get(metric)
    return row


def compute_pair_delta(real: dict[str, Any], control: dict[str, Any], metric: str = "ap75") -> dict[str, Any]:
    deltas: dict[str, float] = {}
    values: list[float] = []
    for scene in FOUR_SCENES:
        key = scene_metric_key(scene, metric)
        if real.get(key) is None or control.get(key) is None:
            continue
        value = float(real[key]) - float(control[key])
        deltas[SCENE_PREFIX[scene]] = value
        values.append(value)
    mean_delta = sum(values) / max(1, len(values))
    return {
        "metric": metric, "mean_delta": mean_delta,
        "positive_scene_count": sum(1 for v in values if v > 0.0),
        "scene_deltas": deltas,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
