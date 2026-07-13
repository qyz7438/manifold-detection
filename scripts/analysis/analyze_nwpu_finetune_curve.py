"""Summarize NWPU fine-tune progress and diagnose validation saturation."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


DECISION_METRICS = (
    "ap50",
    "ap75",
    "precision",
    "recall",
    "false_positive_rate",
    "ece",
    "num_predictions",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_state(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    final_path = root / "eval_metrics.json"
    if final_path.exists():
        return _load_json(final_path)
    progress_path = root / "metrics_history.json"
    if progress_path.exists():
        return _load_json(progress_path)
    raise FileNotFoundError(f"No eval_metrics.json or metrics_history.json under {root}")


def _metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(f"val_{key}", row.get(key))
    return float(value) if isinstance(value, (int, float)) else None


def _material_gain_trace(
    history: list[dict[str, Any]],
    *,
    ap50_gain: float,
    ap75_gain: float,
) -> list[dict[str, Any]]:
    best50: float | None = None
    best75: float | None = None
    trace = []
    for row in history:
        ap50 = _metric(row, "ap50")
        ap75 = _metric(row, "ap75")
        gain50 = None if ap50 is None or best50 is None else ap50 - best50
        gain75 = None if ap75 is None or best75 is None else ap75 - best75
        material = bool(
            (gain50 is not None and gain50 >= ap50_gain)
            or (gain75 is not None and gain75 >= ap75_gain)
        )
        trace.append(
            {
                "epoch": int(row.get("epoch", len(trace) + 1)),
                "ap50_gain_over_prior_best": gain50,
                "ap75_gain_over_prior_best": gain75,
                "material_gain": material,
            }
        )
        if ap50 is not None:
            best50 = ap50 if best50 is None else max(best50, ap50)
        if ap75 is not None:
            best75 = ap75 if best75 is None else max(best75, ap75)
    return trace


def detect_saturation(
    history: list[dict[str, Any]],
    *,
    window: int = 3,
    ap50_gain: float = 0.003,
    ap75_gain: float = 0.005,
) -> dict[str, Any]:
    if window < 1:
        raise ValueError("window must be positive")
    trace = _material_gain_trace(history, ap50_gain=ap50_gain, ap75_gain=ap75_gain)
    recent = trace[-window:]
    latest_material_gain = any(item["material_gain"] for item in recent)
    min_epochs = 16
    block_size = 4
    checked_rows = history[-(block_size * 3):]
    base = {
        "window": window,
        "minimum_epochs": min_epochs,
        "ap50_gain_threshold": ap50_gain,
        "ap75_gain_threshold": ap75_gain,
        "epochs_checked": [int(row.get("epoch", 0)) for row in checked_rows],
        "latest_material_gain": latest_material_gain,
        "gain_trace": trace,
    }
    if len(history) < min_epochs:
        return {
            **base,
            "status": "INSUFFICIENT_HISTORY",
            "saturated": False,
            "safe_to_expand": False,
        }

    blocks = [checked_rows[index:index + block_size] for index in range(0, 12, block_size)]

    def block_median(rows: list[dict[str, Any]], key: str) -> float:
        values = [_metric(row, key) for row in rows]
        return float(statistics.median(value for value in values if value is not None))

    ap75_medians = [block_median(block, "ap75") for block in blocks]
    ap50_medians = [block_median(block, "ap50") for block in blocks]
    ap75_deltas = [ap75_medians[1] - ap75_medians[0], ap75_medians[2] - ap75_medians[1]]
    ap50_deltas = [ap50_medians[1] - ap50_medians[0], ap50_medians[2] - ap50_medians[1]]

    last_eight = history[-8:]
    slopes = []
    for left_index, left in enumerate(last_eight):
        left_value = _metric(left, "ap75")
        left_epoch = int(left.get("epoch", left_index + 1))
        if left_value is None:
            continue
        for right in last_eight[left_index + 1:]:
            right_value = _metric(right, "ap75")
            right_epoch = int(right.get("epoch", left_epoch + 1))
            if right_value is not None and right_epoch != left_epoch:
                slopes.append((right_value - left_value) / (right_epoch - left_epoch))
    ap75_slope = float(statistics.median(slopes)) if slopes else 0.0

    final_ap75 = sorted(_metric(row, "ap75") for row in blocks[-1])
    lower = statistics.median(final_ap75[:2])
    upper = statistics.median(final_ap75[2:])
    final_ap75_iqr = float(upper - lower)

    plateau = bool(
        all(delta <= ap75_gain for delta in ap75_deltas)
        and all(delta <= ap50_gain for delta in ap50_deltas)
        and abs(ap75_slope) <= 0.001
        and final_ap75_iqr <= 0.004
        and not latest_material_gain
    )

    first_block = blocks[0]
    final_block = blocks[-1]
    first_precision = block_median(first_block, "precision")
    final_precision = block_median(final_block, "precision")
    first_recall = block_median(first_block, "recall")
    final_recall = block_median(final_block, "recall")
    first_ece = block_median(first_block, "ece")
    final_ece = block_median(final_block, "ece")
    first_predictions = block_median(first_block, "num_predictions")
    final_predictions = block_median(final_block, "num_predictions")
    risks = {
        "precision_drop": first_precision - final_precision > 0.02,
        "recall_drop": first_recall - final_recall > 0.02,
        "ece_increase": final_ece - first_ece > 0.01,
        "prediction_increase": (
            final_predictions - first_predictions
        ) / max(1.0, first_predictions) > 0.05,
    }
    has_risk = any(risks.values())
    status = "CONTINUE"
    if plateau:
        status = "SATURATED_WITH_RISK" if has_risk else "SATURATED"
    return {
        **base,
        "status": status,
        "saturated": plateau,
        "safe_to_expand": bool(plateau and not has_risk),
        "ap75_block_medians": ap75_medians,
        "ap50_block_medians": ap50_medians,
        "ap75_block_deltas": ap75_deltas,
        "ap50_block_deltas": ap50_deltas,
        "ap75_theil_sen_slope_last8": ap75_slope,
        "final_ap75_iqr": final_ap75_iqr,
        "risks": risks,
    }


def _robust_best_ap75(history: list[dict[str, Any]], window: int = 3) -> dict[str, Any] | None:
    if len(history) < window:
        return None
    candidates = []
    offset = window // 2
    for start in range(0, len(history) - window + 1):
        rows = history[start:start + window]
        values = [_metric(row, "ap75") for row in rows]
        value = float(statistics.median(item for item in values if item is not None))
        candidates.append(
            {
                "epoch": int(rows[offset].get("epoch", start + offset + 1)),
                "value": value,
                "window": window,
            }
        )
    return max(candidates, key=lambda item: item["value"])


def _decision_row(row: dict[str, Any]) -> dict[str, Any]:
    result = {"epoch": int(row.get("epoch", 0))}
    for key in DECISION_METRICS:
        result[key] = _metric(row, key)
    return result


def _rounded_delta(left: float | None, right: Any) -> float | None:
    if left is None or not isinstance(right, (int, float)):
        return None
    return round(left - float(right), 12)


def summarize_curve(
    run: dict[str, Any],
    *,
    baseline: dict[str, Any] | None = None,
    window: int = 3,
    ap50_gain: float = 0.003,
    ap75_gain: float = 0.005,
) -> dict[str, Any]:
    history = list(run.get("history") or [])
    if not history:
        return {
            "completed": bool(run.get("completed", False)),
            "epochs_observed": 0,
            "saturation": detect_saturation([], window=window, ap50_gain=ap50_gain, ap75_gain=ap75_gain),
        }

    best75_row = max(history, key=lambda row: _metric(row, "ap75") or float("-inf"))
    best50_row = max(history, key=lambda row: _metric(row, "ap50") or float("-inf"))
    final = _decision_row(history[-1])
    best_ap75 = {"epoch": int(best75_row["epoch"]), "value": _metric(best75_row, "ap75")}
    best_ap50 = {"epoch": int(best50_row["epoch"]), "value": _metric(best50_row, "ap50")}
    robust_best_ap75 = _robust_best_ap75(history)
    baseline = baseline or {}
    return {
        "completed": bool(run.get("completed", False)),
        "failed_nan": bool(run.get("failed_nan", False)),
        "epochs_observed": len(history),
        "final": final,
        "best_ap75": best_ap75,
        "best_ap50": best_ap50,
        "robust_best_ap75": robust_best_ap75,
        "best_final_ap75_gap": _rounded_delta(best_ap75["value"], final["ap75"]),
        "delta_vs_source": {
            "final_ap50": _rounded_delta(final["ap50"], baseline.get("ap50")),
            "final_ap75": _rounded_delta(final["ap75"], baseline.get("ap75")),
            "best_ap50": _rounded_delta(best_ap50["value"], baseline.get("ap50")),
            "best_ap75": _rounded_delta(best_ap75["value"], baseline.get("ap75")),
        },
        "saturation": detect_saturation(
            history,
            window=window,
            ap50_gain=ap50_gain,
            ap75_gain=ap75_gain,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline-metrics", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--ap50-gain", type=float, default=0.003)
    parser.add_argument("--ap75-gain", type=float, default=0.005)
    args = parser.parse_args()

    run = load_run_state(args.run_dir)
    baseline = _load_json(Path(args.baseline_metrics)) if args.baseline_metrics else None
    summary = summarize_curve(
        run,
        baseline=baseline,
        window=args.window,
        ap50_gain=args.ap50_gain,
        ap75_gain=args.ap75_gain,
    )
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
