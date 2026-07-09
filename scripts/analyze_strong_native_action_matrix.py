"""Compare strong-baseline controls and native action heads on one metric schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = (
    "ap50",
    "ap75",
    "precision",
    "recall",
    "false_positive_rate",
    "ece",
    "num_predictions",
)


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    result = {"epoch": int(row.get("epoch", 0))}
    for key in METRICS:
        value = row.get(f"val_{key}", row.get(key))
        result[key] = float(value) if isinstance(value, (int, float)) else None
    return result


def _round(value: float) -> float:
    return round(float(value), 12)


def selected_strong_baseline(strong: dict[str, Any]) -> dict[str, Any]:
    history = list(strong.get("history") or [])
    if not history:
        raise ValueError("Strong baseline has no history")
    best_epoch = strong.get("best_epoch")
    if best_epoch is not None:
        matches = [row for row in history if int(row.get("epoch", -1)) == int(best_epoch)]
        if matches:
            return _normalize_row(matches[0])
    return _normalize_row(max(history, key=lambda row: float(row.get("val_ap75", -1.0))))


def summarize_run(run: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    history = list(run.get("history") or [])
    if not history:
        raise ValueError("Matrix run has no history")
    normalized = [_normalize_row(row) for row in history]
    best = max(normalized, key=lambda row: row["ap75"] if row["ap75"] is not None else -1.0)
    final = normalized[-1]
    return {
        "completed": bool(run.get("completed", False)),
        "final": final,
        "best_ap75": best,
        "final_ap50_delta": _round(final["ap50"] - baseline["ap50"]),
        "final_ap75_delta": _round(final["ap75"] - baseline["ap75"]),
        "best_ap75_delta": _round(best["ap75"] - baseline["ap75"]),
        "final_ece_delta": _round(final["ece"] - baseline["ece"]),
        "final_recall_delta": _round(final["recall"] - baseline["recall"]),
        "final_prediction_delta": int(final["num_predictions"] - baseline["num_predictions"]),
    }


def analyze_matrix(
    strong: dict[str, Any],
    full: dict[str, Any],
    boxonly: dict[str, Any],
    preserve: dict[str, Any],
    parity: dict[str, Any],
) -> dict[str, Any]:
    baseline = selected_strong_baseline(strong)
    runs = {
        "full_control": summarize_run(full, baseline),
        "boxonly": summarize_run(boxonly, baseline),
        "preserve2": summarize_run(preserve, baseline),
    }
    strict = parity.get("strict_zero_action_parity") or {}
    aggregate = parity.get("aggregate_zero_action_parity") or {}
    parity_passed = bool(
        parity.get("completed") is True
        and strict.get("passed") is True
        and strict.get("mismatched_images") == 0
        and aggregate.get("passed") is True
    )
    preserve_best = runs["preserve2"]["best_ap75"]["ap75"]
    full_best = runs["full_control"]["best_ap75"]["ap75"]
    preserve_margin = _round(preserve_best - full_best)
    return {
        "baseline": baseline,
        "runs": runs,
        "decision": {
            "parity_passed": parity_passed,
            "preserve_beats_full_by_ap75": preserve_margin,
            "information_gain_candidate": bool(parity_passed and preserve_margin >= 0.005),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strong", required=True)
    parser.add_argument("--full-control", required=True)
    parser.add_argument("--boxonly", required=True)
    parser.add_argument("--preserve", required=True)
    parser.add_argument("--parity", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = analyze_matrix(
        _load(args.strong),
        _load(args.full_control),
        _load(args.boxonly),
        _load(args.preserve),
        _load(args.parity),
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
