"""Aggregate eval_metrics.json from runs/ into a markdown table.

Usage:
    python scripts/aggregate_results.py --pattern 'voc_*' --output docs/reports/voc_matrix_interim_results.md
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


def _extract_group(run_name: str) -> dict:
    """Parse a run name like voc_mob_baseline_s42_12ep into group keys."""
    m = re.match(
        r"^(?P<dataset>[^_]+)_(?P<backbone>[^_]+)_(?P<method>[^_]+)_s(?P<seed>\d+)_(?P<epochs>\d+)ep",
        run_name,
    )
    if not m:
        return {}
    return m.groupdict()


def _load_metrics(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--pattern", default="*", help="Glob pattern for run names")
    parser.add_argument("--output", default=None)
    parser.add_argument("--metrics", default="ap50,ap75,ece", help="Comma-separated metric keys")
    args = parser.parse_args()

    metric_keys = [k.strip() for k in args.metrics.split(",")]
    runs_dir = Path(args.runs_dir)
    rows = []
    for run_dir in sorted(runs_dir.glob(args.pattern)):
        metrics = _load_metrics(run_dir / "eval_metrics.json")
        if metrics is None:
            continue
        group = _extract_group(run_dir.name)
        row = {
            "run": run_dir.name,
            **group,
            "epochs": metrics.get("epochs", 0),
        }
        for k in metric_keys:
            row[k] = metrics.get(k)
        row["best_ap50"] = metrics.get("best_ap50")
        rows.append(row)

    # Drop rows that failed or cannot be grouped.
    rows = [
        r for r in rows
        if r.get("ap50") and r.get("ap75")
        and r.get("dataset") is not None
    ]

    if not rows:
        print("No matching runs found.")
        return

    # Group by dataset/backbone/method/epochs for seed aggregation.
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r.get("dataset"), r.get("backbone"), r.get("method"), r.get("epochs"))
        groups.setdefault(key, []).append(r)

    lines = ["# Results aggregation", ""]
    lines.append("## Per-run results")
    lines.append("")
    header = ["run", "dataset", "backbone", "method", "seed", "epochs", "best_ap50", *metric_keys]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        vals = [str(r.get(h, "")) for h in header]
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("## Group statistics (over seeds)")
    lines.append("")
    group_header = ["dataset", "backbone", "method", "epochs", "n", "best_ap50_mean", "best_ap50_std"]
    for k in metric_keys:
        group_header.append(f"{k}_mean")
        group_header.append(f"{k}_std")
    lines.append("| " + " | ".join(group_header) + " |")
    lines.append("|" + "|".join(["---"] * len(group_header)) + "|")
    for key, group_rows in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        n = len(group_rows)
        def mean_std(key2):
            vals = [r[key2] for r in group_rows if isinstance(r[key2], (int, float))]
            if not vals:
                return "", ""
            return f"{statistics.mean(vals):.4f}", f"{statistics.stdev(vals) if len(vals) > 1 else 0.0:.4f}"
        cells = [str(x) for x in key] + [str(n), *mean_std("best_ap50")]
        for k in metric_keys:
            cells.extend(mean_std(k))
        lines.append("| " + " | ".join(cells) + " |")

    md = "\n".join(lines) + "\n"
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"Wrote {out_path}")
    print(md)


if __name__ == "__main__":
    main()
