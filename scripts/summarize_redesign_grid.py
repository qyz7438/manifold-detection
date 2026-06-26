"""Aggregate Penn-Fudan redesign grid results into a markdown/CSV table."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def parse_run_name(name: str) -> dict[str, str]:
    """redesign_<config>_s<seed>"""
    parts = name.split("_")
    seed_idx = next((i for i, p in enumerate(parts) if p.startswith("s") and p[1:].isdigit()), None)
    if seed_idx is None:
        return {}
    config = "_".join(parts[1:seed_idx])
    return {"config": config, "seed": parts[seed_idx][1:]}


def main() -> None:
    root = Path("runs")
    rows = []
    for metrics_path in sorted(root.glob("redesign_*/eval_metrics.json")):
        run_name = metrics_path.parent.name
        info = parse_run_name(run_name)
        if not info:
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({
            "config": info["config"],
            "seed": int(info["seed"]),
            "ap50": data.get("ap50"),
            "ap75": data.get("ap75"),
            "ece": data.get("ece"),
            "precision_at_recall_0_85": data.get("precision_at_recall_0_85"),
            "run": run_name,
        })

    if not rows:
        print("No grid results found in runs/redesign_*/eval_metrics.json", file=sys.stderr)
        return

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["config"], []).append(r)

    print("| config | n | AP50 mean±std | AP75 mean±std | ECE mean±std | P@R=0.85 mean±std |")
    print("|---|---|---|---|---|---|")
    summary_rows = []
    for config in sorted(groups):
        g = groups[config]

        def mean_std(key: str):
            vals = [x[key] for x in g if x[key] is not None]
            if not vals:
                return "-"
            m = sum(vals) / len(vals)
            s = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
            return f"{m:.4f}±{s:.4f}"

        row = {
            "config": config,
            "n": len(g),
            "ap50": mean_std("ap50"),
            "ap75": mean_std("ap75"),
            "ece": mean_std("ece"),
            "p@r85": mean_std("precision_at_recall_0_85"),
        }
        summary_rows.append(row)
        print(f"| {row['config']} | {row['n']} | {row['ap50']} | {row['ap75']} | {row['ece']} | {row['p@r85']} |")

    csv_path = root / "redesign_grid_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "seed", "ap50", "ap75", "ece", "precision_at_recall_0_85", "run"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nPer-run CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
