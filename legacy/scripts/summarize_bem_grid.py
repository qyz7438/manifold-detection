"""Aggregate Penn-Fudan BEM grid results into a markdown/CSV table."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def parse_run_name(name: str) -> dict[str, str]:
    """grid_<config>_s<seed>_e<epochs>"""
    parts = name.split("_")
    # config is everything between 'grid' and 's<seed>'
    seed_idx = next((i for i, p in enumerate(parts) if p.startswith("s") and p[1:].isdigit()), None)
    epoch_idx = next((i for i, p in enumerate(parts) if p.startswith("e") and p[1:].isdigit()), None)
    if seed_idx is None or epoch_idx is None:
        return {}
    config = "_".join(parts[1:seed_idx])
    return {
        "config": config,
        "seed": parts[seed_idx][1:],
        "epochs": parts[epoch_idx][1:],
    }


def main() -> None:
    root = Path("runs")
    rows = []
    for metrics_path in sorted(root.glob("grid_*/eval_metrics.json")):
        run_name = metrics_path.parent.name
        info = parse_run_name(run_name)
        if not info:
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({
            "config": info["config"],
            "seed": int(info["seed"]),
            "epochs": int(info["epochs"]),
            "ap50": data.get("ap50"),
            "ap75": data.get("ap75"),
            "ece": data.get("ece"),
            "precision_at_recall_0_85": data.get("precision_at_recall_0_85"),
            "run": run_name,
        })

    if not rows:
        print("No grid results found in runs/grid_*/eval_metrics.json", file=sys.stderr)
        return

    # Group by config and compute mean/std
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["config"], r["epochs"]), []).append(r)

    print("| config | epochs | n | AP50 mean±std | AP75 mean±std | ECE mean±std | P@R=0.85 mean±std |")
    print("|---|---|---|---|---|---|---|")
    summary_rows = []
    for (config, epochs) in sorted(groups):
        g = groups[(config, epochs)]

        def mean_std(key: str):
            vals = [x[key] for x in g if x[key] is not None]
            if not vals:
                return "-"
            m = sum(vals) / len(vals)
            s = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
            return f"{m:.4f}±{s:.4f}"

        row = {
            "config": config,
            "epochs": epochs,
            "n": len(g),
            "ap50": mean_std("ap50"),
            "ap75": mean_std("ap75"),
            "ece": mean_std("ece"),
            "p@r85": mean_std("precision_at_recall_0_85"),
        }
        summary_rows.append(row)
        print(f"| {row['config']} | {row['epochs']} | {row['n']} | {row['ap50']} | {row['ap75']} | {row['ece']} | {row['p@r85']} |")

    # Per-seed detail CSV
    import csv
    csv_path = root / "bem_grid_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "seed", "epochs", "ap50", "ap75", "ece", "precision_at_recall_0_85", "run"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nPer-run CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
