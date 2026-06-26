"""Aggregate and compare Plan C action-local segmentation results.

Usage:
    python scripts/analyze_planC_results.py --glob "runs/planC_*"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="runs/planC_*", help="Glob pattern for run directories")
    p.add_argument("--metric", default="mIoU", help="Primary metric to compare")
    p.add_argument("--seeds", default="42,123,456", help="Comma-separated seeds to group by")
    return p.parse_args()


def find_runs(pattern: str):
    root = Path(".")
    return sorted(root.glob(pattern))


def load_summary(run_dir: Path):
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def extract_seed(run_name: str) -> int | None:
    import re
    m = re.search(r"_s(\d+)(_|$)", run_name)
    return int(m.group(1)) if m else None


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    runs = find_runs(args.glob)

    rows = []
    for run_dir in runs:
        summary = load_summary(run_dir)
        if summary is None:
            continue
        run_name = summary.get("run_name", run_dir.name)
        baseline = summary.get("baseline_metrics", {})
        final = summary.get("final_metrics", {})
        best = summary.get("best_miou", final.get(args.metric, 0.0))
        rows.append({
            "run": run_name,
            "seed": extract_seed(run_name),
            "baseline": baseline.get(args.metric, 0.0),
            "final": final.get(args.metric, 0.0),
            "best": best,
            "delta_final": final.get(args.metric, 0.0) - baseline.get(args.metric, 0.0),
            "delta_best": best - baseline.get(args.metric, 0.0),
        })

    if not rows:
        print(f"No summaries found matching {args.glob}")
        return

    # Group by run prefix (everything before _s<seed>)
    import re
    groups: dict[str, list[dict]] = {}
    for r in rows:
        prefix = re.sub(r"_s\d+(_|$)", r"\1", r["run"]).rstrip("_")
        groups.setdefault(prefix, []).append(r)

    print(f"\nPlan C results grouped by run prefix (metric={args.metric}):")
    print("-" * 90)
    print(f"{'run':<50} {'baseline':>10} {'final':>10} {'best':>10} {'delta_best':>10}")
    print("-" * 90)

    for prefix in sorted(groups.keys()):
        group_rows = groups[prefix]
        baselines = [r["baseline"] for r in group_rows if r["seed"] in seeds]
        finals = [r["final"] for r in group_rows if r["seed"] in seeds]
        bests = [r["best"] for r in group_rows if r["seed"] in seeds]
        if not baselines:
            continue
        print(f"{prefix:<50} {np.mean(baselines):>10.4f} {np.mean(finals):>10.4f} {np.mean(bests):>10.4f} {np.mean(bests)-np.mean(baselines):>10.4f}")
        for r in group_rows:
            print(f"  {r['run']:<48} {r['baseline']:>10.4f} {r['final']:>10.4f} {r['best']:>10.4f} {r['delta_best']:>10.4f}")


if __name__ == "__main__":
    main()
