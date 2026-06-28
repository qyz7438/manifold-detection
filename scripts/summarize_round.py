#!/usr/bin/env python3
import argparse, json, glob, os, statistics
from collections import defaultdict


def extract_group(run_name):
    base = run_name[:-4] if run_name.endswith("_7ep") else run_name
    if "_s" in base:
        group, seed = base.rsplit("_s", 1)
        return group, int(seed) if seed.isdigit() else -1
    return base, -1


def safe_stdev(vals):
    import math
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="runs/*_7ep")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = []
    group_hists = defaultdict(list)
    for r in sorted(glob.glob(args.glob)):
        p = os.path.join(r, "eval_metrics.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            d = json.load(f)
        name = d.get("run_name", os.path.basename(r))
        group, seed = extract_group(name)
        rows.append({
            "group": group, "seed": seed,
            "ap50": d.get("ap50"), "ap75": d.get("ap75"),
            "ece": d.get("ece"), "best_ap50": d.get("best_ap50"),
            "run": name,
        })
        hist = d.get("history", [])
        if hist:
            group_hists[group].append(hist)

    lines = []
    lines.append("group                       n  AP50_mean   std     AP75_mean   std     ECE_mean  bestAP50")
    agg = defaultdict(list)
    for r in rows:
        agg[r["group"]].append(r)
    for g in sorted(agg):
        vals = agg[g]
        a50 = [v["ap50"] for v in vals]
        a75 = [v["ap75"] for v in vals]
        ece = [v["ece"] for v in vals if v["ece"] is not None]
        best = [v["best_ap50"] for v in vals]
        lines.append(
            "{:<25} {:>3} {:>10.4f} {:>7.4f} {:>10.4f} {:>7.4f} {:>8.4f} {:>9.4f}".format(
                g, len(vals), statistics.mean(a50), safe_stdev(a50),
                statistics.mean(a75), safe_stdev(a75),
                statistics.mean(ece), statistics.mean(best))
        )

    lines.append("")
    lines.append("group                     seed  AP50    AP75    ECE     bestAP50")
    for r in sorted(rows, key=lambda x: (x["group"], x["seed"])):
        lines.append(
            "{:<25} {:>4} {:.4f} {:.4f} {} {:.4f}".format(
                r["group"], r["seed"], r["ap50"], r["ap75"], ("%.4f"%r["ece"] if r["ece"] is not None else "None"), r["best_ap50"])
        )

    lines.append("")
    for g, hists in sorted(group_hists.items()):
        if not hists or not hists[0]:
            continue
        last = max(ep["epoch"] for hist in hists for ep in hist)
        metric_vals = defaultdict(list)
        for hist in hists:
            for ep in hist:
                if ep["epoch"] != last:
                    continue
                for k, v in ep.items():
                    if k in ("epoch", "train_loss", "val_ap50", "val_ap75"):
                        continue
                    if isinstance(v, (int, float)):
                        metric_vals[k].append(v)
        lines.append("=== {} last-epoch diagnostics (n={}) ===".format(g, len(hists)))
        for k in sorted(metric_vals):
            vals = [v for v in metric_vals[k] if v is not None and not __import__("math").isnan(v)]
            if not vals:
                continue
            lines.append(
                "  {:<45} mean={: .6f} std={: .6f}".format(
                    k, statistics.mean(vals), safe_stdev(vals))
            )

    out = "\n".join(lines)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
    print(out)


if __name__ == "__main__":
    main()
