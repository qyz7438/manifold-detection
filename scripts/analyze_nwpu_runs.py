#!/usr/bin/env python
"""Analyze NWPU VHR-10 experiment results from runs/ directories."""
import os, json, sys, glob
from collections import defaultdict

RUNS_DIR = "E:/CLIproject/RLimage/runs"
BASELINE_AP75 = 0.29

def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return None

def scan_dirs():
    """Group runs by experiment type and extract metrics."""
    all_groups = defaultdict(list)

    for d in sorted(os.listdir(RUNS_DIR)):
        full = os.path.join(RUNS_DIR, d)
        if not os.path.isdir(full) or not d.startswith("round2"):
            continue
        if not d[5:9].isdigit():
            continue

        eval_file = os.path.join(full, "eval_metrics.json")
        config_file = os.path.join(full, "round_config.json")

        eval_data = load_json(eval_file)
        config_data = load_json(config_file)

        # Determine experiment type from directory name
        exp_type = classify_exp(d)

        entry = {
            "dir": d,
            "config": config_data or {},
            "eval": eval_data or {},
            "exp_type": exp_type,
            "has_eval": eval_data is not None,
            "has_config": config_data is not None,
        }
        all_groups[exp_type].append(entry)

    return all_groups

def classify_exp(name):
    """Classify experiment by round number and name pattern."""
    # Extract round number
    try:
        num = int(name[5:9])
    except:
        return "other"

    if 2100 <= num <= 2109:
        return "2100-2109_initial_sweeps"
    elif 2110 <= num <= 2119:
        return "2110-2119_manifold_dpo_edge"
    elif 2120 <= num <= 2129:
        return "2120-2129_nwpu_systematic"
    elif 2130 <= num <= 2140:
        return "2130-2140_rescue_density_gate"
    elif 2141 <= num <= 2143:
        return "2141-2143_adapter_bbox"
    elif 2144 <= num <= 2159:
        return "2144-2159_raw_ifft_fft_dim"
    elif 2160 <= num <= 2170:
        return "2160-2170_rpn_chain"
    elif 2171 <= num <= 2188:
        return "2171-2188_nms_duplicate"
    elif 2189 <= num <= 2195:
        return "2189-2195_pre_nms"
    elif 2196 <= num <= 2199:
        return "2196-2199_box_feature_manifold"
    elif 2200 <= num <= 2207:
        return "2200-2207_fusion_scene_fft"
    else:
        return "other"

def extract_metrics(entry):
    """Extract key metrics from eval data."""
    e = entry.get("eval", {})
    if not e:
        return {}

    # Try common metric locations
    metrics = {}

    # Direct keys
    for key in ["mAP", "AP50", "AP75", "map", "ap50", "ap75",
                 "AP", "mAP50", "mAP75"]:
        if key in e:
            metrics[key.lower()] = e[key]

    # Check nested structures
    for sub in ["results", "metrics", "eval", "detection"]:
        if sub in e and isinstance(e[sub], dict):
            for key in ["mAP", "AP50", "AP75", "map", "ap50", "ap75"]:
                if key in e[sub]:
                    metrics[key.lower()] = e[sub][key]

    # Check per-class results
    if "per_class" in e and isinstance(e["per_class"], dict):
        metrics["per_class"] = e["per_class"]

    return metrics

def extract_params(entry):
    """Extract key parameters from config."""
    c = entry.get("config", {})
    params = {}
    if not c:
        return params

    # Training params
    for key in ["learning_rate", "lr", "n_epochs", "epochs", "batch_size",
                 "temperature", "policy_weight", "det_weight", "kl_weight",
                 "gate_strength", "margin", "density_threshold",
                 "score_budget", "theta", "residual_scale",
                 "feature_dim", "dim_reduction", "loss_type"]:
        if key in c:
            params[key] = c[key]

    # From nested structures
    for sub in ["train", "training", "rlvr", "dpo"]:
        if sub in c and isinstance(c[sub], dict):
            s = c[sub]
            for key in ["lr", "learning_rate", "n_epochs", "epochs",
                         "batch_size", "temperature", "policy_weight",
                         "det_weight", "kl_weight", "beta"]:
                if key in s:
                    params[f"{sub}_{key}"] = s[key]

    return params

def print_group_report(group_name, entries, top_n=5):
    """Print analysis for a group of experiments."""
    valid = [e for e in entries if e["has_eval"]]
    if not valid:
        print(f"\n{'='*70}")
        print(f"{group_name} ({len(entries)} total, 0 with eval data)")
        print(f"{'='*70}")
        print("  No eval data available.")
        return

    # Sort by AP75 descending
    entries_with_ap75 = []
    for e in valid:
        m = extract_metrics(e)
        ap75 = m.get("ap75", m.get("mAP75", None))
        ap50 = m.get("ap50", m.get("mAP50", None))
        if ap75 is not None:
            entries_with_ap75.append((e, m, ap75, ap50))

    entries_with_ap75.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'='*70}")
    print(f"{group_name} ({len(valid)}/{len(entries)} with eval)")
    print(f"{'='*70}")

    if not entries_with_ap75:
        print("  No AP75 data found in eval files.")
        # Show what keys are available
        sample = valid[0]
        m = extract_metrics(sample)
        if m:
            print(f"  Available metrics in {sample['dir']}: {list(m.keys())}")
            e = sample["eval"]
            print(f"  Top-level eval keys: {list(e.keys())[:10]}")
        return

    best = entries_with_ap75[0]
    above_baseline = [x for x in entries_with_ap75 if x[2] > BASELINE_AP75]

    print(f"  Best AP75: {best[2]:.4f} (AP50={best[1]} in {best[0]['dir']})")
    print(f"  Total above baseline ({BASELINE_AP75}): {len(above_baseline)}/{len(entries_with_ap75)}")
    if above_baseline:
        baseline_best = above_baseline[0]
        print(f"  Best above baseline: {baseline_best[2]:.4f} in {baseline_best[0]['dir']}")

    # Print top entries
    print(f"\n  Top {min(top_n, len(entries_with_ap75))}:")
    for i, (e, m, ap75, ap50) in enumerate(entries_with_ap75[:top_n]):
        params = extract_params(e)
        param_str = "; ".join(f"{k}={v}" for k, v in sorted(params.items()))
        ap50_str = f"{ap50:.4f}" if ap50 is not None else "N/A"
        marker = " *** ABOVE BASELINE ***" if ap75 > BASELINE_AP75 else ""
        print(f"    {i+1}. {e['dir']}: AP75={ap75:.4f} AP50={ap50_str}{marker}")
        if param_str:
            print(f"       params: {param_str}")

    # Summary stats
    ap75_vals = [x[2] for x in entries_with_ap75]
    print(f"\n  Stats: mean={sum(ap75_vals)/len(ap75_vals):.4f} "
          f"min={min(ap75_vals):.4f} max={max(ap75_vals):.4f} "
          f"median={sorted(ap75_vals)[len(ap75_vals)//2]:.4f}")

    # Compare against det_only baselines within group
    det_only = [x for x in entries_with_ap75 if "det_only" in x[0]["dir"] or "detonly" in x[0]["dir"]]
    if det_only:
        det_ap75 = [x[2] for x in det_only]
        print(f"  det_only baselines: {len(det_ap75)} runs, "
              f"mean AP75={sum(det_ap75)/len(det_ap75):.4f}")

def main():
    print("=" * 70)
    print("NWPU VHR-10 Experiment Analysis (rounds 2100-2207)")
    print(f"Baseline AP75: {BASELINE_AP75}")
    print("=" * 70)

    groups = scan_dirs()

    # Print groups overview
    print(f"\nTotal groups: {len(groups)}")
    for g, entries in sorted(groups.items()):
        total = len(entries)
        with_eval = sum(1 for e in entries if e["has_eval"])
        with_config = sum(1 for e in entries if e["has_config"])
        print(f"  {g}: {total} dirs ({with_eval} eval, {with_config} config)")

    # Detailed reports per group (focus on 2130+)
    target_groups = [
        "2130-2140_rescue_density_gate",
        "2141-2143_adapter_bbox",
        "2160-2170_rpn_chain",
        "2171-2188_nms_duplicate",
        "2189-2195_pre_nms",
        "2200-2207_fusion_scene_fft",
        "2120-2129_nwpu_systematic",
    ]

    for g in target_groups:
        if g in groups:
            print_group_report(g, groups[g])
        else:
            print(f"\n{'='*70}")
            print(f"{g} — NOT FOUND")
            print(f"{'='*70}")

    # Also show some earlier groups for context
    early_groups = ["2100-2109_initial_sweeps", "2110-2119_manifold_dpo_edge",
                     "2144-2159_raw_ifft_fft_dim"]
    for g in early_groups:
        if g in groups:
            print_group_report(g, groups[g], top_n=3)

    # === CROSS-GROUP COMPARISON ===
    print(f"\n\n{'='*70}")
    print("CROSS-GROUP BEST AP75 COMPARISON")
    print("=" * 70)

    all_best = []
    for g, entries in sorted(groups.items()):
        valid = [e for e in entries if e["has_eval"]]
        for e in valid:
            m = extract_metrics(e)
            ap75 = m.get("ap75", m.get("mAP75", None))
            if ap75 is not None:
                all_best.append((ap75, e["dir"], g))

    all_best.sort(key=lambda x: x[0], reverse=True)

    print(f"\nTop 20 experiments across ALL groups:")
    for i, (ap75, d, g) in enumerate(all_best[:20]):
        marker = " ***" if ap75 > BASELINE_AP75 else ""
        print(f"  {i+1}. {d}: AP75={ap75:.4f} [{g}]{marker}")

    print(f"\nAll experiments above baseline ({BASELINE_AP75}):")
    above = [x for x in all_best if x[0] > BASELINE_AP75]
    for ap75, d, g in above:
        print(f"  {d}: AP75={ap75:.4f} [{g}]")

    print(f"\nTotal: {len(all_best)} experiments with AP75 data, "
          f"{len(above)} above baseline")

    # === BASELINE CHECK ===
    print(f"\n\n{'='*70}")
    print("BASELINE RUNS")
    print("=" * 70)
    baseline_dirs = [d for d in os.listdir(RUNS_DIR) if "baseline" in d.lower()
                     and os.path.isdir(os.path.join(RUNS_DIR, d))]
    for bd in sorted(baseline_dirs):
        be = load_json(os.path.join(RUNS_DIR, bd, "eval_metrics.json")) or {}
        bc = load_json(os.path.join(RUNS_DIR, bd, "round_config.json")) or {}
        m = {}
        for k in ["mAP", "AP50", "AP75", "map", "ap50", "ap75"]:
            if k in be:
                m[k] = be[k]
        print(f"  {bd}: {m} | config keys: {list(bc.keys())}")

if __name__ == "__main__":
    main()
