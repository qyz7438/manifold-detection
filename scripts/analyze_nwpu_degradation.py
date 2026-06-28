"""Analyze parameter dynamics where NWPU performance degraded."""
import json, glob, os, statistics
from collections import defaultdict


def load_run(pattern):
    runs = sorted(glob.glob(pattern))
    out = []
    for r in runs:
        p = os.path.join(r, "eval_metrics.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            d = json.load(f)
        out.append((os.path.basename(r), d))
    return out


def extract_seed(name):
    import re
    m = re.search(r"_s(\d+)_", name)
    return int(m.group(1)) if m else -1


def print_per_epoch_table(name, d):
    print(f"\n=== {name} per-epoch metrics ===")
    print("ep  loss    AP50    AP75    ECE     alpha0  eff_rd0")
    for ep in d.get("history", []):
        a0 = ep.get("fpn_sm/0_alpha", float("nan"))
        e0 = ep.get("fpn_sm/0_eff_rel_delta", float("nan"))
        print(f"{ep['epoch']:>2}  {ep['train_loss']:.4f}  {ep['val_ap50']:.4f}  {ep['val_ap75']:.4f}  {ep.get('ece', ep.get('val_ece', float('nan'))):.4f}  {a0:.4f}  {e0:.4f}")


def main():
    groups = {
        "nwpu_mob_baseline": "runs/nwpu_mob_baseline_s*_12ep",
        "nwpu_mob_fpn_sm": "runs/nwpu_mob_fpn_sm_s*_12ep",
        "nwpu_resnet_baseline": "runs/nwpu_resnet_baseline_s*_12ep",
        "nwpu_resnet_fpn_sm": "runs/nwpu_resnet_fpn_sm_s*_12ep",
    }

    data = {}
    for g, pat in groups.items():
        data[g] = load_run(pat)

    # Per-epoch tables for fpn_sm runs
    for g in ["nwpu_mob_fpn_sm", "nwpu_resnet_fpn_sm"]:
        for name, d in data[g]:
            print_per_epoch_table(name, d)

    # Degradation analysis: compare best/worst seeds within resnet fpn_sm
    print("\n=== ResNet50 fpn_sm seed comparison (last epoch) ===")
    rows = []
    for name, d in data["nwpu_resnet_fpn_sm"]:
        last = d["history"][-1]
        rows.append({
            "seed": extract_seed(name),
            "ap50": last["val_ap50"], "ap75": last["val_ap75"], "ece": d["ece"],
            "a0": last["fpn_sm/0_alpha"], "a1": last["fpn_sm/1_alpha"],
            "a2": last["fpn_sm/2_alpha"], "a3": last["fpn_sm/3_alpha"],
            "eff0": last["fpn_sm/0_eff_rel_delta"],
            "cos0": last["fpn_sm/0_cos_delta_x"],
        })
    print("seed  AP50   AP75   ECE    a0      a1      a2      a3      eff0    cos0")
    for r in sorted(rows, key=lambda x: x["seed"]):
        print(f"{r['seed']:>4}  {r['ap50']:.4f} {r['ap75']:.4f} {r['ece']:.4f} "
              f"{r['a0']:.4f} {r['a1']:.4f} {r['a2']:.4f} {r['a3']:.4f} "
              f"{r['eff0']:.4f} {r['cos0']:.4f}")

    # MobileNet ECE vs alpha trend
    print("\n=== MobileNetV3 fpn_sm: ECE vs alpha/eff_rel_delta per epoch (mean over seeds) ===")
    hists = [d["history"] for _, d in data["nwpu_mob_fpn_sm"]]
    epochs = len(hists[0])
    print("ep  AP50   AP75   ECE    a0_mean  eff0_mean")
    for e in range(epochs):
        ep = e + 1
        vals = defaultdict(list)
        for h in hists:
            for item in h:
                if item["epoch"] == ep:
                    vals["ap50"].append(item["val_ap50"])
                    vals["ap75"].append(item["val_ap75"])
                    vals["ece"].append(item.get("ece", float("nan")))
                    vals["a0"].append(item["fpn_sm/0_alpha"])
                    vals["eff0"].append(item["fpn_sm/0_eff_rel_delta"])
                    break
        print(f"{ep:>2}  {statistics.mean(vals['ap50']):.4f} {statistics.mean(vals['ap75']):.4f} "
              f"{statistics.mean(vals['ece']):.4f} {statistics.mean(vals['a0']):.4f} {statistics.mean(vals['eff0']):.4f}")

    # ResNet fpn_sm per-epoch mean
    print("\n=== ResNet50 fpn_sm: per-epoch mean over seeds ===")
    hists = [d["history"] for _, d in data["nwpu_resnet_fpn_sm"]]
    epochs = len(hists[0])
    print("ep  AP50   AP75   ECE    a0_mean  eff0_mean")
    for e in range(epochs):
        ep = e + 1
        vals = defaultdict(list)
        for h in hists:
            for item in h:
                if item["epoch"] == ep:
                    vals["ap50"].append(item["val_ap50"])
                    vals["ap75"].append(item["val_ap75"])
                    vals["ece"].append(item.get("ece", float("nan")))
                    vals["a0"].append(item["fpn_sm/0_alpha"])
                    vals["eff0"].append(item["fpn_sm/0_eff_rel_delta"])
                    break
        print(f"{ep:>2}  {statistics.mean(vals['ap50']):.4f} {statistics.mean(vals['ap75']):.4f} "
              f"{statistics.mean(vals['ece']):.4f} {statistics.mean(vals['a0']):.4f} {statistics.mean(vals['eff0']):.4f}")


if __name__ == "__main__":
    main()
