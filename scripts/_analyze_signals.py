import json, numpy as np
from pathlib import Path
from collections import defaultdict

runs_dir = Path("runs")

# Collect per-epoch diagnostics for each config
diag = defaultdict(lambda: defaultdict(list))

for f in sorted(runs_dir.glob("round280_*/*eval_metrics.json")):
    name = f.parent.name
    if not any(x in name for x in ["_s42", "_s123", "_s456"]):
        continue
    d = json.loads(f.read_text())
    cfg = d.get("config", name.rsplit("_s", 1)[0])
    for row in d["history"]:
        for k in ["reward_std", "rl_loss", "kl_loss", "det_loss", "q_iou_corr",
                   "energy_reward", "pos_count", "total_grad_norm"]:
            v = row.get(k, None)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                diag[cfg][k].append(v)

# Print per-config diagnostic summary
print("=== Reward Signal Diagnostics (averaged over all epochs & seeds) ===")
print()
print(f"{'Config':<22s} {'reward_std':>10s} {'rl_loss':>10s} {'kl_loss':>10s} {'det_loss':>10s} {'rl/det':>8s} {'q_corr':>8s} {'pos':>6s} {'energy':>10s} {'grad_norm':>10s}")
print("-" * 125)

for cfg in sorted(diag.keys()):
    d = diag[cfg]
    def m(k):
        vals = d.get(k, [])
        return np.mean(vals) if vals else float("nan")
    rs = m("reward_std")
    rl = m("rl_loss")
    kl = m("kl_loss")
    dl = m("det_loss")
    qc = m("q_iou_corr")
    en = m("energy_reward")
    pc = m("pos_count")
    gn = m("total_grad_norm")
    ratio = f"{rl/dl*100:.2f}%" if dl and dl > 0 else "N/A"
    rs_s = f"{rs:.4f}" if not np.isnan(rs) else "N/A"
    rl_s = f"{rl:.4f}" if not np.isnan(rl) else "N/A"
    kl_s = f"{kl:.4f}" if not np.isnan(kl) else "N/A"
    dl_s = f"{dl:.2f}" if not np.isnan(dl) else "N/A"
    qc_s = f"{qc:.4f}" if not np.isnan(qc) else "N/A"
    en_s = f"{en:.6f}" if not np.isnan(en) else "N/A"
    pc_s = f"{pc:.0f}" if not np.isnan(pc) else "N/A"
    gn_s = f"{gn:.6f}" if not np.isnan(gn) else "N/A"
    print(f"{cfg:<22s} {rs_s:>10s} {rl_s:>10s} {kl_s:>10s} {dl_s:>10s} {ratio:>8s} {qc_s:>8s} {pc_s:>6s} {en_s:>10s} {gn_s:>10s}")

# Epoch-by-epoch for key RLVR groups
print()
print("=== Epoch-by-epoch signal trace (seed=42, key groups) ===")
print()
for cfg in ["ap75_event", "grpo_adv_g4", "grpo_adv_g8", "fft_loc_only",
            "grpo_fft_g4", "grpo_fft_g8", "per_chan_fft",
            "random_qnorm", "frozen_random", "aligned_verifier"]:
    # Find seed=42 run
    matches = list(runs_dir.glob(f"round280_{cfg}_s42/eval_metrics.json"))
    if not matches:
        # Try alternative naming
        for d in runs_dir.iterdir():
            if d.is_dir() and d.name.startswith(f"round280_{cfg}"):
                ef = d / "eval_metrics.json"
                if ef.exists():
                    dd = json.loads(ef.read_text())
                    if dd.get("seed") == 42:
                        matches = [ef]
                        break
    if not matches:
        continue
    h = json.loads(matches[0].read_text())["history"]
    print(f"--- {cfg} (seed=42) ---")
    header = f"{'Ep':>3s} {'AP75':>8s} {'reward_std':>10s} {'rl_loss':>10s} {'kl_loss':>10s} {'det_loss':>10s} {'rl/det%':>8s} {'q_corr':>8s}"
    if any("energy_reward" in row for row in h):
        header += f" {'energy':>10s}"
    if any("pos_count" in row for row in h):
        header += f" {'pos':>6s}"
    print(header)
    print("-" * len(header))
    for row in h:
        ep = row["epoch"]
        ap75 = row.get("val_ap75", 0)
        rs = row.get("reward_std", float("nan"))
        rl = row.get("rl_loss", float("nan"))
        kl = row.get("kl_loss", float("nan"))
        dl = row.get("det_loss", float("nan"))
        qc = row.get("q_iou_corr", float("nan"))
        en = row.get("energy_reward", float("nan"))
        pc = row.get("pos_count", float("nan"))
        ratio = f"{rl/dl*100:.2f}" if dl and dl > 0 and not np.isnan(rl) else "N/A"
        line = f"{ep:3d} {ap75:8.4f} {rs:10.4f} {rl:10.4f} {kl:10.4f} {dl:10.2f} {ratio:>8s} {qc:8.4f}"
        if not np.isnan(en):
            line += f" {en:10.6f}"
        if not np.isnan(pc):
            line += f" {pc:6.0f}"
        print(line)
    print()

# Reward-to-noise ratio analysis
print("=== Reward Signal-to-Noise Analysis ===")
print()
print(f"{'Config':<22s} {'reward_std':>10s} {'r_std/0.1':>10s} {'rl/det%':>8s} {'interpretation'}")
print("-" * 90)
for cfg in sorted(diag.keys()):
    d = diag[cfg]
    rs = np.mean(d.get("reward_std", [])) if d.get("reward_std") else float("nan")
    rl = np.mean(d.get("rl_loss", [])) if d.get("rl_loss") else float("nan")
    dl = np.mean(d.get("det_loss", [])) if d.get("det_loss") else 1.0
    if np.isnan(rs):
        continue
    # GRPO advantage: adv = (r - mean)/std, so std(adv) ≈ 1.0 is baseline
    # reward_std is std of the advantage within each group
    # If reward_std < 0.3, the reward differences within groups are tiny
    ratio = rs / 0.1  # relative to noise scale s=0.1
    rl_ratio = f"{rl/dl*100:.2f}%" if dl > 0 else "N/A"
    if rs < 0.1:
        interp = "REWARD COLLAPSED (no within-group variation)"
    elif rs < 0.3:
        interp = "Weak signal (advantage barely separates proposals)"
    elif rs < 0.7:
        interp = "Moderate signal"
    else:
        interp = "Strong signal"
    print(f"{cfg:<22s} {rs:10.4f} {ratio:10.2f} {rl_ratio:>8s}  {interp}")
