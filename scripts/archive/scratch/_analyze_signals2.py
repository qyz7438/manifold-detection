import json, numpy as np
from pathlib import Path

runs_dir = Path("runs")

# Read round279 diagnostics (8 epoch unfrozen, has full signal trace)
print("=== Round 279: RLVR Signal Diagnostics (8 epoch, unfrozen) ===")
print()

configs = {
    "det_only_unf": "det_only_unf (baseline, no RL)",
    "ap75_event": "ap75_event (pure IoU reward)",
    "grpo_adv": "grpo_adv (GRPO + KL, G=4)",
    "grpo_fft": "grpo_fft (GRPO + KL + FFT loc-only, G=4)",
    "fft_energy": "fft_energy (GRPO + KL + energy reward, G=4)",
}

for cfg, desc in configs.items():
    matches = list(runs_dir.glob(f"round279_{cfg}_s42/eval_metrics.json"))
    if not matches:
        print(f"  {cfg}: NOT FOUND")
        continue
    d = json.loads(matches[0].read_text())
    h = d["history"]

    print(f"--- {desc} ---")
    keys = list(h[0].keys())
    # Print all keys to see what's available
    print(f"  Available fields: {keys}")

    # Extract signal fields
    has_diag = "rl_loss" in h[0]
    if not has_diag:
        print(f"  No diagnostics in this config")
        print()
        continue

    print(f"  {'Ep':>3s} {'AP75':>8s} {'reward_std':>10s} {'rl_loss':>10s} {'kl_loss':>10s} {'det_loss':>10s} {'rl/det%':>8s} {'q_corr':>8s} {'grad_norm':>10s} {'pos':>6s}")
    print(f"  {'-'*3} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*6}")

    for row in h:
        ep = row["epoch"]
        ap75 = row.get("val_ap75", 0)
        rs = row.get("reward_std", float("nan"))
        rl = row.get("rl_loss", float("nan"))
        kl = row.get("kl_loss", float("nan"))
        dl = row.get("det_loss", float("nan"))
        qc = row.get("q_iou_corr", float("nan"))
        gn = row.get("total_grad_norm", float("nan"))
        pc = row.get("pos_count", float("nan"))
        ratio = f"{rl/dl*100:.2f}%" if dl and dl > 0 and not np.isnan(rl) else "  N/A"

        def fmt(x, w=10):
            if np.isnan(x):
                return f"{'N/A':>{w}s}"
            return f"{x:{w}.4f}"

        print(f"  {ep:3d} {ap75:8.4f} {fmt(rs)} {fmt(rl)} {fmt(kl)} {fmt(dl)} {ratio:>8s} {fmt(qc,8)} {fmt(gn)} {fmt(pc,6)}")

    # Summary stats
    rss = [r["reward_std"] for r in h if not np.isnan(r.get("reward_std", float("nan")))]
    rls = [r["rl_loss"] for r in h if not np.isnan(r.get("rl_loss", float("nan")))]
    kls = [r["kl_loss"] for r in h if not np.isnan(r.get("kl_loss", float("nan")))]
    dls = [r["det_loss"] for r in h if not np.isnan(r.get("det_loss", float("nan")))]
    qcs = [r["q_iou_corr"] for r in h if not np.isnan(r.get("q_iou_corr", float("nan")))]

    if rss:
        print(f"  MEAN: reward_std={np.mean(rss):.4f}, rl_loss={np.mean(rls):.4f}, kl_loss={np.mean(kls):.4f}")
        if rls and dls:
            print(f"        rl/det={np.mean(rls)/np.mean(dls)*100:.4f}%, rl/max(kl,det)={np.mean(rls)/max(np.mean(kls),np.mean(dls))*100:.4f}%")
        if qcs:
            print(f"        q_iou_corr={np.mean(qcs):.4f}")
    print()

# Also check round279_extend (30 epoch det_only_unf)
print("=== Round 279 Extend: 30 epoch det_only_unf ===")
em = json.loads(Path("runs/round279_extend_s42/eval_metrics.json").read_text())
for row in em["history"]:
    if row["epoch"] % 5 == 1 or row["epoch"] == 30:
        print(f"  e{row['epoch']:2d}: AP75={row['val_ap75']:.4f} AP50={row['val_ap50']:.4f}")
print()

# Now check if round278 has diagnostics too
print("=== Round 278: Direct FFT reward signals (8 epoch, frozen box_head) ===")
for f in sorted(Path("runs").glob("round278_*_s42/eval_metrics.json")):
    d = json.loads(f.read_text())
    h = d["history"]
    if "energy_reward" not in h[0]:
        continue
    print(f"--- {d['config']} (seed=42) ---")
    print(f"  {'Ep':>3s} {'AP75':>8s} {'r_std':>8s} {'energy':>10s} {'sim':>10s} {'phase':>10s} {'rl_loss':>10s} {'det_loss':>10s}")
    print(f"  {'-'*3} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for row in h:
        ep = row["epoch"]
        ap75 = row.get("val_ap75", 0)
        rs = row.get("reward_std", float("nan"))
        en = row.get("energy_reward", float("nan"))
        sm = row.get("sim_reward", float("nan"))
        ph = row.get("phase_reward", float("nan"))
        rl = row.get("rl_loss", float("nan"))
        dl = row.get("det_loss", float("nan"))
        print(f"  {ep:3d} {ap75:8.4f} {rs:8.4f} {en:10.6f} {sm:10.6f} {ph:10.6f} {rl:10.4f} {dl:10.2f}")
    ens = [r["energy_reward"] for r in h if not np.isnan(r.get("energy_reward", float("nan")))]
    sms = [r["sim_reward"] for r in h if not np.isnan(r.get("sim_reward", float("nan")))]
    phs = [r["phase_reward"] for r in h if not np.isnan(r.get("phase_reward", float("nan")))]
    if ens:
        print(f"  MEAN: energy={np.mean(ens):.6f}, sim={np.mean(sms):.6f}, phase={np.mean(phs):.6f}")
    print()
