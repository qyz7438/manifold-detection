import json, numpy as np
from pathlib import Path

runs_dir = Path("runs")
results = {}

for f in sorted(runs_dir.glob("round280_*/*eval_metrics.json")):
    name = f.parent.name
    if not any(x in name for x in ["_s42", "_s123", "_s456"]):
        continue
    d = json.loads(f.read_text())
    cfg = d.get("config", name.rsplit("_s", 1)[0])
    seed = d.get("seed", "?")
    best_h = max(d["history"], key=lambda r: r["val_ap75"])
    results.setdefault(cfg, []).append({
        "seed": seed,
        "best_ap75": d["best_ap75"],
        "best_ap50": best_h["val_ap50"],
        "best_epoch": best_h["epoch"],
        "final_ap75": d["history"][-1]["val_ap75"],
        "ece": best_h.get("ece", float("nan")),
        "q_corr": best_h.get("q_iou_corr", float("nan")),
        "reward_std": best_h.get("reward_std", float("nan")),
        "kl_loss": best_h.get("kl_loss", float("nan")),
        "rl_loss": best_h.get("rl_loss", float("nan")),
        "det_loss": best_h.get("det_loss", float("nan")),
    })

baseline = np.mean([r["best_ap75"] for r in results.get("det_only_unf", [])])
print(f"Baseline (det_only_unf) = {baseline:.4f}")
print()
print(f"{'Config':<24s} {'N':>2s} {'AP75':>8s} {'+/-std':>8s} {'Δbaseline':>10s} {'AP50':>8s} {'ECE':>8s} {'q_corr':>8s} {'r_std':>8s}")
print("-" * 110)

for cfg in sorted(results.keys()):
    rows = results[cfg]
    ap75s = [r["best_ap75"] for r in rows]
    ap50s = [r["best_ap50"] for r in rows]
    eces = [r["ece"] for r in rows]
    qc = [r["q_corr"] for r in rows if not np.isnan(r["q_corr"])]
    rs = [r["reward_std"] for r in rows if not np.isnan(r["reward_std"])]
    delta = np.mean(ap75s) - baseline
    qs = f"{np.mean(qc):.4f}" if qc else "N/A"
    rss = f"{np.mean(rs):.4f}" if rs else "N/A"
    print(f"{cfg:<24s} {len(rows):2d} {np.mean(ap75s):8.4f} {np.std(ap75s):8.4f} {delta:+10.4f} {np.mean(ap50s):8.4f} {np.mean(eces):8.4f} {qs:>8s} {rss:>8s}")

print()
print("=== Per-seed ===")
print(f"{'Config':<24s} {'Seed':>5s} {'BestAP75':>10s} {'BestEp':>7s} {'FinalAP75':>10s} {'AP50':>8s}")
for cfg in sorted(results.keys()):
    for r in results[cfg]:
        sd = r['seed']
        if isinstance(sd, int):
            print(f"{cfg:<24s} {sd:5d} {r['best_ap75']:10.4f} {r['best_epoch']:7d} {r['final_ap75']:10.4f} {r['best_ap50']:8.4f}")
        else:
            print(f"{cfg:<24s} {str(sd):>5s} {r['best_ap75']:10.4f} {r['best_epoch']:7d} {r['final_ap75']:10.4f} {r['best_ap50']:8.4f}")

# Signal analysis
print()
print("=== Signal impact analysis ===")
print()

# 1. Unfreezing effect
det_frozen_rows = results.get("det_only_frozen", [])
if det_frozen_rows:
    det_frozen = np.mean([r["best_ap75"] for r in det_frozen_rows])
    print(f"[1] Unfreezing effect (frozen -> unfrozen):")
    print(f"    det_only_frozen = {det_frozen:.4f}")
    print(f"    det_only_unf    = {baseline:.4f}")
    print(f"    delta           = {baseline - det_frozen:+.4f}")
    print()

# 2. RLVR pure groups
print(f"[2] RLVR pure (no FFT, no verifier) vs det_only_unf:")
for cfg in ["ap75_event", "grpo_adv_g4", "grpo_adv_g8"]:
    if cfg in results:
        rows = results[cfg]
        m = np.mean([r["best_ap75"] for r in rows])
        s = np.std([r["best_ap75"] for r in rows])
        print(f"    {cfg:<20s}: {m:.4f} +/-{s:.4f}  delta={m-baseline:+.4f}")

print()

# 3. FFT signal
print(f"[3] FFT variants:")
for cfg in ["per_chan_fft", "fft_loc_only", "grpo_fft_g4", "grpo_fft_g8"]:
    if cfg in results:
        rows = results[cfg]
        m = np.mean([r["best_ap75"] for r in rows])
        s = np.std([r["best_ap75"] for r in rows])
        print(f"    {cfg:<20s}: {m:.4f} +/-{s:.4f}  delta={m-baseline:+.4f}")

print()

# 4. Random / control
print(f"[4] Random / control baselines:")
for cfg in ["random_qnorm", "frozen_random"]:
    if cfg in results:
        rows = results[cfg]
        m = np.mean([r["best_ap75"] for r in rows])
        s = np.std([r["best_ap75"] for r in rows])
        print(f"    {cfg:<20s}: {m:.4f} +/-{s:.4f}  delta={m-baseline:+.4f}")

print()

# 5. Learned verifier
print(f"[5] Learned verifier:")
for cfg in ["aligned_verifier", "select_penalty"]:
    if cfg in results:
        rows = results[cfg]
        m = np.mean([r["best_ap75"] for r in rows])
        s = np.std([r["best_ap75"] for r in rows])
        qc = np.mean([r["q_corr"] for r in rows if not np.isnan(r["q_corr"])])
        print(f"    {cfg:<20s}: {m:.4f} +/-{s:.4f}  delta={m-baseline:+.4f}  q_corr={qc:.4f}")

print()

# 6. AFM groups
print(f"[6] AFM architecture (from baseline checkpoint, AFM layers random init):")
for cfg in ["mid06_frozen", "mid06_unfrozen", "apost_frozen", "cpost_frozen", "phase_frozen"]:
    if cfg in results:
        rows = results[cfg]
        m = np.mean([r["best_ap75"] for r in rows])
        s = np.std([r["best_ap75"] for r in rows])
        print(f"    {cfg:<20s}: {m:.4f} +/-{s:.4f}  delta={m-baseline:+.4f}")

print()

# Effect size ranking
print("=== Effect size ranking (delta from baseline) ===")
all_deltas = []
for cfg, rows in results.items():
    m = np.mean([r["best_ap75"] for r in rows])
    s = np.std([r["best_ap75"] for r in rows])
    all_deltas.append((cfg, m - baseline, s, m))
all_deltas.sort(key=lambda x: -x[1])
for cfg, delta, std, mean in all_deltas:
    sig = " **" if delta > 0.002 else ""
    sig = " !!" if delta < -0.005 else sig
    print(f"  {cfg:<24s}: {delta:+7.4f} +/-{std:.4f}{sig}")
