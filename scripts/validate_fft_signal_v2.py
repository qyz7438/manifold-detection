print(f"\nTotal proposals: {len(all_data)}")

# Extract arrays with new key names
ious = np.array([d["iou"] for d in all_data])
is_best = np.array([d["best_for_gt"] for d in all_data])
nms_survives = np.array([d["nms_survives"] for d in all_data])
confs = np.array([d["conf"] for d in all_data])
gt_ids = np.array([d["gt_id"] for d in all_data])

FFT_KEYS = [
    "amp_lo", "amp_mid", "amp_hi",
    "amp_lo_ratio", "amp_hi_ratio", "amp_var",
    "phase_var", "phase_lo_var", "phase_mid_var", "phase_hi_var",
    "spec_entropy",
]
fft = {name: np.array([d[name] for d in all_data]) for name in FFT_KEYS}

# === 1. Global Cohen's d (best vs non-best) ===
print("\n=== Global discriminability (best vs non-best) ===")
print(f"{'Feature':<18s} {'best_mean':>8s} {'nonbest':>8s} {'gap':>8s} {'cohen_d':>8s}")
for name in FFT_KEYS + ["iou", "conf"]:
    feat = fft[name] if name in fft else (ious if name == "iou" else confs)
    pos = feat[is_best]; neg = feat[~is_best]
    if len(pos) > 0 and len(neg) > 0:
        gap = pos.mean() - neg.mean()
        d = gap / (np.sqrt(pos.var() + neg.var()) / 2 + 1e-8)
        marker = " <<<" if abs(d) > 0.8 else ""
        print(f"{name:<18s} {pos.mean():8.4f} {neg.mean():8.4f} {gap:8.4f} {d:8.3f}{marker}")

# === 2. Within IoU bins: FFT best-vs-nonbest ===
print("\n=== Within IoU bins: FFT Cohen's d (best vs non-best) ===")
for lo, hi in [(0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]:
    mask = (ious >= lo) & (ious < hi) & is_best
    nmask = (ious >= lo) & (ious < hi) & ~is_best
    n_total = mask.sum() + nmask.sum()
    if n_total < 5 or nmask.sum() == 0:
        continue
    strong = []
    for name in FFT_KEYS:
        bp, np_ = fft[name][mask], fft[name][nmask]
        d = (bp.mean() - np_.mean()) / (np.sqrt(bp.var() + np_.var()) / 2 + 1e-8)
        if abs(d) > 0.5:
            strong.append((name, d, bp.mean(), np_.mean()))
    if strong:
        print(f"  IoU[{lo:.1f},{hi:.1f}) n={n_total}:")
        for name, d, bm, nm in sorted(strong, key=lambda x: -abs(x[1]))[:5]:
            print(f"    {name:<16s} d={d:+7.3f}  best={bm:.4f} non={nm:.4f}")

# === 3. FFT tiebreaker by GT group ===
print("\n=== FFT tiebreaker: per-GT ranking improvement ===")
improvements = {name: [] for name in FFT_KEYS}
for gid in np.unique(gt_ids):
    if gid < 0: continue
    gmask = gt_ids == gid
    if gmask.sum() < 2: continue
    gb = is_best[gmask]
    if gb.sum() == 0: continue
    gi = ious[gmask]
    iou_rank = np.argsort(-gi)
    bp_iou = np.where(iou_rank == np.where(gb)[0][0])[0][0]
    for name in FFT_KEYS:
        gf = fft[name][gmask]
        sign = -1 if "hi" in name or "var" in name else 1
        c = -gi * 100 + sign * gf * 0.1
        cr = np.argsort(c)
        bp_c = np.where(cr == np.where(gb)[0][0])[0][0]
        if bp_c < bp_iou:
            improvements[name].append(bp_iou - bp_c)

print(f"  GT groups (>=2 props): {sum(1 for g in np.unique(gt_ids) if g>=0 and (gt_ids==g).sum()>=2)}")
for name in sorted(improvements, key=lambda n: -len(improvements[n])):
    if improvements[name]:
        print(f"  {name:<16s}: {len(improvements[name]):3d} improved, mean +{np.mean(improvements[name]):.2f} rank")

# === 4. NMS survival ===
print("\n=== NMS survival: Cohen's d ===")
print(f"{'Feature':<18s} {'surv':>8s} {'nonsurv':>8s} {'gap':>8s} {'d':>8s}")
for name in FFT_KEYS + ["iou", "conf"]:
    feat = fft[name] if name in fft else (ious if name == "iou" else confs)
    pos = feat[nms_survives]; neg = feat[~nms_survives]
    if len(pos) > 0 and len(neg) > 0:
        gap = pos.mean() - neg.mean()
        d = gap / (np.sqrt(pos.var() + neg.var()) / 2 + 1e-8)
        marker = " <<<" if abs(d) > 0.5 else ""
        print(f"{name:<18s} {pos.mean():8.4f} {neg.mean():8.4f} {gap:8.4f} {d:8.3f}{marker}")

# === 5. FFT → calibration error correlation ===
print("\n=== Calibration: FFT ~ |confidence - IoU| correlation ===")
for lo, hi in [(0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]:
    mask = (ious >= lo) & (ious < hi)
    if mask.sum() < 10: continue
    calib_err = np.abs(confs[mask] - ious[mask])
    for name in FFT_KEYS[:8]:
        corr = np.corrcoef(fft[name][mask], calib_err)[0, 1]
        if abs(corr) > 0.10:
            print(f"  IoU[{lo:.1f},{hi:.1f}) {name:<16s} corr={corr:+.3f}")
