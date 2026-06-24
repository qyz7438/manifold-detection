"""Validate whether FFT features carry signal that IoU doesn't.

Loads baseline model, extracts proposals from val set, computes:
- IoU with GT (per proposal)
- FFT features (energy_lo, energy_hi, phase_var) per ROI crop
- Which proposals survive NMS with GT-confidence ordering

Checks: within same IoU bins, can FFT distinguish NMS-survivors from non-survivors?
"""
import sys, json
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.ops import box_iou, nms
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_penn_fudan_loaders_320,
    decode_boxes,
    extract_perchan_fft,
)
from scripts.round2102_runner import bm
from spectral_detection_posttrain.utils.seed import set_seed

set_seed(42)
DEV = "cuda"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"

model = bm().to(DEV)
ckpt = torch.load(CKPT, map_location=DEV)
model.load_state_dict(ckpt["model"])
model.eval()

# Hooks
box_head_in, roi_crops, sampled_props = {}, {}, {}
model.roi_heads.box_head.register_forward_pre_hook(
    lambda m, args: box_head_in.update({"x": args[0]})
)
model.roi_heads.box_roi_pool.register_forward_pre_hook(
    lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]})
)
model.roi_heads.box_roi_pool.register_forward_hook(
    lambda m, i, o: roi_crops.update({"c": o.clone()})
)

_, vl = build_penn_fudan_loaders_320(batch_size=1)

all_data = []

for img, tgt in tqdm(vl, desc="Extracting proposals"):
    img_d = [img[0].to(DEV)]
    tgt_d = [{k: v.to(DEV) for k, v in tgt[0].items()}]
    box_head_in.clear(); roi_crops.clear(); sampled_props.clear()

    with torch.no_grad():
        model(img_d, tgt_d)  # training mode forward to get all proposals

    rf = box_head_in.get("x")
    crops = roi_crops.get("c")
    sp_raw = sampled_props.get("p")
    if rf is None or crops is None or sp_raw is None or rf.shape[0] == 0:
        continue

    sp_cat = torch.cat(sp_raw, dim=0)
    bf = model.roi_heads.box_head(rf)
    reg_out = model.roi_heads.box_predictor.bbox_pred(bf)
    person_deltas = reg_out[:, 2:6]
    decoded = decode_boxes(sp_cat, person_deltas)
    gt_boxes = tgt_d[0]["boxes"]

    N = sp_cat.shape[0]
    # FFT features: extract ALL 12 raw features + derived
    fft = extract_perchan_fft(crops)  # (N, 6*C)
    ch = fft.shape[1] // 6
    # Amplitude: bands 0-2 (lo, mid, hi)
    amp_lo = fft[:, 0*ch:1*ch].mean(dim=1)   # (N,) low-freq amplitude
    amp_mid = fft[:, 1*ch:2*ch].mean(dim=1)  # (N,) mid-freq amplitude
    amp_hi = fft[:, 2*ch:3*ch].mean(dim=1)   # (N,) high-freq amplitude
    # Phase: bands 3-5 (lo, mid, hi)
    phase_lo = fft[:, 3*ch:4*ch].mean(dim=1)  # (N,) low-freq phase mean
    phase_mid = fft[:, 4*ch:5*ch].mean(dim=1)
    phase_hi = fft[:, 5*ch:6*ch].mean(dim=1)
    # Derived
    amp_total = amp_lo + amp_mid + amp_hi + 1e-8
    amp_lo_ratio = amp_lo / amp_total
    amp_mid_ratio = amp_mid / amp_total
    amp_hi_ratio = amp_hi / amp_total
    # Amplitude variance (texture richness proxy)
    amp_var = fft[:, 0*ch:3*ch].var(dim=1)
    # Phase variance per band (structure consistency)
    phase_lo_var = fft[:, 3*ch:4*ch].var(dim=1)
    phase_mid_var = fft[:, 4*ch:5*ch].var(dim=1)
    phase_hi_var = fft[:, 5*ch:6*ch].var(dim=1)
    phase_var = fft[:, 3*ch:6*ch].var(dim=1)  # total phase variance
    # Spectral entropy across 6 bands (how "peaked" the spectrum is)
    band_means = torch.stack([amp_lo, amp_mid, amp_hi,
                               fft[:, 3*ch:4*ch].abs().mean(dim=1),
                               fft[:, 4*ch:5*ch].abs().mean(dim=1),
                               fft[:, 5*ch:6*ch].abs().mean(dim=1)], dim=1)  # (N, 6)
    band_norm = band_means / (band_means.sum(dim=1, keepdim=True) + 1e-8)
    spec_entropy = -(band_norm * torch.log(band_norm + 1e-8)).sum(dim=1)  # (N,)

    if len(gt_boxes) == 0:
        for i in range(N):
            all_data.append({
                "iou": 0.0, "best_for_gt": False, "nms_survives": False, "gt_id": -1,
                "amp_lo": amp_lo[i].item(), "amp_mid": amp_mid[i].item(), "amp_hi": amp_hi[i].item(),
                "amp_lo_ratio": amp_lo_ratio[i].item(), "amp_hi_ratio": amp_hi_ratio[i].item(),
                "amp_var": amp_var[i].item(), "phase_var": phase_var[i].item(),
                "phase_lo_var": phase_lo_var[i].item(), "phase_mid_var": phase_mid_var[i].item(),
                "phase_hi_var": phase_hi_var[i].item(), "spec_entropy": spec_entropy[i].item(),
                "conf": 0.0,
            })
        continue

    ious = box_iou(decoded, gt_boxes)  # (N, G)

    best_iou, best_gt = ious.max(dim=1)  # (N,)
    conf = F.softmax(model.roi_heads.box_predictor.cls_score(bf), dim=-1)[:, 1]
    keep_nms = nms(decoded, conf, iou_threshold=0.5).cpu()

    for i in range(N):
        matched_gt = best_gt[i].item()
        matched_iou = best_iou[i].item()
        is_best = True
        for j in range(N):
            if j != i and best_gt[j].item() == matched_gt and best_iou[j].item() > matched_iou:
                is_best = False
                break

        all_data.append({
            "iou": matched_iou, "best_for_gt": is_best,
            "nms_survives": i in keep_nms, "gt_id": matched_gt, "conf": conf[i].item(),
            "amp_lo": amp_lo[i].item(), "amp_mid": amp_mid[i].item(), "amp_hi": amp_hi[i].item(),
            "amp_lo_ratio": amp_lo_ratio[i].item(), "amp_hi_ratio": amp_hi_ratio[i].item(),
            "amp_var": amp_var[i].item(), "phase_var": phase_var[i].item(),
            "phase_lo_var": phase_lo_var[i].item(), "phase_mid_var": phase_mid_var[i].item(),
            "phase_hi_var": phase_hi_var[i].item(), "spec_entropy": spec_entropy[i].item(),
        })

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
