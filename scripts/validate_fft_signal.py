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

# === Analysis ===
ious = np.array([d["iou"] for d in all_data])
fft_lo = np.array([d["fft_lo"] for d in all_data])
fft_mid = np.array([d["fft_mid"] for d in all_data])
fft_hi = np.array([d["fft_hi"] for d in all_data])
phase_var = np.array([d["phase_var"] for d in all_data])
is_best = np.array([d["best_for_gt"] for d in all_data])
nms_survives = np.array([d["nms_survives"] for d in all_data])
confs = np.array([d["conf"] for d in all_data])

# 1. Global discriminability
for name, feat in [("fft_lo", fft_lo), ("fft_mid", fft_mid), ("fft_hi", fft_hi),
                     ("phase_var", phase_var), ("conf", confs), ("iou", ious)]:
    pos = feat[is_best]
    neg = feat[~is_best]
    if len(pos) > 0 and len(neg) > 0:
        d = (pos.mean() - neg.mean()) / (np.sqrt(pos.var() + neg.var()) / 2 + 1e-8)
        print(f"  {name:12s} best={pos.mean():.4f} nonbest={neg.mean():.4f} gap={pos.mean()-neg.mean():.4f} cohen_d={d:.3f}")

# 2. Within IoU bins: can FFT separate best from non-best?
print("\n=== Within IoU bins: FFT best-vs-nonbest gap ===")
for lo, hi in [(0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]:
    mask = (ious >= lo) & (ious < hi) & (is_best | True)  # all proposals in bin
    if mask.sum() < 5:
        continue
    bmask = mask & is_best
    nmask = mask & ~is_best
    if nmask.sum() == 0:
        continue
    for fname, feat in [("fft_lo", fft_lo), ("fft_hi", fft_hi), ("phase_var", phase_var)]:
        bp = feat[bmask]; np_ = feat[nmask]
        d = (bp.mean() - np_.mean()) / (np.sqrt(bp.var() + np_.var()) / 2 + 1e-8)
        if abs(d) > 0.3:
            print(f"  IoU[{lo:.1f},{hi:.1f}) {fname:10s}: best={bp.mean():.4f} non={np_.mean():.4f} d={d:.3f}")

# 3. Key question: for proposals with similar IoU, can FFT improve ranking?
print("\n=== FFT tiebreaker: same GT, same IoU bin, ranking improvement ===")
gt_ids = np.array([d["gt_id"] for d in all_data])
improvements = []
for gt_id in np.unique(gt_ids):
    if gt_id < 0:
        continue
    gmask = gt_ids == gt_id
    n_props = gmask.sum()
    if n_props < 2:
        continue
    g_ious = ious[gmask]
    g_fft = fft_hi[gmask]  # use high-freq as quality proxy
    g_best = is_best[gmask]

    # Ranking by IoU
    iou_rank = np.argsort(-g_ious)
    # Ranking by IoU + FFT (FFT as tiebreaker: secondary sort)
    combined = -g_ious * 100 + (-g_fft) * 0.01
    combined_rank = np.argsort(combined)

    # Does FFT move the best proposal higher?
    iou_best_pos = np.where(iou_rank == np.where(g_best)[0][0])[0][0] if g_best.sum() > 0 else -1
    combined_best_pos = np.where(combined_rank == np.where(g_best)[0][0])[0][0] if g_best.sum() > 0 else -1
    if iou_best_pos >= 0 and combined_best_pos < iou_best_pos:
        improvements.append(iou_best_pos - combined_best_pos)

print(f"  GT groups: {(gt_ids >= 0).sum()}")
print(f"  Groups with ≥2 proposals: {sum(1 for g in np.unique(gt_ids) if g >= 0 and (gt_ids == g).sum() >= 2)}")
print(f"  FFT improved best-proposal rank in {len(improvements)} cases")
if improvements:
    print(f"  Mean rank improvement: {np.mean(improvements):.2f} positions")

# 4. What about NMS survival prediction?
print("\n=== NMS survival: can FFT help predict which proposals survive? ===")
for name, feat in [("iou", ious), ("fft_lo", fft_lo), ("fft_hi", fft_hi), ("conf", confs),
                     ("phase_var", phase_var)]:
    surv = feat[nms_survives]
    nonsurv = feat[~nms_survives]
    if len(surv) > 0 and len(nonsurv) > 0:
        d = (surv.mean() - nonsurv.mean()) / (np.sqrt(surv.var() + nonsurv.var()) / 2 + 1e-8)
        print(f"  {name:12s} surv={surv.mean():.4f} nonsurv={nonsurv.mean():.4f} d={d:.3f}")
