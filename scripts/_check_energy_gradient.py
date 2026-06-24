"""Test two approaches to extract energy gradient signal for PG.

1. Temporal-like: energy vs spatial shift — is energy monotonic/smooth around GT?
2. Finite difference: numerical ∂energy/∂box — does -∇E point toward better IoU?
"""
import sys, math
import torch, numpy as np
from pathlib import Path
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
from torchvision.ops import box_iou

DEV = "cuda"; SEED = 42
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
EPS = 3.0  # finite diff epsilon in pixels
set_seed(SEED)

def extract_perchan_fft(x):
    C = x.shape[1]; H, W = x.shape[-2], x.shape[-1]
    fft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    amp = torch.abs(fft)
    freq_h = torch.fft.fftfreq(H, device=x.device)
    freq_w = torch.fft.rfftfreq(W, device=x.device)
    Y, X = torch.meshgrid(freq_h, freq_w, indexing='ij')
    r = torch.sqrt(X**2 + Y**2); R = r.max().clamp_min(1e-6); rn = r / R
    lo = (rn <= 0.3).float(); md = ((rn > 0.3) & (rn <= 0.7)).float(); hi = (rn > 0.7).float()
    a_lo = (amp * lo).flatten(2).sum(2); a_md = (amp * md).flatten(2).sum(2); a_hi = (amp * hi).flatten(2).sum(2)
    return a_lo / (a_lo + a_md + a_hi + 1e-8)

def pool_energy(box_pool, fpn, boxes, img_shape):
    """Pool ROI and compute mean energy."""
    pooled = box_pool(fpn, [boxes], [img_shape])
    return extract_perchan_fft(pooled).mean(dim=1)  # (N,)

model = build_detector({"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
    "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
    "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}}).to(DEV)
ckpt = torch.load(CKPT, map_location=DEV)
model.load_state_dict(ckpt["model"])
model.eval()

fpn_feats = {}
model.backbone.register_forward_hook(
    lambda m, i, o: fpn_feats.update({"f": {k: o[k] for k in o if k != "pool"}}))
box_pool = model.roi_heads.box_roi_pool

tl, vl = build_penn_fudan_loaders({"data": {"root": "./data", "max_size": 320,
    "train_fraction": 0.8, "num_workers": 0}, "train": {"batch_size": 2}})

# ===== 1. Temporal-like: energy vs spatial shift =====
print("=== 1. Temporal-like: Energy vs spatial shift around GT ===")

shifts_px = [-20, -10, -5, -2, 0, 2, 5, 10, 20]
shift_results = {s: [] for s in shifts_px}

for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]
    fpn_feats.clear()
    with torch.no_grad(): _ = model(imgs_d, [{k: v.to(DEV) for k, v in t.items()} for t in tgts])
    fpn = fpn_feats.get("f")
    if fpn is None: continue
    img_shape = (imgs_d[0].shape[-2], imgs_d[0].shape[-1])

    for tgt in tgts:
        for box in tgt["boxes"]:
            w = (box[2] - box[0]).item(); h = (box[3] - box[1]).item()
            cx = (box[0] + box[2])/2; cy = (box[1] + box[3])/2

            for s in shifts_px:
                shifted = torch.tensor([[
                    cx - w/2 + s, cy - h/2 + s,
                    cx + w/2 + s, cy + h/2 + s
                ]], device=DEV).clamp(min=0)
                shifted[0, 2] = shifted[0, 2].clamp(max=img_shape[1]-1)
                shifted[0, 3] = shifted[0, 3].clamp(max=img_shape[0]-1)
                en = pool_energy(box_pool, fpn, shifted, img_shape).item()
                iou = box_iou(shifted.cpu(), box.unsqueeze(0))[0,0].item()
                shift_results[s].append({"en": en, "iou": iou})

print(f"{'Shift':>6s} {'en_mean':>10s} {'en_std':>8s} {'iou_mean':>10s} {'Δen/Δpx':>10s}")
prev_en = None
for s in shifts_px:
    vals = shift_results[s]
    en_m = np.mean([v["en"] for v in vals])
    en_s = np.std([v["en"] for v in vals])
    iou_m = np.mean([v["iou"] for v in vals])
    delta_str = ""
    if prev_en is not None:
        d = s - prev_shift
        if d > 0:
            delta_str = f"{(en_m - prev_en)/d:+.6f}"
    print(f"{s:6d} {en_m:10.4f} {en_s:8.4f} {iou_m:10.4f} {delta_str:>10s}")
    prev_en = en_m; prev_shift = s

# ===== 2. Finite difference gradient =====
print("\n=== 2. Finite difference dEnergy/dBox ===")

fd_results = {"grad_align": []}  # cosine similarity: -∇E vs direction-to-better-IoU
for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]
    fpn_feats.clear()
    with torch.no_grad(): _ = model(imgs_d, [{k: v.to(DEV) for k, v in t.items()} for t in tgts])
    fpn = fpn_feats.get("f")
    if fpn is None: continue
    img_shape = (imgs_d[0].shape[-2], imgs_d[0].shape[-1])

    for tgt in tgts:
        for box in tgt["boxes"]:
            bx1, by1, bx2, by2 = box.tolist()
            bw = bx2 - bx1; bh = by2 - by1
            bcx = bx1 + bw/2; bcy = by1 + bh/2

            # Base energy
            base = torch.tensor([[bx1, by1, bx2, by2]], device=DEV)
            en0 = pool_energy(box_pool, fpn, base, img_shape).item()

            # Perturb cx ± EPS
            p_cx_plus = torch.tensor([[
                bcx + EPS - bw/2, bcy - bh/2,
                bcx + EPS + bw/2, bcy + bh/2
            ]], device=DEV).clamp(min=0)
            en_cx_p = pool_energy(box_pool, fpn, p_cx_plus, img_shape).item()

            # Perturb cx - EPS
            p_cx_minus = torch.tensor([[
                bcx - EPS - bw/2, bcy - bh/2,
                bcx - EPS + bw/2, bcy + bh/2
            ]], device=DEV).clamp(min=0)
            en_cx_m = pool_energy(box_pool, fpn, p_cx_minus, img_shape).item()

            dE_dcx = (en_cx_p - en_cx_m) / (2 * EPS)

            # Compute IoU change in same direction
            iou_cx_p = box_iou(p_cx_plus.cpu(), box.unsqueeze(0))[0,0].item()
            iou_cx_m = box_iou(p_cx_minus.cpu(), box.unsqueeze(0))[0,0].item()
            dIou_dcx = (iou_cx_p - iou_cx_m) / (2 * EPS)

            fd_results["grad_align"].append({
                "dE_dcx": dE_dcx,
                "dIou_dcx": dIou_dcx,
                "en0": en0, "box_area": bw*bh,
            })

# Analyze gradient alignment
align = fd_results["grad_align"]
dE = np.array([a["dE_dcx"] for a in align])
dI = np.array([a["dIou_dcx"] for a in align])

# Count: does -dE sign agree with +dIou sign?
same_sign = sum(1 for i in range(len(dE)) if np.sign(-dE[i]) == np.sign(dI[i]))
opp_sign = sum(1 for i in range(len(dE)) if np.sign(-dE[i]) == -np.sign(dI[i]) and dI[i] != 0)
zero_iou = sum(1 for i in range(len(dE)) if dI[i] == 0)

print(f"  Total GT boxes analyzed: {len(align)}")
print(f"  sign(-dE) == sign(dIou):    {same_sign} ({same_sign/len(align)*100:.1f}%)")
print(f"  sign(-dE) != sign(dIou):    {opp_sign} ({opp_sign/len(align)*100:.1f}%)")
print(f"  dIou == 0 (at boundary):    {zero_iou}")
print(f"  dE_dcx mean: {np.mean(np.abs(dE)):.6f}  dIou_dcx mean: {np.mean(np.abs(dI)):.6f}")

# Per-area breakdown
print(f"\n  {'Area':>10s} {'n':>5s} {'|dE|':>10s} {'|dIou|':>10s} {'align%':>8s}")
for lo, hi, label in [(0, 500, "<500"), (500, 2000, "500-2k"), (2000, 99999, ">2k")]:
    sub = [a for a in align if lo <= a["box_area"] < hi]
    if not sub: continue
    de = np.array([a["dE_dcx"] for a in sub])
    di = np.array([a["dIou_dcx"] for a in sub])
    al = sum(1 for i in range(len(de)) if np.sign(-de[i]) == np.sign(di[i]))
    print(f"  {label:>10s} {len(sub):5d} {np.mean(np.abs(de)):10.6f} {np.mean(np.abs(di)):10.6f} {al/len(sub)*100:7.1f}%")

# Key: if we could use -dE as delta signal, what would happen?
# Compare: random delta direction vs -dE direction
print(f"\n=== 3. -gradE as PG direction vs random delta ===")

random_wins = 0; grad_wins = 0; total = 0
for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]
    fpn_feats.clear()
    with torch.no_grad(): _ = model(imgs_d, [{k: v.to(DEV) for k, v in t.items()} for t in tgts])
    fpn = fpn_feats.get("f")
    if fpn is None: continue
    img_shape = (imgs_d[0].shape[-2], imgs_d[0].shape[-1])

    for tgt in tgts:
        for box in tgt["boxes"]:
            bx1, by1, bx2, by2 = box.tolist()
            bw = bx2 - bx1; bh = by2 - by1
            bcx = bx1 + bw/2; bcy = by1 + bh/2

            # Compute dE/dbox via finite diff (4 params)
            base = torch.tensor([[bx1, by1, bx2, by2]], device=DEV)
            en0 = pool_energy(box_pool, fpn, base, img_shape).item()

            # dE/dcx
            p_plus = torch.tensor([[bcx+EPS-bw/2, bcy-bh/2, bcx+EPS+bw/2, bcy+bh/2]], device=DEV).clamp(min=0)
            p_minus = torch.tensor([[bcx-EPS-bw/2, bcy-bh/2, bcx-EPS+bw/2, bcy+bh/2]], device=DEV).clamp(min=0)
            dE_cx = (pool_energy(box_pool, fpn, p_plus, img_shape).item() -
                     pool_energy(box_pool, fpn, p_minus, img_shape).item()) / (2*EPS)

            # dE/dcy
            p_plus = torch.tensor([[bcx-bw/2, bcy+EPS-bh/2, bcx+bw/2, bcy+EPS+bh/2]], device=DEV).clamp(min=0)
            p_minus = torch.tensor([[bcx-bw/2, bcy-EPS-bh/2, bcx+bw/2, bcy-EPS+bh/2]], device=DEV).clamp(min=0)
            dE_cy = (pool_energy(box_pool, fpn, p_plus, img_shape).item() -
                     pool_energy(box_pool, fpn, p_minus, img_shape).item()) / (2*EPS)

            # dE/dw
            p_plus = torch.tensor([[bcx-(bw+EPS)/2, bcy-bh/2, bcx+(bw+EPS)/2, bcy+bh/2]], device=DEV).clamp(min=0)
            p_minus = torch.tensor([[bcx-(bw-EPS)/2, bcy-bh/2, bcx+(bw-EPS)/2, bcy+bh/2]], device=DEV).clamp(min=0)
            dE_w = (pool_energy(box_pool, fpn, p_plus, img_shape).item() -
                    pool_energy(box_pool, fpn, p_minus, img_shape).item()) / (2*EPS)

            grad_e = np.array([dE_cx, dE_cy, dE_w])
            if np.linalg.norm(grad_e) < 1e-8: continue

            # Move in -∇E direction (small step = 3px equivalent)
            step = 3.0
            dir_e = -grad_e / np.linalg.norm(grad_e)

            new_cx = bcx + step * dir_e[0]
            new_cy = bcy + step * dir_e[1]
            new_w = max(bw + step * dir_e[2], 5)
            new_bh = bh * (new_w / bw) if bw > 0 else bh  # preserve aspect ratio approx

            box_grad = torch.tensor([[
                new_cx - new_w/2, new_cy - new_bh/2,
                new_cx + new_w/2, new_cy + new_bh/2
            ]], device=DEV).clamp(min=0)
            iou_grad = box_iou(box_grad.cpu(), box.unsqueeze(0))[0,0].item()

            # Random delta for comparison
            rx = 3.0 * (2*np.random.random() - 1)
            ry = 3.0 * (2*np.random.random() - 1)
            rw = 3.0 * (2*np.random.random() - 1)
            rdir = np.array([rx, ry, rw])
            rdir = rdir / max(np.linalg.norm(rdir), 1e-8)

            rc_cx = bcx + step * rdir[0]; rc_cy = bcy + step * rdir[1]
            rc_w = max(bw + step * rdir[2], 5); rc_bh = bh * (rc_w / bw) if bw > 0 else bh
            box_rand = torch.tensor([[
                rc_cx - rc_w/2, rc_cy - rc_bh/2,
                rc_cx + rc_w/2, rc_cy + rc_bh/2
            ]], device=DEV).clamp(min=0)
            iou_rand = box_iou(box_rand.cpu(), box.unsqueeze(0))[0,0].item()

            total += 1
            if iou_grad > iou_rand: grad_wins += 1
            elif iou_rand > iou_grad: random_wins += 1

print(f"  Total comparisons: {total}")
print(f"  -∇E direction better: {grad_wins} ({grad_wins/total*100:.1f}%)")
print(f"  Random direction better: {random_wins} ({random_wins/total*100:.1f}%)")
