"""Evaluate ALL reward metrics against detection difficulty (FN/hard cases)."""
import sys, math
import torch, numpy as np
from pathlib import Path
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
from torchvision.ops import box_iou
import torch.nn.functional as F

DEV = "cuda"; SEED = 42
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
set_seed(SEED)

model = build_detector({"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
    "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
    "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}}).to(DEV)
ckpt = torch.load(CKPT, map_location=DEV)
model.load_state_dict(ckpt["model"])
model.eval()

fpn_feats_dict = {}
model.backbone.register_forward_hook(lambda m, i, o: fpn_feats_dict.update({"f": {k: o[k] for k in o if k != "pool"}}))
box_pool = model.roi_heads.box_roi_pool

tl, vl = build_penn_fudan_loaders({"data": {"root": "./data", "max_size": 320,
    "train_fraction": 0.8, "num_workers": 0}, "train": {"batch_size": 2}})

# --- All reward metrics ---

def compute_energy(fft_f):
    ch = fft_f.shape[1] // 6
    a_lo = fft_f[:, 0*ch:1*ch].sum(dim=1)
    a_total = a_lo + fft_f[:, 1*ch:2*ch].sum(dim=1) + fft_f[:, 2*ch:3*ch].sum(dim=1) + 1e-8
    return 2 * (a_lo / a_total) - 1

def extract_perchan_fft(x):
    C = x.shape[1]; H, W = x.shape[-2], x.shape[-1]
    fft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    amp = torch.abs(fft); pha = torch.angle(fft)
    freq_h = torch.fft.fftfreq(H, device=x.device)
    freq_w = torch.fft.rfftfreq(W, device=x.device)
    Y, X = torch.meshgrid(freq_h, freq_w, indexing='ij')
    r = torch.sqrt(X**2+Y**2); R = r.max().clamp_min(1e-6); rn = r/R
    lo = (rn <= 0.3).float(); md = ((rn > 0.3) & (rn <= 0.7)).float(); hi = (rn > 0.7).float()
    a_lo = (amp*lo).flatten(2).sum(2); a_md = (amp*md).flatten(2).sum(2); a_hi = (amp*hi).flatten(2).sum(2)
    return torch.cat([a_lo, a_md, a_hi], dim=1)  # (N, 3*C) without phase for energy

all_gt_boxes = []  # per GT: {iou, energy, area, aspect, edge_dist, ...}

for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]
    img_shape = (imgs_d[0].shape[-2], imgs_d[0].shape[-1])
    fpn_feats_dict.clear()

    with torch.no_grad():
        preds = model(imgs_d)
    fpn = fpn_feats_dict.get("f")
    if fpn is None: continue

    for i_img in range(len(imgs_d)):
        gt_boxes = tgts[i_img]["boxes"].to(DEV)
        pred_boxes = preds[i_img]["boxes"]
        pred_scores = preds[i_img]["scores"]

        if len(gt_boxes) == 0: continue

        # Best IoU for each GT
        if len(pred_boxes) > 0:
            ious = box_iou(pred_boxes, gt_boxes)  # (P, G)
            best_iou_per_gt, best_pred_per_gt = ious.max(dim=0)
            best_score_per_gt = pred_scores[best_pred_per_gt]
        else:
            best_iou_per_gt = torch.zeros(len(gt_boxes))
            best_score_per_gt = torch.zeros(len(gt_boxes))

        # Pool GT boxes through roi_pool to compute FFT metrics
        gt_list = [gt_boxes]
        pooled_gt = box_pool(fpn, gt_list, [img_shape])  # (G, C, 7, 7)
        fft_f = extract_perchan_fft(pooled_gt)
        energy = compute_energy(fft_f)  # (G,)

        for gi in range(len(gt_boxes)):
            box = gt_boxes[gi]
            w = (box[2]-box[0]).item(); h = (box[3]-box[1]).item()
            area = w*h
            iou = best_iou_per_gt[gi].item()
            score = best_score_per_gt[gi].item() if len(pred_boxes) > 0 else 0

            all_gt_boxes.append({
                "iou": iou,
                "score": score,
                "energy": energy[gi].item(),
                "area": area,
                "width": w, "height": h,
                "aspect": w/max(h, 1),
                "edge_dist": min(box[0].item(), img_shape[1]-box[2].item(),
                                 box[1].item(), img_shape[0]-box[3].item()),
                "cx": (box[0].item()+box[2].item())/2,
                "cy": (box[1].item()+box[3].item())/2,
                "img_w": img_shape[1], "img_h": img_shape[0],
            })

print(f"Total GT boxes analyzed: {len(all_gt_boxes)}")

# --- Categorize GT boxes by detection difficulty ---
iou_vals = np.array([g["iou"] for g in all_gt_boxes])

cats = {
    "FN (IoU<0.3)": iou_vals < 0.3,
    "Hard borderline (0.3-0.5)": (iou_vals >= 0.3) & (iou_vals < 0.5),
    "Soft borderline (0.5-0.65)": (iou_vals >= 0.5) & (iou_vals < 0.65),
    "Easy TP (0.65-0.8)": (iou_vals >= 0.65) & (iou_vals < 0.8),
    "Perfect TP (>=0.8)": iou_vals >= 0.8,
}

# --- Metrics to test ---
# 1. Energy (raw, -1 to 1)
# 2. Area
# 3. Aspect ratio
# 4. Edge distance
# 5. Box position (cx, cy)
# 6. Energy x Area (interaction)
# 7. R_loc (the IoU discrete reward itself)

print(f"\n{'Category':<28s} {'n':>5s} {'energy':>8s} {'area':>8s} {'aspect':>8s} {'edge_dist':>8s} {'cx':>8s} {'cy':>8s}")
print("-" * 90)

for cat_name, mask in cats.items():
    subset = [all_gt_boxes[i] for i in range(len(all_gt_boxes)) if mask[i]]
    if not subset: continue
    print(f"{cat_name:<28s} {len(subset):5d} "
          f"{np.mean([g['energy'] for g in subset]):8.4f} "
          f"{np.mean([g['area'] for g in subset]):8.0f} "
          f"{np.mean([g['aspect'] for g in subset]):8.3f} "
          f"{np.mean([g['edge_dist'] for g in subset]):8.1f} "
          f"{np.mean([g['cx'] for g in subset]):8.1f} "
          f"{np.mean([g['cy'] for g in subset]):8.1f}")

# --- Rank correlation with IoU ---
print(f"\n=== Metric vs IoU rank correlation ===")
metrics = {
    "energy": np.array([g["energy"] for g in all_gt_boxes]),
    "-energy": np.array([-g["energy"] for g in all_gt_boxes]),
    "area": np.array([g["area"] for g in all_gt_boxes]),
    "aspect": np.array([g["aspect"] for g in all_gt_boxes]),
    "edge_dist": np.array([g["edge_dist"] for g in all_gt_boxes]),
    "cx": np.array([g["cx"] for g in all_gt_boxes]),
    "cy": np.array([g["cy"] for g in all_gt_boxes]),
}

for name, vals in metrics.items():
    r = np.corrcoef(vals, iou_vals)[0, 1]
    print(f"  {name:<15s}: Pearson r={r:+.4f}")

# --- The big question: which metric best identifies hard cases? ---
print(f"\n=== Separation: FN vs TP (Cohen d) ===")
fn_mask = iou_vals < 0.5
tp_mask = iou_vals >= 0.5
fn_count = fn_mask.sum(); tp_count = tp_mask.sum()

for name, vals in metrics.items():
    fn_vals = vals[fn_mask]; tp_vals = vals[tp_mask]
    delta = np.mean(fn_vals) - np.mean(tp_vals)
    pooled_std = np.sqrt((np.std(fn_vals)**2 + np.std(tp_vals)**2) / 2)
    d = delta / max(pooled_std, 1e-6)
    print(f"  {name:<15s}: FN={np.mean(fn_vals):.4f} TP={np.mean(tp_vals):.4f} Δ={delta:+.4f} Cohen d={d:+.2f}")

# --- Borderline-specific analysis ---
print(f"\n=== Borderline only (0.3 <= IoU < 0.5) — hardest cases ===")
bl_mask = (iou_vals >= 0.3) & (iou_vals < 0.5)
bl_indices = np.where(bl_mask)[0]

if len(bl_indices) > 0:
    bl_energy = np.array([all_gt_boxes[i]["energy"] for i in bl_indices])
    bl_iou = np.array([all_gt_boxes[i]["iou"] for i in bl_indices])
    bl_area = np.array([all_gt_boxes[i]["area"] for i in bl_indices])
    bl_aspect = np.array([all_gt_boxes[i]["aspect"] for i in bl_indices])
    bl_edge = np.array([all_gt_boxes[i]["edge_dist"] for i in bl_indices])

    print(f"  N={len(bl_indices)}")
    print(f"  energy: mean={bl_energy.mean():.4f} std={bl_energy.std():.4f}")
    print(f"  area:   mean={bl_area.mean():.0f} std={bl_area.std():.0f}")
    print(f"  aspect: mean={bl_aspect.mean():.3f} std={bl_aspect.std():.3f}")
    print(f"  edge:   mean={bl_edge.mean():.1f} std={bl_edge.std():.1f}")

    # Within borderline, does any metric predict IoU variation?
    print(f"  Within-borderline correlation with IoU:")
    for name, vals in {"energy": bl_energy, "area": bl_area, "aspect": bl_aspect, "edge": bl_edge}.items():
        r = np.corrcoef(vals, bl_iou)[0, 1]
        print(f"    {name}: r={r:+.4f}")

# --- What combination of metrics best predicts difficulty? ---
print(f"\n=== Feature importance for FN prediction ===")
X = np.column_stack([
    np.array([g["energy"] for g in all_gt_boxes]),
    np.array([g["area"] for g in all_gt_boxes]),
    np.array([g["aspect"] for g in all_gt_boxes]),
    np.array([g["edge_dist"] for g in all_gt_boxes]),
    np.array([g["cx"] for g in all_gt_boxes]),
    np.array([g["cy"] for g in all_gt_boxes]),
    np.log(np.array([g["area"] for g in all_gt_boxes]) + 1),
])
y_fn = (iou_vals < 0.5).astype(float)
y_iou = iou_vals

# Simple univariate AUC for FN classification
from sklearn.metrics import roc_auc_score
print(f"{'Feature':<15s} {'AUC(FN)':>8s}")
for i, name in enumerate(["energy", "area", "aspect", "edge_dist", "cx", "cy", "log_area"]):
    x = X[:, i]
    x_norm = (x - x.mean()) / max(x.std(), 1e-6)
    auc = roc_auc_score(y_fn, np.abs(x_norm) * np.sign(np.corrcoef(x, y_fn)[0,1] if np.abs(np.corrcoef(x,y_fn)[0,1])>1e-6 else 1))
    print(f"  {name:<15s} {auc:8.3f}")

print(f"\nDone.")
