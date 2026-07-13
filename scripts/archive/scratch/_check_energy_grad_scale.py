"""Check: does 14x14 ROI pool improve energy sensitivity to G=4 deltas?"""
import sys, math
import torch, numpy as np
from pathlib import Path
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
from torchvision.ops import MultiScaleRoIAlign

DEV = "cuda"; G = 4; SIGMA = 0.1
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
set_seed(42)

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
    return a_lo / (a_lo + a_md + a_hi + 1e-8)  # energy per channel, (N, C)

# Load model, get FPN features
model = build_detector({"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
    "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
    "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}}).to(DEV)
ckpt = torch.load(CKPT, map_location=DEV)
model.load_state_dict(ckpt["model"])
model.eval()

fpn_feats = {}
model.backbone.register_forward_hook(
    lambda m, i, o: fpn_feats.update({"f": {k: o[k] for k in o if k != "pool"}}))

tl, vl = build_penn_fudan_loaders({"data": {"root": "./data", "max_size": 320,
    "train_fraction": 0.8, "num_workers": 0}, "train": {"batch_size": 2}})

# Poolers
pool_7x7 = model.roi_heads.box_roi_pool  # native 7x7
pool_14x14 = MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'],
    output_size=14, sampling_ratio=2,
).to(DEV)

results = {"7x7": [], "14x14": []}
box_coder = model.roi_heads.box_coder

for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]
    tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
    fpn_feats.clear()

    with torch.no_grad():
        _ = model(imgs_d, tgts_t)
    fpn = fpn_feats.get("f")
    if fpn is None: continue
    image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]

    # Get proposals (use GT boxes with delta perturbations)
    for i_img in range(len(imgs_d)):
        gt_boxes = tgts_t[i_img]["boxes"]
        if len(gt_boxes) == 0: continue

        for gi in range(len(gt_boxes)):
            box = gt_boxes[gi]  # (4,) in [x1,y1,x2,y2]
            w = box[2] - box[0]; h = box[3] - box[1]
            cx = box[0] + 0.5*w; cy = box[1] + 0.5*h

            # G deltas: sigma=0.1 (same as GRPO)
            dx = SIGMA * torch.randn(G, device=DEV)
            dy = SIGMA * torch.randn(G, device=DEV)
            dw = SIGMA * torch.randn(G, device=DEV)
            dh = SIGMA * torch.randn(G, device=DEV)

            # Decode
            g_cx = dx * w + cx; g_cy = dy * h + cy
            g_w = torch.exp(dw) * w; g_h = torch.exp(dh) * h
            boxes_g = torch.stack([
                g_cx - 0.5*g_w, g_cy - 0.5*g_h,
                g_cx + 0.5*g_w, g_cy + 0.5*g_h,
            ], dim=1).clamp(min=0)  # (G, 4)

            # Pool at both resolutions
            box_list = [boxes_g]
            pooled_7 = pool_7x7(fpn, box_list, [image_shapes[i_img]])  # (G, C, 7, 7)
            pooled_14 = pool_14x14(fpn, box_list, [image_shapes[i_img]])  # (G, C, 14, 14)

            # Energy per channel
            en_7 = extract_perchan_fft(pooled_7)  # (G, C)
            en_14 = extract_perchan_fft(pooled_14)  # (G, C)

            # Mean energy across channels
            en_7_avg = en_7.mean(dim=1)  # (G,)
            en_14_avg = en_14.mean(dim=1)  # (G,)

            # Within-group variance
            var_7 = en_7_avg.var().item()
            var_14 = en_14_avg.var().item()

            # Also track per-channel variance
            var_7_perchan = en_7.var(dim=0).mean().item()  # mean across channels
            var_14_perchan = en_14.var(dim=0).mean().item()

            results["7x7"].append({"var": var_7, "var_perchan": var_7_perchan, "nch": en_7.shape[1]})
            results["14x14"].append({"var": var_14, "var_perchan": var_14_perchan, "nch": en_14.shape[1]})

print(f"Analyzed {len(results['7x7'])} proposal groups")
print()
print(f"{'Pool':>6s} {'var_mean':>10s} {'var_median':>10s} {'var>1e-6':>8s} {'var_perchan_mean':>14s}")
print("-" * 55)
for k in ["7x7", "14x14"]:
    vars_ = [r["var"] for r in results[k]]
    vars_mean = np.mean(vars_)
    vars_median = np.median(vars_)
    vars_nonzero = sum(1 for v in vars_ if v > 1e-6)
    vars_pc_mean = np.mean([r["var_perchan"] for r in results[k]])
    print(f"{k:>6s} {vars_mean:10.6f} {vars_median:10.6f} {vars_nonzero:>6d}/{len(vars_):<6d} {vars_pc_mean:14.8f}")

# Also compare: IoU variance in the same groups
iou_vars = []
for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]
    for i_img in range(len(imgs_d)):
        gt_boxes = tgts_t[i_img]["boxes"]
        if len(gt_boxes) == 0: continue
        for gi in range(len(gt_boxes)):
            box = gt_boxes[gi]
            w = box[2] - box[0]; h = box[3] - box[1]
            cx = box[0] + 0.5*w; cy = box[1] + 0.5*h

            dx = SIGMA * torch.randn(G, device=DEV)
            dy = SIGMA * torch.randn(G, device=DEV)
            dw = SIGMA * torch.randn(G, device=DEV)
            dh = SIGMA * torch.randn(G, device=DEV)

            g_cx = dx * w + cx; g_cy = dy * h + cy
            g_w = torch.exp(dw) * w; g_h = torch.exp(dh) * h
            boxes_g = torch.stack([
                g_cx - 0.5*g_w, g_cy - 0.5*g_h,
                g_cx + 0.5*g_w, g_cy + 0.5*g_h,
            ], dim=1).clamp(min=0)

            from torchvision.ops import box_iou
            ious = box_iou(boxes_g, box.unsqueeze(0)).squeeze()  # (G,)
            iou_vars.append(ious.var().item())

print(f"\nIoU within-group variance: mean={np.mean(iou_vars):.6f} median={np.median(iou_vars):.6f}")
