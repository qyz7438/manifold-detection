"""Compare energy sensitivity: delta perturbation vs direct pixel perturbation."""
import sys, math
import torch, numpy as np
from pathlib import Path
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
from torchvision.ops import box_iou

DEV = "cuda"; G = 4; SEED = 42
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
SIGMA_DELTA = 0.1
SIGMA_PIXEL = [2, 5, 10]  # pixel-space jitter in pixels

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
box_pool = model.roi_heads.box_roi_pool

results = {"delta": []}
for sp in SIGMA_PIXEL:
    results[f"pixel_{sp}px"] = []

for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]
    tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
    fpn_feats.clear()
    with torch.no_grad():
        _ = model(imgs_d, tgts_t)
    fpn = fpn_feats.get("f")
    if fpn is None: continue
    image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]

    for i_img in range(len(imgs_d)):
        gt_boxes = tgts_t[i_img]["boxes"]
        if len(gt_boxes) == 0: continue

        for gi in range(len(gt_boxes)):
            box = gt_boxes[gi]
            w = box[2] - box[0]; h = box[3] - box[1]
            cx = box[0] + 0.5*w; cy = box[1] + 0.5*h

            # ===== Method A: Delta perturbation (current) =====
            dx_d = SIGMA_DELTA * torch.randn(G, device=DEV)
            dy_d = SIGMA_DELTA * torch.randn(G, device=DEV)
            dw_d = SIGMA_DELTA * torch.randn(G, device=DEV)
            dh_d = SIGMA_DELTA * torch.randn(G, device=DEV)

            g_cx = dx_d * w + cx; g_cy = dy_d * h + cy
            g_w = torch.exp(dw_d) * w; g_h = torch.exp(dh_d) * h
            boxes_delta = torch.stack([
                g_cx - 0.5*g_w, g_cy - 0.5*g_h,
                g_cx + 0.5*g_w, g_cy + 0.5*g_h,
            ], dim=1).clamp(min=0)

            pooled_d = box_pool(fpn, [boxes_delta], [image_shapes[i_img]])
            en_d = extract_perchan_fft(pooled_d).mean(dim=1)  # (G,)
            iou_d = box_iou(boxes_delta, box.unsqueeze(0)).squeeze()  # (G,)

            results["delta"].append({
                "en_var": en_d.var().item(),
                "en_range": (en_d.max() - en_d.min()).item(),
                "iou_var": iou_d.var().item(),
            })

            # ===== Method B: Direct pixel perturbation =====
            for sp in SIGMA_PIXEL:
                px_noise = sp * torch.randn(G, 4, device=DEV)  # ±N pixels per corner
                boxes_px = box.unsqueeze(0).repeat(G, 1) + px_noise
                boxes_px = boxes_px.clamp(min=0)
                # Keep inside image
                boxes_px[:, 2] = boxes_px[:, 2].clamp(max=image_shapes[i_img][1])
                boxes_px[:, 3] = boxes_px[:, 3].clamp(max=image_shapes[i_img][0])
                # Ensure w,h > 1
                boxes_px[:, 2] = torch.max(boxes_px[:, 2], boxes_px[:, 0] + 1)
                boxes_px[:, 3] = torch.max(boxes_px[:, 3], boxes_px[:, 1] + 1)

                pooled_px = box_pool(fpn, [boxes_px], [image_shapes[i_img]])
                en_px = extract_perchan_fft(pooled_px).mean(dim=1)  # (G,)
                iou_px = box_iou(boxes_px, box.unsqueeze(0)).squeeze()  # (G,)

                results[f"pixel_{sp}px"].append({
                    "en_var": en_px.var().item(),
                    "en_range": (en_px.max() - en_px.min()).item(),
                    "iou_var": iou_px.var().item(),
                })

print(f"Analyzed {len(results['delta'])} proposal groups\n")
print(f"{'Method':<14s} {'en_var':>10s} {'en_range':>10s} {'iou_var':>10s} {'en/iou ratio':>12s}")
print("-" * 60)

for method in ["delta"] + [f"pixel_{sp}px" for sp in SIGMA_PIXEL]:
    vals = results[method]
    en_v = np.mean([v["en_var"] for v in vals])
    en_r = np.mean([v["en_range"] for v in vals])
    iou_v = np.mean([v["iou_var"] for v in vals])
    ratio = en_v / max(iou_v, 1e-8)
    print(f"{method:<14s} {en_v:10.6f} {en_r:10.6f} {iou_v:10.6f} {ratio:11.1f}")

# Key question: at what pixel sigma does energy variance match IoU variance?
print(f"\n=== Trade-off ===")
print(f"{'Method':<14s} {'en_range':>10s} {'iou_range':>10s}")
for method in ["delta"] + [f"pixel_{sp}px" for sp in SIGMA_PIXEL]:
    vals = results[method]
    en_r = np.mean([v["en_range"] for v in vals])
    iou_r = np.mean([np.sqrt(v["iou_var"]) * 2 for v in vals])  # approximate range from std
    print(f"{method:<14s} {en_r:10.6f} {iou_r:10.6f}")
