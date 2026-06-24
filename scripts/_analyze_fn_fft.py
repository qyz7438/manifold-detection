"""Train det_only_unf, then check FFT metrics on FN vs TP boxes."""
import sys, json, math, copy
from pathlib import Path
import torch, torch.nn as nn
import numpy as np
from torchvision.ops import box_iou, nms
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed

DEV = "cuda"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
EPOCHS = 6
SEED = 42
HEAD_LR = 0.001
BODY_LR = 0.0001
SCORE_THRESH = 0.05
IOU_THRESH = 0.5

set_seed(SEED)

# ---------- model ----------
def unfreeze_rlvr(model):
    for p in model.backbone.body.parameters(): p.requires_grad = False
    if hasattr(model.backbone, 'fpn'):
        for p in model.backbone.fpn.parameters(): p.requires_grad = True
    for p in model.rpn.parameters(): p.requires_grad = True
    for p in model.roi_heads.box_head.parameters(): p.requires_grad = True
    for p in model.roi_heads.box_predictor.parameters(): p.requires_grad = True
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            for p in m.parameters(): p.requires_grad = False

def build_opt(model):
    body_params = []; head_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if 'box_head' in n or 'box_predictor' in n: head_params.append(p)
        else: body_params.append(p)
    return torch.optim.SGD([
        {'params': body_params, 'lr': BODY_LR},
        {'params': head_params, 'lr': HEAD_LR},
    ], lr=HEAD_LR, momentum=0.9, weight_decay=0.0005)

def bm():
    return build_detector({"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}})

# ---------- train ----------
print("=== Training det_only_unf 6 epoch ===")
model = bm().to(DEV)
ckpt = torch.load(CKPT, map_location=DEV)
model.load_state_dict(ckpt["model"])
unfreeze_rlvr(model)
opt = build_opt(model)
tl, vl = build_penn_fudan_loaders({"data": {"root": "./data", "max_size": 320, "train_fraction": 0.8, "num_workers": 0}, "train": {"batch_size": 4}})
best_ap75 = -1.0

for ep in range(1, EPOCHS + 1):
    model.train()
    for imgs, tgts in tqdm(tl, desc=f"e{ep}"):
        imgs_d = [i.to(DEV) for i in imgs]
        tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
        ld = model(imgs_d, tgts_t)
        det = sum(ld.values()) if isinstance(ld, dict) else sum(sum(d.values()) for d in ld if isinstance(d, dict))
        opt.zero_grad(set_to_none=True); det.backward(); opt.step()

    model.eval()
    ps, ts = [], []
    for img, tgt in vl:
        out = model([i.to(DEV) for i in img])
        ps.extend([{k: v.cpu() for k, v in o.items()} for o in out])
        ts.extend([{k: v.cpu() for k, v in t.items()} for t in tgt])
    em = evaluate_detection_predictions(ps, ts, iou_threshold=0.5, score_threshold=0.05)
    print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f}")
    if em["ap75"] > best_ap75:
        best_ap75 = em["ap75"]
        torch.save({"model": model.state_dict(), "epoch": ep, "ap75": best_ap75}, "runs/_fn_checkpoint.pth")

print(f"Best AP75: {best_ap75:.4f}")

# ---------- FFT feature extraction ----------
def extract_perchan_fft(x):
    """x: (N, C, H, W) pooled ROI features"""
    C = x.shape[1]; H, W = x.shape[-2], x.shape[-1]
    fft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    amp = torch.abs(fft); pha = torch.angle(fft)
    freq_h = torch.fft.fftfreq(H, device=x.device)
    freq_w = torch.fft.rfftfreq(W, device=x.device)
    Y, X = torch.meshgrid(freq_h, freq_w, indexing='ij')
    r = torch.sqrt(X**2 + Y**2); R = r.max().clamp_min(1e-6); rn = r / R
    lo = (rn <= 0.3).float(); md = ((rn > 0.3) & (rn <= 0.7)).float(); hi = (rn > 0.7).float()
    a_lo = (amp * lo).flatten(2).sum(2); a_md = (amp * md).flatten(2).sum(2); a_hi = (amp * hi).flatten(2).sum(2)
    p_lo = (pha * lo).flatten(2).sum(2); p_md = (pha * md).flatten(2).sum(2); p_hi = (pha * hi).flatten(2).sum(2)
    return torch.cat([a_lo, a_md, a_hi, p_lo, p_md, p_hi], dim=1)

def compute_energy(fft_f):
    """Low-freq energy concentration, higher = more structure"""
    ch = fft_f.shape[1] // 6
    a_lo = fft_f[:, 0*ch:1*ch].sum(dim=1)
    a_md = fft_f[:, 1*ch:2*ch].sum(dim=1)
    a_hi = fft_f[:, 2*ch:3*ch].sum(dim=1)
    low_ratio = a_lo / (a_lo + a_md + a_hi + 1e-8)
    return 2 * low_ratio - 1

def compute_sim_to_ref(fft_f, ref):
    """Cosine similarity to a reference FFT profile"""
    fft_n = torch.nn.functional.normalize(fft_f, dim=-1)
    ref_n = torch.nn.functional.normalize(ref.unsqueeze(0), dim=-1)
    return (fft_n * ref_n).sum(dim=-1)

def compute_phase_dist(fft_f, ref):
    """Phase distance to reference"""
    ch = fft_f.shape[1] // 6
    p_me = fft_f[:, 3*ch:]  # phase bands
    ref_p = ref[3*ch:]
    return ((p_me - ref_p.unsqueeze(0))**2).sum(dim=-1)

# ---------- Analyze FN vs TP ----------
print("\n=== FN vs TP FFT Analysis ===")
model.load_state_dict(torch.load("runs/_fn_checkpoint.pth", map_location=DEV)["model"])
model.eval()

box_pool = model.roi_heads.box_roi_pool

# Hook to capture FPN features
fpn_feats = {}
model.backbone.register_forward_hook(
    lambda m, i, o: fpn_feats.update({"f": {k: o[k] for k in o if k != "pool"}}))

# Hook to capture sampled proposals
sampled_props = {}
model.roi_heads.box_roi_pool.register_forward_pre_hook(
    lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))

fn_metrics = []  # per FN box: {energy, sim, phase, area, edge_dist, ...}
tp_metrics = []  # per TP box
all_pred_metrics = []  # all predictions

# Collect reference: mean FFT of all GT boxes
all_gt_fft = []

# First pass: collect GT box FFT
for img, tgt in vl:
    imgs_d = [i.to(DEV) for i in img]
    tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgt]
    image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]

    for i_img in range(len(imgs_d)):
        gt_boxes_img = tgts_t[i_img]["boxes"]
        if len(gt_boxes_img) == 0: continue
        fpn_feats.clear()
        _ = model(imgs_d, tgts_t)
        fpn = fpn_feats.get("f")
        if fpn is None: continue
        # Pool GT boxes
        gt_list = [gt_boxes_img]
        pooled_gt = box_pool(fpn, gt_list, [image_shapes[i_img]])
        fft_gt = extract_perchan_fft(pooled_gt)
        all_gt_fft.append(fft_gt)

ref_fft = torch.cat(all_gt_fft, dim=0).mean(dim=0)

# Second pass: analyze predictions
for img, tgt in vl:
    imgs_d = [i.to(DEV) for i in img]
    tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgt]
    image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]
    fpn_feats.clear(); sampled_props.clear()

    with torch.no_grad():
        preds = model(imgs_d)
    fpn = fpn_feats.get("f")
    if fpn is None: continue

    for i_img in range(len(imgs_d)):
        gt_boxes = tgts_t[i_img]["boxes"].cpu()
        pred_dict = preds[i_img]
        pred_boxes = pred_dict["boxes"].cpu()
        pred_scores = pred_dict["scores"].cpu()

        keep = pred_scores >= SCORE_THRESH
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]

        if len(pred_boxes) == 0:
            # All GT are FN
            for gi in range(len(gt_boxes)):
                box = gt_boxes[gi]
                w = (box[2]-box[0]).item(); h = (box[3]-box[1]).item()
                fn_metrics.append({
                    "energy": float("nan"), "sim": float("nan"), "phase": float("nan"),
                    "area": w*h, "width": w, "height": h,
                    "edge_dist": min(box[0].item(), 320-box[2].item(), box[1].item(), 320-box[3].item()),
                    "closest_iou": 0.0, "closest_score": 0.0,
                })
            continue

        # Pool predictions through roi_pool to get FFT
        pred_list = [pred_boxes.to(DEV)]
        pooled_pred = box_pool(fpn, pred_list, [image_shapes[i_img]])
        fft_pred = extract_perchan_fft(pooled_pred)  # (N_pred, 6*C)

        energy = compute_energy(fft_pred).cpu()
        sim = compute_sim_to_ref(fft_pred, ref_fft).cpu()
        phase = compute_phase_dist(fft_pred, ref_fft).cpu()

        # Match pred to GT
        ious = box_iou(pred_boxes, gt_boxes)
        matched_gt = set()
        matched_pred = set()

        if ious.numel() > 0:
            for gi in range(len(gt_boxes)):
                best_iou, best_pi = ious[:, gi].max(0)
                if best_iou >= IOU_THRESH:
                    matched_gt.add(gi)
                    matched_pred.add(best_pi.item())

        # TP: predictions matched to GT
        for pi in matched_pred:
            gi = ious[pi].argmax().item()
            box = gt_boxes[gi]
            w = (box[2]-box[0]).item(); h = (box[3]-box[1]).item()
            tp_metrics.append({
                "energy": energy[pi].item(), "sim": sim[pi].item(),
                "phase": phase[pi].item(),
                "iou": ious[pi, gi].item(), "score": pred_scores[pi].item(),
                "area": w*h, "width": w, "height": h,
                "edge_dist": min(box[0].item(), 320-box[2].item(), box[1].item(), 320-box[3].item()),
            })

        # FN: GT boxes not matched
        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                box = gt_boxes[gi]
                # Find closest prediction
                closest_iou = 0.0; closest_score = 0.0; closest_pi = -1
                if ious.numel() > 0:
                    ci = ious[:, gi].argmax().item()
                    closest_iou = ious[ci, gi].item()
                    closest_score = pred_scores[ci].item()
                    closest_pi = ci

                w = (box[2]-box[0]).item(); h = (box[3]-box[1]).item()
                fn_metrics.append({
                    "energy": energy[closest_pi].item() if closest_pi >= 0 else float("nan"),
                    "sim": sim[closest_pi].item() if closest_pi >= 0 else float("nan"),
                    "phase": phase[closest_pi].item() if closest_pi >= 0 else float("nan"),
                    "area": w*h, "width": w, "height": h,
                    "edge_dist": min(box[0].item(), 320-box[2].item(), box[1].item(), 320-box[3].item()),
                    "closest_iou": closest_iou, "closest_score": closest_score,
                })

print(f"TP boxes: {len(tp_metrics)}")
print(f"FN boxes: {len(fn_metrics)}")

# ---------- Comparison ----------
print(f"\n{'Metric':<18s} {'TP_mean':>10s} {'TP_std':>10s} {'FN_mean':>10s} {'FN_std':>10s} {'Δ':>10s} {'Cohen d':>10s}")
print("-" * 80)

for metric_name in ["energy", "sim", "phase"]:
    tp_vals = [m[metric_name] for m in tp_metrics if not np.isnan(m[metric_name])]
    fn_vals = [m[metric_name] for m in fn_metrics if not np.isnan(m[metric_name])]
    if not tp_vals or not fn_vals: continue
    tp_m = np.mean(tp_vals); tp_s = np.std(tp_vals)
    fn_m = np.mean(fn_vals); fn_s = np.std(fn_vals)
    delta = tp_m - fn_m
    pooled_std = np.sqrt((tp_s**2 + fn_s**2) / 2)
    d = delta / pooled_std if pooled_std > 0 else 0
    print(f"{metric_name:<18s} {tp_m:10.4f} {tp_s:10.4f} {fn_m:10.4f} {fn_s:10.4f} {delta:+10.4f} {d:10.3f}")

# Also compare by size
print(f"\n--- By box size ---")
for label, fn_subset in [("Small (<500px)", [m for m in fn_metrics if m["area"] < 500]),
                          ("Medium (500-2000)", [m for m in fn_metrics if 500 <= m["area"] < 2000]),
                          ("Large (>2000)", [m for m in fn_metrics if m["area"] >= 2000])]:
    if not fn_subset: continue
    for metric_name in ["energy", "sim", "phase"]:
        fn_vals = [m[metric_name] for m in fn_subset if not np.isnan(m[metric_name])]
        tp_vals = [m[metric_name] for m in tp_metrics if not np.isnan(m[metric_name])]
        if not fn_vals: continue
        print(f"  {label} n={len(fn_subset)}: {metric_name} TP={np.mean(tp_vals):.4f} FN={np.mean(fn_vals):.4f} Δ={np.mean(tp_vals)-np.mean(fn_vals):+.4f}")

# Per-box detail for all FNs
print(f"\n=== Individual FN cases (with FFT metrics vs TP reference) ===")
tp_e_mean = np.mean([m["energy"] for m in tp_metrics])
tp_s_mean = np.mean([m["sim"] for m in tp_metrics])
tp_p_mean = np.mean([m["phase"] for m in tp_metrics])
print(f"TP reference: energy={tp_e_mean:.4f} sim={tp_s_mean:.4f} phase={tp_p_mean:.4f}")
print(f"{'Area':>7s} {'W':>5s} {'H':>5s} {'Edge':>5s} {'ClsIoU':>7s} {'ClsSc':>7s} {'energy':>8s} {'ΔE/σ':>7s} {'sim':>8s} {'ΔS/σ':>7s} {'phase':>10s}")
print("-" * 95)
tp_e_std = np.std([m["energy"] for m in tp_metrics])
tp_s_std = np.std([m["sim"] for m in tp_metrics])
tp_p_std = np.std([m["phase"] for m in tp_metrics])

for m in sorted(fn_metrics, key=lambda x: -x["area"]):
    e_dev = (m["energy"] - tp_e_mean) / max(tp_e_std, 1e-6) if not np.isnan(m["energy"]) else float("nan")
    s_dev = (m["sim"] - tp_s_mean) / max(tp_s_std, 1e-6) if not np.isnan(m["sim"]) else float("nan")
    p_dev = "N/A" if np.isnan(m["phase"]) else f"{m['phase']:10.2f}"
    print(f"{m['area']:7.0f} {m['width']:5.1f} {m['height']:5.1f} {m['edge_dist']:5.1f} {m['closest_iou']:7.3f} {m['closest_score']:7.4f} {m['energy']:8.4f} {e_dev:+7.2f} {m['sim']:8.4f} {s_dev:+7.2f} {p_dev:>10s}")

# Check: does energy correlate with IoU within predictions?
print(f"\n=== Energy-IoU correlation (all predictions) ===")
all_energies = [m["energy"] for m in tp_metrics]
all_ious = [m["iou"] for m in tp_metrics]
if len(all_energies) > 2:
    corr = np.corrcoef(all_energies, all_ious)[0, 1]
    print(f"  Pearson r(energy, IoU) = {corr:.4f}")

# Save
Path("runs/fn_fft_analysis.json").write_text(json.dumps({
    "tp_count": len(tp_metrics), "fn_count": len(fn_metrics),
    "fn_metrics": fn_metrics, "tp_metrics_summary": {
        "energy_mean": tp_e_mean, "energy_std": tp_e_std,
        "sim_mean": tp_s_mean, "sim_std": tp_s_std,
        "phase_mean": tp_p_mean, "phase_std": tp_p_std,
    }
}, indent=2))
print("\nSaved to runs/fn_fft_analysis.json")
