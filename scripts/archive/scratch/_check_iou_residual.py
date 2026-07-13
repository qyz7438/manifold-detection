"""Check: conditional on IoU, what signals separate TP from FP from FN?"""
import sys, math
import torch, numpy as np
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
from torchvision.ops import box_iou, nms

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

def compute_energy(fft_f):
    ch = fft_f.shape[1] // 6
    a_lo = fft_f[:, 0*ch:1*ch].sum(dim=1)
    a_total = a_lo + fft_f[:, 1*ch:2*ch].sum(dim=1) + fft_f[:, 2*ch:3*ch].sum(dim=1) + 1e-8
    return 2*(a_lo/a_total)-1

def extract_perchan_fft(x):
    C = x.shape[1]; H,W = x.shape[-2], x.shape[-1]
    fft = torch.fft.rfft2(x, dim=(-2,-1), norm="ortho")
    amp = torch.abs(fft); pha = torch.angle(fft)
    freq_h = torch.fft.fftfreq(H, device=x.device); freq_w = torch.fft.rfftfreq(W, device=x.device)
    Y,X = torch.meshgrid(freq_h, freq_w, indexing='ij'); r = torch.sqrt(X**2+Y**2)
    R = r.max().clamp_min(1e-6); rn = r/R
    lo = (rn<=0.3).float(); md = ((rn>0.3)&(rn<=0.7)).float(); hi = (rn>0.7).float()
    a_lo = (amp*lo).flatten(2).sum(2); a_md = (amp*md).flatten(2).sum(2); a_hi = (amp*hi).flatten(2).sum(2)
    return torch.cat([a_lo, a_md, a_hi], dim=1)

all_boxes = []

for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]
    img_shape = (imgs_d[0].shape[-2], imgs_d[0].shape[-1])
    fpn_feats_dict.clear()
    with torch.no_grad(): preds = model(imgs_d)
    fpn = fpn_feats_dict.get("f")
    if fpn is None: continue

    for i_img in range(len(imgs_d)):
        pred_boxes = preds[i_img]["boxes"]
        pred_scores = preds[i_img]["scores"]
        gt_boxes = tgts[i_img]["boxes"].to(DEV)
        if len(pred_boxes) == 0: continue

        # Pool pred boxes
        pooled = box_pool(fpn, [pred_boxes], [img_shape])
        fft = extract_perchan_fft(pooled)
        energy = compute_energy(fft)

        # IoU to GTs
        ious = box_iou(pred_boxes, gt_boxes)  # (P, G)
        best_iou, best_gt = ious.max(dim=1)  # (P,)

        # Classify each prediction
        matched_gt = set()
        for gi in range(len(gt_boxes)):
            iou_to_g = ious[:, gi]
            if iou_to_g.max() >= 0.5:
                best_p = iou_to_g.argmax().item()
                matched_gt.add(best_p)

        for pi in range(len(pred_boxes)):
            iou = best_iou[pi].item()
            score = pred_scores[pi].item()
            en = energy[pi].item()
            box = pred_boxes[pi].cpu()
            w = (box[2]-box[0]).item(); h = (box[3]-box[1]).item()

            if iou >= 0.5:
                if pi in matched_gt:
                    label = "TP"
                else:
                    label = "FP-duplicate"  # high IoU but another box matched first
            else:
                if iou >= 0.3:
                    label = "FP-borderline"
                else:
                    label = "FP-noise"

            all_boxes.append({
                "label": label, "iou": iou, "score": score, "energy": en,
                "area": w*h, "cx": (box[0]+box[2])/2, "cy": (box[1]+box[3])/2,
            })

print(f"Total predictions: {len(all_boxes)}")
labels = [b["label"] for b in all_boxes]
for lbl in ["TP", "FP-duplicate", "FP-borderline", "FP-noise"]:
    print(f"  {lbl:<20s}: {labels.count(lbl)}")

# Key question: conditional on IoU, what separates TP from FP-duplicate?
# (These have similar IoU but one is "chosen", the other is suppressed)
print(f"\n=== TP vs FP-duplicate (both IoU>=0.5) ===")
tp_dup = [b for b in all_boxes if b["label"] in ("TP", "FP-duplicate")]

if len(tp_dup) > 2:
    from sklearn.linear_model import LogisticRegression
    X = np.column_stack([[b["iou"] for b in tp_dup],
                          [b["score"] for b in tp_dup],
                          [b["energy"] for b in tp_dup],
                          [b["area"] for b in tp_dup],
                          [b["cx"] for b in tp_dup],
                          [b["cy"] for b in tp_dup]])
    y = np.array([1 if b["label"] == "TP" else 0 for b in tp_dup])

    tp = [b for b in tp_dup if b["label"] == "TP"]
    fp = [b for b in tp_dup if b["label"] == "FP-duplicate"]

    for metric in ["iou", "score", "energy", "area", "cx", "cy"]:
        tv = np.array([b[metric] for b in tp])
        fv = np.array([b[metric] for b in fp])
        delta = np.mean(tv) - np.mean(fv)
        ps = np.sqrt((tv.var() + fv.var())/2)
        d = delta / max(ps, 1e-6)
        print(f"  {metric:<8s}: TP={np.mean(tv):.4f} FP={np.mean(fv):.4f} Δ={delta:+.4f} d={d:+.2f}")

    # Conditional: same IoU bin, what separates?
    print(f"\n  === Conditional on IoU (same bin) ===")
    for lo, hi in [(0.5, 0.6), (0.6, 0.75), (0.75, 1.0)]:
        bin_boxes = [b for b in tp_dup if lo <= b["iou"] < hi]
        bin_tp = [b for b in bin_boxes if b["label"] == "TP"]
        bin_fp = [b for b in bin_boxes if b["label"] == "FP-duplicate"]
        if len(bin_tp) < 2 or len(bin_fp) < 2: continue
        print(f"  IoU [{lo:.1f},{hi:.1f}): n_tp={len(bin_tp)} n_fp={len(bin_fp)}")
        for metric in ["score", "energy", "area", "cx", "cy"]:
            tv = np.array([b[metric] for b in bin_tp])
            fv = np.array([b[metric] for b in bin_fp])
            delta = np.mean(tv) - np.mean(fv)
            ps = np.sqrt((tv.var() + fv.var())/2)
            d = delta / max(ps, 1e-6)
            print(f"    {metric:<8s}: TP={np.mean(tv):.4f} FP={np.mean(fv):.4f} d={d:+.2f}")

# What about NMS: among FP-duplicates, which would NMS have removed?
print(f"\n=== NMS analysis on FP-duplicates ===")
for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]
    with torch.no_grad(): preds = model(imgs_d)
    for i_img in range(len(imgs_d)):
        boxes = preds[i_img]["boxes"]
        scores = preds[i_img]["scores"]
        if len(boxes) == 0: continue
        keep = nms(boxes, scores, 0.5)
        n_suppressed = len(boxes) - len(keep)
        if n_suppressed > 0:
            suppressed = [i for i in range(len(boxes)) if i not in keep]
            # Among suppressed, how many had IoU>=0.5 with any GT?
            gt_boxes = tgts[i_img]["boxes"].to(DEV)
            ious = box_iou(boxes, gt_boxes)
            best_iou = ious.max(dim=1).values
            suppressed_high_iou = [i for i in suppressed if best_iou[i] >= 0.5]
            if suppressed_high_iou:
                print(f"  Img{i_img}: {len(suppressed_high_iou)}/{n_suppressed} NMS-suppressed boxes had IoU>=0.5")
