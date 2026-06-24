"""Diagnostic: CPS P_max sensitivity to box displacement.

Key question: does P_max change measurably (>0.05) for 1-3px box shifts?
If yes → CPS+GRPO is viable. If no → abandon.
"""
import sys, torch, torch.nn.functional as F, numpy as np
from torchvision.ops import box_iou
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
set_seed(42)
DEV = "cuda"; PIX = 64

def cps_peak(patches_a, patches_b):
    """Cross-power spectrum peak between two patch batches."""
    ga = patches_a.float().mean(dim=1)  # (N,64,64)
    gb = patches_b.float().mean(dim=1)
    Fa = torch.fft.fft2(ga); Fb = torch.fft.fft2(gb)
    denom = Fa.abs() * Fb.abs() + 1e-6
    R = (Fa * Fb.conj()) / denom
    C = torch.fft.ifft2(R).real
    return C.flatten(1).max(dim=1).values  # (N,) peak height

cfg = {"model":{"name":"fasterrcnn_mobilenet_v3_large_320_fpn",
        "model_name":"fasterrcnn_mobilenet_v3_large_320_fpn",
        "pretrained":True,"num_classes":2,"min_size":320,"max_size":320}}
model = build_detector(cfg).to(DEV)
ckpt = torch.load("runs/round227_v1_baseline_20ep/checkpoint_best.pth", map_location=DEV)
model.load_state_dict(ckpt["model"]); model.eval()
loaders = build_penn_fudan_loaders({
    "data":{"root":"./data","max_size":320,"train_fraction":0.8,"num_workers":0},
    "train":{"batch_size":2}})

pc = {}; rc = {}
model.rpn.register_forward_hook(lambda m,i,o: pc.update({"p":o[0]}))
model.roi_heads.box_head.register_forward_pre_hook(lambda m,i: rc.update({"x":i[0]}))

results = {}  # shift_px → list of P_max values

for images, targets in loaders[0]:
    pc.clear(); rc.clear()
    model([img.to(DEV) for img in images], [{k:v.to(DEV) for k,v in t.items()} for t in targets])
    rf = rc.get("x"); pr = pc.get("p")
    if rf is None or rf.shape[0]==0: continue

    N = rf.shape[0]; bf = model.roi_heads.box_head(rf)
    mu = model.roi_heads.box_predictor.bbox_pred(bf)[:,-4:]

    proposals_cat = torch.cat(pr, dim=0)
    N = min(N, proposals_cat.shape[0])
    mu = mu[:N]; proposals_cat = proposals_cat[:N]

    # Decode boxes from mu (no noise, use mean prediction)
    widths = proposals_cat[:,2]-proposals_cat[:,0]
    heights = proposals_cat[:,3]-proposals_cat[:,1]
    cx = proposals_cat[:,0]+0.5*widths; cy = proposals_cat[:,1]+0.5*heights
    px_ = mu[:,0]*widths + cx - 0.5*torch.exp(mu[:,2])*widths
    py_ = mu[:,1]*heights + cy - 0.5*torch.exp(mu[:,3])*heights
    px2 = mu[:,0]*widths + cx + 0.5*torch.exp(mu[:,2])*widths
    py2 = mu[:,1]*heights + cy + 0.5*torch.exp(mu[:,3])*heights
    decoded = torch.stack([px_, py_, px2, py2], dim=1).clamp(min=0)

    # Find matched proposals (IoU > 0.3 with any GT)
    npi = [p.shape[0] for p in pr]
    img_map = []
    for img_i, p in enumerate(pr):
        for _ in range(p.shape[0]):
            img_map.append(img_i)
    img_map = img_map[:N]

    for pi in range(N):
        img_i = img_map[pi]
        gt_boxes = targets[img_i]["boxes"].to(DEV)
        if len(gt_boxes) == 0: continue
        ious = box_iou(decoded[pi:pi+1], gt_boxes)
        best_iou, best_idx = ious.max(dim=1)
        if best_iou[0] < 0.3: continue

        gt_box = gt_boxes[best_idx[0]]
        base_box = decoded[pi]

        # Get GT patch
        img = images[img_i]
        gx1,gy1,gx2,gy2 = gt_box.round().long().clamp(min=0)
        gx1,gx2 = max(0,min(gx1,img.shape[-1]-1)), max(gx1+1,min(gx2,img.shape[-1]))
        gy1,gy2 = max(0,min(gy1,img.shape[-2]-1)), max(gy1+1,min(gy2,img.shape[-2]))
        gt_crop = img[:,gy1:gy2,gx1:gx2]
        if gt_crop.shape[-1]<4 or gt_crop.shape[-2]<4: continue
        gt_patch = F.interpolate(gt_crop.unsqueeze(0).float(),(PIX,PIX),mode='bilinear',align_corners=False).squeeze(0)

        # Test shifts
        for shift in [0, 1, 2, 3, 5]:
            dx, dy = shift, 0  # pure horizontal shift
            shifted = base_box.clone()
            shifted[0] += dx; shifted[2] += dx; shifted[1] += dy; shifted[3] += dy
            shifted = shifted.clamp(min=0)

            sx1,sy1,sx2,sy2 = shifted.round().long().clamp(min=0)
            sx1,sx2 = max(0,min(sx1,img.shape[-1]-1)), max(sx1+1,min(sx2,img.shape[-1]))
            sy1,sy2 = max(0,min(sy1,img.shape[-2]-1)), max(sy1+1,min(sy2,img.shape[-2]))
            sh_crop = img[:,sy1:sy2,sx1:sx2]
            if sh_crop.shape[-1]<4 or sh_crop.shape[-2]<4: continue
            sh_patch = F.interpolate(sh_crop.unsqueeze(0).float(),(PIX,PIX),mode='bilinear',align_corners=False).squeeze(0)

            p_max = cps_peak(gt_patch.unsqueeze(0), sh_patch.unsqueeze(0))
            results.setdefault(shift, []).append(p_max.item())

    if sum(len(v) for v in results.values()) > 2000: break

print(f"\n=== CPS P_max vs Box Shift ({len(results.get(0,[]))} samples) ===")
print(f"  {'Shift':>6s}  {'P_max':>8s}  {'Std':>8s}  {'vs 0px':>8s}  {'grad':>8s}")
p0 = np.mean(results[0])
for s in sorted(results.keys()):
    vals = np.array(results[s])
    delta = np.mean(vals) - p0
    grad = delta / max(s, 1)
    print(f"  {s:4d}px  {np.mean(vals):8.4f}  {np.std(vals):8.4f}  {delta:+8.4f}  {grad:+8.4f}")
print(f"\n  Resolution: P_max drops {(p0 - np.mean(results.get(1,[0]))):.4f} per 1px")
print(f"  Verdict: {'VIABLE (>0.05)' if abs(p0 - np.mean(results.get(1,[0]))) > 0.05 else 'NOT viable (<0.05)'}")
