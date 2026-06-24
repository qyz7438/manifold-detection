"""Diagnostic: verify three RLVR approaches simultaneously.

Approach A: Proposal-level reward (RPN scoring)
Approach B: Discrete action space (dx/dy grid)
Approach C: ARS (parameter-space perturbation, batch-level reward)
"""
import sys, torch, torch.nn.functional as F, numpy as np, copy
from torchvision.ops import box_iou
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
set_seed(42)
DEV = "cuda"; PIX = 64; NUM_BATCHES = 5

cfg = {"model":{"name":"fasterrcnn_mobilenet_v3_large_320_fpn",
        "model_name":"fasterrcnn_mobilenet_v3_large_320_fpn",
        "pretrained":True,"num_classes":2,"min_size":320,"max_size":320}}
model = build_detector(cfg).to(DEV)
ckpt = torch.load("runs/round227_v1_baseline_20ep/checkpoint_best.pth", map_location=DEV)
model.load_state_dict(ckpt["model"]); model.eval()
loaders = build_penn_fudan_loaders({
    "data":{"root":"./data","max_size":320,"train_fraction":0.8,"num_workers":0},
    "train":{"batch_size":2}})

def pixel_fft_quality(patches):
    gray = patches.mean(dim=1); fft = torch.fft.fft2(gray.float()).abs()
    mf = fft.flatten(1); t = mf.sum(dim=1,keepdim=True).clamp_min(1e-6)
    hf = mf[:,mf.shape[1]//2:].sum(dim=1)/t.squeeze(1)
    mn = mf/t; ent = -(mn*torch.log(mn+1e-6)).sum(dim=1)
    me = torch.log(torch.tensor(float(mf.shape[1]),device=DEV))
    en = 1.0 - ent/me
    pv = torch.angle(torch.fft.fft2(gray.float())+1e-6).flatten(1).std(dim=1).clamp_max(1.0)
    return (0.3*hf + 0.4*en + 0.3*(1.0-pv)).clamp(0,1)

# ============================================================
# TEST A: Proposal-level reward differentiation
# ============================================================
print("="*60)
print("TEST A: Proposal-level reward")
print("  Can quality distinguish good vs bad proposals?")
print("="*60)

pc = {}; rc = {}
model.rpn.register_forward_hook(lambda m,i,o: pc.update({"p":o[0]}))
model.roi_heads.box_head.register_forward_pre_hook(lambda m,i: rc.update({"x":i[0]}))

prop_qualities = []
prop_ious = []
for images, targets in loaders[0]:
    pc.clear(); rc.clear()
    model([img.to(DEV) for img in images], [{k:v.to(DEV) for k,v in t.items()} for t in targets])
    proposals = pc.get("p"); roi_feats = rc.get("x")
    if proposals is None: continue

    npi = [p.shape[0] for p in proposals]
    img_idx = torch.cat([torch.full((n,),i,dtype=torch.long) for i,n in enumerate(npi)], dim=0)

    for img_i, props_img in enumerate(proposals):
        gt_boxes = targets[img_i]["boxes"]
        if len(gt_boxes) == 0: continue
        Np = props_img.shape[0]
        for pi in range(min(Np, 50)):  # limit per image
            box = props_img[pi]
            x1,y1,x2,y2 = box.round().long().clamp(min=0)
            x1,x2 = max(0,min(x1,images[img_i].shape[-1]-1)), max(x1+1,min(x2,images[img_i].shape[-1]))
            y1,y2 = max(0,min(y1,images[img_i].shape[-2]-1)), max(y1+1,min(y2,images[img_i].shape[-2]))
            crop = images[img_i][:,y1:y2,x1:x2]
            if crop.shape[-1]<4 or crop.shape[-2]<4: continue
            patch = F.interpolate(crop.unsqueeze(0).float(),(PIX,PIX),mode='bilinear',align_corners=False)
            q = pixel_fft_quality(patch.to(DEV)).item()
            prop_qualities.append(q)
            ious = box_iou(box.unsqueeze(0).to(DEV), gt_boxes.to(DEV))
            prop_ious.append(ious.max().item())

props_q = np.array(prop_qualities)
props_i = np.array(prop_ious)
good = props_i > 0.3; bad = props_i < 0.1
print(f"  Proposals: {len(props_q)}")
print(f"  Quality[IoU>0.3, n={good.sum()}]: {props_q[good].mean():.4f} ± {props_q[good].std():.4f}")
print(f"  Quality[IoU<0.1, n={bad.sum()}]: {props_q[bad].mean():.4f} ± {props_q[bad].std():.4f}")
print(f"  Gap: {props_q[good].mean()-props_q[bad].mean():.4f}")
print(f"  r(IoU): {np.corrcoef(props_q, props_i)[0,1]:.4f}")
# skip top1 — not applicable for proposal-level
print(f"  VERDICT A: {'PASS' if props_q[good].mean()-props_q[bad].mean() > 0.05 else 'FAIL'}")

# ============================================================
# TEST B: Discrete action space
# ============================================================
print("\n" + "="*60)
print("TEST B: Discrete action space")
print("  Can large (±2/5px) discrete shifts produce measurable quality diff?")
print("="*60)

# Find matched proposals
matched = []
for images, targets in loaders[0]:
    pc.clear(); rc.clear()
    model([img.to(DEV) for img in images], [{k:v.to(DEV) for k,v in t.items()} for t in targets])
    rf = rc.get("x"); pr = pc.get("p")
    if rf is None: continue
    N = rf.shape[0]; bf = model.roi_heads.box_head(rf)
    mu = model.roi_heads.box_predictor.bbox_pred(bf)[:,-4:]
    pc_ = torch.cat(pr, dim=0); N = min(N, pc_.shape[0], 200); mu = mu[:N]
    # Decode boxes
    w = pc_[:N,2]-pc_[:N,0]; h = pc_[:N,3]-pc_[:N,1]
    cx = pc_[:N,0]+0.5*w; cy = pc_[:N,1]+0.5*h
    px = mu[:,0]*w+cx-0.5*torch.exp(mu[:,2])*w
    py = mu[:,1]*h+cy-0.5*torch.exp(mu[:,3])*h
    px2 = mu[:,0]*w+cx+0.5*torch.exp(mu[:,2])*w
    py2 = mu[:,1]*h+cy+0.5*torch.exp(mu[:,3])*h
    decoded = torch.stack([px,py,px2,py2],dim=1).clamp(min=0)
    npi = [p.shape[0] for p in pr]
    img_map = []; [img_map.extend([img_i]*p.shape[0]) for img_i,p in enumerate(pr)]
    img_map = img_map[:N]
    for pi in range(N):
        img_i = img_map[pi]
        gt_boxes = targets[img_i]["boxes"]
        if len(gt_boxes)==0: continue
        ious = box_iou(decoded[pi:pi+1], gt_boxes.to(DEV))
        if ious.max()>0.3:
            matched.append((img_i, decoded[pi], gt_boxes[ious.argmax()]))
    if len(matched)>50: break

# Test discrete shifts
DISCRETE_SHIFTS = [0, 2, 5, -2, -5]
shift_results = {s: [] for s in DISCRETE_SHIFTS}
for img_i, base_box, gt_box in matched[:50]:
    img = images[img_i]
    # GT patch
    gx1,gy1,gx2,gy2 = gt_box.round().long().clamp(min=0)
    gx1,gx2 = max(0,min(gx1,img.shape[-1]-1)), max(gx1+1,min(gx2,img.shape[-1]))
    gy1,gy2 = max(0,min(gy1,img.shape[-2]-1)), max(gy1+1,min(gy2,img.shape[-2]))
    gt_crop = img[:,gy1:gy2,gx1:gx2]
    if gt_crop.shape[-1]<4 or gt_crop.shape[-2]<4: continue
    gt_patch = F.interpolate(gt_crop.unsqueeze(0).float(),(PIX,PIX),mode='bilinear',align_corners=False)
    gt_q = pixel_fft_quality(gt_patch.to(DEV)).item()
    # Proposal patches at different shifts
    for dx in DISCRETE_SHIFTS:
        shifted = base_box.clone()
        shifted = shifted.to(DEV)
        shifted[0] += dx; shifted[2] += dx; shifted = shifted.clamp(min=0)
        sx1,sy1,sx2,sy2 = shifted.round().long().clamp(min=0)
        sx1,sx2 = max(0,min(sx1,img.shape[-1]-1)), max(sx1+1,min(sx2,img.shape[-1]))
        sy1,sy2 = max(0,min(sy1,img.shape[-2]-1)), max(sy1+1,min(sy2,img.shape[-2]))
        sh_crop = img[:,sy1:sy2,sx1:sx2]
        if sh_crop.shape[-1]<4 or sh_crop.shape[-2]<4: continue
        sh_patch = F.interpolate(sh_crop.unsqueeze(0).float(),(PIX,PIX),mode='bilinear',align_corners=False)
        sh_q = pixel_fft_quality(sh_patch.to(DEV)).item()
        shift_results[dx].append(sh_q)

print(f"  Matched proposals tested: {len(matched[:50])}")
print(f"  GT patch quality: {gt_q:.4f}")
for dx in DISCRETE_SHIFTS:
    vals = np.array(shift_results[dx])
    print(f"  dx={dx:+3d}px: q={vals.mean():.4f} ± {vals.std():.4f}  (n={len(vals)})")
d0 = np.mean(shift_results[0]); d2 = np.mean(shift_results[2])
print(f"  Gap (0px vs 2px): {abs(d0-d2):.4f}")
print(f"  VERDICT B: {'PASS' if abs(d0-d2) > 0.01 else 'FAIL'}")

# ============================================================
# TEST C: ARS (parameter-space perturbation, batch-level)
# ============================================================
print("\n" + "="*60)
print("TEST C: ARS parameter-space perturbation")
print("  Can batch-level reward differentiate θ+ν from θ-ν?")
print("="*60)

def sum_losses(ld):
    if isinstance(ld, dict): return sum(ld.values()).item()
    if isinstance(ld, (list, tuple)):
        t = 0.0
        for d in ld:
            if isinstance(d, dict):
                for v in d.values(): t += v.sum().item()
        return t
    return sum(ld).item()

# Get bbox_pred weights
bbox_pred = model.roi_heads.box_predictor.bbox_pred
orig_weights = {n: p.clone() for n, p in bbox_pred.named_parameters()}
n_params = sum(p.numel() for p in orig_weights.values())
print(f"  bbox_pred params: {n_params}")

ars_results = []
for batch_i, (images, targets) in enumerate(loaders[0]):
    if batch_i >= NUM_BATCHES: break
    images_dev = [img.to(DEV) for img in images]
    targets_t = [{k:v.to(DEV) for k,v in t.items()} for t in targets]

    # Baseline
    model.eval()
    with torch.no_grad():
        ld = model(images_dev, targets_t)
        r_base = sum_losses(ld)

    # Generate perturbation
    perturbation = {}
    total_norm = 0
    for n, p in bbox_pred.named_parameters():
        pert = torch.randn_like(p) * 0.01  # σ=0.01 on weights
        perturbation[n] = pert
        total_norm += pert.norm().item()**2

    # Positive direction
    with torch.no_grad():
        for n, p in bbox_pred.named_parameters():
            p.add_(perturbation[n])
        ld_p = model(images_dev, targets_t)
        r_pos = sum_losses(ld_p)
        # Restore
        for n, p in bbox_pred.named_parameters():
            p.copy_(orig_weights[n])

    # Negative direction
    with torch.no_grad():
        for n, p in bbox_pred.named_parameters():
            p.sub_(perturbation[n])
        ld_n = model(images_dev, targets_t)
        r_neg = sum_losses(ld_n)
        # Restore
        for n, p in bbox_pred.named_parameters():
            p.copy_(orig_weights[n])

    diff_pos = r_pos - r_base
    diff_neg = r_neg - r_base
    gap = r_pos - r_neg
    ars_results.append({"pos": diff_pos, "neg": diff_neg, "gap": gap, "norm": total_norm**0.5})
    print(f"  batch {batch_i}: base={r_base:.1f} +ν={r_pos:.1f}({diff_pos:+.1f}) -ν={r_neg:.1f}({diff_neg:+.1f}) gap={gap:.1f}")

pos_d = np.array([r["pos"] for r in ars_results])
neg_d = np.array([r["neg"] for r in ars_results])
gaps = np.array([r["gap"] for r in ars_results])
print(f"\n  Mean +ν diff: {pos_d.mean():.1f} ± {pos_d.std():.1f}")
print(f"  Mean -ν diff: {neg_d.mean():.1f} ± {neg_d.std():.1f}")
print(f"  Mean gap: {gaps.mean():.1f} ± {gaps.std():.1f}")
print(f"  Median gap: {np.median(gaps):.1f}")
print(f"  Stable batches (|gap|<100): {[f'{g:.0f}' for g in gaps if abs(g)<100]}")
signal_mean = abs(gaps.mean()) / max(gaps.std(), 1.0)
# Median-based: use IQR as robust std estimate
iqr = np.percentile(gaps, 75) - np.percentile(gaps, 25)
median_gap = abs(np.median(gaps))
robust_std = iqr / 1.35  # IQR→std for normal distribution
signal_median = median_gap / max(robust_std, 1.0)
print(f"  Signal/Noise (mean): {signal_mean:.2f}")
print(f"  Signal/Noise (median+IQR): {signal_median:.2f}")
print(f"  VERDICT C: {'PASS (>1.5)' if signal_median > 1.5 else 'FAIL (<1.5)'}")
