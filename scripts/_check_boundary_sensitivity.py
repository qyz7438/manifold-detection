"""Check if boundary feature sampling is more sensitive to box shifts than ROI pool."""
import sys, math
import torch, numpy as np
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
from torchvision.ops import box_iou
import torch.nn.functional as F

DEV = "cuda"; SEED = 42; SIGMA = 0.1; G = 4
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
set_seed(SEED)

model = build_detector({"model":{"name":"fasterrcnn_mobilenet_v3_large_320_fpn","model_name":"fasterrcnn_mobilenet_v3_large_320_fpn","pretrained":True,"num_classes":2,"min_size":320,"max_size":320}}).to(DEV)
ckpt = torch.load(CKPT, map_location=DEV); model.load_state_dict(ckpt["model"]); model.eval()

fpn_feats = {}
model.backbone.register_forward_hook(lambda m,i,o: fpn_feats.update({"f":{k:o[k] for k in o if k!="pool"}}))
sampled_props = {}
model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m,args: sampled_props.update({"p":[a.clone() for a in args[1]]}))
box_head_in = {}
model.roi_heads.box_head.register_forward_pre_hook(lambda m,args: box_head_in.update({"x":args[0]}))
box_pool = model.roi_heads.box_roi_pool

tl, vl = build_penn_fudan_loaders({"data":{"root":"./data","max_size":320,"train_fraction":0.8,"num_workers":0},"train":{"batch_size":2}})

def boundary_features(fpn_features, boxes, img_shape, n_pts=8):
    """Sample FPN features along box boundaries. Returns (M, C*4*n_pts) vector per box.
    For multi-scale FPN, routes each box to appropriate level by box area."""
    H, W = img_shape
    M = boxes.shape[0]
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]

    # Normalize coordinates to [-1, 1] for grid_sample
    def norm(x, y): return torch.stack([2*x/W-1, 2*y/H-1], dim=-1)

    # Sample points along 4 edges
    all_pts = []
    for i in range(n_pts):
        t = i / (n_pts-1)
        # Top edge: (x1->x2, y1)
        px = x1 + t*(x2-x1); py = y1 + 0*x1
        all_pts.append(norm(px, py))
        # Bottom edge
        px = x1 + t*(x2-x1); py = y2 + 0*x1
        all_pts.append(norm(px, py))
        # Left edge
        px = x1 + 0*x1; py = y1 + t*(y2-y1)
        all_pts.append(norm(px, py))
        # Right edge
        px = x2 + 0*x1; py = y1 + t*(y2-y1)
        all_pts.append(norm(px, py))

    coords = torch.stack(all_pts, dim=1)  # (M, 4*n_pts, 2)

    # Use P3 level (stride=8, highest resolution) - FPN keys are strings '0','1','2','3'
    feat_p3 = fpn_features['0']  # (B, C, H_f, W_f)
    feat = feat_p3[0:1].expand(M, -1, -1, -1)  # (M, C, H_f, W_f)

    # grid_sample needs (N, H_out, W_out, 2) grid
    grid = coords.view(M, -1, 2).unsqueeze(1)  # (M, 1, 4*n_pts, 2)

    # Interpolate at boundary points
    sampled = F.grid_sample(feat, grid, align_corners=True)  # (M, C, 1, 4*n_pts)
    return sampled.squeeze(2).reshape(M, -1)  # (M, C*4*n_pts)

def corner_features(fpn_features, boxes, img_shape, patch=5):
    """Extract 5x5 patches around 4 corners. Returns (M, C*4*25) vector."""
    H, W = img_shape
    M = boxes.shape[0]
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]

    features = []
    for name, cx, cy in [("tl",x1,y1), ("tr",x2,y1), ("bl",x1,y2), ("br",x2,y2)]:
        # Generate patch grid around corner
        offsets = torch.linspace(-(patch//2), patch//2, patch, device=DEV)
        gy, gx = torch.meshgrid(offsets, offsets, indexing='ij')
        gx = gx.flatten(); gy = gy.flatten()  # (25,)

        px = (cx.unsqueeze(1) + gx.unsqueeze(0)) / W * 2 - 1  # (M, 25)
        py = (cy.unsqueeze(1) + gy.unsqueeze(0)) / H * 2 - 1  # (M, 25)
        grid = torch.stack([px, py], dim=-1).view(M, 1, -1, 2)  # (M, 1, 25, 2)

        # Use P3 FPN level
        feat = fpn_features['0'][0:1].expand(M, -1, -1, -1)
        sampled = F.grid_sample(feat, grid, align_corners=True)  # (M, C, 1, 25)
        features.append(sampled.reshape(M, -1))  # (M, C*25)

    return torch.cat(features, dim=1)  # (M, C*4*25)


def compute_energy_fft(x):
    """Compute FFT energy from pooled ROI features."""
    M, C, Hf, Wf = x.shape
    fft = torch.fft.rfft2(x, dim=(-2,-1), norm="ortho"); amp = torch.abs(fft)
    fh = torch.fft.fftfreq(Hf,device=DEV); fw = torch.fft.rfftfreq(Wf,device=DEV)
    Y,X = torch.meshgrid(fh,fw,indexing='ij'); r=torch.sqrt(X**2+Y**2); R=r.max().clamp_min(1e-6); rn=r/R
    lo=(rn<=0.3).float()
    al=(amp*lo).flatten(2).sum(2); at=al+(amp*((rn>0.3)&(rn<=0.7)).float()).flatten(2).sum(2)+(amp*(rn>0.7).float()).flatten(2).sum(2)+1e-8
    return (al/at).mean(dim=1)  # (M,)

# Test sensitivity: delta perturbation → energy variance
results = {"roi_pool": [], "boundary": [], "corner": []}

for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]; img_shape = (imgs_d[0].shape[-2], imgs_d[0].shape[-1])
    fpn_feats.clear(); sampled_props.clear(); box_head_in.clear()
    with torch.no_grad(): _ = model(imgs_d, [{k:v.to(DEV) for k,v in t.items()} for t in tgts])
    fpn = fpn_feats.get("f"); sp_raw = sampled_props.get("p")
    if fpn is None or sp_raw is None: continue

    for tgt in tgts:
        for box in tgt["boxes"]:
            w=box[2]-box[0]; h=box[3]-box[1]; cx=box[0]+0.5*w; cy=box[1]+0.5*h
            # G random perturbations
            dx = SIGMA*torch.randn(G,device=DEV); dy = SIGMA*torch.randn(G,device=DEV)
            dw = SIGMA*torch.randn(G,device=DEV); dh = SIGMA*torch.randn(G,device=DEV)

            g_cx = dx*w + cx; g_cy = dy*h + cy
            g_w = torch.exp(dw)*w; g_h = torch.exp(dh)*h
            boxes_g = torch.stack([g_cx-0.5*g_w, g_cy-0.5*g_h, g_cx+0.5*g_w, g_cy+0.5*g_h], dim=1).clamp(min=0)

            # 1. ROI pool energy variance
            pooled = box_pool(fpn, [boxes_g.to(DEV)], [img_shape])
            en_roi = compute_energy_fft(pooled)
            results["roi_pool"].append({"var": en_roi.var().item()})

            # 2. Boundary feature energy (use 1D FFT on boundary vectors)
            bf = boundary_features(fpn, boxes_g.to(DEV), img_shape, n_pts=4)
            # Reshape to (G, C, 16) then 1D FFT along last dim
            bf_r = bf.view(G, 256, 16)  # (G, C, 4*4)
            fft_b = torch.fft.rfft(bf_r, dim=-1)  # 1D FFT along boundary
            amp_b = torch.abs(fft_b)
            en_b = amp_b[:,:,:2].sum(dim=(1,2)) / amp_b.sum(dim=(1,2)).clamp_min(1e-8)
            results["boundary"].append({"var": en_b.var().item()})

            # 3. Corner feature energy
            cf = corner_features(fpn, boxes_g.to(DEV), img_shape, patch=3)
            cf_r = cf.view(G, 256, 4*9)
            fft_c = torch.fft.rfft(cf_r, dim=-1)
            amp_c = torch.abs(fft_c)
            en_c = amp_c[:,:,:2].sum(dim=(1,2)) / amp_c.sum(dim=(1,2)).clamp_min(1e-8)
            results["corner"].append({"var": en_c.var().item()})

print(f"Analyzed {len(results['roi_pool'])} GT boxes\n")
print(f"{'Method':<15s} {'energy_var':>12s} {'energy_mean':>12s} {'variance_ratio':>12s}")
print("-" * 55)
for method in ["roi_pool", "boundary", "corner"]:
    vars_ = np.array([r["var"] for r in results[method]])
    print(f"{method:<15s} {vars_.mean():12.6f} {vars_.mean():12.4f} {vars_.mean()/results['roi_pool'][0]['var'] if results['roi_pool'] else 0:12.1f}")

# Also check: IoU variance for reference
iou_vars = []
for imgs, tgts in vl:
    for tgt in tgts:
        for box in tgt["boxes"]:
            w=box[2]-box[0]; h=box[3]-box[1]; cx=box[0]+0.5*w; cy=box[1]+0.5*h
            dx=SIGMA*torch.randn(G,device=DEV); dy=SIGMA*torch.randn(G); dw=SIGMA*torch.randn(G); dh=SIGMA*torch.randn(G)
            g_cx=dx*w+cx; g_cy=dy*h+cy; g_w=torch.exp(dw)*w; g_h=torch.exp(dh)*h
            boxes_g=torch.stack([g_cx-0.5*g_w,g_cy-0.5*g_h,g_cx+0.5*g_w,g_cy+0.5*g_h],dim=1).clamp(min=0)
            ious=box_iou(boxes_g, box.unsqueeze(0)).squeeze()
            iou_vars.append(ious.var().item())
print(f"\nIoU within-group variance: {np.mean(iou_vars):.6f}")
