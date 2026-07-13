"""Test: P2 vs P3 FPN, and raw image pixel features for box shift sensitivity."""
import sys, math
import torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
from torchvision.ops import box_iou

DEV = "cuda"; SEED = 42; SIGMA = 0.1; G = 4
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
set_seed(SEED)

model = build_detector({"model":{"name":"fasterrcnn_mobilenet_v3_large_320_fpn","model_name":"fasterrcnn_mobilenet_v3_large_320_fpn","pretrained":True,"num_classes":2,"min_size":320,"max_size":320}}).to(DEV)
ckpt = torch.load(CKPT, map_location=DEV); model.load_state_dict(ckpt["model"]); model.eval()

fpn_feats = {}
model.backbone.register_forward_hook(lambda m,i,o: fpn_feats.update({"f":{k:o[k] for k in o if k!="pool"}}))
sampled_props = {}
model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m,args: sampled_props.update({"p":[a.clone() for a in args[1]]}))
box_pool = model.roi_heads.box_roi_pool

tl, vl = build_penn_fudan_loaders({"data":{"root":"./data","max_size":320,"train_fraction":0.8,"num_workers":0},"train":{"batch_size":2}})

def compute_energy_fft(x):
    M, C, Hf, Wf = x.shape
    fft = torch.fft.rfft2(x, dim=(-2,-1), norm="ortho"); amp = torch.abs(fft)
    fh = torch.fft.fftfreq(Hf,device=DEV); fw = torch.fft.rfftfreq(Wf,device=DEV)
    Y,X = torch.meshgrid(fh,fw,indexing='ij'); r=torch.sqrt(X**2+Y**2); R=r.max().clamp_min(1e-6); rn=r/R
    lo=(rn<=0.3).float()
    al=(amp*lo).flatten(2).sum(2); at=al+(amp*((rn>0.3)&(rn<=0.7)).float()).flatten(2).sum(2)+(amp*(rn>0.7).float()).flatten(2).sum(2)+1e-8
    return (al/at).mean(dim=1)

def crop_and_pool(image, boxes, out_size=7):
    """Crop raw image at box locations and resize. image: (C, H, W). Returns (M, C, out_size, out_size)."""
    M = boxes.shape[0]
    _, H, W = image.shape
    crops = []
    for i in range(M):
        x1, y1, x2, y2 = boxes[i].long().clamp(min=0)
        x1 = x1.clamp(max=W-2); x2 = x2.clamp(max=W-1); y1 = y1.clamp(max=H-2); y2 = y2.clamp(max=H-1)
        if x2 <= x1+1: x2 = x1+2
        if y2 <= y1+1: y2 = y1+2
        crop = image[:, y1:y2, x1:x2]
        crop = F.interpolate(crop.unsqueeze(0), size=(out_size, out_size), mode='bilinear', align_corners=False)
        crops.append(crop.squeeze(0))
    return torch.stack(crops)  # (M, 3, out_size, out_size)

def fpn_grid_sample(fpn_feat, boxes, img_shape, out_h=7, out_w=7):
    """grid_sample from FPN features: for each box, sample out_hxout_w grid points inside the box.
    Returns (M, C, out_h, out_w)."""
    M = boxes.shape[0]
    H, W = img_shape
    C = fpn_feat.shape[1]

    # Normalize box coords to [-1, 1]
    x1 = 2*boxes[:,0]/W - 1; y1 = 2*boxes[:,1]/H - 1
    x2 = 2*boxes[:,2]/W - 1; y2 = 2*boxes[:,3]/H - 1

    # Grid of sampling points inside box
    gy = torch.linspace(-1, 1, out_h, device=DEV).view(1, out_h, 1).expand(M, out_h, out_w)
    gx = torch.linspace(-1, 1, out_w, device=DEV).view(1, 1, out_w).expand(M, out_h, out_w)

    # Map from [-1,1] grid to box-local coordinates
    px = (gx + 1)/2 * (x2 - x1).view(M,1,1) + x1.view(M,1,1)
    py = (gy + 1)/2 * (y2 - y1).view(M,1,1) + y1.view(M,1,1)

    grid = torch.stack([px, py], dim=-1)  # (M, out_h, out_w, 2)
    feat = fpn_feat[0:1].expand(M, -1, -1, -1)  # (M, C, H_f, W_f)
    return F.grid_sample(feat, grid, align_corners=True)  # (M, C, out_h, out_w)

results = {"roi_pool_P3": [], "roi_pool_P2": [], "raw_image": [], "fpn_P2_grid": [], "fpn_P3_grid": []}

# Pre-build P2 pooler if needed (using P3's pooler with manual P2 extraction)
# Actually, the box_pool routes to correct level automatically based on box area.
# For P2: we need to force pooling from P2 level. Let's just use grid_sample on P2.

for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]; img_shape = (imgs_d[0].shape[-2], imgs_d[0].shape[-1])
    img_raw = imgs_d[0]
    fpn_feats.clear(); sampled_props.clear()
    with torch.no_grad(): _ = model(imgs_d, [{k:v.to(DEV) for k,v in t.items()} for t in tgts])
    fpn = fpn_feats.get("f")
    if fpn is None: continue

    fpn_p2 = fpn['0']  # P2, stride=2, highest res
    fpn_p3 = fpn['1']  # P3, stride=4

    for tgt in tgts:
        for box in tgt["boxes"]:
            w=box[2]-box[0]; h=box[3]-box[1]; cx=box[0]+0.5*w; cy=box[1]+0.5*h
            dx=SIGMA*torch.randn(G,device=DEV); dy=SIGMA*torch.randn(G,device=DEV)
            dw=SIGMA*torch.randn(G,device=DEV); dh=SIGMA*torch.randn(G,device=DEV)
            g_cx=dx*w+cx; g_cy=dy*h+cy; g_w=torch.exp(dw)*w; g_h=torch.exp(dh)*h
            boxes_g=torch.stack([g_cx-0.5*g_w,g_cy-0.5*g_h,g_cx+0.5*g_w,g_cy+0.5*g_h],dim=1).clamp(min=0).to(DEV)
            boxes_g[:,2]=boxes_g[:,2].clamp(max=img_shape[1]-1)
            boxes_g[:,3]=boxes_g[:,3].clamp(max=img_shape[0]-1)

            # 1. Standard ROI pool (routes to P3 typically for pedestrian-sized boxes)
            pooled = box_pool(fpn, [boxes_g], [img_shape])
            en_roi = compute_energy_fft(pooled)
            results["roi_pool_P3"].append({"var": en_roi.var().item()})

            # 2. Raw image crop + resize to 7x7
            img_crops = crop_and_pool(img_raw, boxes_g, out_size=7)  # (G, 3, 7, 7)
            en_img = compute_energy_fft(img_crops)
            results["raw_image"].append({"var": en_img.var().item()})

            # 3. FPN P2 (stride=2) grid sample 7x7
            fpn_p2_7x7 = fpn_grid_sample(fpn_p2, boxes_g, img_shape, 7, 7)  # (G, 256, 7, 7)
            en_p2 = compute_energy_fft(fpn_p2_7x7)
            results["fpn_P2_grid"].append({"var": en_p2.var().item()})

            # 4. FPN P3 (stride=4) grid sample 7x7
            fpn_p3_7x7 = fpn_grid_sample(fpn_p3, boxes_g, img_shape, 7, 7)  # (G, 256, 7, 7)
            en_p3 = compute_energy_fft(fpn_p3_7x7)
            results["fpn_P3_grid"].append({"var": en_p3.var().item()})

            # 5. std ROI pool on P2 (if we can force it — skip for now)
            results["roi_pool_P2"].append({"var": float('nan')})

print(f"Analyzed {len(results['roi_pool_P3'])} GT boxes\n")
print(f"{'Method':<20s} {'energy_var':>12s} {'vs roi_pool':>10s}")
print("-" * 45)
roi_baseline = np.nanmean([r["var"] for r in results["roi_pool_P3"]])
for method in ["roi_pool_P3", "fpn_P3_grid", "fpn_P2_grid", "raw_image"]:
    vars_ = np.array([r["var"] for r in results[method]])
    v = np.nanmean(vars_)
    print(f"{method:<20s} {v:12.6f} {v/roi_baseline:9.1f}x")
