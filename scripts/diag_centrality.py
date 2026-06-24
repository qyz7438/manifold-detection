"""Diagnostic: edge centrality vs IoU correlation."""
import sys, torch, torch.nn.functional as F, numpy as np
from torchvision.ops import box_iou
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
set_seed(42)
DEV = "cuda"; M = 8; PIX = 64; MAX_P = 200

def edge_centrality(patches):
    """Gradient magnitude weighted by distance to center. High = edges near center."""
    gray = patches.float().mean(dim=1, keepdim=True)
    gx = gray[:,:,:,1:] - gray[:,:,:,:-1]
    gy = gray[:,:,1:,:] - gray[:,:,:-1,:]
    edge = torch.sqrt(gx[:,:,:-1,:].pow(2) + gy[:,:,:,:-1].pow(2) + 1e-6).squeeze(1)  # (N, H-1, W-1)
    H, W = edge.shape[-2], edge.shape[-1]
    cy, cx = (H-1)/2, (W-1)/2
    ys = torch.arange(H, device=patches.device).float()
    xs = torch.arange(W, device=patches.device).float()
    Y, X = torch.meshgrid(ys, xs, indexing='ij')
    dist = torch.sqrt((Y - cy)**2 + (X - cx)**2)
    d_max = dist.max()
    weight = 1.0 - dist / d_max  # 1 at center, 0 at corners
    weighted = edge * weight.unsqueeze(0)
    total = edge.flatten(1).sum(dim=1).clamp_min(1e-6)
    centrality = weighted.flatten(1).sum(dim=1) / total
    return centrality.clamp(0, 1)

def decode_boxes(proposals, deltas):
    w = proposals[:,2] - proposals[:,0]; h = proposals[:,3] - proposals[:,1]
    cx = proposals[:,0] + 0.5*w; cy = proposals[:,1] + 0.5*h
    px = deltas[:,0]*w + cx - 0.5*torch.exp(deltas[:,2])*w
    py = deltas[:,1]*h + cy - 0.5*torch.exp(deltas[:,3])*h
    return torch.stack([px, py, deltas[:,0]*w+cx+0.5*torch.exp(deltas[:,2])*w,
                        deltas[:,1]*h+cy+0.5*torch.exp(deltas[:,3])*h], dim=1).clamp(min=0)

cfg = {"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                 "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                 "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}}
model = build_detector(cfg).to(DEV)
ckpt = torch.load("runs/round227_v1_baseline_20ep/checkpoint_best.pth", map_location=DEV)
model.load_state_dict(ckpt["model"]); model.eval()
loaders = build_penn_fudan_loaders({
    "data": {"root": "./data", "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
    "train": {"batch_size": 2}})

pc = {}; rc = {}
model.rpn.register_forward_hook(lambda m,i,o: pc.update({"p": o[0]}))
model.roi_heads.box_head.register_forward_pre_hook(lambda m,i: rc.update({"x": i[0]}))

all_c, all_i = [], []

for images, targets in loaders[0]:
    pc.clear(); rc.clear()
    model([img.to(DEV) for img in images], [{k: v.to(DEV) for k, v in t.items()} for t in targets])
    rf = rc.get("x"); pr = pc.get("p")
    if rf is None or rf.shape[0] == 0: continue
    N = rf.shape[0]; bf = model.roi_heads.box_head(rf)
    mu = model.roi_heads.box_predictor.bbox_pred(bf)[:, -4:]
    pc_ = torch.cat(pr, dim=0); N = min(N, pc_.shape[0], MAX_P); mu = mu[:N]
    eps = torch.randn(N, M, 4, device=DEV)
    deltas = mu.unsqueeze(1) + 0.1 * eps
    ad = deltas.reshape(N*M, 4)
    pe = pc_[:N].unsqueeze(1).expand(-1, M, -1).reshape(N*M, 4)
    boxes = decode_boxes(pe, ad)
    npi = [p.shape[0] for p in pr]
    ii = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(npi)], dim=0)[:N]
    patches = []; Kt = N * M; sl = min(Kt, 256)
    for idx in range(sl):
        pi_idx = min(idx // M, N - 1); img_i = ii[pi_idx].item(); img = images[img_i]; b = boxes[idx]
        x1, y1 = max(0, int(b[0].round().item())), max(0, int(b[1].round().item()))
        x2, y2 = min(img.shape[-1], max(x1+1, int(b[2].round().item()))), min(img.shape[-2], max(y1+1, int(b[3].round().item())))
        crop = img[:, y1:y2, x1:x2]
        patches.append(F.interpolate(crop.unsqueeze(0).float(), size=(PIX, PIX), mode='bilinear', align_corners=False).squeeze(0) if crop.shape[-1]>=4 and crop.shape[-2]>=4 else torch.zeros(3, PIX, PIX))
    pb = torch.stack(patches).to(DEV)
    c = edge_centrality(pb)
    cp = torch.zeros(Kt, device=DEV); cp[:sl] = c; cm = cp.view(N, M)[:N]
    pim = []; nb = 0
    for ip, p in enumerate(pr):
        for _ in range(p.shape[0]):
            if nb < N: pim.append(ip)
            nb += 1
    pim = pim[:N]; im = torch.zeros(N, M)
    for pi in range(N):
        gt_ = targets[pim[pi]]["boxes"].to(DEV)
        if len(gt_) > 0: ious = box_iou(boxes[pi*M:(pi+1)*M], gt_); im[pi] = ious.max(dim=1).values
    for pi in range(N):
        all_c.extend(cm[pi].tolist()); all_i.extend(im[pi].tolist())
    if len(all_c) > 3000: break

ca = np.array(all_c); ia = np.array(all_i)
print(f"\nTotal pairwise: {len(ca)}")
print(f"  r(IoU):                       {np.corrcoef(ca, ia)[0,1]:.4f}")
print(f"  centrality mean:              {ca.mean():.4f}")
print(f"  centrality[IoU<0.2]:          {ca[ia<0.2].mean():.4f}")
print(f"  centrality[IoU>0.4]:          {ca[ia>0.4].mean():.4f}")
print(f"  centrality[IoU>0.6]:          {ca[ia>0.6].mean():.4f}")
print(f"  good/bad gap:                 {ca[ia>0.4].mean()-ca[ia<0.2].mean():.4f}")
top1 = np.mean([ca.reshape(-1,M)[i].argmax()==ia.reshape(-1,M)[i].argmax() for i in range(min(len(ca)//M, 200))])
print(f"  top1-match (random=0.125):    {top1:.3f}")
