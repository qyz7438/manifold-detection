"""Diagnostic: edge-FFT quality vs IoU correlation, compare with raw pixel FFT."""
import sys, torch, torch.nn.functional as F, numpy as np
from torchvision.ops import box_iou
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
set_seed(42)
DEV = "cuda"; M = 8; PIX = 64; MAX_P = 200

def edge_fft_quality(patches):
    gray = patches.float().mean(dim=1, keepdim=True)
    gx = gray[:,:,:,1:] - gray[:,:,:,:-1]
    gy = gray[:,:,1:,:] - gray[:,:,:-1,:]
    edge = torch.sqrt(gx[:,:,:-1,:].pow(2) + gy[:,:,:,:-1].pow(2) + 1e-6).squeeze(1)
    fft = torch.fft.fft2(edge).abs(); mf = fft.flatten(1)
    t = mf.sum(dim=1, keepdim=True).clamp_min(1e-6)
    hf = mf[:, mf.shape[1]//2:].sum(dim=1) / t.squeeze(1)
    mn = mf / t; ent = -(mn * torch.log(mn + 1e-6)).sum(dim=1)
    me = torch.log(torch.tensor(float(mf.shape[1]), device=DEV))
    en = 1.0 - ent / me
    pv = torch.angle(torch.fft.fft2(edge) + 1e-6).flatten(1).std(dim=1).clamp_max(1.0)
    return (0.3*hf + 0.4*en + 0.3*(1.0-pv)).clamp(0, 1)

def pixel_fft_quality(patches):
    gray = patches.float().mean(dim=1)
    fft = torch.fft.fft2(gray).abs(); mf = fft.flatten(1)
    t = mf.sum(dim=1, keepdim=True).clamp_min(1e-6)
    hf = mf[:, mf.shape[1]//2:].sum(dim=1) / t.squeeze(1)
    mn = mf / t; ent = -(mn * torch.log(mn + 1e-6)).sum(dim=1)
    me = torch.log(torch.tensor(float(mf.shape[1]), device=DEV))
    en = 1.0 - ent / me
    pv = torch.angle(torch.fft.fft2(gray) + 1e-6).flatten(1).std(dim=1).clamp_max(1.0)
    return (0.3*hf + 0.4*en + 0.3*(1.0-pv)).clamp(0, 1)

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

all_eq, all_pq, all_ei, all_pi = [], [], [], []

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

    patches = []
    K_total = N * M
    sample_limit = min(K_total, 256)
    for idx in range(sample_limit):
        pi_idx = min(idx // M, N - 1)
        img_i = ii[pi_idx].item()
        img = images[img_i]
        b = boxes[idx]
        x1, y1 = max(0, int(b[0].round().item())), max(0, int(b[1].round().item()))
        x2, y2 = min(img.shape[-1], max(x1+1, int(b[2].round().item()))), min(img.shape[-2], max(y1+1, int(b[3].round().item())))
        crop = img[:, y1:y2, x1:x2]
        if crop.shape[-1] >= 4 and crop.shape[-2] >= 4:
            crop = F.interpolate(crop.unsqueeze(0).float(), size=(PIX, PIX), mode='bilinear', align_corners=False).squeeze(0)
            patches.append(crop)
        else:
            patches.append(torch.zeros(3, PIX, PIX))

    pb = torch.stack(patches).to(DEV)
    eq = edge_fft_quality(pb)
    pq = pixel_fft_quality(pb)

    eq_pad = torch.zeros(K_total, device=DEV); eq_pad[:len(eq)] = eq
    pq_pad = torch.zeros(K_total, device=DEV); pq_pad[:len(pq)] = pq
    eq_m = eq_pad.view(N, M)[:N]
    pq_m = pq_pad.view(N, M)[:N]

    pim = []; nb = 0
    for ip, p in enumerate(pr):
        for _ in range(p.shape[0]):
            if nb < N: pim.append(ip)
            nb += 1
    pim = pim[:N]

    im = torch.zeros(N, M)
    for pi in range(N):
        gt_ = targets[pim[pi]]["boxes"].to(DEV)
        if len(gt_) > 0:
            ious = box_iou(boxes[pi*M:(pi+1)*M], gt_)
            im[pi] = ious.max(dim=1).values

    for pi in range(N):
        all_eq.extend(eq_m[pi].tolist()); all_pq.extend(pq_m[pi].tolist())
        all_ei.extend(im[pi].tolist()); all_pi.extend(im[pi].tolist())

    if len(all_eq) > 3000: break

eq_a = np.array(all_eq); pq_a = np.array(all_pq); ei_a = np.array(all_ei)

print(f"\nTotal pairwise samples: {len(eq_a)}")
print(f"\n{'':20s}  {'r(IoU)':>8s}  {'q@IoU<0.2':>10s}  {'q@IoU>0.4':>10s}  {'top1-match':>10s}")
print(f"{'EDGE-FFT':20s}  {np.corrcoef(eq_a, ei_a)[0,1]:8.4f}  {eq_a[ei_a<0.2].mean():10.4f}  {eq_a[ei_a>0.4].mean():10.4f}  {np.mean([eq_a.reshape(-1,M)[i].argmax()==ei_a.reshape(-1,M)[i].argmax() for i in range(min(len(eq_a)//M,200))]):10.3f}")
print(f"{'PIXEL-FFT':20s}  {np.corrcoef(pq_a, ei_a)[0,1]:8.4f}  {pq_a[ei_a<0.2].mean():10.4f}  {pq_a[ei_a>0.4].mean():10.4f}  {np.mean([pq_a.reshape(-1,M)[i].argmax()==ei_a.reshape(-1,M)[i].argmax() for i in range(min(len(pq_a)//M,200))]):10.3f}")

print(f"\nEdge quality:  mean={eq_a.mean():.4f}  std={eq_a.std():.4f}  min={eq_a.min():.4f}  max={eq_a.max():.4f}")
print(f"Pixel quality: mean={pq_a.mean():.4f}  std={pq_a.std():.4f}  min={pq_a.min():.4f}  max={pq_a.max():.4f}")
