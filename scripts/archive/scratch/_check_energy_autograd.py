"""Check: can autograd through roi_pool->FFT->energy produce useful delta gradients?"""
import sys
import torch, numpy as np
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed

DEV = "cuda"; SEED = 42; SIGMA = 0.1
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
set_seed(SEED)

# Pre-compute frequency masks for 7x7 (roi_pool output size)
H, W = 7, 7
fh = torch.fft.fftfreq(H, device=DEV)
fw = torch.fft.rfftfreq(W, device=DEV)  # 4 values for W=7
Y, X = torch.meshgrid(fh, fw, indexing='ij')
rn = torch.sqrt(X**2 + Y**2) / torch.sqrt(X**2 + Y**2).max().clamp_min(1e-6)
LO = (rn <= 0.3).float()
MD = ((rn > 0.3) & (rn <= 0.7)).float()
HI = (rn > 0.7).float()

def compute_energy_value(pooled_roi):
    """Compute scalar energy from pooled ROI features. Works on (1,C,7,7)."""
    fft = torch.fft.rfft2(pooled_roi, dim=(-2, -1), norm="ortho")
    amp = torch.abs(fft)
    a_lo = (amp * LO).sum(); a_md = (amp * MD).sum(); a_hi = (amp * HI).sum()
    return a_lo / (a_lo + a_md + a_hi + 1e-8)

model = build_detector({"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
    "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
    "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}}).to(DEV)
ckpt = torch.load(CKPT, map_location=DEV)
model.load_state_dict(ckpt["model"])
for p in model.parameters(): p.requires_grad = False

fpn_feats_dict = {}
model.backbone.register_forward_hook(lambda m, i, o: fpn_feats_dict.update({"f": {k: o[k] for k in o if k != "pool"}}))
sampled_props_dict = {}
model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m, args: sampled_props_dict.update({"p": [a.clone() for a in args[1]]}))
box_head_in_dict = {}
model.roi_heads.box_head.register_forward_pre_hook(lambda m, args: box_head_in_dict.update({"x": args[0]}))
box_pool = model.roi_heads.box_roi_pool

tl, vl = build_penn_fudan_loaders({"data": {"root": "./data", "max_size": 320,
    "train_fraction": 0.8, "num_workers": 0}, "train": {"batch_size": 2}})

all_gnorms = []; total = 0; autograd_w = 0; rand_w = 0

for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]
    tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
    sampled_props_dict.clear(); box_head_in_dict.clear(); fpn_feats_dict.clear()
    img_shape = (imgs_d[0].shape[-2], imgs_d[0].shape[-1])

    with torch.no_grad(): _ = model(imgs_d, tgts_t)
    rf = box_head_in_dict.get("x"); sp_raw = sampled_props_dict.get("p"); fpn = fpn_feats_dict.get("f")
    if rf is None or sp_raw is None or fpn is None: continue

    n = min(rf.shape[0], 16)
    rf_g = rf[:n].clone().detach().requires_grad_(True)
    bf = model.roi_heads.box_head(rf_g)
    mu = model.roi_heads.box_predictor.bbox_pred(bf)[:, -4:]

    sp = torch.cat(sp_raw, dim=0)[:n]
    bw = sp[:,2]-sp[:,0]; bh = sp[:,3]-sp[:,1]; bcx = sp[:,0]+0.5*bw; bcy = sp[:,1]+0.5*bh

    dx = mu[:,0]/10.0; dy = mu[:,1]/10.0; dw = mu[:,2]/5.0; dh = mu[:,3]/5.0
    decoded = torch.stack([
        dx*bw+bcx-0.5*torch.exp(dw)*bw, dy*bh+bcy-0.5*torch.exp(dh)*bh,
        dx*bw+bcx+0.5*torch.exp(dw)*bw, dy*bh+bcy+0.5*torch.exp(dh)*bh,
    ], dim=1).clamp(min=0)

    pooled = box_pool(fpn, [decoded], [img_shape])  # (N, C, 7, 7)

    for i in range(n):
        en = compute_energy_value(pooled[i:i+1])
        grad_i = torch.autograd.grad(en, mu, retain_graph=True, allow_unused=True)[0]
        if grad_i is None or i >= grad_i.shape[0]: continue
        gn = grad_i[i].norm().item(); all_gnorms.append(gn)
        if gn < 1e-8: continue

        dir_auto = -grad_i[i] / gn; delta_auto = SIGMA * dir_auto
        mu_auto = mu[i].detach() + delta_auto

        dir_rand = torch.randn(4, device=DEV); dir_rand = dir_rand / dir_rand.norm()
        mu_rand = mu[i].detach() + SIGMA * dir_rand

        def decode_box(mu_val):
            d = mu_val / torch.tensor([10.0,10.0,5.0,5.0], device=DEV)
            return torch.stack([
                d[0]*bw[i]+bcx[i]-0.5*torch.exp(d[2])*bw[i],
                d[1]*bh[i]+bcy[i]-0.5*torch.exp(d[3])*bh[i],
                d[0]*bw[i]+bcx[i]+0.5*torch.exp(d[2])*bw[i],
                d[1]*bh[i]+bcy[i]+0.5*torch.exp(d[3])*bh[i],
            ]).clamp(min=0).unsqueeze(0)

        with torch.no_grad():
            en_a = compute_energy_value(box_pool(fpn, [decode_box(mu_auto)], [img_shape]))
            en_r = compute_energy_value(box_pool(fpn, [decode_box(mu_rand)], [img_shape]))

        total += 1
        if en_a < en_r: autograd_w += 1
        else: rand_w += 1

print(f"\n=== Autograd energy gradient ===")
print(f"Total proposals tested: {n_test if 'n' in dir() else '?'}")
print(f"Proposals with grad: {len(all_gnorms)}")
if all_gnorms:
    g = np.array(all_gnorms)
    print(f"N={len(g)}  mean={g.mean():.8f}  median={np.median(g):.8f}  max={g.max():.8f}")
    nz = (g > 1e-8).sum()
    print(f"nonzero: {nz}/{len(g)} ({100*nz/len(g):.1f}%)")

print(f"\n-gradE achieves lower energy: {autograd_w}/{total} ({autograd_w/total*100:.1f}%)")
print(f"Random achieves lower energy: {rand_w}/{total} ({rand_w/total*100:.1f}%)")
