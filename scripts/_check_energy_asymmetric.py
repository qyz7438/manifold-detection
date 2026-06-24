"""Test asymmetric energy penalty: only penalize HIGH energy, ignore low."""
import sys
import torch, numpy as np
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
from torchvision.ops import box_iou

DEV = "cuda"; SEED = 42; G = 4; SIGMA = 0.1
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
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
sampled_props = {}
model.roi_heads.box_roi_pool.register_forward_pre_hook(
    lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
box_head_in = {}
model.roi_heads.box_head.register_forward_pre_hook(
    lambda m, args: box_head_in.update({"x": args[0]}))
box_pool = model.roi_heads.box_roi_pool

import math
def compute_loc_reward(iou):
    r = torch.zeros_like(iou)
    r[iou >= 0.75] = 1.0
    r[(iou >= 0.5) & (iou < 0.75)] = 0.3
    r[iou < 0.5] = -0.5
    return r

def grpo_advantage(reward):
    r_mean = reward.mean(dim=1, keepdim=True)
    r_std = reward.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (reward - r_mean) / r_std

def glp(d, m, s):
    e = (d - m.unsqueeze(1)) / s.unsqueeze(1)
    return -0.5 * (e.pow(2) + 2 * torch.log(s.unsqueeze(1)) + math.log(2 * math.pi)).sum(dim=-1)

tl, vl = build_penn_fudan_loaders({"data": {"root": "./data", "max_size": 320,
    "train_fraction": 0.8, "num_workers": 0}, "train": {"batch_size": 2}})

# Test asymmetric penalty: ReLU(energy - threshold) only
# Compare symmetric (linear -energy) vs asymmetric (ReLU only)
results_sym = []
results_asym = []

for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]
    tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
    sampled_props.clear(); box_head_in.clear(); fpn_feats.clear()
    image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]

    with torch.no_grad():
        _ = model(imgs_d, tgts_t)

    rf = box_head_in.get("x"); sp_raw = sampled_props.get("p"); fpn = fpn_feats.get("f")
    if rf is None or sp_raw is None or fpn is None or rf.shape[0] == 0: continue

    N_rf = rf.shape[0]
    bf = model.roi_heads.box_head(rf)
    mu = model.roi_heads.box_predictor.bbox_pred(bf)[:, -4:]
    s = torch.full_like(mu, 0.1)
    deltas = mu.unsqueeze(1) + s.unsqueeze(1) * torch.randn(N_rf, G, 4, device=DEV)

    sp_cat = torch.cat(sp_raw, dim=0); N = min(N_rf, sp_cat.shape[0])
    deltas = deltas[:N]

    box_list, delta_list, img_map = [], [], []
    offset = 0
    for i_img, p_img in enumerate(sp_raw):
        n_a = min(p_img.shape[0], N - offset)
        if n_a <= 0: break
        box_list.append(sp_cat[offset:offset + n_a])
        delta_list.append(deltas[offset:offset + n_a].reshape(-1, 4))
        img_map.extend([i_img] * (n_a * G))
        offset += n_a

    sp_exp = torch.cat([p.repeat_interleave(G, dim=0) for p in box_list], dim=0)
    delta_cat = torch.cat(delta_list, dim=0)
    bw = sp_exp[:, 2] - sp_exp[:, 0]; bh = sp_exp[:, 3] - sp_exp[:, 1]
    bcx = sp_exp[:, 0] + 0.5 * bw; bcy = sp_exp[:, 1] + 0.5 * bh
    dx = delta_cat[:, 0] / 10.0; dy = delta_cat[:, 1] / 10.0
    dw = delta_cat[:, 2] / 5.0;  dh = delta_cat[:, 3] / 5.0
    decoded_cat = torch.stack([
        dx * bw + bcx - 0.5 * torch.exp(dw) * bw,
        dy * bh + bcy - 0.5 * torch.exp(dh) * bh,
        dx * bw + bcx + 0.5 * torch.exp(dw) * bw,
        dy * bh + bcy + 0.5 * torch.exp(dh) * bh,
    ], dim=1).clamp(min=0)

    decoded_list, off = [], 0
    for di in delta_list: n = di.shape[0]; decoded_list.append(decoded_cat[off:off + n]); off += n

    pooled = box_pool(fpn, decoded_list, image_shapes)
    energy = extract_perchan_fft(pooled).mean(dim=1).view(offset, G)  # (N, G)

    iou_r = torch.zeros(offset, G, device=DEV)
    for pi in range(offset):
        i_img = img_map[pi * G]
        gt = tgts_t[i_img]["boxes"]
        if len(gt) > 0:
            iou_r[pi] = box_iou(decoded_cat[pi * G:(pi + 1) * G], gt).max(dim=1).values

    reward_loc = compute_loc_reward(iou_r)

    # Per-group z-energy
    en_mean = energy.mean(dim=1, keepdim=True)
    en_std = energy.std(dim=1, keepdim=True).clamp_min(1e-6)
    z_energy = (energy - en_mean) / en_std

    # Symmetric (what we tried before): -z_energy — penalize high, reward low
    reward_sym = reward_loc + 0.02 * torch.clamp(-z_energy, -1, 1)

    # Asymmetric: ReLU only — penalty when z > 0 (above mean), zero when z < 0
    # Tuneable threshold: use z > 0 (above group mean) or z > 1 (1 sigma above)
    reward_asym = reward_loc + 0.02 * torch.clamp(-torch.relu(z_energy), -1, 0)
    # This is equivalent to: if z > 0 (above mean), penalize; if z < 0, do nothing

    # Compare: which reward variant better ranks top IoU?
    max_iou = iou_r.max(dim=1).values
    for pi in range(offset):
        if max_iou[pi] < 0.1: continue
        iou_g = iou_r[pi]  # (G,)
        best_iou = iou_g.argmax().item()

        # Best by different rewards
        best_loc = reward_loc[pi].argmax().item()
        best_sym = reward_sym[pi].argmax().item()
        best_asym = reward_asym[pi].argmax().item()

        results_sym.append(iou_g[best_sym] >= iou_g[best_loc] - 1e-6)
        results_asym.append(iou_g[best_asym] >= iou_g[best_loc] - 1e-6)

print(f"Total comparable proposal groups: {len(results_sym)}\n")
print(f"{'Reward variant':<25s} {'Selects best IoU':>20s}")
print("-" * 50)
print(f"{'R_loc (baseline)':<25s} {'(reference)':>20s}")
print(f"{'R_loc + sym(-z_energy)':<25s} {f'{sum(results_sym)}/{len(results_sym)} = {100*sum(results_sym)/len(results_sym):.1f}%':>20s}")
print(f"{'R_loc + asym(ReLU(-z))':<25s} {f'{sum(results_asym)}/{len(results_asym)} = {100*sum(results_asym)/len(results_asym):.1f}%':>20s}")

# Check: what fraction of proposals have high energy?
# Also: for FN proposals, what's the energy distribution?
print(f"\n=== Energy distribution by IoU category ===")
all_en = []; all_iou = []
for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]
    sampled_props.clear(); box_head_in.clear(); fpn_feats.clear()
    with torch.no_grad(): _ = model(imgs_d, [{k: v.to(DEV) for k, v in t.items()} for t in tgts])
    fpn = fpn_feats.get("f")
    if fpn is None: continue
    img_shape = (imgs_d[0].shape[-2], imgs_d[0].shape[-1])
    for tgt in tgts:
        for box in tgt["boxes"]:
            w = box[2] - box[0]; h = box[3] - box[1]
            for shift in torch.linspace(-10, 10, 21):
                cx = (box[0]+box[2])/2 + shift
                shifted = torch.tensor([[cx-w/2, (box[1]+box[3])/2-h/2, cx+w/2, (box[1]+box[3])/2+h/2]], device=DEV).clamp(min=0)
                shifted[0, 2] = shifted[0, 2].clamp(max=img_shape[1]-1)
                en = extract_perchan_fft(box_pool(fpn, [shifted], [img_shape])).mean().item()
                iou = box_iou(shifted.cpu(), box.unsqueeze(0))[0,0].item()
                all_en.append(en); all_iou.append(iou)

all_en = np.array(all_en); all_iou = np.array(all_iou)

for label, (lo, hi) in [("FN (IoU<0.5)", (0, 0.5)), ("Borderline (0.4-0.55)", (0.4, 0.55)), ("TP (IoU>=0.75)", (0.75, 1.0))]:
    mask = (all_iou >= lo) & (all_iou < hi)
    subset = all_en[mask]
    if len(subset) > 0:
        print(f"  {label:<22s}: n={len(subset):4d}  mean={subset.mean():.4f}  std={subset.std():.4f}  "
              f"p5={np.percentile(subset,5):.4f}  p50={np.percentile(subset,50):.4f}  p95={np.percentile(subset,95):.4f}")
