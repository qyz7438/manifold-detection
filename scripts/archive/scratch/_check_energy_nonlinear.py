"""Compare non-linear penalty shapes for asymmetric energy reward."""
import sys, math
import torch, numpy as np
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
from torchvision.ops import box_iou

DEV = "cuda"; SEED = 42; G = 4; SIGMA = 0.1; THR = 0.05
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

# Collect energy vs IoU data across all proposals
model = build_detector({"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
    "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
    "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}}).to(DEV)
ckpt = torch.load(CKPT, map_location=DEV)
model.load_state_dict(ckpt["model"])
model.eval()

fpn_feats = {}
model.backbone.register_forward_hook(lambda m, i, o: fpn_feats.update({"f": {k: o[k] for k in o if k != "pool"}}))
sampled_props = {}
model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
box_head_in = {}
model.roi_heads.box_head.register_forward_pre_hook(lambda m, args: box_head_in.update({"x": args[0]}))
box_pool = model.roi_heads.box_roi_pool

def compute_loc_reward(iou):
    r = torch.zeros_like(iou)
    r[iou >= 0.75] = 1.0
    r[(iou >= 0.5) & (iou < 0.75)] = 0.3
    r[iou < 0.5] = -0.5
    return r

tl, vl = build_penn_fudan_loaders({"data": {"root": "./data", "max_size": 320,
    "train_fraction": 0.8, "num_workers": 0}, "train": {"batch_size": 2}})

all_pairs = []  # (energy, iou, reward_loc) per sample
all_groups = []  # per proposal group: [(en_G, iou_G, loc_r_G)]

for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]
    tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
    sampled_props.clear(); box_head_in.clear(); fpn_feats.clear()
    image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]
    with torch.no_grad(): _ = model(imgs_d, tgts_t)
    rf = box_head_in.get("x"); sp_raw = sampled_props.get("p"); fpn = fpn_feats.get("f")
    if rf is None or sp_raw is None or fpn is None or rf.shape[0] == 0: continue

    N_rf = rf.shape[0]; bf = model.roi_heads.box_head(rf)
    mu = model.roi_heads.box_predictor.bbox_pred(bf)[:, -4:]
    s = torch.full_like(mu, 0.1)
    deltas = mu.unsqueeze(1) + s.unsqueeze(1) * torch.randn(N_rf, G, 4, device=DEV)
    sp_cat = torch.cat(sp_raw, dim=0); N = min(N_rf, sp_cat.shape[0]); deltas = deltas[:N]

    box_list, delta_list, img_map = [], [], []; offset = 0
    for i_img, p_img in enumerate(sp_raw):
        n_a = min(p_img.shape[0], N - offset)
        if n_a <= 0: break
        box_list.append(sp_cat[offset:offset + n_a])
        delta_list.append(deltas[offset:offset + n_a].reshape(-1, 4))
        img_map.extend([i_img] * (n_a * G)); offset += n_a

    sp_exp = torch.cat([p.repeat_interleave(G, dim=0) for p in box_list], dim=0)
    delta_cat = torch.cat(delta_list, dim=0)
    bw = sp_exp[:, 2] - sp_exp[:, 0]; bh = sp_exp[:, 3] - sp_exp[:, 1]
    bcx = sp_exp[:, 0] + 0.5 * bw; bcy = sp_exp[:, 1] + 0.5 * bh
    dx = delta_cat[:, 0] / 10.0; dy = delta_cat[:, 1] / 10.0
    dw = delta_cat[:, 2] / 5.0;  dh = delta_cat[:, 3] / 5.0
    decoded_cat = torch.stack([
        dx * bw + bcx - 0.5 * torch.exp(dw) * bw, dy * bh + bcy - 0.5 * torch.exp(dh) * bh,
        dx * bw + bcx + 0.5 * torch.exp(dw) * bw, dy * bh + bcy + 0.5 * torch.exp(dh) * bh,
    ], dim=1).clamp(min=0)

    decoded_list, off = [], 0
    for di in delta_list: n = di.shape[0]; decoded_list.append(decoded_cat[off:off + n]); off += n

    pooled = box_pool(fpn, decoded_list, image_shapes)
    energy = extract_perchan_fft(pooled).mean(dim=1).view(offset, G)

    iou_r = torch.zeros(offset, G, device=DEV)
    for pi in range(offset):
        gt = tgts_t[img_map[pi * G]]["boxes"]
        if len(gt) > 0:
            iou_r[pi] = box_iou(decoded_cat[pi * G:(pi + 1) * G], gt).max(dim=1).values

    reward_loc = compute_loc_reward(iou_r)
    for pi in range(offset):
        g_en = energy[pi].detach().cpu().numpy()
        g_iou = iou_r[pi].detach().cpu().numpy()
        g_loc = reward_loc[pi].detach().cpu().numpy()
        all_groups.append({"en": g_en, "iou": g_iou, "loc": g_loc})
        for j in range(G):
            all_pairs.append((g_en[j], g_iou[j]))

all_en = np.array([p[0] for p in all_pairs])
all_iou = np.array([p[1] for p in all_pairs])

# Find optimal threshold and shape
print("=== Non-linear penalty shapes ===")
print(f"\nEnergy stats: mean={all_en.mean():.4f} std={all_en.std():.4f}")
print(f"IoU stats:   mean={all_iou.mean():.4f} std={all_iou.std():.4f}")

# Test various penalty functions
penalties = {
    "linear_relu(0.5)": lambda e: -np.maximum(e - 0.50, 0),
    "linear_relu(0.55)": lambda e: -np.maximum(e - 0.55, 0),
    "softplus(0.5, k=10)": lambda e: -np.log(1 + np.exp(10 * (e - 0.50))) / 10,
    "softplus(0.55, k=15)": lambda e: -np.log(1 + np.exp(15 * (e - 0.55))) / 15,
    "sigmoid(0.5, k=15)": lambda e: -1 / (1 + np.exp(-15 * (e - 0.50))),
    "sigmoid(0.53, k=20)": lambda e: -1 / (1 + np.exp(-20 * (e - 0.53))),
    "exp_penalty(a=2)": lambda e: -np.exp(np.clip(2*(e-0.5), -5, 5)) / np.exp(1),
    "exp_penalty(a=3)": lambda e: -np.exp(np.clip(3*(e-0.53), -5, 5)) / np.exp(1),
}

print(f"\n{'Penalty':<25s} {'select_best':>12s} {'rank_corr':>10s} {'TP_penalty':>12s} {'FN_penalty':>12s} {'TP/FN_ratio':>12s}")
print("-" * 90)

for name, fn in penalties.items():
    # Test on per-group ranking
    correct = 0; total = 0
    all_pen = []
    for g in all_groups:
        en_g = g["en"]; loc_g = g["loc"]; iou_g = g["iou"]
        pen_g = fn(en_g) * 0.01  # small weight
        reward = loc_g + pen_g
        best = reward.argmax()
        if iou_g[best] >= iou_g.max() - 1e-6:
            correct += 1
        total += 1

        # Track penalty on TP vs FN
        max_iou = iou_g.max()
        if max_iou >= 0.75:
            all_pen.append(("TP", pen_g.mean()))
        elif max_iou < 0.5:
            all_pen.append(("FN", pen_g.mean()))

    tp_pen = np.mean([p[1] for p in all_pen if p[0] == "TP"]) if any(p[0]=="TP" for p in all_pen) else 0
    fn_pen = np.mean([p[1] for p in all_pen if p[0] == "FN"]) if any(p[0]=="FN" for p in all_pen) else 0
    ratio = abs(fn_pen) / max(abs(tp_pen), 1e-8) if tp_pen != 0 else float('inf')

    print(f"{name:<25s} {correct:>5d}/{total:<5d} {correct/total*100:>8.1f}% {tp_pen:>12.6f} {fn_pen:>12.6f} {ratio:>11.1f}x")

# Now try same on borderline groups only
print(f"\n=== Borderline groups only (max IoU 0.35-0.55) ===")
border_groups = [g for g in all_groups if 0.35 <= g["iou"].max() < 0.55]
print(f"Borderline groups: {len(border_groups)}")

for name, fn in penalties.items():
    correct = 0
    for g in border_groups:
        en_g = g["en"]; loc_g = g["loc"]; iou_g = g["iou"]
        pen_g = fn(en_g) * 0.01
        reward = loc_g + pen_g
        best = reward.argmax()
        if iou_g[best] >= iou_g.max() - 1e-6:
            correct += 1
    print(f"{name:<25s} {correct:>5d}/{len(border_groups):<5d} {correct/len(border_groups)*100:>8.1f}%")
