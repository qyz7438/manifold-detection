"""Counterfactual energy verifier analysis on baseline checkpoint.
Tests: 1) conditional energy-IoU correlation, 2) rank flip rate,
3) best-IoU selection by different reward variants, 4) negative controls.
"""
import sys, json, math
from pathlib import Path
import torch, numpy as np
from torchvision.ops import box_iou, nms
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed

DEV = "cuda"
CKPT = "runs/_fn_checkpoint.pth"  # from our 6-epoch det_only_unf training
G_SAMPLES = 4
SEED = 42
set_seed(SEED)

# Load model
model = build_detector({"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
    "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
    "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}}).to(DEV)
ckpt = torch.load(CKPT, map_location=DEV)
model.load_state_dict(ckpt["model"])
model.eval()

box_pool = model.roi_heads.box_roi_pool

# Hooks
fpn_feats = {}
sampled_props = {}
box_head_in = {}
model.roi_heads.box_roi_pool.register_forward_pre_hook(
    lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
model.roi_heads.box_head.register_forward_pre_hook(
    lambda m, args: box_head_in.update({"x": args[0]}))
model.backbone.register_forward_hook(
    lambda m, i, o: fpn_feats.update({"f": {k: o[k] for k in o if k != "pool"}}))

# FFT
def extract_perchan_fft(x):
    C = x.shape[1]; H, W = x.shape[-2], x.shape[-1]
    fft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    amp = torch.abs(fft); pha = torch.angle(fft)
    freq_h = torch.fft.fftfreq(H, device=x.device)
    freq_w = torch.fft.rfftfreq(W, device=x.device)
    Y, X = torch.meshgrid(freq_h, freq_w, indexing='ij')
    r = torch.sqrt(X**2 + Y**2); R = r.max().clamp_min(1e-6); rn = r / R
    lo = (rn <= 0.3).float(); md = ((rn > 0.3) & (rn <= 0.7)).float(); hi = (rn > 0.7).float()
    a_lo = (amp * lo).flatten(2).sum(2); a_md = (amp * md).flatten(2).sum(2); a_hi = (amp * hi).flatten(2).sum(2)
    p_lo = (pha * lo).flatten(2).sum(2); p_md = (pha * md).flatten(2).sum(2); p_hi = (pha * hi).flatten(2).sum(2)
    return torch.cat([a_lo, a_md, a_hi, p_lo, p_md, p_hi], dim=1)

def compute_energy(fft_f):
    ch = fft_f.shape[1] // 6
    a_lo = fft_f[:, 0*ch:1*ch].sum(dim=1)
    a_total = a_lo + fft_f[:, 1*ch:2*ch].sum(dim=1) + fft_f[:, 2*ch:3*ch].sum(dim=1) + 1e-8
    return 2 * (a_lo / a_total) - 1

def compute_loc_reward(iou):
    r = torch.zeros_like(iou)
    r[iou >= 0.75] = 1.0
    r[(iou >= 0.5) & (iou < 0.75)] = 0.3
    r[iou < 0.5] = -0.5
    return r

def glp(d, m, s):
    e = (d - m.unsqueeze(1)) / s.unsqueeze(1)
    return -0.5 * (e.pow(2) + 2 * torch.log(s.unsqueeze(1)) + math.log(2 * math.pi)).sum(dim=-1)

tl, vl = build_penn_fudan_loaders({"data": {"root": "./data", "max_size": 320,
    "train_fraction": 0.8, "num_workers": 0}, "train": {"batch_size": 2}})

# Collect: for each proposal group (same image+GT), collect all G samples with IoU, energy, etc.
all_groups = []  # list of {gt_idx, iou_vec, energy_vec, loc_reward_vec}

rng_shuf = torch.Generator(device=DEV).manual_seed(SEED + 7777)

for imgs, tgts in tqdm(vl, desc="Collecting proposals"):
    imgs_d = [i.to(DEV) for i in imgs]
    tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
    sampled_props.clear(); box_head_in.clear(); fpn_feats.clear()
    image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]

    with torch.no_grad():
        _ = model(imgs_d, tgts_t)

    rf = box_head_in.get("x"); sp_raw = sampled_props.get("p"); fpn = fpn_feats.get("f")
    if rf is None or sp_raw is None or fpn is None or rf.shape[0] == 0:
        continue

    N_rf = rf.shape[0]
    bf = model.roi_heads.box_head(rf)
    mu = model.roi_heads.box_predictor.bbox_pred(bf)[:, -4:]

    s = torch.full_like(mu, 0.1)
    deltas = mu.unsqueeze(1) + s.unsqueeze(1) * torch.randn(N_rf, G_SAMPLES, 4, device=DEV)

    sp_cat = torch.cat(sp_raw, dim=0); N = min(N_rf, sp_cat.shape[0])
    mu = mu[:N]; deltas = deltas[:N]

    # Build decoded boxes per proposal
    box_list, delta_list, img_map = [], [], []
    offset = 0
    for i_img, p_img in enumerate(sp_raw):
        n_a = min(p_img.shape[0], N - offset)
        if n_a <= 0: break
        box_list.append(sp_cat[offset:offset + n_a])
        delta_list.append(deltas[offset:offset + n_a].reshape(-1, 4))
        img_map.extend([i_img] * (n_a * G_SAMPLES))
        offset += n_a

    sp_exp = torch.cat([p.repeat_interleave(G_SAMPLES, dim=0) for p in box_list], dim=0)
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
    for di in delta_list:
        n = di.shape[0]; decoded_list.append(decoded_cat[off:off + n]); off += n

    # Pool FFT energy
    pooled = box_pool(fpn, decoded_list, image_shapes)
    energy_all = compute_energy(extract_perchan_fft(pooled))

    # Compute IoU with GT
    for pi in range(offset):
        i_img = img_map[pi * G_SAMPLES]
        gt = tgts_t[i_img]["boxes"]
        if len(gt) == 0:
            continue
        iou_g = box_iou(decoded_cat[pi * G_SAMPLES:(pi + 1) * G_SAMPLES].detach(), gt)  # (G, #GT)
        # For each GT that this proposal might target, create a group
        for gi in range(len(gt)):
            iou_vec = iou_g[:, gi].detach()  # (G,)
            max_iou = iou_vec.max().item()
            if max_iou < 0.1:  # proposal doesn't target this GT at all
                continue
            energy_vec = energy_all[pi * G_SAMPLES:(pi + 1) * G_SAMPLES].detach()
            loc_r = compute_loc_reward(iou_vec)
            all_groups.append({
                "img_idx": i_img,
                "gt_idx": gi,
                "iou": iou_vec.cpu().numpy(),
                "energy": energy_vec.cpu().numpy(),
                "loc_reward": loc_r.cpu().numpy(),
            })

print(f"\nTotal proposal groups: {len(all_groups)}")
print(f"Average G samples per group: {np.mean([g['iou'].shape[0] for g in all_groups]):.1f}")

# === ANALYSIS ===

# 1. Conditional energy-IoU correlation (within groups, by IoU bin)
print("\n=== 1. Conditional Energy-IoU Correlation ===")
iou_bins = [(0.1, 0.35), (0.35, 0.5), (0.5, 0.75), (0.75, 1.0)]
for lo, hi in iou_bins:
    pairs = []
    for g in all_groups:
        for j in range(len(g["iou"])):
            if lo <= g["iou"][j] < hi:
                pairs.append((g["iou"][j], g["energy"][j]))
    if len(pairs) < 2:
        print(f"  IoU [{lo:.2f},{hi:.2f}): n={len(pairs)} insufficient")
        continue
    ious = np.array([p[0] for p in pairs])
    energies = np.array([p[1] for p in pairs])
    corr = np.corrcoef(ious, energies)[0, 1]
    en_tp = energies[ious >= 0.5].mean() if (ious >= 0.5).any() else float("nan")
    en_fn = energies[ious < 0.5].mean() if (ious < 0.5).any() else float("nan")
    print(f"  IoU [{lo:.2f},{hi:.2f}): n={len(pairs):5d}  corr={corr:+.4f}  en_mean={energies.mean():+.4f}")

# Within-group correlation (each group = same GT, same original proposal)
print("\n  Within-group energy-IoU correlation distribution:")
group_corrs = []
for g in all_groups:
    if len(g["iou"]) < 2: continue
    c = np.corrcoef(g["iou"], g["energy"])[0, 1]
    if not np.isnan(c):
        group_corrs.append(c)
if group_corrs:
    print(f"  Mean={np.mean(group_corrs):+.4f}  Median={np.median(group_corrs):+.4f}  Std={np.std(group_corrs):.4f}")
    print(f"  % negative: {100*sum(1 for c in group_corrs if c < 0)/len(group_corrs):.1f}%")
    print(f"  % significant negative (r<-0.5): {100*sum(1 for c in group_corrs if c < -0.5)/len(group_corrs):.1f}%")

# 2. Rank flip rate
print("\n=== 2. Rank Flip Analysis ===")
BETA = 0.05

flips_loc_vs_energy = 0
flips_loc_vs_cgated = 0
total_comparable = 0
loc_wins = 0
energy_wins = 0
cgated_wins = 0

for g in all_groups:
    iou = g["iou"]
    if len(iou) < 2: continue

    loc_r = g["loc_reward"]
    energy = g["energy"]

    # Normalize energy within group (z-score)
    en_mean = energy.mean(); en_std = energy.std()
    if en_std < 1e-8: en_std = 1.0
    z_energy = (energy - en_mean) / en_std

    # C-gated: only apply to borderline proposals
    r_cgated = loc_r.copy()
    for j in range(len(iou)):
        if 0.35 <= iou[j] < 0.55:
            r_cgated[j] = r_cgated[j] + BETA * np.clip(-z_energy[j], -1, 1)

    r_energy_raw = loc_r + BETA * (-energy)  # raw energy penalty (what we ran)
    r_energy_z = loc_r + BETA * np.clip(-z_energy, -1, 1)  # z-scored

    # Best by each reward
    best_loc = loc_r.argmax()
    best_energy_raw = r_energy_raw.argmax()
    best_cgated = r_cgated.argmax()
    best_true = iou.argmax()

    # Count wins: does the reward variant pick the highest IoU?
    max_iou = iou.max()
    loc_picks_max = (iou[best_loc] >= max_iou - 1e-6)
    energy_picks_max = (iou[best_energy_raw] >= max_iou - 1e-6)
    cgated_picks_max = (iou[best_cgated] >= max_iou - 1e-6)

    total_comparable += 1
    if loc_picks_max: loc_wins += 1
    if energy_picks_max: energy_wins += 1
    if cgated_picks_max: cgated_wins += 1

    # Rank flips: does energy-based reward pick different top than loc?
    if best_loc != best_energy_raw:
        flips_loc_vs_energy += 1
    if best_loc != best_cgated:
        flips_loc_vs_cgated += 1

print(f"  Total comparable groups: {total_comparable}")
print(f"  Best IoU selection rate:")
print(f"    R_loc only:          {loc_wins}/{total_comparable} = {loc_wins/total_comparable*100:.1f}%")
print(f"    R_loc + energy(raw): {energy_wins}/{total_comparable} = {energy_wins/total_comparable*100:.1f}%")
print(f"    R_loc + C-gated:     {cgated_wins}/{total_comparable} = {cgated_wins/total_comparable*100:.1f}%")
print(f"  Rank flip rate:")
print(f"    raw energy: {flips_loc_vs_energy}/{total_comparable} = {flips_loc_vs_energy/total_comparable*100:.1f}%")
print(f"    C-gated:    {flips_loc_vs_cgated}/{total_comparable} = {flips_loc_vs_cgated/total_comparable*100:.1f}%")

# 3. Negative controls
print("\n=== 3. Negative Controls ===")
shuffle_wins = 0
signflip_wins = 0
shuffle_flips = 0
signflip_flips = 0
rng = np.random.RandomState(42)

for g in all_groups:
    iou = g["iou"]
    if len(iou) < 2: continue
    loc_r = g["loc_reward"]
    energy = g["energy"]

    # Shuffle energy within group
    en_shuf = energy.copy(); rng.shuffle(en_shuf)
    z_shuf = (en_shuf - en_shuf.mean()) / max(en_shuf.std(), 1e-8)
    r_shuffle = loc_r + BETA * np.clip(-z_shuf, -1, 1)
    best_shuf = r_shuffle.argmax()
    if iou[best_shuf] >= iou.max() - 1e-6: shuffle_wins += 1
    if best_shuf != loc_r.argmax(): shuffle_flips += 1

    # Sign-flip energy
    z_flip = (energy - energy.mean()) / max(energy.std(), 1e-8)
    r_signflip = loc_r + BETA * np.clip(+z_flip, -1, 1)  # reward HIGH energy instead of penalizing
    best_flip = r_signflip.argmax()
    if iou[best_flip] >= iou.max() - 1e-6: signflip_wins += 1
    if best_flip != loc_r.argmax(): signflip_flips += 1

print(f"  Shuffle energy:  selection={shuffle_wins}/{total_comparable}={shuffle_wins/total_comparable*100:.1f}%  flips={shuffle_flips}")
print(f"  Sign-flip energy: selection={signflip_wins}/{total_comparable}={signflip_wins/total_comparable*100:.1f}%  flips={signflip_flips}")
print(f"  (Reference: R_loc only = {loc_wins/total_comparable*100:.1f}%)")

# 4. Borderline-specific analysis
print("\n=== 4. Borderline Group Analysis (max IoU in [0.35, 0.55]) ===")
border_groups = [g for g in all_groups if 0.35 <= g["iou"].max() < 0.55]
print(f"  Borderline groups: {len(border_groups)}")

if border_groups:
    bl_loc = 0; bl_energy = 0; bl_cgated = 0
    for g in border_groups:
        iou = g["iou"]; loc_r = g["loc_reward"]; energy = g["energy"]
        en_mean = energy.mean(); en_std = max(energy.std(), 1e-8)
        z_en = (energy - en_mean) / en_std
        r_cg = loc_r.copy()
        for j in range(len(iou)):
            if 0.35 <= iou[j] < 0.55:
                r_cg[j] += BETA * np.clip(-z_en[j], -1, 1)
        r_en = loc_r + BETA * (-energy)
        if iou[loc_r.argmax()] >= iou.max() - 1e-6: bl_loc += 1
        if iou[r_en.argmax()] >= iou.max() - 1e-6: bl_energy += 1
        if iou[r_cg.argmax()] >= iou.max() - 1e-6: bl_cgated += 1
    print(f"  Best-IoU selection in borderline groups:")
    print(f"    R_loc only:  {bl_loc}/{len(border_groups)} = {bl_loc/len(border_groups)*100:.1f}%")
    print(f"    +energy raw: {bl_energy}/{len(border_groups)} = {bl_energy/len(border_groups)*100:.1f}%")
    print(f"    +C-gated:    {bl_cgated}/{len(border_groups)} = {bl_cgated/len(border_groups)*100:.1f}%")

    # Within borderline groups, does energy correlate with IoU?
    bl_pairs = []
    for g in border_groups:
        for j in range(len(g["iou"])):
            if 0.35 <= g["iou"][j] < 0.55:
                bl_pairs.append((g["iou"][j], g["energy"][j]))
    if len(bl_pairs) >= 2:
        bl_iou = np.array([p[0] for p in bl_pairs])
        bl_en = np.array([p[1] for p in bl_pairs])
        print(f"  Borderline energy-IoU corr: {np.corrcoef(bl_iou, bl_en)[0,1]:+.4f}")

# 5. Summary recommendation
print("\n=== 5. Recommendation ===")
if cgated_wins > loc_wins and cgated_wins > energy_wins:
    print("  C-gated IMPROVES selection -> worth training")
elif cgated_wins < loc_wins:
    print(f"  C-gated does NOT improve selection ({cgated_wins} vs {loc_wins})")
    if shuffle_wins >= loc_wins - 2:
        print("  Shuffle matches loc -> energy has NO causal signal within groups")
    print("  Likely energy correlates with difficulty, NOT with within-group IoU")

# Save
Path("runs/counterfactual_energy.json").write_text(json.dumps({
    "total_groups": len(all_groups),
    "total_comparable": total_comparable,
    "loc_wins": loc_wins, "energy_wins": energy_wins, "cgated_wins": cgated_wins,
    "shuffle_wins": shuffle_wins, "signflip_wins": signflip_wins,
    "flips_loc_vs_energy": flips_loc_vs_energy,
    "flips_loc_vs_cgated": flips_loc_vs_cgated,
    "flips_shuffle": shuffle_flips, "flips_signflip": signflip_flips,
    "group_corrs_mean": np.mean(group_corrs) if group_corrs else None,
    "group_corrs_median": np.median(group_corrs) if group_corrs else None,
    "pct_negative_corr": 100*sum(1 for c in group_corrs if c < 0)/len(group_corrs) if group_corrs else None,
    "bl_loc_wins": bl_loc, "bl_energy_wins": bl_energy, "bl_cgated_wins": bl_cgated,
    "n_border_groups": len(border_groups),
}, indent=2))
print("\nSaved to runs/counterfactual_energy.json")
