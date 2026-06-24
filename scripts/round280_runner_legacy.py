"""Plan 2.80: All 20 groups, unfrozen, seed=42, 5 epoch.

20 groups unified for fair comparison:
  RLVR: ap75_event, select_penalty, grpo_adv_g4, grpo_adv_g8
  Hybrid: per_chan_fft, random_qnorm, frozen_random, aligned_verifier,
          fft_loc_only, grpo_fft_g4, grpo_fft_g8
  AFM:   det_only_frozen, det_only_unfrozen,
          mid06_frozen, mid06_unfrozen,
          apost_frozen, cpost_frozen, phase_frozen

All unfrozen (FPN+RPN+box_head+box_predictor), seed=42 only, 5 epochs.
"""
import sys, json, subprocess, math, copy, shutil
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torchvision.ops import box_iou, nms
import numpy as np
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import save_json, ensure_run_dir
from spectral_detection_posttrain.utils.seed import set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
G_SAMPLES_DEFAULT = 4
EPOCHS = 20
SEEDS = [42, 123, 456]
RL_WEIGHT = 0.05
KL_WEIGHT = 0.1
FFT_WEIGHT = 0.1
ENERGY_WEIGHT = 0.05
HEAD_LR = 0.001
BODY_LR = 0.0001


def extract_perchan_fft(x):
    C = x.shape[1]; H, W = x.shape[-2], x.shape[-1]
    fft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    amp = torch.abs(fft); pha = torch.angle(fft)
    freq_h = torch.fft.fftfreq(H, device=x.device)
    freq_w = torch.fft.rfftfreq(W, device=x.device)
    Y, X = torch.meshgrid(freq_h, freq_w, indexing='ij')
    r = torch.sqrt(X ** 2 + Y ** 2); R = r.max().clamp_min(1e-6); rn = r / R
    lo = (rn <= 0.3).float(); md = ((rn > 0.3) & (rn <= 0.7)).float(); hi = (rn > 0.7).float()
    a_lo = (amp * lo).flatten(2).sum(2); a_md = (amp * md).flatten(2).sum(2); a_hi = (amp * hi).flatten(2).sum(2)
    p_lo = (pha * lo).flatten(2).sum(2); p_md = (pha * md).flatten(2).sum(2); p_hi = (pha * hi).flatten(2).sum(2)
    return torch.cat([a_lo, a_md, a_hi, p_lo, p_md, p_hi], dim=1)


def compute_loc_reward(iou_img):
    r = torch.zeros_like(iou_img)
    r[iou_img >= 0.75] = 1.0
    r[(iou_img >= 0.5) & (iou_img < 0.75)] = 0.3
    r[iou_img < 0.5] = -0.5
    return r


def grpo_advantage(reward):
    r_mean = reward.mean(dim=1, keepdim=True)
    r_std = reward.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (reward - r_mean) / r_std


def glp(d, m, s):
    e = (d - m.unsqueeze(1)) / s.unsqueeze(1)
    return -0.5 * (e.pow(2) + 2 * torch.log(s.unsqueeze(1)) + math.log(2 * math.pi)).sum(dim=-1)


class BaseVerifier(nn.Module):
    def __init__(self, roi_dim, geo_dim=4, hidden=128):
        super().__init__()
        self.roi_net = nn.Sequential(nn.Linear(roi_dim, hidden), nn.ReLU())
        self.geo_net = nn.Sequential(nn.Linear(geo_dim, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden + 32, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, roi_feat, geo_feat):
        return self.head(torch.cat([self.roi_net(roi_feat), self.geo_net(geo_feat)], dim=1)).squeeze(-1)


class FFTResidualVerifier(nn.Module):
    def __init__(self, fft_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fft_dim, hidden), nn.ReLU(), nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, fft_feat): return self.net(fft_feat).squeeze(-1)


class PerChanFFT(nn.Module):
    def __init__(self, fft_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fft_dim, hidden), nn.ReLU(), nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x).squeeze(-1)


class AlignedVerifier(nn.Module):
    def __init__(self, roi_dim, fft_dim, geo_dim=4, hidden=128):
        super().__init__()
        self.roi_net = nn.Sequential(nn.Linear(roi_dim, hidden), nn.ReLU())
        self.fft_net = nn.Sequential(nn.Linear(fft_dim, hidden), nn.ReLU())
        self.geo_net = nn.Sequential(nn.Linear(geo_dim, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden * 2 + 32, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, roi_feat, fft_feat, geo_feat):
        r = self.roi_net(roi_feat); f = self.fft_net(fft_feat); g = self.geo_net(geo_feat)
        return self.head(torch.cat([r, f, g], dim=1)).squeeze(-1)


def unfreeze(model, is_afm=False):
    """Unfreeze FPN+RPN+box_head+box_predictor. Freeze backbone."""
    for p in model.backbone.body.parameters(): p.requires_grad = False
    if hasattr(model.backbone, 'fpn'):
        for p in model.backbone.fpn.parameters(): p.requires_grad = True
    for p in model.rpn.parameters(): p.requires_grad = True
    for p in model.roi_heads.box_head.parameters(): p.requires_grad = True
    for p in model.roi_heads.box_predictor.parameters(): p.requires_grad = True
    if is_afm:
        # Unfreeze AFM gate if present
        for n, p in model.named_parameters():
            if 'afm' in n.lower() or 'gate' in n.lower() or 'mag' in n.lower():
                p.requires_grad = True
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            for p in m.parameters(): p.requires_grad = False


def build_opt(model, vrf=None, vrf2=None):
    body_params, head_params, vrf_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if 'box_head' in n or 'box_predictor' in n: head_params.append(p)
        else: body_params.append(p)
    for v in [vrf, vrf2]:
        if v is not None: vrf_params.extend([p for p in v.parameters() if p.requires_grad])
    return torch.optim.SGD([
        {'params': body_params, 'lr': BODY_LR}, {'params': head_params, 'lr': HEAD_LR},
        {'params': vrf_params, 'lr': HEAD_LR},
    ], lr=HEAD_LR, momentum=0.9, weight_decay=0.0005)


def bl(batch_size=2):
    return build_penn_fudan_loaders({"data": {"root": "./data", "max_size": 320, "train_fraction": 0.8, "num_workers": 0}, "train": {"batch_size": batch_size}})


def bm(afm_type="none", afm_residual_mode="current"):
    is_afm = afm_type != "none"
    return build_detector({"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn", "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn", "pretrained": not is_afm, "num_classes": 2, "min_size": 320, "max_size": 320, "afm_channels": 256 if is_afm else 0, "afm_type": afm_type, "afm_residual_mode": afm_residual_mode}})


@torch.no_grad()
def ev(model, vl):
    model.eval(); ps, ts = [], []
    for img, tgt in vl:
        out = model([i.to(DEV) for i in img])
        ps.extend([{k: v.cpu() for k, v in o.items()} for o in out])
        ts.extend([{k: v.cpu() for k, v in t.items()} for t in tgt])
    return evaluate_detection_predictions(ps, ts, iou_threshold=0.5, score_threshold=0.05)


def run_one(cfg_name, mode, G=G_SAMPLES_DEFAULT, afm_type="none", afm_rm="current", seed=42):
    run_name = f"round280_{cfg_name}_s{seed}"
    set_seed(seed)
    model = bm(afm_type=afm_type, afm_residual_mode=afm_rm).to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    is_afm = afm_type != "none"
    if is_afm:
        model.load_state_dict(ckpt["model"], strict=False)  # AFM layers random init
    else:
        model.load_state_dict(ckpt["model"])
    unfreeze(model, is_afm=is_afm)
    box_pool = model.roi_heads.box_roi_pool

    baseline_model = copy.deepcopy(model); baseline_model.eval()
    for p in baseline_model.parameters(): p.requires_grad = False

    sampled_props, box_head_in, fpn_feats = {}, {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(lambda m, args: box_head_in.update({"x": args[0]}))
    model.backbone.register_forward_hook(lambda m, i, o: fpn_feats.update({"f": {k: o[k] for k in o if k != "pool"}}))

    rng_shuf = torch.Generator(device=DEV).manual_seed(seed + 7777)
    bs = 2 if G == 8 else 4
    tl, vl = bl(batch_size=bs)
    run_dir = ensure_run_dir(run_name)
    shutil.copy(__file__, run_dir / "runner_snapshot.py")

    is_det = mode == "det_only"
    is_afm_only = mode == "afm_only"
    use_grpo = not is_det and not is_afm_only
    needs_verifier = mode in ("fft_loc_only", "grpo_fft")
    needs_aligned = mode == "aligned_verifier"
    needs_perchan = mode == "per_chan_fft"
    use_random = mode == "random_qnorm"
    use_frozen = mode == "frozen_random"
    vrf = None; vrf2 = None
    opt = build_opt(model, vrf, vrf2)

    h = []
    baseline_bbox_w = baseline_model.roi_heads.box_predictor.bbox_pred.weight.detach().clone()
    baseline_bbox_b = baseline_model.roi_heads.box_predictor.bbox_pred.bias.detach().clone()

    for ep in range(1, EPOCHS + 1):
        model.train()
        for v in [vrf, vrf2]:
            if v is not None: v.train()
        td, trl, tv, tkl = 0.0, 0.0, 0.0, 0.0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}"):
            imgs_d = [i.to(DEV) for i in imgs]; tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear(); box_head_in.clear(); fpn_feats.clear()

            ld = model(imgs_d, tgts_t)
            det = sum(ld.values()) if isinstance(ld, dict) else sum(sum(d.values()) for d in ld if isinstance(d, dict))
            rf = box_head_in.get("x"); sp_raw = sampled_props.get("p"); fpn = fpn_feats.get("f")
            rl = torch.tensor(0.0, device=DEV); vloss = torch.tensor(0.0, device=DEV); kl_loss = torch.tensor(0.0, device=DEV)
            total_gn = 0.0

            if use_grpo and rf is not None and sp_raw is not None and rf.shape[0] > 0 and fpn is not None:
                N_rf = rf.shape[0]
                bf = model.roi_heads.box_head(rf)
                mu = model.roi_heads.box_predictor.bbox_pred(bf)[:, -4:]
                curr_w = model.roi_heads.box_predictor.bbox_pred.weight
                curr_b = model.roi_heads.box_predictor.bbox_pred.bias
                kl_loss = KL_WEIGHT * ((curr_w - baseline_bbox_w).pow(2).sum() + (curr_b - baseline_bbox_b).pow(2).sum())

                s = torch.full_like(mu, 0.1, requires_grad=False)
                deltas = mu.detach().unsqueeze(1) + s.unsqueeze(1) * torch.randn(N_rf, G, 4, device=DEV)
                log_probs = glp(deltas, mu, s)

                sp_cat = torch.cat(sp_raw, dim=0); N = min(N_rf, sp_cat.shape[0])
                deltas = deltas[:N]; log_probs = log_probs[:N]

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
                decoded_cat = torch.stack([dx*bw+bcx-0.5*torch.exp(dw)*bw, dy*bh+bcy-0.5*torch.exp(dh)*bh, dx*bw+bcx+0.5*torch.exp(dw)*bw, dy*bh+bcy+0.5*torch.exp(dh)*bh], dim=1).clamp(min=0)
                decoded_list, off = [], 0
                for di in delta_list: n = di.shape[0]; decoded_list.append(decoded_cat[off:off + n]); off += n

                iou_r = torch.zeros(offset, G, device=DEV)
                for pi in range(offset):
                    i_img = img_map[pi * G]
                    gt = tgts_t[i_img]["boxes"]
                    if len(gt) > 0: iou_r[pi] = box_iou(decoded_cat[pi * G:(pi + 1) * G], gt).max(dim=1).values

                # --- Mode-specific reward ---
                reward_img = compute_loc_reward(iou_r)
                use_learned = needs_verifier or needs_aligned or needs_perchan or use_frozen

                if use_learned or needs_verifier:
                    image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]
                    with torch.no_grad(): pooled = box_pool(fpn, decoded_list, image_shapes)
                    fft_f = extract_perchan_fft(pooled)
                    geo = torch.cat([torch.stack([(b[:, 0]+b[:, 2])/(2*image_shapes[i_img][1]), (b[:, 1]+b[:, 3])/(2*image_shapes[i_img][0]), torch.log((b[:, 2]-b[:, 0]).clamp_min(1)), torch.log((b[:, 3]-b[:, 1]).clamp_min(1))], dim=1) for i_img, b in enumerate(decoded_list)], dim=0)

                if needs_aligned:
                    if vrf is None:
                        rd = pooled.shape[1]*pooled.shape[2]*pooled.shape[3]; fd = fft_f.shape[1]
                        vrf = AlignedVerifier(rd, fd).to(DEV); opt = build_opt(model, vrf)
                    q_pred = vrf(pooled.flatten(1), fft_f, geo).view(offset, G)
                    q_target = iou_r.clamp(0, 1)
                    vloss = F.mse_loss(q_pred, q_target.detach())
                    q_norm = (q_pred - q_pred.mean(dim=1, keepdim=True)) / (q_pred.std(dim=1, keepdim=True).clamp_min(1e-6))
                    adv = q_norm
                elif needs_perchan:
                    if vrf is None:
                        vrf = PerChanFFT(fft_f.shape[1]).to(DEV); opt = build_opt(model, vrf)
                    q_pred = vrf(fft_f).view(offset, G)
                    q_target = iou_r.clamp(0, 1)
                    vloss = F.mse_loss(q_pred, q_target.detach())
                    q_norm = (q_pred - q_pred.mean(dim=1, keepdim=True)) / (q_pred.std(dim=1, keepdim=True).clamp_min(1e-6))
                    adv = q_norm
                elif use_frozen:
                    if vrf is None:
                        rd = pooled.shape[1]*pooled.shape[2]*pooled.shape[3]; fd = fft_f.shape[1]
                        vrf = AlignedVerifier(rd, fd).to(DEV)
                        for p in vrf.parameters(): p.requires_grad = False
                        opt = build_opt(model)
                    q_pred = vrf(pooled.flatten(1), fft_f, geo).view(offset, G)
                    q_norm = (q_pred - q_pred.mean(dim=1, keepdim=True)) / (q_pred.std(dim=1, keepdim=True).clamp_min(1e-6))
                    adv = q_norm
                elif use_random:
                    adv = torch.randn(offset, G, generator=rng_shuf, device=DEV)
                    adv = (adv - adv.mean(dim=1, keepdim=True)) / (adv.std(dim=1, keepdim=True).clamp_min(1e-6))
                elif needs_verifier:
                    roi_flat = pooled.flatten(1)
                    fft_shuf = fft_f[torch.randperm(fft_f.shape[0], generator=rng_shuf, device=DEV)]
                    if vrf is None:
                        rd = pooled.shape[1]*pooled.shape[2]*pooled.shape[3]; fd = fft_f.shape[1]
                        vrf = BaseVerifier(rd).to(DEV); vrf2 = FFTResidualVerifier(fd).to(DEV)
                        opt = build_opt(model, vrf, vrf2)
                    q_base = vrf(roi_flat, geo).view(offset, G)
                    q_fft_r = vrf2(fft_f).view(offset, G); q_fft_s = vrf2(fft_shuf).view(offset, G)
                    ft = reward_img.clamp(-1, 1)
                    vloss = F.mse_loss(q_base, ft.detach())
                    vloss = vloss + F.mse_loss(q_fft_r, (ft - q_base.detach()).clamp(-1, 1).detach())
                    him = iou_r.max(dim=1).values > 0.5
                    if him.any(): vloss = vloss + 0.1 * F.relu(0.1 - (q_fft_r[him].mean() - q_fft_s[him].mean()))
                    pos_mask = (iou_r.max(dim=1).values >= 0.5).float().unsqueeze(1)
                    reward_img = reward_img + FFT_WEIGHT * q_fft_r.detach() * pos_mask
                    adv = grpo_advantage(reward_img)
                elif mode == "select_penalty":
                    # Per-image NMS-aware select reward using cls_probs
                    cls_logits = model.roi_heads.box_predictor.cls_score(bf)
                    cls_probs = F.softmax(cls_logits, dim=1)[:, 1]
                    reward_img = compute_loc_reward(iou_r)
                    adv_img = torch.zeros(offset, G, device=DEV)
                    for i_img in range(len(tgts_t)):
                        pis = [pi for pi in range(offset) if img_map[pi * G] == i_img]
                        if not pis: continue
                        dec_img = torch.cat([decoded_cat[pi * G:(pi + 1) * G] for pi in pis], dim=0)
                        scores = torch.cat([cls_probs[pi].repeat(G) for pi in pis], dim=0)
                        keep = nms(dec_img, scores, iou_threshold=0.5)
                        keep_set = set(keep.tolist())
                        for j, pi in enumerate(pis):
                            for g in range(G):
                                idx = j * G + g
                                iou_val = iou_r[pi, g].item()
                                if idx in keep_set:
                                    if iou_val >= 0.75: adv_img[pi, g] = 1.0
                                    elif iou_val >= 0.5: adv_img[pi, g] = 0.3
                                    else: adv_img[pi, g] = -0.7
                                else:
                                    if iou_val >= 0.5: adv_img[pi, g] = -0.3
                                    elif iou_val < 0.3: adv_img[pi, g] = -1.0
                                    else: adv_img[pi, g] = -0.5
                    adv = grpo_advantage(adv_img)
                elif use_grpo:
                    adv = grpo_advantage(reward_img)

                soft_w = iou_r.max(dim=1).values.clamp(0, 1).unsqueeze(1)
                rl = -(adv.detach() * log_probs * soft_w).mean()

            vloss_term = vloss if vrf is not None else torch.tensor(0.0, device=DEV)
            loss = det + vloss_term + RL_WEIGHT * rl + kl_loss
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            td += det.item(); trl += rl.item(); tv += vloss.item(); tkl += kl_loss.item()

        em = ev(model, vl)
        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"], "precision": em.get("precision", 0), "recall": em.get("recall", 0), "ece": em.get("ece", 0)}
        h.append(row)
        print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f}")

    best_h = max(h, key=lambda r: r["val_ap75"])
    em.update({"run_name": run_name, "config": cfg_name, "mode": mode, "seed": seed, "epochs": EPOCHS, "best_ap50": best_h["val_ap50"], "best_ap75": best_h["val_ap75"], "history": h, "git_hash": GIT})
    save_json(em, run_dir / "eval_metrics.json")
    return em


if __name__ == "__main__":
    all_results = []
    configs = [
        # --- RLVR pure ---
        ("ap75_event", "ap75_event", 4),
        ("select_penalty", "select_penalty", 4),
        ("grpo_adv_g4", "grpo_adv", 4),
        ("grpo_adv_g8", "grpo_adv", 8),
        # --- RLVR↔RLHF hybrid ---
        ("per_chan_fft", "per_chan_fft", 4),
        ("random_qnorm", "random_qnorm", 4),
        ("frozen_random", "frozen_random", 4),
        ("aligned_verifier", "aligned_verifier", 4),
        ("fft_loc_only", "fft_loc_only", 4),
        ("grpo_fft_g4", "grpo_fft", 4),
        ("grpo_fft_g8", "grpo_fft", 8),
        # --- det_only baselines ---
        ("det_only_unf", "det_only", 4),
    ]

    for cfg_name, mode, G, *afm_args in configs:
        afm_type = afm_args[0] if afm_args else "none"
        afm_rm = afm_args[1] if len(afm_args) > 1 else "current"
        for s in SEEDS:
            r = run_one(cfg_name, mode, G=G, afm_type=afm_type, afm_rm=afm_rm, seed=s)
            all_results.append(r)

    print("\n## Plan 2.80 All 20 Groups (unfrozen, 3 seeds, 5ep)")
    print(f"  {'Config':<20s} {'Seed':>5s} {'BestAP75':>8s} {'AP50':>8s}")
    for r in all_results:
        print(f"  {r['config']:<20s} {r['seed']:5d} {r['best_ap75']:8.4f} {r['best_ap50']:8.4f}")

    config_names = [c[0] for c in configs]
    for cfg in config_names:
        vals = [r for r in all_results if r["config"] == cfg]
        if not vals: continue
        bv = [r["best_ap75"] for r in vals]
        ap50s = [r["best_ap50"] for r in vals]
        print(f"  {cfg}: bestAP75={np.mean(bv):.4f}±{np.std(bv):.4f}  AP50={np.mean(ap50s):.4f}")
