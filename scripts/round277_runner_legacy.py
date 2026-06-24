"""Plan 2.77: GRPO + KL Anchor — G=8 clean test.

Per GPT-5.5 audit of 2.76:
  - G=8: most direct way to improve GRPO group-advantage stability
  - Keep KL anchor (det_kl=0.716 proved KL doesn't harm baseline)
  - Keep RL_WEIGHT=0.05 (grpo_adv already below baseline, don't amplify noise)
  - Add FFT diagnostics: reward hit rate, IoU>=0.5 ratio

2 groups × 3 seeds = 6 experiments:
  A  grpo_adv    GRPO group advantage, G=8, no verifier, IoU reward
  B  grpo_fft    GRPO + FFT loc-only residual, G=8

Plus det_only baseline for reference (from 2.76, same deterministic setup).

Pass: grpo_fft - grpo_adv >= +0.01 AND >= 2/3 wins AND grpo_fft >= 0.721 (det_only+0.005)
"""
import sys, json, subprocess, math, copy, shutil
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torchvision.ops import box_iou
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
G_SAMPLES = 8
EPOCHS = 8
SEEDS = [42, 123, 456]
RL_WEIGHT = 0.05
KL_WEIGHT = 0.1
FFT_WEIGHT = 0.1
DET_ONLY_BEST = 0.7158  # from 2.76, for pass criterion


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

    def forward(self, fft_feat):
        return self.net(fft_feat).squeeze(-1)


def glp(d, m, s):
    e = (d - m.unsqueeze(1)) / s.unsqueeze(1)
    return -0.5 * (e.pow(2) + 2 * torch.log(s.unsqueeze(1)) + math.log(2 * math.pi)).sum(dim=-1)


def bl():
    return build_penn_fudan_loaders({"data": {"root": "./data", "max_size": 320, "train_fraction": 0.8, "num_workers": 0}, "train": {"batch_size": 2}})


def bm():
    return build_detector({"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn", "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn", "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}})


def fe(m, parts):
    for p in m.parameters():
        p.requires_grad = False
    for part in parts:
        if isinstance(part, nn.Module):
            for p in part.parameters():
                p.requires_grad = True


@torch.no_grad()
def ev(model, vl):
    model.eval()
    ps, ts = [], []
    for img, tgt in vl:
        out = model([i.to(DEV) for i in img])
        ps.extend([{k: v.cpu() for k, v in o.items()} for o in out])
        ts.extend([{k: v.cpu() for k, v in t.items()} for t in tgt])
    return evaluate_detection_predictions(ps, ts, iou_threshold=0.5, score_threshold=0.05)


def run_one(cfg_name, mode, seed):
    run_name = f"round277_{cfg_name}_s{seed}"
    set_seed(seed)
    model = bm().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"])
    fe(model, [model.roi_heads.box_head, model.roi_heads.box_predictor])
    box_pool = model.roi_heads.box_roi_pool

    baseline_model = copy.deepcopy(model)
    baseline_model.eval()
    for p in baseline_model.parameters():
        p.requires_grad = False

    sampled_props, box_head_in, fpn_feats = {}, {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(
        lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(
        lambda m, args: box_head_in.update({"x": args[0]}))
    model.backbone.register_forward_hook(
        lambda m, i, o: fpn_feats.update({"f": {k: o[k] for k in o if k != "pool"}}))

    rng_shuf = torch.Generator(device=DEV).manual_seed(seed + 7777)
    tl, vl = bl()
    run_dir = ensure_run_dir(run_name)

    needs_fft = mode == "grpo_fft"
    vrf_base = None; vrf_fft = None

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    bbox_pred_weight = model.roi_heads.box_predictor.bbox_pred.weight

    h = []; best_ap75 = -1.0
    diag = {"q_ious": [], "total_grad_norm": [], "reward_std": [],
            "fft_hit_rate": [], "iou50_ratio": [], "fft_reward_mean": []}

    baseline_bbox_w = baseline_model.roi_heads.box_predictor.bbox_pred.weight.detach().clone()
    baseline_bbox_b = baseline_model.roi_heads.box_predictor.bbox_pred.bias.detach().clone()

    for ep in range(1, EPOCHS + 1):
        model.train()
        for v in [vrf_base, vrf_fft]:
            if v is not None: v.train()
        td, trl, tv, tkl, pos = 0.0, 0.0, 0.0, 0.0, 0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}"):
            imgs_d = [i.to(DEV) for i in imgs]
            tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear(); box_head_in.clear(); fpn_feats.clear()

            ld = model(imgs_d, tgts_t)
            det = sum(ld.values()) if isinstance(ld, dict) else sum(sum(d.values()) for d in ld if isinstance(d, dict))

            rf = box_head_in.get("x"); sp_raw = sampled_props.get("p"); fpn = fpn_feats.get("f")
            rl = torch.tensor(0.0, device=DEV); vloss = torch.tensor(0.0, device=DEV)
            kl_loss = torch.tensor(0.0, device=DEV); total_gn_batch = 0.0

            if rf is not None and sp_raw is not None and rf.shape[0] > 0 and fpn is not None:
                N_rf = rf.shape[0]
                bf = model.roi_heads.box_head(rf)
                mu = model.roi_heads.box_predictor.bbox_pred(bf)[:, -4:]

                # KL
                curr_w = model.roi_heads.box_predictor.bbox_pred.weight
                curr_b = model.roi_heads.box_predictor.bbox_pred.bias
                kl_loss = KL_WEIGHT * ((curr_w - baseline_bbox_w).pow(2).sum() + (curr_b - baseline_bbox_b).pow(2).sum())

                s = torch.full_like(mu, 0.1, requires_grad=False)
                deltas = mu.detach().unsqueeze(1) + s.unsqueeze(1) * torch.randn(N_rf, G_SAMPLES, 4, device=DEV)
                log_probs = glp(deltas, mu, s)

                sp_cat = torch.cat(sp_raw, dim=0); N = min(N_rf, sp_cat.shape[0])
                mu = mu[:N]; deltas = deltas[:N]; log_probs = log_probs[:N]

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

                iou_r = torch.zeros(offset, G_SAMPLES, device=DEV)
                for pi in range(offset):
                    i_img = img_map[pi * G_SAMPLES]
                    gt = tgts_t[i_img]["boxes"]
                    if len(gt) > 0:
                        iou_r[pi] = box_iou(decoded_cat[pi * G_SAMPLES:(pi + 1) * G_SAMPLES], gt).max(dim=1).values

                reward_img = compute_loc_reward(iou_r)

                if needs_fft:
                    image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]
                    with torch.no_grad():
                        pooled = box_pool(fpn, decoded_list, image_shapes)
                    roi_flat = pooled.flatten(1)
                    fft_f = extract_perchan_fft(pooled)
                    fft_shuf = fft_f[torch.randperm(fft_f.shape[0], generator=rng_shuf, device=DEV)]
                    geo = torch.cat([
                        torch.stack([
                            (b[:, 0] + b[:, 2]) / (2 * image_shapes[i_img][1]),
                            (b[:, 1] + b[:, 3]) / (2 * image_shapes[i_img][0]),
                            torch.log((b[:, 2] - b[:, 0]).clamp_min(1)),
                            torch.log((b[:, 3] - b[:, 1]).clamp_min(1)),
                        ], dim=1) for i_img, b in enumerate(decoded_list)], dim=0)

                    if vrf_base is None:
                        roi_dim = pooled.shape[1] * pooled.shape[2] * pooled.shape[3]
                        vrf_base = BaseVerifier(roi_dim).to(DEV)
                        vrf_fft = FFTResidualVerifier(fft_f.shape[1]).to(DEV)
                        params = [p for p in list(model.parameters()) + list(vrf_base.parameters()) + list(vrf_fft.parameters()) if p.requires_grad]
                        opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

                    q_base = vrf_base(roi_flat, geo).view(offset, G_SAMPLES)
                    q_fft_real = vrf_fft(fft_f).view(offset, G_SAMPLES)
                    q_fft_shuf = vrf_fft(fft_shuf).view(offset, G_SAMPLES)

                    fft_target = reward_img.clamp(-1, 1)
                    vloss = F.mse_loss(q_base, fft_target.detach())
                    residual_target = (fft_target - q_base.detach()).clamp(-1, 1)
                    vloss = vloss + F.mse_loss(q_fft_real, residual_target.detach())
                    high_iou_mask = iou_r.max(dim=1).values > 0.5
                    if high_iou_mask.any():
                        vloss = vloss + 0.1 * F.relu(0.1 - (q_fft_real[high_iou_mask].mean() - q_fft_shuf[high_iou_mask].mean()))

                    # FFT diagnostics
                    iou50_mask = iou_r.max(dim=1).values >= 0.5
                    diag["iou50_ratio"].append(iou50_mask.float().mean().item())
                    diag["fft_reward_mean"].append(q_fft_real[iou50_mask].mean().item() if iou50_mask.any() else 0.0)
                    diag["fft_hit_rate"].append((q_fft_real.abs() > 0.1).float().mean().item())

                    pos_mask = iou50_mask.float().unsqueeze(1)
                    reward_img = reward_img + FFT_WEIGHT * q_fft_real.detach() * pos_mask

                    diag["q_ious"].extend(list(zip(q_fft_real.flatten().tolist(), iou_r.flatten().tolist())))

                adv = grpo_advantage(reward_img)
                diag["reward_std"].append(adv.std().item())
                soft_w = iou_r.max(dim=1).values.clamp(0, 1).unsqueeze(1)
                rl = -(adv.detach() * log_probs * soft_w).mean()
                pos += (soft_w > 0.3).sum().item()

            vloss_term = vloss if vrf_base is not None else torch.tensor(0.0, device=DEV)
            loss = det + vloss_term + RL_WEIGHT * rl + kl_loss
            opt.zero_grad(set_to_none=True); loss.backward()
            total_gn_batch = bbox_pred_weight.grad.norm().item() if bbox_pred_weight.grad is not None else 0.0
            opt.step()

            diag["total_grad_norm"].append(total_gn_batch)
            td += det.item(); trl += rl.item(); tv += vloss.item(); tkl += kl_loss.item()

        em = ev(model, vl)
        q_corr = 0.0
        if len(diag["q_ious"]) > 1:
            qs = np.array([x[0] for x in diag["q_ious"]]); iis = np.array([x[1] for x in diag["q_ious"]])
            q_corr = np.corrcoef(qs, iis)[0, 1]

        tgn = np.mean(diag["total_grad_norm"]) if diag["total_grad_norm"] else 0.0
        rs_m = np.mean(diag["reward_std"]) if diag["reward_std"] else 0.0
        fft_hr = np.mean(diag["fft_hit_rate"]) if diag["fft_hit_rate"] else 0.0
        iou50_r = np.mean(diag["iou50_ratio"]) if diag["iou50_ratio"] else 0.0
        fft_rm = np.mean(diag["fft_reward_mean"]) if diag["fft_reward_mean"] else 0.0

        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "precision": em.get("precision", 0), "recall": em.get("recall", 0),
               "ece": em.get("ece", 0), "q_iou_corr": float(q_corr),
               "reward_std": float(rs_m), "total_grad_norm": float(tgn),
               "fft_hit_rate": float(fft_hr), "iou50_ratio": float(iou50_r),
               "fft_reward_mean": float(fft_rm),
               "det_loss": td, "rl_loss": trl, "vloss": tv, "kl_loss": tkl, "pos_count": int(pos)}
        h.append(row)
        print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} "
              f"q_corr={q_corr:.4f} tgn={tgn:.6f} r_std={rs_m:.4f} "
              f"FFThit={fft_hr:.3f} IoU50={iou50_r:.3f} FFTmu={fft_rm:.4f}")
        if em["ap75"] > best_ap75: best_ap75 = em["ap75"]
        for k in diag: diag[k].clear()

    best_h = max(h, key=lambda r: r["val_ap75"])
    em.update({"run_name": run_name, "config": cfg_name, "mode": mode, "seed": seed,
               "epochs": len(h), "best_ap50": best_h["val_ap50"], "best_ap75": best_ap75,
               "history": h, "git_hash": GIT})
    save_json(em, run_dir / "eval_metrics.json")
    return em


if __name__ == "__main__":
    all_results = []
    configs = {"grpo_adv": "grpo_adv", "grpo_fft": "grpo_fft"}
    for cfg, mode in configs.items():
        for s in SEEDS:
            r = run_one(cfg, mode, s)
            all_results.append(r)

    print("\n## Plan 2.77 GRPO G=8")
    print(f"  {'Config':<10s} {'Seed':>5s} {'AP75':>8s} {'BestAP75':>8s} {'AP50':>8s} {'q_corr':>8s} {'r_std':>8s} {'FFThit':>8s} {'IoU50':>8s} {'FFTmu':>8s}")
    for r in all_results:
        best_h = max(r["history"], key=lambda x: x["val_ap75"])
        print(f"  {r['config']:<10s} {r['seed']:5d} {r['ap75']:8.4f} {r['best_ap75']:8.4f} {best_h['val_ap50']:8.4f} {best_h.get('q_iou_corr', 0):8.4f} {best_h.get('reward_std', 0):8.4f} {best_h.get('fft_hit_rate', 0):8.4f} {best_h.get('iou50_ratio', 0):8.4f} {best_h.get('fft_reward_mean', 0):8.4f}")

    for cfg in configs:
        vals = [r for r in all_results if r["config"] == cfg]
        if not vals: continue
        bv = [r["best_ap75"] for r in vals]; fv = [r["ap75"] for r in vals]
        ap50s = [max(r["history"], key=lambda x: x["val_ap75"])["val_ap50"] for r in vals]
        print(f"  {cfg}: bestAP75={np.mean(bv):.4f}±{np.std(bv):.4f}  finalAP75={np.mean(fv):.4f}  AP50={np.mean(ap50s):.4f}")

    va = [r["best_ap75"] for r in all_results if r["config"] == "grpo_fft"]
    vb = [r["best_ap75"] for r in all_results if r["config"] == "grpo_adv"]
    if va and vb:
        delta = np.mean(va) - np.mean(vb)
        wins = sum(1 for i in range(3) if va[i] > vb[i])
        pass1 = delta >= 0.01 and wins >= 2
        pass2 = np.mean(va) >= DET_ONLY_BEST + 0.005
        status = "PASS" if pass1 and pass2 else "FAIL"
        print(f"\n  grpo_fft vs grpo_adv: Δ={delta:+.4f} wins={wins}/3")
        print(f"  grpo_fft vs det_only({DET_ONLY_BEST}): Δ={np.mean(va)-DET_ONLY_BEST:+.4f}")
        print(f"  → {status} (require Δ>=0.01 AND >=2/3 AND >=det_only+0.005)")
