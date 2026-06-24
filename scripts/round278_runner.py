"""Plan 2.78: Direct FFT Reward — energy + similarity + phase consistency.

Three FFT-based reward terms added to R_loc (no learned verifier):
  1. R_energy: low-frequency energy concentration = 2*amp_low/total - 1
     → reward for "box contains structured object"
  2. R_sim: cosine similarity to group-mean FFT profile within each proposal
     → reward for "frequency profile consistency across G samples"
  3. R_phase: phase consistency = -clamp(phase_dist_to_group_mean, 0, 1)
     → penalty for phase deviation from group mean (not circular phase stat)

GRPO + KL anchor (from 2.76), G=4, 3 seeds.energy/sim/phase weights=0.05/0.05/0.02.
"""
import copy
import shutil
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_mobv3_detector,
    build_penn_fudan_loaders_320,
    decode_boxes,
    evaluate_model,
    extract_perchan_fft,
    gaussian_log_prob,
    grpo_advantage,
)
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
G_SAMPLES = 4
EPOCHS = 8
SEEDS = [42, 123, 456]
RL_WEIGHT = 0.05
KL_WEIGHT = 0.1
ENERGY_WEIGHT = 0.05
SIM_WEIGHT = 0.05
PHASE_WEIGHT = 0.02


def compute_energy_reward(fft_f):
    """R_energy: low-frequency energy concentration. Higher = more structured object.
    fft_f: (N*G, 6*C) — per-channel 3-band amp+phase features.
    Returns: (N*G,) reward in [-1, 1].
    """
    ch_per_band = fft_f.shape[1] // 6  # 3 amp + 3 phase
    a_lo = fft_f[:, 0 * ch_per_band:1 * ch_per_band].sum(dim=1)  # low freq amp
    a_md = fft_f[:, 1 * ch_per_band:2 * ch_per_band].sum(dim=1)  # mid freq amp
    a_hi = fft_f[:, 2 * ch_per_band:3 * ch_per_band].sum(dim=1)  # high freq amp
    a_total = a_lo + a_md + a_hi + 1e-8
    low_ratio = a_lo / a_total  # [0, 1], higher = more structured
    return 2 * low_ratio - 1  # map to [-1, 1]


def compute_similarity_reward(fft_f, N, G):
    """R_sim: cosine similarity to best-IoU sample within each proposal group.
    fft_f: (N*G, D). Returns: (N, G) reward in [-1, 1].
    """
    fft_r = fft_f.view(N, G, -1)  # (N, G, D)
    # Norm each sample
    fft_n = F.normalize(fft_r, dim=-1)  # (N, G, D)
    # Mean vector as reference (or use most consistent pair)
    ref = fft_n.mean(dim=1, keepdim=True)  # (N, 1, D)
    sim = (fft_n * ref).sum(dim=-1)  # (N, G), cosine sim to mean
    return sim


def compute_phase_reward(fft_f, N, G):
    """R_phase: phase variance across G samples. Lower variance = more consistent.
    fft_f: (N*G, 6*C). Uses phase bands (last 3 of 6).
    Returns: (N, G) reward in [-1, 1].
    """
    ch_per_band = fft_f.shape[1] // 6
    # Extract phase features (bands 3-5)
    p_feat = fft_f[:, 3 * ch_per_band:]  # (N*G, 3*C)
    p_r = p_feat.view(N, G, -1)  # (N, G, 3*C)
    # Phase variance within group
    p_mean = p_r.mean(dim=1, keepdim=True)  # (N, 1, 3*C)
    p_dist = ((p_r - p_mean) ** 2).sum(dim=-1)  # (N, G), distance to group mean
    # Normalize: max reward when close to mean, penalty when far
    p_dist_norm = p_dist / (p_dist.mean(dim=1, keepdim=True).clamp_min(1e-6) + 1e-8)
    return -p_dist_norm.clamp(-1, 1)  # negative distance → reward for consistency


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


def bl():
    return build_penn_fudan_loaders_320(batch_size=2)


def bm():
    return build_mobv3_detector(num_classes=2, pretrained=True)


def fe(m, parts):
    for p in m.parameters():
        p.requires_grad = False
    for part in parts:
        if isinstance(part, nn.Module):
            for p in part.parameters():
                p.requires_grad = True


@torch.no_grad()
def ev(model, vl):
    return evaluate_model(model, vl, DEV, iou_threshold=0.5, score_threshold=0.05)


def run_one(cfg_name, seed):
    run_name = f"round278_{cfg_name}_s{seed}"
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

    tl, vl = bl()
    run_dir = ensure_run_dir(run_name)
    shutil.copy(__file__, run_dir / "runner_snapshot.py")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    bbox_pred_weight = model.roi_heads.box_predictor.bbox_pred.weight

    h = []; best_ap75 = -1.0
    diag = {"total_grad_norm": [], "reward_std": [],
            "energy_mean": [], "sim_mean": [], "phase_mean": []}

    baseline_bbox_w = baseline_model.roi_heads.box_predictor.bbox_pred.weight.detach().clone()
    baseline_bbox_b = baseline_model.roi_heads.box_predictor.bbox_pred.bias.detach().clone()

    for ep in range(1, EPOCHS + 1):
        model.train()
        td, trl, tkl, pos = 0.0, 0.0, 0.0, 0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}"):
            imgs_d = [i.to(DEV) for i in imgs]
            tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear(); box_head_in.clear(); fpn_feats.clear()

            ld = model(imgs_d, tgts_t)
            det = sum(ld.values()) if isinstance(ld, dict) else sum(sum(d.values()) for d in ld if isinstance(d, dict))

            rf = box_head_in.get("x"); sp_raw = sampled_props.get("p"); fpn = fpn_feats.get("f")
            rl = torch.tensor(0.0, device=DEV); total_gn_batch = 0.0

            if rf is not None and sp_raw is not None and rf.shape[0] > 0 and fpn is not None:
                N_rf = rf.shape[0]
                bf = model.roi_heads.box_head(rf)
                mu = model.roi_heads.box_predictor.bbox_pred(bf)[:, -4:]

                curr_w = model.roi_heads.box_predictor.bbox_pred.weight
                curr_b = model.roi_heads.box_predictor.bbox_pred.bias
                kl_loss = KL_WEIGHT * ((curr_w - baseline_bbox_w).pow(2).sum() + (curr_b - baseline_bbox_b).pow(2).sum())

                s = torch.full_like(mu, 0.1, requires_grad=False)
                deltas = mu.detach().unsqueeze(1) + s.unsqueeze(1) * torch.randn(N_rf, G_SAMPLES, 4, device=DEV)
                log_probs = gaussian_log_prob(deltas, mu, s)

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
                decoded_cat = decode_boxes(sp_exp, delta_cat)

                decoded_list, off = [], 0
                for di in delta_list:
                    n = di.shape[0]; decoded_list.append(decoded_cat[off:off + n]); off += n

                iou_r = torch.zeros(offset, G_SAMPLES, device=DEV)
                for pi in range(offset):
                    i_img = img_map[pi * G_SAMPLES]
                    gt = tgts_t[i_img]["boxes"]
                    if len(gt) > 0:
                        iou_r[pi] = box_iou(decoded_cat[pi * G_SAMPLES:(pi + 1) * G_SAMPLES], gt).max(dim=1).values

                # R_loc
                reward_img = compute_loc_reward(iou_r)

                # FFT-based reward terms
                image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]
                with torch.no_grad():
                    pooled = box_pool(fpn, decoded_list, image_shapes)
                fft_f = extract_perchan_fft(pooled)  # (N*G, 6*C)

                # 1. R_energy: low-frequency concentration
                r_energy = ENERGY_WEIGHT * compute_energy_reward(fft_f).view(offset, G_SAMPLES)
                diag["energy_mean"].append(r_energy.abs().mean().item())

                # 2. R_sim: cosine similarity within group
                r_sim = SIM_WEIGHT * compute_similarity_reward(fft_f, offset, G_SAMPLES)
                diag["sim_mean"].append(r_sim.abs().mean().item())

                # 3. R_phase: phase consistency
                r_phase = PHASE_WEIGHT * compute_phase_reward(fft_f, offset, G_SAMPLES)
                diag["phase_mean"].append(r_phase.abs().mean().item())

                # Combined reward
                reward_img = reward_img + r_energy + r_sim + r_phase

                # GRPO advantage
                adv = grpo_advantage(reward_img)
                diag["reward_std"].append(adv.std().item())

                soft_w = iou_r.max(dim=1).values.clamp(0, 1).unsqueeze(1)
                rl = -(adv.detach() * log_probs * soft_w).mean()
                pos += (soft_w > 0.3).sum().item()

            loss = det + RL_WEIGHT * rl + kl_loss
            opt.zero_grad(set_to_none=True); loss.backward()
            total_gn_batch = bbox_pred_weight.grad.norm().item() if bbox_pred_weight.grad is not None else 0.0
            opt.step()

            diag["total_grad_norm"].append(total_gn_batch)
            td += det.item(); trl += rl.item(); tkl += kl_loss.item()

        em = ev(model, vl)
        tgn = np.mean(diag["total_grad_norm"]) if diag["total_grad_norm"] else 0.0
        rs_m = np.mean(diag["reward_std"]) if diag["reward_std"] else 0.0
        en_m = np.mean(diag["energy_mean"]) if diag["energy_mean"] else 0.0
        sm_m = np.mean(diag["sim_mean"]) if diag["sim_mean"] else 0.0
        ph_m = np.mean(diag["phase_mean"]) if diag["phase_mean"] else 0.0

        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "precision": em.get("precision", 0), "recall": em.get("recall", 0),
               "ece": em.get("ece", 0), "reward_std": float(rs_m), "total_grad_norm": float(tgn),
               "energy_reward": float(en_m), "sim_reward": float(sm_m), "phase_reward": float(ph_m),
               "det_loss": td, "rl_loss": trl, "kl_loss": tkl, "pos_count": int(pos)}
        h.append(row)
        print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} "
              f"tgn={tgn:.6f} r_std={rs_m:.4f} En={en_m:.4f} Sim={sm_m:.4f} Ph={ph_m:.4f}")
        if em["ap75"] > best_ap75: best_ap75 = em["ap75"]
        for k in diag: diag[k].clear()

    best_h = max(h, key=lambda r: r["val_ap75"])
    em.update({"run_name": run_name, "config": cfg_name, "seed": seed,
               "epochs": len(h), "best_ap50": best_h["val_ap50"], "best_ap75": best_ap75,
               "history": h, "git_hash": GIT})
    save_json(em, run_dir / "eval_metrics.json")
    return em


if __name__ == "__main__":
    all_results = []
    configs = {"fft_reward": "fft_reward"}
    for cfg in configs:
        for s in SEEDS:
            r = run_one(cfg, s)
            all_results.append(r)

    print("\n## Plan 2.78 Direct FFT Reward")
    print(f"  {'Config':<14s} {'Seed':>5s} {'AP75':>8s} {'BestAP75':>8s} {'AP50':>8s} {'r_std':>8s} {'En':>8s} {'Sim':>8s} {'Ph':>8s}")
    for r in all_results:
        best_h = max(r["history"], key=lambda x: x["val_ap75"])
        print(f"  {r['config']:<14s} {r['seed']:5d} {r['ap75']:8.4f} {r['best_ap75']:8.4f} {best_h['val_ap50']:8.4f} {best_h.get('reward_std', 0):8.4f} {best_h.get('energy_reward', 0):8.4f} {best_h.get('sim_reward', 0):8.4f} {best_h.get('phase_reward', 0):8.4f}")

    vals = [r for r in all_results if r["config"] == "fft_reward"]
    bv = [r["best_ap75"] for r in vals]
    print(f"\n  fft_reward: bestAP75={np.mean(bv):.4f}±{np.std(bv):.4f}")
