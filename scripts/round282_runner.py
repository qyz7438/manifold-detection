"""Plan 2.82: C-gated Energy Residual Verifier.
Key insight: energy helps select best IoU in borderline proposals (45.2% vs 33.5%),
but globally it creates perverse PG incentives. C-gate: only penalize high energy
when IoU is borderline (0.35-0.55), as a tie-breaker, not a global penalty.
Formulation: R = R_loc + beta * clip(-z_energy_group, -1, 1) * 1[0.35 <= IoU < 0.55]
"""
import copy
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_mobv3_detector,
    build_penn_fudan_loaders_320,
    build_sgd_optimizer,
    compute_loc_reward,
    decode_boxes,
    evaluate_model,
    extract_perchan_fft,
    gaussian_log_prob,
    grpo_advantage,
    unfreeze_rlvr,
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
BETA = 0.02  # small, per GPT-5.5 recommendation
HEAD_LR = 0.001
BODY_LR = 0.0001
IOU_LO = 0.35
IOU_HI = 0.55


def compute_energy(fft_f):
    ch = fft_f.shape[1] // 6
    a_lo = fft_f[:, 0*ch:1*ch].sum(dim=1)
    a_total = a_lo + fft_f[:, 1*ch:2*ch].sum(dim=1) + fft_f[:, 2*ch:3*ch].sum(dim=1) + 1e-8
    return 2 * (a_lo / a_total) - 1


def run_one(cfg_name, mode, seed):
    run_name = f"round282_{cfg_name}_s{seed}"
    set_seed(seed)
    model = build_mobv3_detector(num_classes=2, pretrained=True).to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"])
    unfreeze_rlvr(model)
    box_pool = model.roi_heads.box_roi_pool

    baseline_model = copy.deepcopy(model)
    baseline_model.eval()
    for p in baseline_model.parameters(): p.requires_grad = False

    sampled_props, box_head_in, fpn_feats = {}, {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(
        lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(
        lambda m, args: box_head_in.update({"x": args[0]}))
    model.backbone.register_forward_hook(
        lambda m, i, o: fpn_feats.update({"f": {k: o[k] for k in o if k != "pool"}}))

    tl, vl = build_penn_fudan_loaders_320(batch_size=4)
    run_dir = ensure_run_dir(run_name)
    shutil.copy(__file__, run_dir / "runner_snapshot.py")

    is_det = mode == "det_only_unf"
    is_cgated = mode == "c_gated"
    is_shuffle = mode == "c_gated_shuffle"
    rng_shuf = torch.Generator(device=DEV).manual_seed(seed + 9999)

    opt = build_sgd_optimizer(model, head_lr=HEAD_LR, body_lr=BODY_LR)
    bbox_pred_weight = model.roi_heads.box_predictor.bbox_pred.weight

    h = []; best_ap75 = -1.0
    diag = {"reward_std": [], "energy_vals": [], "energy_tp": [], "energy_fn": [],
            "n_gated": [], "en_gated_mean": []}

    baseline_bbox_w = baseline_model.roi_heads.box_predictor.bbox_pred.weight.detach().clone()
    baseline_bbox_b = baseline_model.roi_heads.box_predictor.bbox_pred.bias.detach().clone()

    for ep in range(1, EPOCHS + 1):
        model.train()
        td, trl, tkl = 0.0, 0.0, 0.0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}"):
            imgs_d = [i.to(DEV) for i in imgs]
            tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear(); box_head_in.clear(); fpn_feats.clear()

            ld = model(imgs_d, tgts_t)
            det = sum(ld.values()) if isinstance(ld, dict) else sum(sum(d.values()) for d in ld if isinstance(d, dict))

            rf = box_head_in.get("x"); sp_raw = sampled_props.get("p"); fpn = fpn_feats.get("f")
            rl = torch.tensor(0.0, device=DEV)
            kl_loss = torch.tensor(0.0, device=DEV)

            if not is_det and rf is not None and sp_raw is not None and rf.shape[0] > 0 and fpn is not None:
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

                # C-gated energy residual
                if is_cgated or is_shuffle:
                    image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]
                    with torch.no_grad():
                        pooled = box_pool(fpn, decoded_list, image_shapes)
                    fft_f = extract_perchan_fft(pooled)
                    energy = compute_energy(fft_f).view(offset, G_SAMPLES)

                    # Track energy stats
                    max_iou = iou_r.max(dim=1).values
                    tp_mask = max_iou >= 0.5
                    fn_mask = max_iou < 0.5
                    diag["energy_vals"].append(energy.mean().item())
                    if tp_mask.any(): diag["energy_tp"].append(energy[tp_mask].mean().item())
                    if fn_mask.any(): diag["energy_fn"].append(energy[fn_mask].mean().item())

                    # Per-group z-score normalize energy
                    en_mean = energy.mean(dim=1, keepdim=True)
                    en_std = energy.std(dim=1, keepdim=True).clamp_min(1e-6)
                    z_energy = (energy - en_mean) / en_std

                    if is_shuffle:
                        # Shuffle energy within each proposal group (negative control)
                        z_shuf = torch.zeros_like(z_energy)
                        for pi in range(offset):
                            idx = torch.randperm(G_SAMPLES, generator=rng_shuf, device=DEV)
                            z_shuf[pi] = z_energy[pi][idx]
                        z_energy = z_shuf

                    # Per-group C-gate: gate on max IoU, not per-sample IoU
                    # (fix: per-sample gate was mismatched with counterfactual evidence)
                    group_max_iou = iou_r.max(dim=1).values  # (offset,)
                    border_group_mask = (group_max_iou >= IOU_LO) & (group_max_iou < IOU_HI)  # (offset,)

                    n_gated = border_group_mask.sum().item()
                    diag["n_gated"].append(n_gated * G_SAMPLES)
                    if border_group_mask.any():
                        diag["en_gated_mean"].append(energy[border_group_mask].mean().item())

                # GRPO on R_loc only (not R_loc + energy)
                adv = grpo_advantage(reward_img)

                # Energy as residual advantage bias AFTER GRPO
                # (fix: BETA before GRPO normalizes away when all G samples have same R_loc)
                if is_cgated or is_shuffle:
                    gated_bias = z_energy.new_zeros(offset, G_SAMPLES)
                    gated_bias[border_group_mask] = torch.clamp(-z_energy[border_group_mask], -1, 1)
                    # gated_bias mean over G samples is ~0 per group, so it acts as tie-breaker
                    adv = adv + BETA * gated_bias
                diag["reward_std"].append(adv.std().item())
                soft_w = iou_r.max(dim=1).values.clamp(0, 1).unsqueeze(1)
                rl = -(adv.detach() * log_probs * soft_w).mean()

            loss = det + RL_WEIGHT * rl + kl_loss
            opt.zero_grad(set_to_none=True); loss.backward()
            opt.step()

            td += det.item(); trl += rl.item(); tkl += kl_loss.item()

        em = evaluate_model(model, vl, DEV, iou_threshold=0.5, score_threshold=0.05)
        rs_m = np.mean(diag["reward_std"]) if diag["reward_std"] else 0.0
        en_m = np.mean(diag["energy_vals"]) if diag["energy_vals"] else 0.0
        en_tp = np.mean(diag["energy_tp"]) if diag["energy_tp"] else float("nan")
        en_fn = np.mean(diag["energy_fn"]) if diag["energy_fn"] else float("nan")
        en_gap = en_tp - en_fn if not np.isnan(en_tp) and not np.isnan(en_fn) else float("nan")
        ng = np.sum(diag["n_gated"]) if diag["n_gated"] else 0

        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "precision": em.get("precision", 0), "recall": em.get("recall", 0),
               "ece": em.get("ece", 0), "reward_std": float(rs_m),
               "energy_mean": float(en_m), "energy_tp": float(en_tp), "energy_fn": float(en_fn),
               "energy_gap": float(en_gap), "n_gated": int(ng),
               "det_loss": td, "rl_loss": trl, "kl_loss": tkl}
        h.append(row)
        summary = (f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} "
                   f"r_std={rs_m:.4f} en_gap={en_gap:.4f} n_gated={ng}")
        if diag["en_gated_mean"]:
            summary += f" en_gated={np.mean(diag['en_gated_mean']):.4f}"
        print(summary)
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
    configs = {
        "det_only_unf": "det_only_unf",
        "c_gated": "c_gated",
        "c_gated_shuffle": "c_gated_shuffle",
    }
    for cfg, mode in configs.items():
        for s in SEEDS:
            r = run_one(cfg, mode, s)
            all_results.append(r)

    print("\n## Plan 2.82 C-gated Energy Residual Verifier")
    print(f"  {'Config':<18s} {'Seed':>5s} {'AP75':>8s} {'BestAP75':>8s} {'AP50':>8s} {'r_std':>8s} {'en_gap':>8s} {'n_gated':>8s}")
    for r in all_results:
        best_h = max(r["history"], key=lambda x: x["val_ap75"])
        print(f"  {r['config']:<18s} {r['seed']:5d} {r['ap75']:8.4f} {r['best_ap75']:8.4f} "
              f"{best_h['val_ap50']:8.4f} {best_h.get('reward_std', 0):8.4f} "
              f"{best_h.get('energy_gap', 0):8.4f} {best_h.get('n_gated', 0):8d}")

    for cfg in configs:
        vals = [r for r in all_results if r["config"] == cfg]
        if not vals: continue
        bv = [r["best_ap75"] for r in vals]
        print(f"  {cfg}: bestAP75={np.mean(bv):.4f} +/- {np.std(bv):.4f}")
