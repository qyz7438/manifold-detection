"""Plan 2.79: Unfrozen FPN + RPN + box_head — RLVR validation.

Per GPT-5.5 audit: current RLVR is head-only (~100K params). Must verify if
unfreezing representation layers (FPN+RPN+box_head, ~2M params) changes
the RL signal before scaling to full backbone.

5 groups × 3 seeds = 15 experiments:
  A  det_only_unf    supervised fine-tune baseline (unfrozen)
  B  ap75_event      pure IoU discrete reward + GRPO G=4
  C  grpo_adv        GRPO + KL + IoU reward G=4
  D  grpo_fft        GRPO + KL + FFT loc-only G=4 (best from 2.76)
  E  fft_energy      GRPO + KL + energy reward G=4

Unfreeze: FPN + RPN head + box_head + box_predictor
Frozen: backbone (MobileNetV3)
Low LR: 0.0001 for FPN/RPN, 0.001 for box_head/verifier
"""
import copy
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
from spectral_detection_posttrain.models.verifiers import (
    BaseVerifier,
    FFTResidualVerifier,
    build_geo_features,
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
FFT_WEIGHT = 0.1
ENERGY_WEIGHT = 0.05
HEAD_LR = 0.001
BODY_LR = 0.0001


def compute_energy_reward(fft_f):
    ch = fft_f.shape[1] // 6
    a_lo = fft_f[:, 0 * ch:1 * ch].sum(dim=1)
    a_md = fft_f[:, 1 * ch:2 * ch].sum(dim=1)
    a_hi = fft_f[:, 2 * ch:3 * ch].sum(dim=1)
    low_ratio = a_lo / (a_lo + a_md + a_hi + 1e-8)
    return 2 * low_ratio - 1


def run_one(cfg_name, mode, seed):
    run_name = f"round279_{cfg_name}_s{seed}"
    set_seed(seed)
    model = build_mobv3_detector(num_classes=2, pretrained=True).to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"])
    unfreeze_rlvr(model)
    box_pool = model.roi_heads.box_roi_pool

    baseline_model = copy.deepcopy(model)
    baseline_model.eval()
    for p in baseline_model.parameters():
        p.requires_grad = False

    sampled_props, box_head_in, fpn_feats = {}, {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(
        lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]})
    )
    model.roi_heads.box_head.register_forward_pre_hook(
        lambda m, args: box_head_in.update({"x": args[0]})
    )
    model.backbone.register_forward_hook(
        lambda m, i, o: fpn_feats.update({"f": {k: o[k] for k in o if k != "pool"}})
    )

    rng_shuf = torch.Generator(device=DEV).manual_seed(seed + 7777)
    tl, vl = build_penn_fudan_loaders_320(batch_size=4)
    run_dir = ensure_run_dir(run_name)
    shutil.copy(__file__, run_dir / "runner_snapshot.py")

    needs_verifier = mode == "grpo_fft"
    use_energy = mode == "fft_energy"
    is_det = mode == "det_only_unf"
    use_grpo = not is_det
    vrf_base = None
    vrf_fft = None

    opt = build_sgd_optimizer(model, head_lr=HEAD_LR, body_lr=BODY_LR, extra_modules=[vrf_base, vrf_fft])
    bbox_pred_weight = model.roi_heads.box_predictor.bbox_pred.weight

    h = []
    best_ap75 = -1.0
    diag = {"total_grad_norm": [], "reward_std": [], "q_ious": [], "energy_mean": []}

    baseline_bbox_w = baseline_model.roi_heads.box_predictor.bbox_pred.weight.detach().clone()
    baseline_bbox_b = baseline_model.roi_heads.box_predictor.bbox_pred.bias.detach().clone()

    for ep in range(1, EPOCHS + 1):
        model.train()
        for v in [vrf_base, vrf_fft]:
            if v is not None:
                v.train()
        td, trl, tv, tkl, pos = 0.0, 0.0, 0.0, 0.0, 0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}"):
            imgs_d = [i.to(DEV) for i in imgs]
            tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear()
            box_head_in.clear()
            fpn_feats.clear()

            ld = model(imgs_d, tgts_t)
            det = sum(ld.values()) if isinstance(ld, dict) else sum(sum(d.values()) for d in ld if isinstance(d, dict))

            rf = box_head_in.get("x")
            sp_raw = sampled_props.get("p")
            fpn = fpn_feats.get("f")
            rl = torch.tensor(0.0, device=DEV)
            vloss = torch.tensor(0.0, device=DEV)
            kl_loss = torch.tensor(0.0, device=DEV)
            total_gn_batch = 0.0

            if use_grpo and rf is not None and sp_raw is not None and rf.shape[0] > 0 and fpn is not None:
                N_rf = rf.shape[0]
                bf = model.roi_heads.box_head(rf)
                mu = model.roi_heads.box_predictor.bbox_pred(bf)[:, -4:]

                curr_w = model.roi_heads.box_predictor.bbox_pred.weight
                curr_b = model.roi_heads.box_predictor.bbox_pred.bias
                kl_loss = KL_WEIGHT * ((curr_w - baseline_bbox_w).pow(2).sum() + (curr_b - baseline_bbox_b).pow(2).sum())

                s = torch.full_like(mu, 0.1, requires_grad=False)
                deltas = mu.detach().unsqueeze(1) + s.unsqueeze(1) * torch.randn(N_rf, G_SAMPLES, 4, device=DEV)
                log_probs = gaussian_log_prob(deltas, mu, s)

                sp_cat = torch.cat(sp_raw, dim=0)
                N = min(N_rf, sp_cat.shape[0])
                mu = mu[:N]
                deltas = deltas[:N]
                log_probs = log_probs[:N]

                box_list, delta_list, img_map = [], [], []
                offset = 0
                for i_img, p_img in enumerate(sp_raw):
                    n_a = min(p_img.shape[0], N - offset)
                    if n_a <= 0:
                        break
                    box_list.append(sp_cat[offset:offset + n_a])
                    delta_list.append(deltas[offset:offset + n_a].reshape(-1, 4))
                    img_map.extend([i_img] * (n_a * G_SAMPLES))
                    offset += n_a

                sp_exp = torch.cat([p.repeat_interleave(G_SAMPLES, dim=0) for p in box_list], dim=0)
                delta_cat = torch.cat(delta_list, dim=0)
                decoded_cat = decode_boxes(sp_exp, delta_cat)
                decoded_list, off = [], 0
                for di in delta_list:
                    n = di.shape[0]
                    decoded_list.append(decoded_cat[off:off + n])
                    off += n

                iou_r = torch.zeros(offset, G_SAMPLES, device=DEV)
                for pi in range(offset):
                    i_img = img_map[pi * G_SAMPLES]
                    gt = tgts_t[i_img]["boxes"]
                    if len(gt) > 0:
                        iou_r[pi] = box_iou(decoded_cat[pi * G_SAMPLES:(pi + 1) * G_SAMPLES], gt).max(dim=1).values

                reward_img = compute_loc_reward(iou_r)

                if needs_verifier or use_energy:
                    image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]
                    with torch.no_grad():
                        pooled = box_pool(fpn, decoded_list, image_shapes)
                    fft_f = extract_perchan_fft(pooled)
                    geo = build_geo_features(torch.cat(decoded_list, dim=0), image_shapes, img_map)

                if needs_verifier:
                    roi_flat = pooled.flatten(1)
                    fft_shuf = fft_f[torch.randperm(fft_f.shape[0], generator=rng_shuf, device=DEV)]

                    if vrf_base is None:
                        roi_dim = pooled.shape[1] * pooled.shape[2] * pooled.shape[3]
                        vrf_base = BaseVerifier(roi_dim).to(DEV)
                        vrf_fft = FFTResidualVerifier(fft_f.shape[1]).to(DEV)
                        opt = build_sgd_optimizer(model, head_lr=HEAD_LR, body_lr=BODY_LR, extra_modules=[vrf_base, vrf_fft])

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

                    pos_mask = (iou_r.max(dim=1).values >= 0.5).float().unsqueeze(1)
                    reward_img = reward_img + FFT_WEIGHT * q_fft_real.detach() * pos_mask
                    diag["q_ious"].extend(list(zip(q_fft_real.flatten().tolist(), iou_r.flatten().tolist())))

                elif use_energy:
                    r_energy = ENERGY_WEIGHT * compute_energy_reward(fft_f).view(offset, G_SAMPLES)
                    diag["energy_mean"].append(r_energy.abs().mean().item())
                    reward_img = reward_img + r_energy

                adv = grpo_advantage(reward_img)
                diag["reward_std"].append(adv.std().item())
                soft_w = iou_r.max(dim=1).values.clamp(0, 1).unsqueeze(1)
                rl = -(adv.detach() * log_probs * soft_w).mean()
                pos += (soft_w > 0.3).sum().item()

            vloss_term = vloss if vrf_base is not None else torch.tensor(0.0, device=DEV)
            loss = det + vloss_term + RL_WEIGHT * rl + kl_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            total_gn_batch = bbox_pred_weight.grad.norm().item() if bbox_pred_weight.grad is not None else 0.0
            opt.step()

            diag["total_grad_norm"].append(total_gn_batch)
            td += det.item()
            trl += rl.item()
            tv += vloss.item()
            tkl += kl_loss.item()

        em = evaluate_model(model, vl, DEV, iou_threshold=0.5, score_threshold=0.05)
        q_corr = 0.0
        if len(diag["q_ious"]) > 1:
            qs = np.array([x[0] for x in diag["q_ious"]])
            iis = np.array([x[1] for x in diag["q_ious"]])
            q_corr = np.corrcoef(qs, iis)[0, 1]
        tgn = np.mean(diag["total_grad_norm"]) if diag["total_grad_norm"] else 0.0
        rs_m = np.mean(diag["reward_std"]) if diag["reward_std"] else 0.0
        en_m = np.mean(diag["energy_mean"]) if diag["energy_mean"] else 0.0

        row = {
            "epoch": ep,
            "val_ap50": em["ap50"],
            "val_ap75": em["ap75"],
            "precision": em.get("precision", 0),
            "recall": em.get("recall", 0),
            "ece": em.get("ece", 0),
            "q_iou_corr": float(q_corr),
            "reward_std": float(rs_m),
            "total_grad_norm": float(tgn),
            "energy_reward": float(en_m),
            "det_loss": td,
            "rl_loss": trl,
            "vloss": tv,
            "kl_loss": tkl,
            "pos_count": int(pos),
        }
        h.append(row)
        print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} "
              f"q_corr={q_corr:.4f} tgn={tgn:.6f} r_std={rs_m:.4f} En={en_m:.4f}")
        if em["ap75"] > best_ap75:
            best_ap75 = em["ap75"]
        for k in diag:
            diag[k].clear()

    best_h = max(h, key=lambda r: r["val_ap75"])
    em.update({
        "run_name": run_name,
        "config": cfg_name,
        "mode": mode,
        "seed": seed,
        "epochs": len(h),
        "best_ap50": best_h["val_ap50"],
        "best_ap75": best_ap75,
        "history": h,
        "git_hash": GIT,
    })
    save_json(em, run_dir / "eval_metrics.json")
    return em


if __name__ == "__main__":
    all_results = []
    configs = {
        "det_only_unf": "det_only_unf",
        "ap75_event": "ap75_event",
        "grpo_adv": "grpo_adv",
        "grpo_fft": "grpo_fft",
        "fft_energy": "fft_energy",
    }
    for cfg, mode in configs.items():
        for s in SEEDS:
            r = run_one(cfg, mode, s)
            all_results.append(r)

    print("\n## Plan 2.79 Unfrozen RLVR (FPN+RPN+box_head)")
    print(f"  {'Config':<15s} {'Seed':>5s} {'AP75':>8s} {'BestAP75':>8s} {'AP50':>8s} {'q_corr':>8s} {'r_std':>8s} {'En':>8s}")
    for r in all_results:
        best_h = max(r["history"], key=lambda x: x["val_ap75"])
        print(f"  {r['config']:<15s} {r['seed']:5d} {r['ap75']:8.4f} {r['best_ap75']:8.4f} {best_h['val_ap50']:8.4f} {best_h.get('q_iou_corr', 0):8.4f} {best_h.get('reward_std', 0):8.4f} {best_h.get('energy_reward', 0):8.4f}")

    for cfg in configs:
        vals = [r for r in all_results if r["config"] == cfg]
        if not vals:
            continue
        bv = [r["best_ap75"] for r in vals]
        fv = [r["ap75"] for r in vals]
        ap50s = [max(r["history"], key=lambda x: x["val_ap75"])["val_ap50"] for r in vals]
        print(f"  {cfg}: bestAP75={np.mean(bv):.4f}±{np.std(bv):.4f}  finalAP75={np.mean(fv):.4f}  AP50={np.mean(ap50s):.4f}")

    def judge(name_a, name_b, threshold=0.01):
        va = [r["best_ap75"] for r in all_results if r["config"] == name_a]
        vb = [r["best_ap75"] for r in all_results if r["config"] == name_b]
        if not va or not vb:
            return
        delta = np.mean(va) - np.mean(vb)
        wins = sum(1 for i in range(3) if va[i] > vb[i])
        status = "PASS" if delta > threshold and wins >= 2 else "FAIL"
        print(f"\n  {name_a} vs {name_b}: Δ={delta:+.4f} wins={wins}/3 → {status}")

    print("\n## RLVR signal check")
    judge("grpo_adv", "det_only_unf")
    judge("ap75_event", "det_only_unf")
    judge("grpo_fft", "grpo_adv")
