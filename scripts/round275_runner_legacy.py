"""Plan 2.75 v3: Loc/Select Split + FFT Loc-Only + Early-Stop.

  1. Split reward: R_loc (IoU-based) + R_select (NMS-based survival, includes FP penalty)
  2. FFT loc-only: only participates in localization residual (pure R_loc target)
  3. Soft-weight policy gradient: uses max IoU clamp(0,1) as weight (not hard pm > 0.3)
  4. Early-stop: AP75 no-improve 3 epochs OR AP50/Recall drop > 0.02

3 groups × 3 seeds:
  A  select_penalty       R_loc + R_select
  B  fft_loc_only         R_loc + FFT residual (IoU>=0.5 only)
  C  combined             R_loc + R_select + FFT loc-only

Plus det_only baseline. All use image-level advantage normalization.
Runner snapshot saved to run_dir for reproducibility.
"""
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, nms
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
    compute_loc_reward,
)
from spectral_detection_posttrain.models.verifiers import BaseVerifier, FFTResidualVerifier, build_geo_features
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
G_SAMPLES = 4
EPOCHS = 8
SEEDS = [42, 123, 456]
RL_WEIGHT = 0.05
RANK_WEIGHT = 0.1
FFT_WEIGHT = 0.1
SELECT_WEIGHT = 0.5
EARLY_STOP_PATIENCE = 3
MAX_AP50_DROP = 0.02
MAX_RECALL_DROP = 0.02


def fe(m, parts):
    for p in m.parameters():
        p.requires_grad = False
    for part in parts:
        if isinstance(part, nn.Module):
            for p in part.parameters():
                p.requires_grad = True


def compute_select_reward(decoded_img, gt_boxes, cls_scores):
    """R_select: NMS-based survival reward. Soft margins, not hard {-1,+1}.
    Returns: per-box reward tensor (N_boxes,), diagnostics dict.
    """
    total = decoded_img.shape[0]
    reward = torch.zeros(total, device=decoded_img.device)
    diag = {"ap75_tp": 0, "ap50_tp": 0, "high_fp": 0, "duplicate": 0,
            "nms_survivors": 0, "matched_tp": 0, "matched_fp": 0, "suppressed_dup": 0}

    if len(gt_boxes) == 0:
        reward[:] = -1.0
        diag["high_fp"] = total
        return reward, diag

    keep = nms(decoded_img, cls_scores, iou_threshold=0.5)
    keep_set = set(keep.tolist())
    diag["nms_survivors"] = len(keep_set)

    if len(keep) > 0:
        iou_surv = box_iou(decoded_img[keep], gt_boxes)
        best_iou_surv, best_gt_surv = iou_surv.max(dim=1)
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=decoded_img.device)

        for si, ki in enumerate(keep.tolist()):
            iou_val = best_iou_surv[si].item()
            gt_i = best_gt_surv[si].item()
            if iou_val >= 0.5 and not gt_matched[gt_i]:
                reward[ki] = 1.0  # matched TP
                gt_matched[gt_i] = True
                diag["matched_tp"] += 1
                if iou_val >= 0.75:
                    diag["ap75_tp"] += 1
                else:
                    diag["ap50_tp"] += 1
            else:
                reward[ki] = -0.7  # FP survivor
                diag["matched_fp"] += 1

    # Non-surviving boxes
    iou_flat = box_iou(decoded_img, gt_boxes).max(dim=1).values
    for idx in range(total):
        if idx not in keep_set:
            if iou_flat[idx] >= 0.5:
                reward[idx] = -0.3  # duplicate (suppressed, high IoU)
                diag["suppressed_dup"] += 1
            elif iou_flat[idx] < 0.3:
                reward[idx] = -1.0  # hard FP
                diag["high_fp"] += 1
            else:
                reward[idx] = -0.5  # middling FP
                diag["duplicate"] += 1

    return reward, diag


def per_image_normalize(x, offset, img_map, G):
    """Normalize per-image. Returns (normed_tensor, mad_vs_batch) for diagnostic."""
    x_norm = torch.zeros_like(x)
    # Batch-level baseline for comparison
    x_flat_batch = x.view(-1)
    batch_norm = (x_flat_batch - x_flat_batch.mean()) / (x_flat_batch.std().clamp_min(1e-6))
    batch_norm = batch_norm.view(-1, G)
    mad_sum = 0.0; n_img = 0
    for i_img in set(img_map[::G]):
        mask = torch.tensor([img_map[pi * G] == i_img for pi in range(offset)], device=x.device)
        idx = mask.nonzero(as_tuple=True)[0]
        if len(idx) == 0: continue
        x_img = x[idx]
        x_flat = x_img.view(-1)
        x_norm[idx] = ((x_flat - x_flat.mean()) / (x_flat.std().clamp_min(1e-6))).view(-1, G)
        mad_sum += (x_norm[idx] - batch_norm[idx]).abs().mean().item()
        n_img += 1
    return x_norm, mad_sum / max(n_img, 1)



def ranking_loss(q_pred, iou_r, margin=0.1):
    N, G = q_pred.shape
    loss = torch.tensor(0.0, device=q_pred.device); count = 0
    for i in range(N):
        ious = iou_r[i]
        for a in range(G):
            for b in range(a + 1, G):
                if ious[a] - ious[b] > margin:
                    loss += F.relu(margin - (q_pred[i, a] - q_pred[i, b])); count += 1
                elif ious[b] - ious[a] > margin:
                    loss += F.relu(margin - (q_pred[i, b] - q_pred[i, a])); count += 1
    return loss / max(count, 1)


def run_one(cfg_name, mode, seed):
    """mode: 'select_penalty' | 'fft_loc_only' | 'combined' | 'det_only'"""
    run_name = f"round275_{cfg_name}_s{seed}"
    set_seed(seed)
    model = build_mobv3_detector(num_classes=2, pretrained=True).to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"])
    fe(model, [model.roi_heads.box_head, model.roi_heads.box_predictor])
    box_pool = model.roi_heads.box_roi_pool

    sampled_props, box_head_in, fpn_feats = {}, {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(
        lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(
        lambda m, args: box_head_in.update({"x": args[0]}))
    model.backbone.register_forward_hook(
        lambda m, i, o: fpn_feats.update({"f": {k: o[k] for k in o if k != "pool"}}))

    rng_gen = torch.Generator(device=DEV).manual_seed(seed + 9999)
    rng_shuf = torch.Generator(device=DEV).manual_seed(seed + 7777)
    tl, vl = build_penn_fudan_loaders_320(batch_size=4)
    run_dir = ensure_run_dir(run_name)
    shutil.copy(__file__, run_dir / "runner_snapshot.py")  # reproducibility

    needs_verifier = mode in ("fft_loc_only", "combined")
    needs_select = mode in ("select_penalty", "combined")
    needs_fft_loc = mode in ("fft_loc_only", "combined")
    # FP penalty removed (dead code with cls_scores > 0.7 threshold)
    is_det = mode == "det_only"

    vrf_base = None; vrf_fft = None

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    bbox_pred_weight = model.roi_heads.box_predictor.bbox_pred.weight

    h = []; best_ap75 = -1.0
    baseline_ap50 = None; baseline_recall = None
    no_improve_count = 0
    stopped_early = False
    diag = {"q_ious": [], "total_grad_norm": [], "reward_std": [], "q_std": [],
            "ap75_tp": [], "ap50_tp": [], "high_fp": [], "duplicate": [],
            "matched_tp": [], "matched_fp": [], "suppressed_dup": [], "nms_survivors": [],
            "norm_mad": []}

    for ep in range(1, EPOCHS + 1):
        model.train()
        for v in [vrf_base, vrf_fft]:
            if v is not None: v.train()
        td, trl, tv, pos = 0.0, 0.0, 0.0, 0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}"):
            imgs_d = [i.to(DEV) for i in imgs]
            tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear(); box_head_in.clear(); fpn_feats.clear()

            ld = model(imgs_d, tgts_t)
            det = sum(ld.values()) if isinstance(ld, dict) else sum(sum(d.values()) for d in ld if isinstance(d, dict))

            rf = box_head_in.get("x"); sp_raw = sampled_props.get("p"); fpn = fpn_feats.get("f")
            rl = torch.tensor(0.0, device=DEV); vloss = torch.tensor(0.0, device=DEV)
            total_gn_batch = 0.0

            if not is_det and rf is not None and sp_raw is not None and rf.shape[0] > 0 and fpn is not None:
                N_rf = rf.shape[0]
                bf = model.roi_heads.box_head(rf)
                mu = model.roi_heads.box_predictor.bbox_pred(bf)[:, -4:]
                cls_logits = model.roi_heads.box_predictor.cls_score(bf)
                cls_probs = F.softmax(cls_logits, dim=1)[:, 1]

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

                # --- Per-image reward computation ---
                reward_img = torch.zeros(offset, G_SAMPLES, device=DEV)
                reward_loc_img = torch.zeros(offset, G_SAMPLES, device=DEV)  # R_loc only, for FFT target
                rd_tot = {"ap75_tp": 0, "ap50_tp": 0, "high_fp": 0, "duplicate": 0,
                          "matched_tp": 0, "matched_fp": 0, "suppressed_dup": 0, "nms_survivors": 0}
                for i_img in range(len(tgts_t)):
                    mask = torch.tensor([img_map[pi * G_SAMPLES] == i_img for pi in range(offset)], device=DEV)
                    pi_list = mask.nonzero(as_tuple=True)[0].tolist()
                    if not pi_list: continue

                    dec_img = torch.cat([decoded_cat[pi * G_SAMPLES:(pi + 1) * G_SAMPLES] for pi in pi_list], dim=0)
                    iou_img = torch.stack([iou_r[pi] for pi in pi_list], dim=0)
                    cls_img = torch.cat([cls_probs[pi].repeat(G_SAMPLES) for pi in pi_list], dim=0)

                    # R_loc: always computed
                    r_loc = compute_loc_reward(iou_img)

                    # R_select: computed for select_penalty and combined
                    if needs_select:
                        r_select, rd_s = compute_select_reward(dec_img, tgts_t[i_img]["boxes"], cls_img)
                        r_select = r_select.view(len(pi_list), G_SAMPLES)
                        for k in rd_s: rd_tot[k] += rd_s[k]

                    # Combine reward
                    if mode == "select_penalty":
                        rwd = (1.0 - SELECT_WEIGHT) * r_loc + SELECT_WEIGHT * r_select
                    elif mode == "fft_loc_only":
                        rwd = r_loc  # base, FFT added later
                    elif mode == "combined":
                        rwd = (1.0 - SELECT_WEIGHT) * r_loc + SELECT_WEIGHT * r_select

                    for j, pi in enumerate(pi_list):
                        reward_img[pi] = rwd[j]
                        reward_loc_img[pi] = r_loc[j]  # pure R_loc, for FFT target

                for k in rd_tot: diag[k].append(rd_tot[k])

                # --- Verifier (fft_loc_only / combined) ---
                if needs_verifier:
                    image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]
                    with torch.no_grad():
                        pooled = box_pool(fpn, decoded_list, image_shapes)
                    roi_flat = pooled.flatten(1)
                    fft_f = extract_perchan_fft(pooled)
                    fft_shuf = fft_f[torch.randperm(fft_f.shape[0], generator=rng_shuf, device=DEV)]
                    geo = build_geo_features(torch.cat(decoded_list, dim=0), image_shapes, img_map)

                    if vrf_base is None:
                        roi_dim = pooled.shape[1] * pooled.shape[2] * pooled.shape[3]
                        vrf_base = BaseVerifier(roi_dim).to(DEV)
                        vrf_fft = FFTResidualVerifier(fft_f.shape[1]).to(DEV)
                        params = [p for p in list(model.parameters()) + list(vrf_base.parameters()) + list(vrf_fft.parameters()) if p.requires_grad]
                        opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

                    q_base = vrf_base(roi_flat, geo).view(offset, G_SAMPLES)
                    q_fft_real = vrf_fft(fft_f).view(offset, G_SAMPLES)
                    q_fft_shuf = vrf_fft(fft_shuf).view(offset, G_SAMPLES)

                    # combined: FFT targets pure R_loc; fft_loc_only: same as reward_img (R_loc only already)
                    fft_target = reward_loc_img if mode == "combined" else reward_img
                    base_target = fft_target.clamp(-1, 1)
                    vloss = F.mse_loss(q_base, base_target.detach())
                    residual_target = (base_target - q_base.detach()).clamp(-1, 1)
                    vloss = vloss + F.mse_loss(q_fft_real, residual_target.detach())
                    high_iou_mask = iou_r.max(dim=1).values > 0.5
                    if high_iou_mask.any():
                        vloss = vloss + 0.1 * F.relu(0.1 - (q_fft_real[high_iou_mask].mean() - q_fft_shuf[high_iou_mask].mean()))

                    # FIX 2: FFT loc-only — only apply residual to positive samples (IoU >= 0.5)
                    pos_mask = (iou_r.max(dim=1).values >= 0.5).float().unsqueeze(1)
                    fft_contrib = FFT_WEIGHT * q_fft_real.detach() * pos_mask if needs_fft_loc else torch.zeros_like(reward_img)

                    final_reward = reward_img + fft_contrib
                    q_norm, mad = per_image_normalize(final_reward, offset, img_map, G_SAMPLES)

                    diag["q_ious"].extend(list(zip(q_fft_real.flatten().tolist(), iou_r.flatten().tolist())))
                    diag["q_std"].append(q_fft_real.std().item())
                    diag["norm_mad"].append(mad)
                else:
                    # select_penalty: no verifier
                    q_norm, mad = per_image_normalize(reward_img, offset, img_map, G_SAMPLES)
                    diag["norm_mad"].append(mad)

                diag["reward_std"].append(q_norm.std().item())
                # Soft weight: max IoU clamped [0,1]. Low IoU proposals still get weak RL signal.
                soft_w = iou_r.max(dim=1).values.clamp(0, 1).unsqueeze(1)
                rl = -(q_norm.detach() * log_probs * soft_w).mean()
                pos += (soft_w > 0.3).sum().item()

            # --- Backward ---
            if is_det:
                loss = det
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            else:
                vloss_term = vloss if (vrf_base is not None) else torch.tensor(0.0, device=DEV)
                loss = det + vloss_term + RL_WEIGHT * rl
                opt.zero_grad(set_to_none=True); loss.backward()
                total_gn_batch = bbox_pred_weight.grad.norm().item() if bbox_pred_weight.grad is not None else 0.0
                opt.step()

            diag["total_grad_norm"].append(total_gn_batch)
            td += det.item(); trl += rl.item(); tv += vloss.item()

        # --- Epoch diagnostics ---
        em = evaluate_model(model, vl, DEV, iou_threshold=0.5, score_threshold=0.05)
        if len(diag["q_ious"]) > 0:
            qs = np.array([x[0] for x in diag["q_ious"]])
            iis = np.array([x[1] for x in diag["q_ious"]])
            q_corr = np.corrcoef(qs, iis)[0, 1] if len(qs) > 1 else 0.0
        else:
            q_corr = 0.0
        tgn = np.mean(diag["total_grad_norm"]) if len(diag["total_grad_norm"]) > 0 else 0.0
        qs_m = np.mean(diag["q_std"]) if len(diag["q_std"]) > 0 else 0.0
        rs_m = np.mean(diag["reward_std"]) if len(diag["reward_std"]) > 0 else 0.0
        ap75_tp = np.sum(diag["ap75_tp"]); dup = np.sum(diag["duplicate"]); hfp = np.sum(diag["high_fp"])
        mtp = np.sum(diag["matched_tp"]); mfp = np.sum(diag["matched_fp"]); sd = np.sum(diag["suppressed_dup"])
        nms_s = np.sum(diag["nms_survivors"])

        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "precision": em.get("precision", 0), "recall": em.get("recall", 0),
               "ece": em.get("ece", 0), "q_iou_corr": float(q_corr), "q_std": float(qs_m),
               "reward_std": float(rs_m), "total_grad_norm": float(tgn),
               "ap75_tp": int(ap75_tp), "duplicate_fp": int(dup), "high_conf_fp": int(hfp),
               "matched_tp": int(mtp), "matched_fp": int(mfp), "suppressed_dup": int(sd),
               "nms_survivors": int(nms_s), "det_loss": td, "rl_loss": trl, "vloss": tv,
               "pos_count": int(pos), "norm_mad": np.mean(diag["norm_mad"]) if len(diag["norm_mad"]) > 0 else 0.0}
        h.append(row)
        print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} Rec={em.get('recall', 0):.4f} "
              f"q_corr={q_corr:.4f} tgn={tgn:.6f} r_std={rs_m:.4f} MAD={h[-1]['norm_mad']:.4f}")
        # --- Early-stop ---
        if baseline_ap50 is None and not is_det:
            baseline_ap50 = em["ap50"]
            baseline_recall = em.get("recall", 0)
        if not is_det and baseline_ap50 is not None:
            ap50_drop = baseline_ap50 - em["ap50"]
            recall_drop = baseline_recall - em.get("recall", 0)
            if em["ap75"] > best_ap75:
                best_ap75 = em["ap75"]
                no_improve_count = 0
            else:
                no_improve_count += 1
            if (no_improve_count >= EARLY_STOP_PATIENCE or ap50_drop > MAX_AP50_DROP or recall_drop > MAX_RECALL_DROP):
                if ep >= 3:
                    print(f"  Early stop at e{ep}: no_imp={no_improve_count} ap50_drop={ap50_drop:.4f} rec_drop={recall_drop:.4f}")
                    stopped_early = True
        elif is_det:
            if em["ap75"] > best_ap75: best_ap75 = em["ap75"]

        for k in diag: diag[k].clear()
        if stopped_early:
            break

    best_h = max(h, key=lambda r: r["val_ap75"])
    em.update({"run_name": run_name, "config": cfg_name, "mode": mode, "seed": seed,
               "epochs": len(h), "best_ap50": best_h["val_ap50"], "best_ap75": best_ap75,
               "stopped_early": stopped_early, "history": h, "git_hash": GIT,
               "q_iou_corr_final": h[-1]["q_iou_corr"], "total_grad_final": h[-1]["total_grad_norm"]})
    save_json(em, run_dir / "eval_metrics.json")
    return em


if __name__ == "__main__":
    all_results = []
    configs = {"select_penalty": "select_penalty", "fft_loc_only": "fft_loc_only",
               "combined": "combined", "det_only": "det_only"}
    for cfg, mode in configs.items():
        for s in SEEDS:
            r = run_one(cfg, mode, s)
            all_results.append(r)

    print("\n## Plan 2.75 Loc/Select Split + FFT Loc-Only")
    print(f"  {'Config':<18s} {'Seed':>5s} {'AP75':>8s} {'BestAP75':>8s} {'AP50':>8s} {'Rec':>8s} {'Prec':>8s} {'q_corr':>8s} {'r_std':>8s} {'TP':>6s} {'mTP':>6s} {'FP':>6s} {'Epochs':>6s}")
    for r in all_results:
        best_h = max(r["history"], key=lambda x: x["val_ap75"])  # best-epoch metrics
        print(f"  {r['config']:<18s} {r['seed']:5d} {r['ap75']:8.4f} {r['best_ap75']:8.4f} {best_h['val_ap50']:8.4f} {best_h.get('recall', 0):8.4f} {best_h.get('precision', 0):8.4f} {r.get('q_iou_corr_final', 0):8.4f} {best_h.get('reward_std', 0):8.4f} {best_h.get('ap75_tp', 0):6d} {best_h.get('matched_tp', 0):6d} {best_h.get('high_conf_fp', 0):6d} {r.get('epochs', 0):6d}")

    for cfg in configs:
        vals = [r for r in all_results if r["config"] == cfg]
        if not vals: continue
        bv = [r["best_ap75"] for r in vals]; fv = [r["ap75"] for r in vals]
        ap50s = [max(r["history"], key=lambda x: x["val_ap75"])["val_ap50"] for r in vals]
        recs = [max(r["history"], key=lambda x: x["val_ap75"]).get("recall", 0) for r in vals]
        precs = [max(r["history"], key=lambda x: x["val_ap75"]).get("precision", 0) for r in vals]
        fps = [np.mean([x["high_conf_fp"] for x in r["history"]]) for r in vals]
        print(f"  {cfg}: bestAP75={np.mean(bv):.4f}±{np.std(bv):.4f}  finalAP75={np.mean(fv):.4f}  AP50={np.mean(ap50s):.4f}  Rec={np.mean(recs):.4f}  Prec={np.mean(precs):.4f}  avgFP={np.mean(fps):.0f}")

    def judge(name_a, name_b, threshold=0.01):
        va = [r["best_ap75"] for r in all_results if r["config"] == name_a]
        vb = [r["best_ap75"] for r in all_results if r["config"] == name_b]
        if not va or not vb: return
        delta = np.mean(va) - np.mean(vb)
        wins = sum(1 for i in range(3) if va[i] > vb[i])
        status = "PASS" if delta > threshold and wins >= 2 else "FAIL"
        print(f"\n  {name_a} vs {name_b}: Δ={delta:+.4f} wins={wins}/3 → {status}")

    print("\n## Key judgments")
    judge("fft_loc_only", "select_penalty")
    judge("combined", "fft_loc_only")
    judge("combined", "select_penalty")
    judge("select_penalty", "det_only")
