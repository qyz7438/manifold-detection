"""Plan 2.85: Cross-proposal GRPO — energy at image scale, not per-proposal.

Root cause (round281-284): G=4 perturbations of ONE proposal produce near-zero
energy variance (Δen/Δpx=-0.0005). But across proposals in the same image,
energy separates FN from TP with Cohen d=0.99.

Fix: GRPO advantage computed across ALL proposals in the same image, not
within one proposal's G perturbations. Energy becomes meaningful.

adv = (reward - mean_all_proposals_in_image) / std_all_proposals_in_image
"""
import copy
import shutil
import subprocess
import sys

import numpy as np
import torch
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_mobv3_detector,
    build_penn_fudan_loaders_320,
    compute_loc_reward,
    decode_boxes,
    evaluate_model,
    extract_perchan_fft,
    gaussian_log_prob,
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
BETA = 0.02
HEAD_LR = 0.001
BODY_LR = 0.0001
SIGMA = 0.1


def compute_energy(fft_f):
    ch = fft_f.shape[1] // 6
    a_lo = fft_f[:, 0*ch:1*ch]
    a_total = a_lo + fft_f[:, 1*ch:2*ch] + fft_f[:, 2*ch:3*ch] + 1e-8
    return (a_lo / a_total).mean(dim=1)  # mean low-frequency ratio over channels, (N,)


def bl():
    return build_penn_fudan_loaders_320(batch_size=2)


def bm():
    return build_mobv3_detector(num_classes=2, pretrained=True)


@torch.no_grad()
def ev(model, vl):
    return evaluate_model(model, vl, DEV, iou_threshold=0.5, score_threshold=0.05)


def cross_proposal_grpo(reward, n_proposals_per_img):
    """GRPO advantage across proposals in the same image.
    reward: (total_N,) flattened across all images
    n_proposals_per_img: list of N_i per image
    """
    adv = torch.zeros_like(reward)
    offset = 0
    for n_p in n_proposals_per_img:
        if n_p <= 1:  # need at least 2 proposals for meaningful GRPO
            if n_p == 1:
                adv[offset] = 0.0
            offset += n_p
            continue
        r_img = reward[offset:offset + n_p]
        r_mean = r_img.mean()
        r_std = r_img.std().clamp_min(1e-6)
        adv[offset:offset + n_p] = (r_img - r_mean) / r_std
        offset += n_p
    return adv


def build_opt(model):
    body_params = []; head_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if 'box_head' in n or 'box_predictor' in n: head_params.append(p)
        else: body_params.append(p)
    return torch.optim.SGD([
        {'params': body_params, 'lr': BODY_LR},
        {'params': head_params, 'lr': HEAD_LR},
    ], lr=HEAD_LR, momentum=0.9, weight_decay=0.0005)


# Penalty functions
def penalty_sigmoid(energy):
    return -torch.sigmoid(15 * (energy - 0.5))


def penalty_asymmetric(energy):
    """Only penalize energy > 0.5 (FN-like), ignore low energy"""
    return -torch.relu(energy - 0.5)


def run_one(cfg_name, mode, seed):
    run_name = f"round285_{cfg_name}_s{seed}"
    set_seed(seed)
    model = bm().to(DEV)
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

    tl, vl = bl()
    run_dir = ensure_run_dir(run_name)
    shutil.copy(__file__, run_dir / "runner_snapshot.py")

    is_det = mode == "det_only_unf"
    use_energy = mode in ("crossproposal_sigmoid", "crossproposal_asym")
    is_shuffle = mode == "crossproposal_shuffle"
    rng_shuf = torch.Generator(device=DEV).manual_seed(seed + 9999)

    opt = build_opt(model)

    h = []; best_ap75 = -1.0
    diag = {"reward_std": [], "energy_cross_std": [], "energy_cross_gap": []}

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
                N = rf.shape[0]
                bf = model.roi_heads.box_head(rf)
                mu = model.roi_heads.box_predictor.bbox_pred(bf)[:, -4:]

                curr_w = model.roi_heads.box_predictor.bbox_pred.weight
                curr_b = model.roi_heads.box_predictor.bbox_pred.bias
                kl_loss = KL_WEIGHT * ((curr_w - baseline_bbox_w).pow(2).sum() + (curr_b - baseline_bbox_b).pow(2).sum())

                # G perturbations per proposal (for gradient diversity)
                s = torch.full_like(mu, SIGMA)
                deltas = mu.detach().unsqueeze(1) + s.unsqueeze(1) * torch.randn(N, G_SAMPLES, 4, device=DEV)
                log_probs = gaussian_log_prob(deltas, mu, s)  # (N, G)

                sp_cat = torch.cat(sp_raw, dim=0)
                sp_exp = sp_cat.repeat_interleave(G_SAMPLES, dim=0)  # (N*G, 4)
                delta_flat = deltas.reshape(-1, 4)  # (N*G, 4)
                decoded_flat = decode_boxes(sp_exp, delta_flat)

                # Compute IoU: for each proposal, find max IoU with any GT in its image
                image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]
                iou_flat = torch.zeros(N * G_SAMPLES, device=DEV)
                img_id_per_sample = []
                for i_img, p_img in enumerate(sp_raw):
                    img_id_per_sample.extend([i_img] * (p_img.shape[0] * G_SAMPLES))

                for i in range(N * G_SAMPLES):
                    gt = tgts_t[img_id_per_sample[i]]["boxes"]
                    if len(gt) > 0:
                        iou_flat[i] = box_iou(decoded_flat[i:i+1], gt).max()

                reward_loc = compute_loc_reward(iou_flat)  # (N*G,)

                # Energy penalty
                gated_bias = torch.zeros(N * G_SAMPLES, device=DEV)
                if use_energy or is_shuffle:
                    decoded_list = [decoded_flat]  # single list for box_pool
                    with torch.no_grad():
                        pooled = box_pool(fpn, decoded_list, image_shapes)
                    fft_f = extract_perchan_fft(pooled)
                    energy = compute_energy(fft_f)  # (N*G,)

                    if is_shuffle:
                        energy = energy[torch.randperm(energy.shape[0], generator=rng_shuf, device=DEV)]

                    gated_bias = BETA * penalty_sigmoid(energy)

                    # Track cross-proposal energy stats
                    en_std_img = []
                    en_tp_img = []; en_fn_img = []
                    off = 0
                    for i_img, p_img in enumerate(sp_raw):
                        np_i = p_img.shape[0] * G_SAMPLES
                        if np_i > 0:
                            en_img = energy[off:off+np_i]
                            iou_img = iou_flat[off:off+np_i]
                            en_std_img.append(en_img.std().item())
                            tp = en_img[iou_img >= 0.5]
                            fn = en_img[iou_img < 0.5]
                            if len(tp) > 0: en_tp_img.append(tp.mean().item())
                            if len(fn) > 0: en_fn_img.append(fn.mean().item())
                            off += np_i
                    diag["energy_cross_std"].append(np.mean(en_std_img) if en_std_img else 0)
                    if en_tp_img and en_fn_img:
                        diag["energy_cross_gap"].append(np.mean(en_tp_img) - np.mean(en_fn_img))

                # Reward = R_loc + energy penalty
                reward_total = reward_loc + gated_bias

                # Cross-proposal GRPO (per image)
                n_props = [p.shape[0] * G_SAMPLES for p in sp_raw]
                adv = cross_proposal_grpo(reward_total, n_props)  # (N*G,)
                adv = adv.view(N, G_SAMPLES)  # (N, G)

                # PG loss
                diag["reward_std"].append(adv.std().item())
                soft_w = iou_flat.view(N, G_SAMPLES).max(dim=1).values.clamp(0, 1).unsqueeze(1)
                rl = -(adv.detach() * log_probs * soft_w).mean()

            loss = det + RL_WEIGHT * rl + kl_loss
            opt.zero_grad(set_to_none=True); loss.backward()
            opt.step()

            td += det.item(); trl += rl.item(); tkl += kl_loss.item()

        em = ev(model, vl)
        rs_m = np.mean(diag["reward_std"]) if diag["reward_std"] else 0.0
        ec_std = np.mean(diag["energy_cross_std"]) if diag["energy_cross_std"] else 0.0
        ec_gap = np.mean(diag["energy_cross_gap"]) if diag["energy_cross_gap"] else 0.0

        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "precision": em.get("precision", 0), "recall": em.get("recall", 0),
               "ece": em.get("ece", 0), "reward_std": float(rs_m),
               "energy_cross_std": float(ec_std), "energy_cross_gap": float(ec_gap),
               "det_loss": td, "rl_loss": trl, "kl_loss": tkl}
        h.append(row)
        print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} "
              f"r_std={rs_m:.4f} en_cross_std={ec_std:.4f} en_gap={ec_gap:.4f}")
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
        "crossproposal_sigmoid": "crossproposal_sigmoid",
        "crossproposal_asym": "crossproposal_asym",
        "crossproposal_shuffle": "crossproposal_shuffle",
    }
    for cfg, mode in configs.items():
        for s in SEEDS:
            r = run_one(cfg, mode, s)
            all_results.append(r)

    print("\n## Plan 2.85 Cross-proposal GRPO")
    print(f"  {'Config':<24s} {'Seed':>5s} {'AP75':>8s} {'BestAP75':>8s} {'AP50':>8s} {'r_std':>8s} {'en_cross_std':>12s} {'en_gap':>8s}")
    for r in all_results:
        best_h = max(r["history"], key=lambda x: x["val_ap75"])
        print(f"  {r['config']:<24s} {r['seed']:5d} {r['ap75']:8.4f} {r['best_ap75']:8.4f} "
              f"{best_h['val_ap50']:8.4f} {best_h.get('reward_std', 0):8.4f} "
              f"{best_h.get('energy_cross_std', 0):12.4f} {best_h.get('energy_cross_gap', 0):8.4f}")

    for cfg in configs:
        vals = [r for r in all_results if r["config"] == cfg]
        if not vals: continue
        bv = [r["best_ap75"] for r in vals]
        print(f"  {cfg}: bestAP75={np.mean(bv):.4f} +/- {np.std(bv):.4f}")
