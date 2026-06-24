"""Plan 2.85: Cross-proposal GRPO — energy at image scale, not per-proposal.

Root cause (round281-284): G=4 perturbations of ONE proposal produce near-zero
energy variance (Δen/Δpx=-0.0005). But across proposals in the same image,
energy separates FN from TP with Cohen d=0.99.

Fix: GRPO advantage computed across ALL proposals in the same image, not
within one proposal's G perturbations. Energy becomes meaningful.

adv = (reward - mean_all_proposals_in_image) / std_all_proposals_in_image
"""
import sys, json, subprocess, math, copy, shutil
from pathlib import Path
import torch, torch.nn as nn
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
G_SAMPLES = 4
EPOCHS = 8
SEEDS = [42, 123, 456]
RL_WEIGHT = 0.05
KL_WEIGHT = 0.1
BETA = 0.02
HEAD_LR = 0.001
BODY_LR = 0.0001
SIGMA = 0.1


def extract_perchan_fft(x):
    C = x.shape[1]; H, W = x.shape[-2], x.shape[-1]
    fft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    amp = torch.abs(fft)
    freq_h = torch.fft.fftfreq(H, device=x.device)
    freq_w = torch.fft.rfftfreq(W, device=x.device)
    Y, X = torch.meshgrid(freq_h, freq_w, indexing='ij')
    r = torch.sqrt(X**2+Y**2); R = r.max().clamp_min(1e-6); rn = r/R
    lo = (rn <= 0.3).float(); md = ((rn > 0.3) & (rn <= 0.7)).float(); hi = (rn > 0.7).float()
    a_lo = (amp*lo).flatten(2).sum(2); a_md = (amp*md).flatten(2).sum(2); a_hi = (amp*hi).flatten(2).sum(2)
    return a_lo / (a_lo+a_md+a_hi+1e-8)


def compute_energy(fft_f):
    return fft_f.mean(dim=1)  # mean over channels, (N,)


def compute_loc_reward(iou):
    r = torch.zeros_like(iou)
    r[iou >= 0.75] = 1.0
    r[(iou >= 0.5) & (iou < 0.75)] = 0.3
    r[iou < 0.5] = -0.5
    return r


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


def glp(d, m, s):
    e = (d - m.unsqueeze(1)) / s.unsqueeze(1)
    return -0.5*(e.pow(2)+2*torch.log(s.unsqueeze(1))+math.log(2*math.pi)).sum(dim=-1)


def unfreeze_rlvr(model):
    for p in model.backbone.body.parameters(): p.requires_grad = False
    if hasattr(model.backbone, 'fpn'):
        for p in model.backbone.fpn.parameters(): p.requires_grad = True
    for p in model.rpn.parameters(): p.requires_grad = True
    for p in model.roi_heads.box_head.parameters(): p.requires_grad = True
    for p in model.roi_heads.box_predictor.parameters(): p.requires_grad = True
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            for p in m.parameters(): p.requires_grad = False


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


def bl():
    return build_penn_fudan_loaders({"data": {"root": "./data", "max_size": 320, "train_fraction": 0.8, "num_workers": 0}, "train": {"batch_size": 2}})


def bm():
    return build_detector({"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn", "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn", "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}})


@torch.no_grad()
def ev(model, vl):
    model.eval()
    ps, ts = [], []
    for img, tgt in vl:
        out = model([i.to(DEV) for i in img])
        ps.extend([{k: v.cpu() for k, v in o.items()} for o in out])
        ts.extend([{k: v.cpu() for k, v in t.items()} for t in tgt])
    return evaluate_detection_predictions(ps, ts, iou_threshold=0.5, score_threshold=0.05)


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
    penalty_fn = penalty_sigmoid if mode == "crossproposal_sigmoid" else (penalty_asymmetric if mode == "crossproposal_asym" else None)

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
                log_probs = glp(deltas, mu, s)  # (N, G)

                sp_cat = torch.cat(sp_raw, dim=0)
                sp_exp = sp_cat.repeat_interleave(G_SAMPLES, dim=0)  # (N*G, 4)
                delta_flat = deltas.reshape(-1, 4)  # (N*G, 4)

                bw = sp_exp[:, 2] - sp_exp[:, 0]; bh = sp_exp[:, 3] - sp_exp[:, 1]
                bcx = sp_exp[:, 0] + 0.5 * bw; bcy = sp_exp[:, 1] + 0.5 * bh
                dx = delta_flat[:, 0] / 10.0; dy = delta_flat[:, 1] / 10.0
                dw = delta_flat[:, 2] / 5.0;  dh = delta_flat[:, 3] / 5.0
                decoded_flat = torch.stack([
                    dx*bw + bcx - 0.5*torch.exp(dw)*bw, dy*bh + bcy - 0.5*torch.exp(dh)*bh,
                    dx*bw + bcx + 0.5*torch.exp(dw)*bw, dy*bh + bcy + 0.5*torch.exp(dh)*bh,
                ], dim=1).clamp(min=0)  # (N*G, 4)

                # Compute IoU: for each proposal, find max IoU with any GT in its image
                image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]
                iou_flat = torch.zeros(N * G_SAMPLES, device=DEV)
                img_id_per_proposal = []

                # Map each proposal to its image
                offset_p = 0
                for i_img, p_img in enumerate(sp_raw):
                    np_i = p_img.shape[0]
                    for j in range(np_i):
                        img_id_per_proposal.extend([i_img] * G_SAMPLES)
                    offset_p += np_i

                # IoU per sample
                for i in range(N * G_SAMPLES):
                    i_img = img_id_per_proposal[i]
                    gt = tgts_t[i_img]["boxes"]
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
