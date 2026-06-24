"""Plan 2.81: Energy Residual Verifier — penalize high-energy proposals.
Key finding: FFT energy on ROI features separates FN from TP (Cohen d=0.99),
and correlates negatively with IoU (r=-0.49). Higher energy = more FN-like.
R_residual = -energy (penalize high energy), added to R_loc with small weight.
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
G_SAMPLES = 4
EPOCHS = 8
SEEDS = [42, 123, 456]
RL_WEIGHT = 0.05
KL_WEIGHT = 0.1
ENERGY_PENALTY_WEIGHT = 0.05  # small so it doesn't dominate R_loc
HEAD_LR = 0.001
BODY_LR = 0.0001


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
    """Low-freq energy concentration. Higher = more FN-like (Cohen d=0.99)."""
    ch = fft_f.shape[1] // 6
    a_lo = fft_f[:, 0*ch:1*ch].sum(dim=1)
    a_md = fft_f[:, 1*ch:2*ch].sum(dim=1)
    a_hi = fft_f[:, 2*ch:3*ch].sum(dim=1)
    low_ratio = a_lo / (a_lo + a_md + a_hi + 1e-8)
    return 2 * low_ratio - 1  # [-1, 1], higher = more FN-like


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
    return build_penn_fudan_loaders({"data": {"root": "./data", "max_size": 320, "train_fraction": 0.8, "num_workers": 0}, "train": {"batch_size": 4}})


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


def run_one(cfg_name, mode, seed):
    run_name = f"round281_{cfg_name}_s{seed}"
    set_seed(seed)
    model = bm().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"])
    unfreeze_rlvr(model)
    box_pool = model.roi_heads.box_roi_pool

    # KL baseline
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
    use_energy = mode == "energy_residual"
    opt = build_opt(model)
    bbox_pred_weight = model.roi_heads.box_predictor.bbox_pred.weight

    h = []; best_ap75 = -1.0
    diag = {"reward_std": [], "energy_vals": [], "energy_tp": [], "energy_fn": []}

    baseline_bbox_w = baseline_model.roi_heads.box_predictor.bbox_pred.weight.detach().clone()
    baseline_bbox_b = baseline_model.roi_heads.box_predictor.bbox_pred.bias.detach().clone()

    for ep in range(1, EPOCHS + 1):
        model.train()
        td, trl, tkl, tenergy = 0.0, 0.0, 0.0, 0.0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}"):
            imgs_d = [i.to(DEV) for i in imgs]
            tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear(); box_head_in.clear(); fpn_feats.clear()

            ld = model(imgs_d, tgts_t)
            det = sum(ld.values()) if isinstance(ld, dict) else sum(sum(d.values()) for d in ld if isinstance(d, dict))

            rf = box_head_in.get("x"); sp_raw = sampled_props.get("p"); fpn = fpn_feats.get("f")
            rl = torch.tensor(0.0, device=DEV)
            kl_loss = torch.tensor(0.0, device=DEV)
            total_gn_batch = 0.0

            if not is_det and rf is not None and sp_raw is not None and rf.shape[0] > 0 and fpn is not None:
                N_rf = rf.shape[0]
                bf = model.roi_heads.box_head(rf)
                mu = model.roi_heads.box_predictor.bbox_pred(bf)[:, -4:]

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

                # R_loc: base reward from IoU
                reward_img = compute_loc_reward(iou_r)

                # Energy residual penalty
                if use_energy:
                    image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]
                    with torch.no_grad():
                        pooled = box_pool(fpn, decoded_list, image_shapes)
                    fft_f = extract_perchan_fft(pooled)
                    energy = compute_energy(fft_f).view(offset, G_SAMPLES)  # (N, G), higher = more FN-like
                    r_energy_penalty = -ENERGY_PENALTY_WEIGHT * energy  # penalty for high energy
                    reward_img = reward_img + r_energy_penalty

                    # Track energy stats
                    max_iou = iou_r.max(dim=1).values
                    tp_mask = max_iou >= 0.5
                    fn_mask = max_iou < 0.5
                    diag["energy_vals"].append(energy.mean().item())
                    if tp_mask.any():
                        diag["energy_tp"].append(energy[tp_mask].mean().item())
                    if fn_mask.any():
                        diag["energy_fn"].append(energy[fn_mask].mean().item())

                adv = grpo_advantage(reward_img)
                diag["reward_std"].append(adv.std().item())
                soft_w = iou_r.max(dim=1).values.clamp(0, 1).unsqueeze(1)
                rl = -(adv.detach() * log_probs * soft_w).mean()

            loss = det + RL_WEIGHT * rl + kl_loss
            opt.zero_grad(set_to_none=True); loss.backward()
            total_gn_batch = bbox_pred_weight.grad.norm().item() if bbox_pred_weight.grad is not None else 0.0
            opt.step()

            td += det.item(); trl += rl.item(); tkl += kl_loss.item()
            if use_energy and len(diag["energy_vals"]) > 0:
                tenergy += diag["energy_vals"][-1]

        em = ev(model, vl)
        rs_m = np.mean(diag["reward_std"]) if diag["reward_std"] else 0.0
        en_m = np.mean(diag["energy_vals"]) if diag["energy_vals"] else 0.0
        en_tp = np.mean(diag["energy_tp"]) if diag["energy_tp"] else float("nan")
        en_fn = np.mean(diag["energy_fn"]) if diag["energy_fn"] else float("nan")
        en_gap = en_tp - en_fn if not np.isnan(en_tp) and not np.isnan(en_fn) else float("nan")

        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "precision": em.get("precision", 0), "recall": em.get("recall", 0),
               "ece": em.get("ece", 0), "reward_std": float(rs_m),
               "energy_mean": float(en_m), "energy_tp": float(en_tp), "energy_fn": float(en_fn),
               "energy_gap": float(en_gap),
               "det_loss": td, "rl_loss": trl, "kl_loss": tkl}
        h.append(row)
        print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} "
              f"r_std={rs_m:.4f} en={en_m:.4f} en_tp={en_tp:.4f} en_fn={en_fn:.4f} en_gap={en_gap:.4f}")
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
    # 3 configs: baseline, energy residual G=4
    configs = {
        "det_only_unf": "det_only_unf",
        "energy_residual": "energy_residual",
    }
    for cfg, mode in configs.items():
        for s in SEEDS:
            r = run_one(cfg, mode, s)
            all_results.append(r)

    print("\n## Plan 2.81 Energy Residual Verifier")
    print(f"  {'Config':<18s} {'Seed':>5s} {'AP75':>8s} {'BestAP75':>8s} {'AP50':>8s} {'r_std':>8s} {'en':>8s} {'en_tp':>8s} {'en_fn':>8s} {'en_gap':>8s}")
    for r in all_results:
        best_h = max(r["history"], key=lambda x: x["val_ap75"])
        print(f"  {r['config']:<18s} {r['seed']:5d} {r['ap75']:8.4f} {r['best_ap75']:8.4f} "
              f"{best_h['val_ap50']:8.4f} {best_h.get('reward_std', 0):8.4f} "
              f"{best_h.get('energy_mean', 0):8.4f} {best_h.get('energy_tp', 0):8.4f} "
              f"{best_h.get('energy_fn', 0):8.4f} {best_h.get('energy_gap', 0):8.4f}")

    for cfg in configs:
        vals = [r for r in all_results if r["config"] == cfg]
        if not vals: continue
        bv = [r["best_ap75"] for r in vals]
        print(f"  {cfg}: bestAP75={np.mean(bv):.4f} +/- {np.std(bv):.4f}")
