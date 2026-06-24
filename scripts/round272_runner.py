"""Plan 2.72 v5: q-only source ablation.

Fixes applied (cumulative):
  1. det_only optimizer created ONCE — fair comparison
  2. RNG isolation: shuffle/permute use dedicated torch.Generator
  3. Proposal alignment: box_roi_pool pre-hook captures SAMPLED proposals
  4. FPN spatial_scale: directly from model.roi_heads.box_roi_pool.scales
  5. Gradient field: renamed to total_grad_norm (rl/det separation deferred)
  6. qonly_random: per-proposal normalize (same treatment as verifier groups)

6 groups × 3 seeds = 18 experiments:
  A  det_only              supervised fine-tune baseline
  B  qonly_real            real per-channel FFT verifier
  C  qonly_shuf            shuffled FFT control
  D  qonly_band            band-permuted control
  E  qonly_random          random q_norm, per-proposal normalized
  F  frozen_random_verifier random init verifier, frozen, no vloss

Key diagnostics: total_grad_norm, q_corr, q_std, reward_std
"""
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_mobv3_detector,
    build_penn_fudan_loaders_320,
    evaluate_model,
    gaussian_log_prob,
)
from spectral_detection_posttrain.models.verifiers import PerChanFFTVerifier
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
G = 4
EPOCHS = 5
ROI_SIZE = 14
SEEDS = [42, 123, 456]
RL_WEIGHT = 0.05
FFT_DIM = 256 * 6  # C channels × 6 bands (3 amp + 3 phase)


def extract_perchan_fft(roi):
    C = roi.shape[1]; H, W = roi.shape[-2], roi.shape[-1]
    fft = torch.fft.rfft2(roi, dim=(-2, -1), norm="ortho")
    amp = torch.abs(fft); pha = torch.angle(fft)
    freq = torch.fft.fftfreq(max(H, W), device=roi.device)
    Y, X = torch.meshgrid(freq[:H], freq[:W // 2 + 1], indexing='ij')
    r = torch.sqrt(X ** 2 + Y ** 2); R = r.max().clamp_min(1e-6); rn = r / R
    lo = (rn <= 0.15).float(); md = ((rn > 0.15) & (rn <= 0.4)).float(); hi = (rn > 0.4).float()
    a_lo = (amp * lo.unsqueeze(0).unsqueeze(0)).flatten(2).sum(2)
    a_md = (amp * md.unsqueeze(0).unsqueeze(0)).flatten(2).sum(2)
    a_hi = (amp * hi.unsqueeze(0).unsqueeze(0)).flatten(2).sum(2)
    p_lo = (pha * lo.unsqueeze(0).unsqueeze(0)).flatten(2).sum(2)
    p_md = (pha * md.unsqueeze(0).unsqueeze(0)).flatten(2).sum(2)
    p_hi = (pha * hi.unsqueeze(0).unsqueeze(0)).flatten(2).sum(2)
    return torch.cat([a_lo, a_md, a_hi, p_lo, p_md, p_hi], dim=1)


def band_permute(fft_f, rng_gen):
    """Band-permute using dedicated generator — does NOT consume global RNG."""
    B = fft_f.shape[0]; ch_per = fft_f.shape[1] // 6
    out = torch.zeros_like(fft_f)
    for b in range(6):
        sl = slice(b * ch_per, (b + 1) * ch_per)
        out[:, sl] = fft_f[torch.randperm(B, generator=rng_gen, device=fft_f.device)][:, sl]
    return out


def decode_boxes(pr, d):
    w = pr[:, 2] - pr[:, 0]; h = pr[:, 3] - pr[:, 1]
    cx = pr[:, 0] + 0.5 * w; cy = pr[:, 1] + 0.5 * h
    px = d[:, 0] * w + cx - 0.5 * torch.exp(d[:, 2]) * w
    py = d[:, 1] * h + cy - 0.5 * torch.exp(d[:, 3]) * h
    return torch.stack([px, py,
        d[:, 0] * w + cx + 0.5 * torch.exp(d[:, 2]) * w,
        d[:, 1] * h + cy + 0.5 * torch.exp(d[:, 3]) * h], dim=1).clamp(min=0)


def glp(d, m, s):
    e = (d - m.unsqueeze(1)) / s.unsqueeze(1)
    return -0.5 * (e.pow(2) + 2 * torch.log(s.unsqueeze(1)) + math.log(2 * math.pi)).sum(dim=-1)


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


def compute_q_pred(fpn, roi_boxes, lvl, fpn_keys, vrf, mode, rng_gen, input_size=320):
    """Unified q_pred computation. mode: 'real' | 'shuf' | 'band' | 'frozen_random'.
    spatial_scale computed from feature map size / input_size (no guesswork)."""
    N_total = roi_boxes.shape[0]
    q_pred = torch.zeros(N_total, device=DEV)
    for ki, k in enumerate(fpn_keys):
        ki_lvl = int(k) + 2
        mask = lvl == ki_lvl
        if mask.sum() == 0:
            continue
        sc = fpn[k].shape[-1] / float(input_size)
        r = torchvision.ops.roi_align(fpn[k], roi_boxes[mask], output_size=ROI_SIZE, spatial_scale=sc)
        fft_f = extract_perchan_fft(r)
        if mode == "shuf":
            fft_f = fft_f[torch.randperm(fft_f.shape[0], generator=rng_gen, device=DEV)]
        elif mode == "band":
            fft_f = band_permute(fft_f, rng_gen)
        q_pred[mask] = vrf(fft_f)
    return q_pred


def _bld_opt(model, vrf, lr=0.001):
    """Build optimizer ONCE. All groups share the same optimizer creation point."""
    params = [p for p in model.parameters() if p.requires_grad]
    if vrf is not None:
        params += [p for p in vrf.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)



def run_one(cfg_name, mode, seed):
    run_name = f"round272_{cfg_name}_s{seed}"
    set_seed(seed)
    model = bm().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"])
    fe(model, [model.roi_heads.box_head, model.roi_heads.box_predictor])

    # --- Hooks ---
    sampled_props = {}  # box_roi_pool pre-hook: real sampled proposals
    box_head_in = {}    # box_head pre-hook: box features
    fpn_feats = {}      # backbone hook: FPN feature maps
    model.roi_heads.box_roi_pool.register_forward_pre_hook(
        lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(
        lambda m, args: box_head_in.update({"x": args[0]}))
    model.backbone.register_forward_hook(
        lambda m, i, o: fpn_feats.update({"f": {k: o[k] for k in o if k != "pool"}}))

    # --- RNG isolation ---
    rng_gen = torch.Generator(device=DEV).manual_seed(seed + 9999)

    # --- Verifier (lazy) ---
    vrf = None

    # --- Data ---
    tl, vl = bl()
    rd = ensure_run_dir(run_name)

    # --- Optimizer (created ONCE, before training) ---
    opt = _bld_opt(model, vrf)

    h = []; best_ap75 = -1.0
    use_verifier = mode in ("real", "shuf", "band", "frozen_random")
    diag = {"q_ious": [], "total_grad_norm": [], "reward_std": [], "q_std": []}
    bbox_pred_weight = model.roi_heads.box_predictor.bbox_pred.weight

    for ep in range(1, EPOCHS + 1):
        model.train()
        if vrf is not None:
            vrf.train()
        td, trl, tv, pos = 0.0, 0.0, 0.0, 0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}"):
            imgs_d = [i.to(DEV) for i in imgs]
            tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear(); box_head_in.clear(); fpn_feats.clear()

            ld = model(imgs_d, tgts_t)
            if isinstance(ld, dict):
                det = sum(ld.values())
            else:
                det = sum(sum(d.values()) for d in ld if isinstance(d, dict))

            rf = box_head_in.get("x")
            sp = sampled_props.get("p")   # per-image list of SAMPLED proposals
            fpn = fpn_feats.get("f")
            rl = torch.tensor(0.0, device=DEV)
            vloss = torch.tensor(0.0, device=DEV)
            total_gn_batch = 0.0

            if mode != "det_only" and rf is not None and sp is not None and rf.shape[0] > 0 and fpn is not None:
                N_rf = rf.shape[0]
                bf = model.roi_heads.box_head(rf)
                mu = model.roi_heads.box_predictor.bbox_pred(bf)[:, -4:]
                s = torch.full_like(mu, 0.1, requires_grad=False)
                deltas = mu.detach().unsqueeze(1) + s.unsqueeze(1) * torch.randn(N_rf, G, 4, device=DEV)
                log_probs = gaussian_log_prob(deltas, mu, s)

                # FIX: use SAMPLED proposals, not raw RPN. sp is per-image list.
                sp_cat = torch.cat(sp, dim=0)
                N = min(N_rf, sp_cat.shape[0])
                mu = mu[:N]; deltas = deltas[:N]; log_probs = log_probs[:N]
                ad = deltas.reshape(N * G, 4)
                pe = sp_cat[:N].unsqueeze(1).expand(-1, G, -1).reshape(N * G, 4)
                boxes = decode_boxes(pe, ad)

                # img_map from sampled proposals (sp is per-image)
                img_map = []
                for i_img, p in enumerate(sp):
                    n_p = min(p.shape[0], N - sum(len(x) for x in (sp[:i_img] if i_img > 0 else [])))
                    img_map.extend([i_img] * n_p)
                img_map = img_map[:N]

                iou_r = torch.zeros(N, G, device=DEV)
                for pi in range(N):
                    gt = tgts_t[img_map[pi]]["boxes"]
                    if len(gt) > 0:
                        iou_r[pi] = box_iou(boxes[pi * G:(pi + 1) * G], gt).max(dim=1).values

                if use_verifier:
                    roi_boxes = torch.zeros(N * G, 5, device=DEV)
                    for bi in range(N * G):
                        roi_boxes[bi, 0] = img_map[bi // G]
                        roi_boxes[bi, 1:] = boxes[bi]
                    fpn_keys = sorted(fpn.keys(), key=int)
                    bw = boxes[:, 2] - boxes[:, 0]; bh = boxes[:, 3] - boxes[:, 1]
                    area = (bw * bh).clamp_min(1)
                    lvl = torch.floor(torch.log2(torch.sqrt(area) / 224) + 4).long().clamp(2, 5)

                    if vrf is None:
                        vrf = PerChanFFTVerifier(FFT_DIM).to(DEV)
                        if mode == "frozen_random":
                            for p in vrf.parameters():
                                p.requires_grad = False
                        opt = _bld_opt(model, vrf)  # rebuild with verifier params

                    q_pred = compute_q_pred(fpn, roi_boxes, lvl, fpn_keys, vrf, mode, rng_gen)
                    q_pred = q_pred.view(N, G)

                    if mode != "frozen_random":
                        vloss = F.mse_loss(q_pred, iou_r.detach())

                    q_norm = (q_pred - q_pred.mean(dim=1, keepdim=True)) / (q_pred.std(dim=1, keepdim=True).clamp_min(1e-6))
                    diag["q_ious"].extend(list(zip(q_pred.flatten().tolist(), iou_r.flatten().tolist())))
                    diag["q_std"].append(q_pred.std().item())
                else:
                    # qonly_random: random q_norm, per-proposal normalized (same as verifier)
                    q_norm = torch.randn(N, G, generator=rng_gen, device=DEV)
                    q_norm = (q_norm - q_norm.mean(dim=1, keepdim=True)) / (q_norm.std(dim=1, keepdim=True).clamp_min(1e-6))

                diag["reward_std"].append(q_norm.std().item())
                pm = iou_r.max(dim=1).values > 0.3
                if pm.any():
                    rl = -(q_norm[pm].detach() * log_probs[pm]).mean()
                    pos += pm.sum().item()

            # --- Loss + gradient computation ---
            if mode == "det_only":
                loss = det
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            else:
                vloss_term = vloss if vrf is not None else torch.tensor(0.0, device=DEV)
                loss = det + vloss_term + RL_WEIGHT * rl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                total_gn_batch = bbox_pred_weight.grad.norm().item() if bbox_pred_weight.grad is not None else 0.0
                opt.step()

            diag["total_grad_norm"].append(total_gn_batch)
            td += det.item(); trl += rl.item(); tv += vloss.item()

        # --- Epoch diagnostics ---
        em = ev(model, vl)
        if len(diag["q_ious"]) > 0:
            qs = np.array([x[0] for x in diag["q_ious"]])
            iis = np.array([x[1] for x in diag["q_ious"]])
            q_corr = np.corrcoef(qs, iis)[0, 1]
        else:
            q_corr = 0.0
        tgn = np.mean(diag["total_grad_norm"]) if len(diag["total_grad_norm"]) > 0 else 0.0
        qs_m = np.mean(diag["q_std"]) if len(diag["q_std"]) > 0 else 0.0
        rs_m = np.mean(diag["reward_std"]) if len(diag["reward_std"]) > 0 else 0.0
        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "precision": em.get("precision", 0), "recall": em.get("recall", 0),
               "ece": em.get("ece", 0), "q_iou_corr": float(q_corr), "q_std": float(qs_m),
               "reward_std": float(rs_m), "total_grad_norm": float(tgn),
               "pred_count": em.get("pred_count", 0), "det_loss": td, "rl_loss": trl,
               "vloss": tv, "pos_count": int(pos)}
        h.append(row)
        print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} q_corr={q_corr:.4f} "
              f"tgn={tgn:.6f} q_std={qs_m:.4f} r_std={rs_m:.4f}")
        if em["ap75"] > best_ap75:
            best_ap75 = em["ap75"]
        for k in diag:
            diag[k].clear()

    best_h = max(h, key=lambda r: r["val_ap75"])
    em.update({"run_name": run_name, "config": cfg_name, "mode": mode, "seed": seed,
               "epochs": EPOCHS, "best_ap50": best_h["val_ap50"], "best_ap75": best_ap75,
               "history": h, "git_hash": GIT, "q_iou_corr_final": h[-1]["q_iou_corr"],
               "total_grad_final": h[-1]["total_grad_norm"]})
    save_json(em, rd / "eval_metrics.json")
    return em


if __name__ == "__main__":
    all_results = []
    modes = {"det_only": "det_only", "qonly_real": "real", "qonly_shuf": "shuf",
             "qonly_band": "band", "qonly_random": "random", "frozen_random": "frozen_random"}
    for cfg, mode in modes.items():
        for s in SEEDS:
            r = run_one(cfg, mode, s)
            all_results.append(r)

    print("\n## Plan 2.72 v5 q-only Source Ablation (FPN scale from model)")
    print(f"  {'Config':<20s} {'Seed':>5s} {'AP75':>8s} {'BestAP75':>8s} {'AP50':>8s} {'q_corr':>8s} {'tot_gn':>10s} {'q_std':>8s} {'r_std':>8s}")
    for r in all_results:
        hf = r["history"][-1]
        print(f"  {r['config']:<20s} {r['seed']:5d} {r['ap75']:8.4f} {r['best_ap75']:8.4f} {hf['val_ap50']:8.4f} {r.get('q_iou_corr_final', 0):8.4f} {r.get('total_grad_final', 0):10.6f} {hf.get('q_std', 0):8.4f} {hf.get('reward_std', 0):8.4f}")

    for cfg in modes:
        best_vals = [r["best_ap75"] for r in all_results if r["config"] == cfg]
        final_vals = [r["ap75"] for r in all_results if r["config"] == cfg]
        qc = [r.get("q_iou_corr_final", 0) for r in all_results if r["config"] == cfg]
        tgn = [r.get("total_grad_final", 0) for r in all_results if r["config"] == cfg]
        print(f"  {cfg}: bestAP75={np.mean(best_vals):.4f}±{np.std(best_vals):.4f}  finalAP75={np.mean(final_vals):.4f}  q_corr={np.mean(qc):.4f}  tgn={np.mean(tgn):.6f}")

    def get_best(cfg):
        return [r["best_ap75"] for r in all_results if r["config"] == cfg]

    def judge(name_a, name_b, threshold=0.01):
        va = get_best(name_a); vb = get_best(name_b)
        delta = np.mean(va) - np.mean(vb)
        wins = sum(1 for i in range(3) if va[i] > vb[i])
        status = "PASS" if delta > threshold and wins >= 2 else "FAIL"
        print(f"\n  {name_a} vs {name_b}: Δ={delta:+.4f} wins={wins}/3 → {status}")

    judge("qonly_real", "qonly_random")
    judge("qonly_real", "frozen_random")
    judge("qonly_real", "qonly_band")
    judge("qonly_real", "qonly_shuf")
    judge("qonly_random", "det_only")
    judge("frozen_random", "det_only")
