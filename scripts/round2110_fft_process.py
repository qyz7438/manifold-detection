"""Plan 2.110: FFT as process reward — exploration modulator for RLVR.

Instead of FFT as outcome reward (failed), FFT guides exploration:
  - Proposals with high spectral ambiguity → larger sigma → more exploration
  - Proposals with high spectral confidence → smaller sigma → exploit current knowledge

This makes RLVR explore more where classifier and FFT disagree.
Based on k2.7's discovery: process reward is untapped in detection RLVR.
"""
import copy, shutil, subprocess, sys
import numpy as np, torch, torch.nn.functional as F
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_penn_fudan_loaders_320, decode_boxes, evaluate_model,
    gaussian_log_prob, unfreeze_rlvr, extract_perchan_fft,
)
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import set_seed

DEV, SEED = "cuda", 42; SEEDS = [42, 123, 456]
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
G_SAMPLES, EPOCHS = 4, 8
RL_WEIGHT, KL_WEIGHT, FFT_WEIGHT = 0.0005, 0.01, 0.1

def bm():
    return build_detector({"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}})

def build_opt(model):
    body, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (head if "box_head" in n or "box_predictor" in n else body).append(p)
    return torch.optim.SGD([{"params": body, "lr": 0.0001}, {"params": head, "lr": 0.001}],
                           lr=0.001, momentum=0.9, weight_decay=0.0005)

def spectral_ambiguity(crops):
    """How uncertain is the spectral evidence? High = explore more.
    Uses entropy of frequency band distribution — peaked spectrum = confident, flat = ambiguous.
    """
    f = extract_perchan_fft(crops); ch = f.shape[1] // 6
    amp = f[:, 0*ch:3*ch]  # (N, 3*ch, 7, 4)
    amp_bands = amp.reshape(amp.shape[0], 3, -1).mean(dim=-1)  # (N, 3)
    band_prob = amp_bands / (amp_bands.sum(dim=1, keepdim=True) + 1e-8)
    entropy = -(band_prob * torch.log(band_prob + 1e-8)).sum(dim=1) / np.log(3)  # [0,1]
    return entropy  # (N,) high=ambiguous=explore more

def cross_proposal_grpo(reward, n_props):
    adv = torch.zeros_like(reward); off = 0
    for n_p in n_props:
        if n_p <= 0: continue
        if n_p == 1: adv[off] = 0.0; off += n_p; continue
        r = reward[off:off+n_p]; m = r.mean(); s = r.std().clamp_min(1e-6)
        adv[off:off+n_p] = (r - m) / s; off += n_p
    return adv


def run_one(cfg_name, mode, seed):
    run_name = f"round2110_{cfg_name}_s{seed}"; set_seed(seed)
    model = bm().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV); model.load_state_dict(ckpt["model"])
    unfreeze_rlvr(model)
    baseline_model = copy.deepcopy(model); baseline_model.eval()
    for p in baseline_model.parameters(): p.requires_grad = False

    sampled_props, box_head_in, roi_crops = {}, {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(lambda m, args: box_head_in.update({"x": args[0]}))
    use_fft = mode != "det_only"
    if use_fft:
        model.roi_heads.box_roi_pool.register_forward_hook(lambda m, i, o: roi_crops.update({"c": o.clone()}))

    tl, vl = build_penn_fudan_loaders_320(batch_size=2)
    run_dir = ensure_run_dir(run_name); shutil.copy(__file__, run_dir / "runner_snapshot.py")
    opt = build_opt(model); baseline_bp = baseline_model.roi_heads.box_predictor
    is_det = mode == "det_only"; h, best_ap75 = [], -1.0; diag = {"entropy": [], "conf_shift": []}

    for ep in range(1, EPOCHS + 1):
        model.train(); td, trl, tkl = 0.0, 0.0, 0.0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}", leave=False):
            imgs_d = [i.to(DEV) for i in imgs]; tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear(); box_head_in.clear()
            if use_fft: roi_crops.clear()
            ld = model(imgs_d, tgts_t); det = sum(ld.values())
            rf = box_head_in.get("x"); sp_raw = sampled_props.get("p")
            if use_fft: crops = roi_crops.get("c")
            rl = torch.tensor(0.0, device=DEV); kl = torch.tensor(0.0, device=DEV)

            if not is_det and rf is not None and sp_raw is not None and rf.shape[0] > 0:
                bf = model.roi_heads.box_head(rf); cls_logits = model.roi_heads.box_predictor.cls_score(bf)
                with torch.no_grad():
                    baseline_bf = baseline_model.roi_heads.box_head(rf)
                    baseline_cls_conf = F.softmax(baseline_bp.cls_score(baseline_bf), dim=-1)[:, 1]

                # FFT process reward: spectral ambiguity → modulates exploration sigma
                if use_fft and crops is not None:
                    entropy = spectral_ambiguity(crops)  # (N,) [0,1]
                else:
                    entropy = torch.zeros(cls_logits.shape[0], device=DEV)

                # Per-proposal sigma modulated by spectral ambiguity
                sp_sigma = 0.05 + 0.2 * (1.0 - baseline_cls_conf) + FFT_WEIGHT * entropy

                with torch.no_grad():
                    s_base = sp_sigma.unsqueeze(1).expand(-1, cls_logits.shape[1])
                    bl_logits = baseline_bp.cls_score(baseline_bf)
                    perturbed = bl_logits.unsqueeze(1) + s_base.unsqueeze(1) * torch.randn(
                        bl_logits.shape[0], G_SAMPLES, bl_logits.shape[1], device=DEV)
                pert_conf = F.softmax(perturbed, dim=-1)[:, :, 1]
                s_cls = sp_sigma.unsqueeze(1).expand(-1, cls_logits.shape[1])
                log_probs = gaussian_log_prob(perturbed, cls_logits, s_cls)

                # Standard IoU×conf reward
                sp_cat = torch.cat(sp_raw, dim=0)
                N = min(cls_logits.shape[0], sp_cat.shape[0])
                with torch.no_grad():
                    reg = baseline_bp.bbox_pred(baseline_bf[:N])
                    decoded = decode_boxes(sp_cat[:N], reg[:, 2:6])
                iou_p = torch.zeros(N, device=DEV); off = 0
                for i_img, p_img in enumerate(sp_raw):
                    np_ = p_img.shape[0]
                    if np_ > 0 and off + np_ <= N:
                        gt = tgts_t[i_img]["boxes"]
                        if len(gt) > 0:
                            iou_p[off:off+np_] = box_iou(decoded[off:off+np_], gt).max(dim=1).values
                    off += np_
                iou_p = iou_p[:N]

                quality = (2 * iou_p - 1).unsqueeze(1)
                reward = pert_conf[:N] * quality
                reward_flat = reward.reshape(-1)
                npp = [p.shape[0] * G_SAMPLES for p in sp_raw]
                adv = cross_proposal_grpo(reward_flat, npp).view(N, G_SAMPLES)
                rl = -(adv.detach() * log_probs[:N]).mean()
                kl = KL_WEIGHT * (pert_conf[:N] - baseline_cls_conf[:N].unsqueeze(1)).pow(2).mean()
                diag["entropy"].append(entropy[:N].mean().item())
                diag["conf_shift"].append((pert_conf[:N].mean() - baseline_cls_conf[:N].mean()).item())

            loss = det + RL_WEIGHT * rl + kl
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            td += det.item(); trl += rl.item(); tkl += kl.item()

        em = evaluate_model(model, vl, DEV)
        ent_m = np.mean(diag["entropy"]) if diag["entropy"] else 0
        cs_m = np.mean(diag["conf_shift"]) if diag["conf_shift"] else 0
        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "ece": em.get("ece", 0), "entropy": float(ent_m), "conf_shift": float(cs_m),
               "det": td, "rl": trl, "kl": tkl}
        h.append(row); print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} ent={ent_m:.3f}")
        if em["ap75"] > best_ap75: best_ap75 = em["ap75"]
        for k in diag: diag[k].clear()

    best_h = max(h, key=lambda r: r["val_ap75"])
    em.update({"run_name": run_name, "config": cfg_name, "mode": mode, "seed": seed,
               "epochs": len(h), "best_ap50": best_h["val_ap50"], "best_ap75": best_ap75, "history": h})
    save_json(em, run_dir / "eval_metrics.json"); return em


if __name__ == "__main__":
    results = []
    for cfg, mode in [("det_only", "det_only"), ("fft_process", "fft_process")]:
        for s in SEEDS: results.append(run_one(cfg, mode, s))
    print("\n## 2.110 FFT Process Reward")
    for r in results:
        bh = max(r["history"], key=lambda x: x["val_ap75"])
        print(f"  {r['config']:<12s} s{r['seed']} AP75={r['best_ap75']:.4f} AP50={bh['val_ap50']:.4f}")
    for cfg in ["det_only", "fft_process"]:
        vals = [r for r in results if r["config"] == cfg]
        if vals: print(f"  {cfg}: bestAP75={np.mean([v['best_ap75'] for v in vals]):.4f} +/- {np.std([v['best_ap75'] for v in vals]):.4f}")
