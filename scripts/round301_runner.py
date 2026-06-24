"""Plan 3.1: RLVR on classifier — IoU-calibrated confidence reward.

Round 3.0 proved the RLVR framework is healthy (r_std=0.499 stable, conf_shift grows)
but geo reward doesn't align with classification quality.

Round 3.1 fix: reward directly tied to IoU quality.
For proposal i with IoU_i and confidence c_i:
  R_i = c_i * (2 * IoU_i - 1)
  → High IoU: reward = +c (push confidence up, more for higher IoU)
  → Low IoU:  reward = -c (push confidence down, more for lower IoU)

This teaches classifier calibration: confidence should track IoU quality.
Cross-proposal GRPO ranks proposals within each image.
"""
import copy
import shutil
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_penn_fudan_loaders_320,
    decode_boxes,
    evaluate_model,
    gaussian_log_prob,
    unfreeze_rlvr,
)
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
G_SAMPLES = 4
EPOCHS = 8
SEEDS = [42, 123, 456]
RL_WEIGHT = 0.1
KL_WEIGHT = 0.03
HEAD_LR = 0.001
BODY_LR = 0.0001
SIGMA = 0.1


def cross_proposal_grpo(reward, n_proposals_per_img):
    adv = torch.zeros_like(reward)
    offset = 0
    for n_p in n_proposals_per_img:
        if n_p <= 1:
            if n_p == 1:
                adv[offset] = 0.0
            offset += n_p
            continue
        r_img = reward[offset : offset + n_p]
        r_mean = r_img.mean()
        r_std = r_img.std().clamp_min(1e-6)
        adv[offset : offset + n_p] = (r_img - r_mean) / r_std
        offset += n_p
    return adv


def bl():
    return build_penn_fudan_loaders_320(batch_size=2)


def bm():
    return build_detector(
        {
            "model": {
                "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                "pretrained": True,
                "num_classes": 2,
                "min_size": 320,
                "max_size": 320,
            }
        }
    )


def build_opt(model):
    body_params = []
    head_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "box_head" in n or "box_predictor" in n:
            head_params.append(p)
        else:
            body_params.append(p)
    return torch.optim.SGD(
        [{"params": body_params, "lr": BODY_LR}, {"params": head_params, "lr": HEAD_LR}],
        lr=HEAD_LR, momentum=0.9, weight_decay=0.0005,
    )


@torch.no_grad()
def ev(model, vl):
    return evaluate_model(model, vl, DEV, iou_threshold=0.5, score_threshold=0.05)


def compute_iou_per_proposal(sp_raw, bf, box_predictor, tgts_t, image_shapes):
    """Compute max IoU with GT for each proposal using a given box_predictor.
    Uses the box_predictor from baseline to isolate box regression drift from RLVR signal.
    Returns: iou (N,) — max IoU per proposal with any GT in its image.
    """
    sp_cat = torch.cat(sp_raw, dim=0)
    N = sp_cat.shape[0]
    reg_out = box_predictor.bbox_pred(bf[:N])
    person_deltas = reg_out[:, 2:6]  # class 1
    decoded = decode_boxes(sp_cat, person_deltas)

    iou_out = torch.zeros(N, device=DEV)
    offset = 0
    for i_img, p_img in enumerate(sp_raw):
        n_p = p_img.shape[0]
        if n_p == 0:
            continue
        gt = tgts_t[i_img]["boxes"]
        if len(gt) > 0:
            iou_out[offset : offset + n_p] = box_iou(decoded[offset : offset + n_p], gt).max(dim=1).values
        offset += n_p
    return iou_out


def run_one(cfg_name, mode, seed):
    run_name = f"round301_{cfg_name}_s{seed}"
    set_seed(seed)
    model = bm().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"])
    unfreeze_rlvr(model)

    baseline_model = copy.deepcopy(model)
    baseline_model.eval()
    for p in baseline_model.parameters():
        p.requires_grad = False

    sampled_props, box_head_in = {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(
        lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]})
    )
    model.roi_heads.box_head.register_forward_pre_hook(
        lambda m, args: box_head_in.update({"x": args[0]})
    )

    tl, vl = bl()
    run_dir = ensure_run_dir(run_name)
    shutil.copy(__file__, run_dir / "runner_snapshot.py")
    is_det = mode == "det_only_unf"
    is_rlvr = mode == "rlvr_iou_cls"
    opt = build_opt(model)

    h = []
    best_ap75 = -1.0
    diag = {"adv_std": [], "reward_raw_std": [], "conf_shift": [], "mean_iou": []}

    for ep in range(1, EPOCHS + 1):
        model.train()
        td, trl, tkl = 0.0, 0.0, 0.0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}"):
            imgs_d = [i.to(DEV) for i in imgs]
            tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear()
            box_head_in.clear()
            image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]

            ld = model(imgs_d, tgts_t)
            det = sum(ld.values()) if isinstance(ld, dict) else sum(sum(d.values()) for d in ld if isinstance(d, dict))
            rf = box_head_in.get("x")
            sp_raw = sampled_props.get("p")
            rl = torch.tensor(0.0, device=DEV)
            kl_loss = torch.tensor(0.0, device=DEV)

            if not is_det and rf is not None and sp_raw is not None and rf.shape[0] > 0:
                bf = model.roi_heads.box_head(rf)
                cls_logits = model.roi_heads.box_predictor.cls_score(bf)

                with torch.no_grad():
                    baseline_bf = baseline_model.roi_heads.box_head(rf)
                    baseline_cls_conf = F.softmax(
                        baseline_model.roi_heads.box_predictor.cls_score(baseline_bf), dim=-1
                    )[:, 1]

                # Sample G perturbations of classifier logits
                s_cls = torch.full_like(cls_logits, SIGMA)
                perturbed_logits = cls_logits.unsqueeze(1) + s_cls.unsqueeze(1) * torch.randn(
                    cls_logits.shape[0], G_SAMPLES, cls_logits.shape[1], device=DEV
                )  # (N, G, 2)
                perturbed_conf = F.softmax(perturbed_logits, dim=-1)[:, :, 1]  # (N, G)

                log_probs = gaussian_log_prob(perturbed_logits, cls_logits, s_cls)  # (N, G)

                # IoU per proposal — used as ground-truth quality signal
                iou_p = compute_iou_per_proposal(
                    sp_raw, baseline_bf, baseline_model.roi_heads.box_predictor, tgts_t, image_shapes
                )  # (N,)

                # Reward: R = conf * (2*IoU - 1)
                #   IoU=1 → reward=+conf (max push-up)
                #   IoU=0.5 → reward=0 (no signal, neutral)
                #   IoU=0 → reward=-conf (push-down)
                N = min(cls_logits.shape[0], iou_p.shape[0])
                quality = (2 * iou_p[:N] - 1).unsqueeze(1)  # (N, 1)
                reward_img = perturbed_conf[:N] * quality  # (N, G)

                # Cross-proposal GRPO: per-image normalization
                reward_flat = reward_img.reshape(-1)  # (N*G,)
                n_props_per_img = [p.shape[0] * G_SAMPLES for p in sp_raw]
                adv = cross_proposal_grpo(reward_flat, n_props_per_img).view(N, G_SAMPLES)

                diag["adv_std"].append(adv.std().item())
                # Raw reward std BEFORE GRPO normalization (actual diagnostic)
                raw_std_per_img = []
                off_r = 0
                for p in sp_raw:
                    n_r = p.shape[0] * G_SAMPLES
                    if n_r > 0:
                        raw_std_per_img.append(reward_flat[off_r : off_r + n_r].std().item())
                    off_r += n_r
                diag["reward_raw_std"].append(np.mean(raw_std_per_img) if raw_std_per_img else 0.0)
                rl = -(adv.detach() * log_probs[:N]).mean()

                # Function-space KL
                kl_loss = KL_WEIGHT * (perturbed_conf[:N] - baseline_cls_conf[:N].unsqueeze(1)).pow(2).mean()

                diag["conf_shift"].append((perturbed_conf[:N].mean() - baseline_cls_conf[:N].mean()).item())
                diag["mean_iou"].append(iou_p[:N].mean().item())

            loss = det + RL_WEIGHT * rl + kl_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            td += det.item()
            trl += rl.item()
            tkl += kl_loss.item()

        em = ev(model, vl)
        rs_m = np.mean(diag["adv_std"]) if diag["adv_std"] else 0.0
        rr_m = np.mean(diag["reward_raw_std"]) if diag["reward_raw_std"] else 0.0
        cs_m = np.mean(diag["conf_shift"]) if diag["conf_shift"] else 0.0
        mi_m = np.mean(diag["mean_iou"]) if diag["mean_iou"] else 0.0

        row = {
            "epoch": ep,
            "val_ap50": em["ap50"], "val_ap75": em["ap75"],
            "pre": em.get("precision", 0), "rec": em.get("recall", 0),
            "ece": em.get("ece", 0),
            "adv_std": float(rs_m), "reward_raw_std": float(rr_m),
            "conf_shift": float(cs_m), "mean_iou": float(mi_m),
            "det_loss": td, "rl_loss": trl, "kl_loss": tkl,
        }
        h.append(row)
        print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} "
              f"adv_std={rs_m:.4f} raw_r_std={rr_m:.4f} conf_shift={cs_m:.4f} mean_iou={mi_m:.4f}")
        if em["ap75"] > best_ap75:
            best_ap75 = em["ap75"]
        for k in diag:
            diag[k].clear()

    best_h = max(h, key=lambda r: r["val_ap75"])
    em.update({
        "run_name": run_name, "config": cfg_name, "mode": mode, "seed": seed,
        "epochs": len(h), "best_ap50": best_h["val_ap50"], "best_ap75": best_ap75,
        "history": h, "git_hash": GIT,
    })
    save_json(em, run_dir / "eval_metrics.json")
    return em


if __name__ == "__main__":
    all_results = []
    for cfg, mode in [("det_only_unf", "det_only_unf"), ("rlvr_iou_cls", "rlvr_iou_cls")]:
        for s in SEEDS:
            r = run_one(cfg, mode, s)
            all_results.append(r)

    print("\n## Plan 3.1 RLVR on Classifier — IoU-Calibrated Confidence Reward")
    fmt = "  {:<18s} s{} AP75={:.4f} best={:.4f} AP50={:.4f}"
    for r in all_results:
        bh = max(r["history"], key=lambda x: x["val_ap75"])
        print(fmt.format(r["config"], r["seed"], r["ap75"], r["best_ap75"], bh["val_ap50"]))
    for cfg in ["det_only_unf", "rlvr_iou_cls"]:
        vals = [r for r in all_results if r["config"] == cfg]
        if vals:
            bv = [v["best_ap75"] for v in vals]
            print(f"  {cfg}: bestAP75={np.mean(bv):.4f} +/- {np.std(bv):.4f}")
