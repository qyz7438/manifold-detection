"""Plan 2.86: Geometric reward (area + cx + cy + aspect).

AUC analysis found this combination (without energy) has AUC=0.9966 for FN detection.
Test: does penalizing small/narrow/edge boxes via PG improve AP75?
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
    build_penn_fudan_loaders_320,
    compute_loc_reward,
    decode_boxes,
    evaluate_model,
    gaussian_log_prob,
    grpo_advantage,
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
RL_WEIGHT = 0.05
KL_WEIGHT = 0.1
BETA = 0.02
HEAD_LR = 0.001
BODY_LR = 0.0001
IOU_LO = 0.3
IOU_HI = 0.55


def geo_penalty(boxes, img_w, img_h):
    """Penalize small, narrow, edge-positioned boxes. Normalized per-group."""
    # boxes: (N, 4) in [x1,y1,x2,y2]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    area = w * h
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    aspect = w / torch.clamp(h, min=1)

    # Normalize each metric to [-1, 1] range
    area_n = (area - 500) / 1000  # ~N(0,1) scale
    cx_n = (cx - img_w / 2) / (img_w / 4)  # center = 0, edge = +1
    cy_n = (cy - img_h / 2) / (img_h / 4)
    aspect_n = (aspect - 0.4) / 0.2  # narrow (<0.3) = negative

    # Combine: penalize non-center, small, narrow
    penalty = torch.stack(
        [
            torch.abs(cx_n),
            torch.abs(cy_n),
            -area_n.clamp(-1, 1),
            -aspect_n.clamp(-1, 1),
        ],
        dim=1,
    ).mean(dim=1)
    return -torch.sigmoid(3 * (penalty - 0.5))  # sigmoid makes it nonlinear


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
        [
            {"params": body_params, "lr": BODY_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        lr=HEAD_LR,
        momentum=0.9,
        weight_decay=0.0005,
    )


def bl():
    return build_penn_fudan_loaders_320(batch_size=2)


def bm():
    return build_detector({"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn", "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn", "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}})


@torch.no_grad()
def ev(model, vl):
    return evaluate_model(model, vl, DEV, iou_threshold=0.5, score_threshold=0.05)


def run_one(cfg_name, mode, seed):
    run_name = f"round286_{cfg_name}_s{seed}"
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
    is_geo = mode == "geo"
    opt = build_opt(model)

    h = []
    best_ap75 = -1.0
    diag = {"reward_std": []}

    bw_base = baseline_model.roi_heads.box_predictor.bbox_pred.weight.detach().clone()
    bb_base = baseline_model.roi_heads.box_predictor.bbox_pred.bias.detach().clone()

    for ep in range(1, EPOCHS + 1):
        model.train()
        td, trl, tkl = 0.0, 0.0, 0.0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}"):
            imgs_d = [i.to(DEV) for i in imgs]
            tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear()
            box_head_in.clear()
            img_shape = (imgs_d[0].shape[-2], imgs_d[0].shape[-1])

            ld = model(imgs_d, tgts_t)
            det = sum(ld.values()) if isinstance(ld, dict) else sum(sum(d.values()) for d in ld if isinstance(d, dict))
            rf = box_head_in.get("x")
            sp_raw = sampled_props.get("p")
            rl = torch.tensor(0.0, device=DEV)
            kl_loss = torch.tensor(0.0, device=DEV)

            if not is_det and rf is not None and sp_raw is not None and rf.shape[0] > 0:
                N = rf.shape[0]
                bf = model.roi_heads.box_head(rf)
                mu = model.roi_heads.box_predictor.bbox_pred(bf)[:, -4:]
                kl_loss = KL_WEIGHT * (
                    (model.roi_heads.box_predictor.bbox_pred.weight - bw_base).pow(2).sum()
                    + (model.roi_heads.box_predictor.bbox_pred.bias - bb_base).pow(2).sum()
                )

                s = torch.full_like(mu, 0.1)
                deltas = mu.detach().unsqueeze(1) + s.unsqueeze(1) * torch.randn(N, G_SAMPLES, 4, device=DEV)
                log_probs = gaussian_log_prob(deltas, mu, s)

                sp_cat = torch.cat(sp_raw, dim=0)
                N = min(N, sp_cat.shape[0])
                deltas = deltas[:N]
                log_probs = log_probs[:N]

                box_list, delta_list, img_map = [], [], []
                offset = 0
                for i_img, p_img in enumerate(sp_raw):
                    n_a = min(p_img.shape[0], N - offset)
                    if n_a <= 0:
                        break
                    box_list.append(sp_cat[offset : offset + n_a])
                    delta_list.append(deltas[offset : offset + n_a].reshape(-1, 4))
                    img_map.extend([i_img] * (n_a * G_SAMPLES))
                    offset += n_a

                sp_exp = torch.cat([p.repeat_interleave(G_SAMPLES, dim=0) for p in box_list], dim=0)
                delta_flat = torch.cat(delta_list, dim=0)
                decoded_cat = decode_boxes(sp_exp, delta_flat)

                iou_r = torch.zeros(offset, G_SAMPLES, device=DEV)
                for pi in range(offset):
                    gt = tgts_t[img_map[pi * G_SAMPLES]]["boxes"]
                    if len(gt) > 0:
                        iou_r[pi] = box_iou(decoded_cat[pi * G_SAMPLES : (pi + 1) * G_SAMPLES], gt).max(dim=1).values

                reward_img = compute_loc_reward(iou_r)

                if is_geo:
                    geo_pen = geo_penalty(decoded_cat, img_shape[1], img_shape[0])
                    geo_pen = geo_pen.view(offset, G_SAMPLES)
                    # Per-group z-score
                    gp_mean = geo_pen.mean(dim=1, keepdim=True)
                    gp_std = geo_pen.std(dim=1, keepdim=True).clamp_min(1e-6)
                    z_geo = (geo_pen - gp_mean) / gp_std
                    group_max_iou = iou_r.max(dim=1).values
                    border_mask = ((group_max_iou >= IOU_LO) & (group_max_iou < IOU_HI)).unsqueeze(1)
                    gated_bias = z_geo * border_mask.float()
                    reward_img = reward_img + BETA * gated_bias

                adv = grpo_advantage(reward_img)
                diag["reward_std"].append(adv.std().item())
                soft_w = iou_r.max(dim=1).values.clamp(0, 1).unsqueeze(1)
                rl = -(adv.detach() * log_probs * soft_w).mean()

            loss = det + RL_WEIGHT * rl + kl_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            td += det.item()
            trl += rl.item()
            tkl += kl_loss.item()

        em = ev(model, vl)
        rs_m = np.mean(diag["reward_std"]) if diag["reward_std"] else 0
        row = {
            "epoch": ep,
            "val_ap50": em["ap50"],
            "val_ap75": em["ap75"],
            "pre": em.get("precision", 0),
            "rec": em.get("recall", 0),
            "ece": em.get("ece", 0),
            "reward_std": float(rs_m),
            "det_loss": td,
            "rl_loss": trl,
            "kl_loss": tkl,
        }
        h.append(row)
        print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} r_std={rs_m:.4f}")
        if em["ap75"] > best_ap75:
            best_ap75 = em["ap75"]
        for k in diag:
            diag[k].clear()

    best_h = max(h, key=lambda r: r["val_ap75"])
    em.update(
        {
            "run_name": run_name,
            "config": cfg_name,
            "mode": mode,
            "seed": seed,
            "epochs": len(h),
            "best_ap50": best_h["val_ap50"],
            "best_ap75": best_ap75,
            "history": h,
            "git_hash": GIT,
        }
    )
    save_json(em, run_dir / "eval_metrics.json")
    return em


if __name__ == "__main__":
    all_results = []
    for cfg, mode in [("det_only_unf", "det_only_unf"), ("geo", "geo")]:
        for s in SEEDS:
            r = run_one(cfg, mode, s)
            all_results.append(r)

    print("\n## Plan 2.86 Geometric Reward (area+cx+cy+aspect)")
    for r in all_results:
        bh = max(r["history"], key=lambda x: x["val_ap75"])
        print(f"  {r['config']:<18s} s{r['seed']} AP75={r['ap75']:.4f} best={r['best_ap75']:.4f} AP50={bh['val_ap50']:.4f}")
    for cfg in ["det_only_unf", "geo"]:
        vals = [r for r in all_results if r["config"] == cfg]
        print(f"  {cfg}: {np.mean([v['best_ap75'] for v in vals]):.4f} +/- {np.std([v['best_ap75'] for v in vals]):.4f}")
