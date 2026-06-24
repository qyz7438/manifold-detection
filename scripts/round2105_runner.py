"""Plan 2.105: Discrete RLVR — Bernoulli keep/reject per proposal + NMS reward.

Maps detection RLVR to LLM RLVR pattern:
  discrete action (keep/reject each proposal) + discrete reward (NMS hit count) + PG

This is the "right" RLVR for detection — the reward depends on NMS outcome which
is non-differentiable and requires exploration.
"""
import copy
import shutil
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou, nms
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_penn_fudan_loaders_320,
    decode_boxes,
    evaluate_model,
    unfreeze_rlvr,
)
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
G_SAMPLES = 8
EPOCHS = 8
SEEDS = [42, 123, 456]
RL_WEIGHT = 0.001
KL_WEIGHT = 0.01
HEAD_LR = 0.001
BODY_LR = 0.0001


def bl():
    return build_penn_fudan_loaders_320(batch_size=2)


def bm():
    return build_detector(
        {"model": {"name": "fasterrcnn_mobilenet_v3_large_fpn",
                    "model_name": "fasterrcnn_mobilenet_v3_large_fpn",
                    "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}}
    )


def build_opt(model):
    body_params, head_params = [], []
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


def discrete_nms_reward(sp_raw, kept_mask, conf, box_predictor, bf, tgts_t, image_shapes):
    """Run greedy match on kept proposals, return hit count per sample G.

    kept_mask: (N, G) — 1=keep, 0=reject
    conf: (N, G) — confidence scores for kept proposals
    Returns: (G,) hit_count for each sample
    """
    G = kept_mask.shape[1]
    sp_cat = torch.cat(sp_raw, dim=0)
    reg_out = box_predictor.bbox_pred(bf)
    person_deltas = reg_out[:, 2:6]
    decoded = decode_boxes(sp_cat, person_deltas)
    hits = torch.zeros(G, device=DEV)

    offset = 0
    for i_img, p_img in enumerate(sp_raw):
        n_p = p_img.shape[0]
        if n_p == 0:
            continue
        gt = tgts_t[i_img]["boxes"]
        if len(gt) == 0:
            offset += n_p
            continue

        for g in range(G):
            kept_idx = kept_mask[offset : offset + n_p, g].bool()
            if not kept_idx.any():
                continue
            k_boxes = decoded[offset : offset + n_p][kept_idx]
            k_conf = conf[offset : offset + n_p, g][kept_idx]

            # Greedy match: sort by confidence, match to best unmatched GT
            _, sort_idx = k_conf.sort(descending=True)
            matched = torch.zeros(len(gt), dtype=torch.bool, device=DEV)
            for idx in sort_idx:
                ious = box_iou(k_boxes[idx:idx+1], gt).squeeze(0)
                valid = ious * (~matched).float()
                if valid.max() > 0.5:
                    best_gt = valid.argmax()
                    matched[best_gt] = True
                    if ious[best_gt] >= 0.75:
                        hits[g] += 1

        offset += n_p
    return hits


def run_one(cfg_name, mode, seed):
    run_name = f"round2105_{cfg_name}_s{seed}"
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
    is_rlvr = mode == "rlvr_discrete"
    opt = build_opt(model)

    h = []
    best_ap75 = -1.0
    diag = {"mean_hit": [], "conf_shift": []}

    for ep in range(1, EPOCHS + 1):
        model.train()
        td, trl, tkl = 0.0, 0.0, 0.0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}"):
            imgs_d = [i.to(DEV) for i in imgs]
            tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear(); box_head_in.clear()
            image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]

            ld = model(imgs_d, tgts_t)
            det = sum(ld.values()) if isinstance(ld, dict) else sum(sum(d.values()) for d in ld if isinstance(d, dict))
            rf = box_head_in.get("x"); sp_raw = sampled_props.get("p")
            rl = torch.tensor(0.0, device=DEV)
            kl_loss = torch.tensor(0.0, device=DEV)

            if not is_det and rf is not None and sp_raw is not None and rf.shape[0] > 0:
                bf = model.roi_heads.box_head(rf)
                cls_logits = model.roi_heads.box_predictor.cls_score(bf)
                current_conf = F.softmax(cls_logits, dim=-1)[:, 1]

                with torch.no_grad():
                    baseline_bf = baseline_model.roi_heads.box_head(rf)
                    baseline_logits = baseline_model.roi_heads.box_predictor.cls_score(baseline_bf)
                    baseline_conf = F.softmax(baseline_logits, dim=-1)[:, 1]

                # Discrete Bernoulli: sample keep/reject from baseline
                with torch.no_grad():
                    kept = torch.bernoulli(
                        baseline_conf.unsqueeze(1).expand(-1, G_SAMPLES)
                    )  # (N, G)

                # Log-prob of each decision under CURRENT model (off-policy)
                eps = 1e-8
                current_conf_g = current_conf.unsqueeze(1).expand(-1, G_SAMPLES)
                log_probs_per_box = kept * torch.log(current_conf_g + eps) + \
                                    (1 - kept) * torch.log(1 - current_conf_g + eps)  # (N, G)
                log_probs = log_probs_per_box.sum(dim=0)  # (G,) — image-level sum

                # Reward: greedy-match hit count for each sample
                hits = discrete_nms_reward(
                    sp_raw, kept, current_conf_g,
                    baseline_model.roi_heads.box_predictor, baseline_bf, tgts_t, image_shapes
                )  # (G,)

                # GRPO across G samples
                r_mean = hits.mean()
                r_std = hits.std().clamp_min(1e-6)
                adv = (hits - r_mean) / r_std

                rl = -(adv.detach() * log_probs).mean()
                diag["mean_hit"].append(hits.mean().item())

                # KL: keep confidence close to baseline
                kl_loss = KL_WEIGHT * (current_conf - baseline_conf).pow(2).mean()

                diag["conf_shift"].append((current_conf.mean() - baseline_conf.mean()).item())

            loss = det + RL_WEIGHT * rl + kl_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()

            td += det.item(); trl += rl.item(); tkl += kl_loss.item()

        em = ev(model, vl)
        mh_m = np.mean(diag["mean_hit"]) if diag["mean_hit"] else 0.0
        cs_m = np.mean(diag["conf_shift"]) if diag["conf_shift"] else 0.0

        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "ece": em.get("ece", 0), "mean_hit": float(mh_m),
               "conf_shift": float(cs_m), "det_loss": td, "rl_loss": trl, "kl_loss": tkl}
        h.append(row)
        print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} "
              f"mean_hit={mh_m:.2f} rl={trl:.2f} kl={tkl:.2f} cs={cs_m:.4f}")
        if em["ap75"] > best_ap75:
            best_ap75 = em["ap75"]
        for k in diag:
            diag[k].clear()

    best_h = max(h, key=lambda r: r["val_ap75"])
    em.update({"run_name": run_name, "config": cfg_name, "mode": mode, "seed": seed,
               "epochs": len(h), "best_ap50": best_h["val_ap50"], "best_ap75": best_ap75,
               "history": h, "git_hash": GIT})
    save_json(em, run_dir / "eval_metrics.json")
    return em


if __name__ == "__main__":
    all_results = []
    for cfg, mode in [("det_only_unf", "det_only_unf"), ("rlvr_discrete", "rlvr_discrete")]:
        for s in SEEDS:
            r = run_one(cfg, mode, s)
            all_results.append(r)

    print("\n## Plan 2.105 Discrete RLVR — Bernoulli keep/reject + NMS reward")
    for r in all_results:
        bh = max(r["history"], key=lambda x: x["val_ap75"])
        print(f"  {r['config']:<18s} s{r['seed']} AP75={r['ap75']:.4f} best={r['best_ap75']:.4f} AP50={bh['val_ap50']:.4f}")
    for cfg in ["det_only_unf", "rlvr_discrete"]:
        vals = [r for r in all_results if r["config"] == cfg]
        if vals:
            print(f"  {cfg}: bestAP75={np.mean([v['best_ap75'] for v in vals]):.4f} +/- {np.std([v['best_ap75'] for v in vals]):.4f}")
