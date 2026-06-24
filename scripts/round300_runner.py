"""Plan 3.0: RLVR for classifier confidence — geo reward + cross-proposal GRPO.

Root cause (round281-286): RLVR on box regression is redundant with SFT smooth L1.
The reward signals that SFT CANNOT encode (geo, energy, cross-proposal comparison)
are unique — but they've been routed to the wrong parameters.

Fix: RLVR updates classifier logits (confidence scores), not box regression deltas.
Reward is geo_penalty. Cross-proposal GRPO ranks proposals within an image.
Function-space KL replaces weight-space KL.
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
KL_WEIGHT = 0.05
HEAD_LR = 0.001
BODY_LR = 0.0001
SIGMA = 0.1


def geo_penalty(boxes, img_w, img_h):
    """Penalize small, narrow, edge-positioned boxes. Higher = better/more central."""
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    area = w * h
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    aspect = w / torch.clamp(h, min=1)

    area_n = (area - 500) / 1000
    cx_n = (cx - img_w / 2) / (img_w / 4)
    cy_n = (cy - img_h / 2) / (img_h / 4)
    aspect_n = (aspect - 0.4) / 0.2

    penalty = torch.stack(
        [torch.abs(cx_n), torch.abs(cy_n), -area_n.clamp(-1, 1), -aspect_n.clamp(-1, 1)],
        dim=1,
    ).mean(dim=1)
    return -torch.sigmoid(3 * (penalty - 0.5))


def cross_proposal_grpo(reward, n_proposals_per_img):
    """GRPO advantage across proposals in the same image."""
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
        lr=HEAD_LR,
        momentum=0.9,
        weight_decay=0.0005,
    )


@torch.no_grad()
def ev(model, vl):
    return evaluate_model(model, vl, DEV, iou_threshold=0.5, score_threshold=0.05)


def run_one(cfg_name, mode, seed):
    run_name = f"round300_{cfg_name}_s{seed}"
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
    is_rlvr = mode == "rlvr_geo_cls"
    opt = build_opt(model)

    h = []
    best_ap75 = -1.0
    diag = {"reward_std": [], "conf_shift": []}

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
            det = (
                sum(ld.values())
                if isinstance(ld, dict)
                else sum(sum(d.values()) for d in ld if isinstance(d, dict))
            )
            rf = box_head_in.get("x")
            sp_raw = sampled_props.get("p")
            rl = torch.tensor(0.0, device=DEV)
            kl_loss = torch.tensor(0.0, device=DEV)

            if not is_det and rf is not None and sp_raw is not None and rf.shape[0] > 0:
                # --- RLVR on classifier logits ---
                bf = model.roi_heads.box_head(rf)  # (N, 1024) after FC layers
                cls_logits = model.roi_heads.box_predictor.cls_score(bf)  # (N, 2)

                with torch.no_grad():
                    baseline_bf = baseline_model.roi_heads.box_head(rf)
                    baseline_cls = F.softmax(
                        baseline_model.roi_heads.box_predictor.cls_score(baseline_bf), dim=-1
                    )[:, 1]  # (N,)

                # Sample G perturbations of classifier logits
                s_cls = torch.full_like(cls_logits, SIGMA)
                perturbed_logits = cls_logits.unsqueeze(1) + s_cls.unsqueeze(1) * torch.randn(
                    cls_logits.shape[0], G_SAMPLES, cls_logits.shape[1], device=DEV
                )  # (N, G, 2)
                perturbed_conf = F.softmax(perturbed_logits, dim=-1)[:, :, 1]  # (N, G)

                log_probs = gaussian_log_prob(perturbed_logits, cls_logits, s_cls)  # (N, G)

                # Compute geo reward per proposal (same for all G samples)
                sp_cat = torch.cat(sp_raw, dim=0)
                N = min(cls_logits.shape[0], sp_cat.shape[0])
                reg_out = model.roi_heads.box_predictor.bbox_pred(bf[:N])
                person_deltas = reg_out[:, 2:6]  # class 1 (person)
                decoded = decode_boxes(sp_cat[:N], person_deltas)
                geo_r = geo_penalty(decoded, img_shape[1], img_shape[0])  # (N,) higher=better

                # Reward: geo score, flattened across proposals × samples
                reward_flat = geo_r.unsqueeze(1).expand(N, G_SAMPLES).reshape(-1)  # (N*G,)
                n_props = [N]
                adv = cross_proposal_grpo(reward_flat, n_props).view(N, G_SAMPLES)  # (N, G)

                # PG loss: update classifier to prefer proposals with higher geo scores
                diag["reward_std"].append(adv.std().item())
                rl = -(adv.detach() * log_probs[:N]).mean()

                # Function-space KL: keep confidence distribution close to baseline
                kl_loss = KL_WEIGHT * (perturbed_conf[:N] - baseline_cls[:N].unsqueeze(1)).pow(2).mean()

                diag["conf_shift"].append((perturbed_conf[:N].mean() - baseline_cls[:N].mean()).item())

            loss = det + RL_WEIGHT * rl + kl_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            td += det.item()
            trl += rl.item()
            tkl += kl_loss.item()

        em = ev(model, vl)
        rs_m = np.mean(diag["reward_std"]) if diag["reward_std"] else 0.0
        cs_m = np.mean(diag["conf_shift"]) if diag["conf_shift"] else 0.0

        row = {
            "epoch": ep,
            "val_ap50": em["ap50"],
            "val_ap75": em["ap75"],
            "pre": em.get("precision", 0),
            "rec": em.get("recall", 0),
            "ece": em.get("ece", 0),
            "reward_std": float(rs_m),
            "conf_shift": float(cs_m),
            "det_loss": td,
            "rl_loss": trl,
            "kl_loss": tkl,
        }
        h.append(row)
        print(
            f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} "
            f"r_std={rs_m:.4f} conf_shift={cs_m:.4f}"
        )
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
    for cfg, mode in [("det_only_unf", "det_only_unf"), ("rlvr_geo_cls", "rlvr_geo_cls")]:
        for s in SEEDS:
            r = run_one(cfg, mode, s)
            all_results.append(r)

    print("\n## Plan 3.0 RLVR on Classifier with Geo Reward + Cross-Proposal GRPO")
    fmt = "  {:<18s} s{} AP75={:.4f} best={:.4f} AP50={:.4f}"
    for r in all_results:
        bh = max(r["history"], key=lambda x: x["val_ap75"])
        print(fmt.format(r["config"], r["seed"], r["ap75"], r["best_ap75"], bh["val_ap50"]))
    for cfg in ["det_only_unf", "rlvr_geo_cls"]:
        vals = [r for r in all_results if r["config"] == cfg]
        if vals:
            bv = [v["best_ap75"] for v in vals]
            print(f"  {cfg}: bestAP75={np.mean(bv):.4f} +/- {np.std(bv):.4f}")
