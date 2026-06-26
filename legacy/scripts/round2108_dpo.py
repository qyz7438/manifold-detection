"""Plan 2.108: DPO on classifier — pairwise best-vs-worst within each GT.

For each GT with >=2 matching proposals:
  chosen  = highest IoU match  → should get highest logit
  rejected = lowest IoU match  → should get lowest logit

DPO loss: -log σ(β * (logit_c - logit_c_ref - logit_r + logit_r_ref))

No PG, no sampling, no GRPO. Direct contrastive loss on logit margins.
This models what CE can't: cross-proposal ranking within the same GT.
"""
import copy, shutil, subprocess, sys
import numpy as np, torch, torch.nn.functional as F
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_penn_fudan_loaders_320, decode_boxes, evaluate_model, unfreeze_rlvr,
)
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import set_seed

DEV = "cuda"; CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
SEED, EPOCHS = 42, 8; SEEDS = [42, 123, 456]
DPO_WEIGHT, KL_WEIGHT, BETA = 0.1, 0.01, 2.0

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


def compute_iou_and_gt(sp_raw, bf, box_predictor, tgts_t):
    """Per-proposal max IoU + which GT it matches. Returns (iou, gt_idx)."""
    sp_cat = torch.cat(sp_raw, dim=0); N = sp_cat.shape[0]
    with torch.no_grad():
        reg = box_predictor.bbox_pred(bf[:N]); decoded = decode_boxes(sp_cat, reg[:, 2:6])
    iou = torch.zeros(N, device=DEV); gt_idx = torch.zeros(N, dtype=torch.long, device=DEV)
    off = 0
    for i_img, p_img in enumerate(sp_raw):
        n_p = p_img.shape[0]
        if n_p == 0: continue
        gt = tgts_t[i_img]["boxes"]
        if len(gt) > 0:
            i = box_iou(decoded[off:off+n_p], gt)  # (n_p, G)
            iou[off:off+n_p], gt_idx[off:off+n_p] = i.max(dim=1)
        off += n_p
    return iou, gt_idx


def run_one(cfg_name, mode, seed):
    run_name = f"round2108_{cfg_name}_s{seed}"; set_seed(seed)
    model = bm().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV); model.load_state_dict(ckpt["model"])
    unfreeze_rlvr(model)
    baseline_model = copy.deepcopy(model); baseline_model.eval()
    for p in baseline_model.parameters(): p.requires_grad = False

    sampled_props, box_head_in = {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(lambda m, args: box_head_in.update({"x": args[0]}))

    tl, vl = build_penn_fudan_loaders_320(batch_size=2)
    run_dir = ensure_run_dir(run_name); shutil.copy(__file__, run_dir / "runner_snapshot.py")
    is_det = mode == "det_only_unf"; is_dpo = mode == "dpo"
    opt = build_opt(model); baseline_bp = baseline_model.roi_heads.box_predictor
    h, best_ap75 = [], -1.0
    diag = {"dpo_loss": [], "conf_shift": [], "n_pairs": []}

    for ep in range(1, EPOCHS + 1):
        model.train(); td, tdpo, tkl = 0.0, 0.0, 0.0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}", leave=False):
            imgs_d = [i.to(DEV) for i in imgs]; tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear(); box_head_in.clear()
            ld = model(imgs_d, tgts_t); det = sum(ld.values())
            rf = box_head_in.get("x"); sp_raw = sampled_props.get("p")
            dpo = torch.tensor(0.0, device=DEV); kl = torch.tensor(0.0, device=DEV)

            if not is_det and rf is not None and sp_raw is not None and rf.shape[0] > 0:
                bf = model.roi_heads.box_head(rf); cls_logits = model.roi_heads.box_predictor.cls_score(bf)
                person_logit = cls_logits[:, 1]  # (N,) logit of person class

                with torch.no_grad():
                    baseline_bf = baseline_model.roi_heads.box_head(rf)
                    baseline_logits = baseline_bp.cls_score(baseline_bf)
                    baseline_person = baseline_logits[:, 1]

                # IoU + GT matching
                iou_p, gt_idx = compute_iou_and_gt(sp_raw, baseline_bf, baseline_bp, tgts_t)
                N = min(cls_logits.shape[0], iou_p.shape[0])

                # DPO: pair best-vs-worst within each GT group
                n_pairs = 0
                for gid in torch.unique(gt_idx[:N]):
                    if gid < 0: continue
                    mask = gt_idx[:N] == gid
                    if mask.sum() < 2: continue
                    ious = iou_p[:N][mask]
                    logits = person_logit[:N][mask]
                    ref_logits = baseline_person[:N][mask]
                    best_idx = ious.argmax(); worst_idx = ious.argmin()
                    if best_idx == worst_idx: continue
                    lc, lr = logits[best_idx], logits[worst_idx]
                    rc, rr = ref_logits[best_idx], ref_logits[worst_idx]
                    margin = (lc - rc) - (lr - rr)
                    dpo = dpo - F.logsigmoid(BETA * margin)
                    n_pairs += 1

                if n_pairs > 0:
                    dpo = dpo / n_pairs
                diag["dpo_loss"].append(dpo.item()); diag["n_pairs"].append(n_pairs)
                kl = KL_WEIGHT * (person_logit[:N] - baseline_person[:N]).pow(2).mean()
                diag["conf_shift"].append((person_logit[:N].mean() - baseline_person[:N].mean()).item())

            loss = det + DPO_WEIGHT * dpo + kl
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            td += det.item(); tdpo += dpo.item(); tkl += kl.item()

        em = evaluate_model(model, vl, DEV)
        np_m = np.mean(diag["n_pairs"]) if diag["n_pairs"] else 0
        dp_m = np.mean(diag["dpo_loss"]) if diag["dpo_loss"] else 0
        cs_m = np.mean(diag["conf_shift"]) if diag["conf_shift"] else 0
        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "ece": em.get("ece", 0), "n_pairs": float(np_m), "dpo_loss": float(dp_m),
               "conf_shift": float(cs_m), "det": td, "dpo": tdpo, "kl": tkl}
        h.append(row); print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} pairs={np_m:.1f} dpo={dp_m:.3f}")
        if em["ap75"] > best_ap75: best_ap75 = em["ap75"]
        for k in diag: diag[k].clear()

    best_h = max(h, key=lambda r: r["val_ap75"])
    em.update({"run_name": run_name, "config": cfg_name, "mode": mode, "seed": seed,
               "epochs": len(h), "best_ap50": best_h["val_ap50"], "best_ap75": best_ap75, "history": h})
    save_json(em, run_dir / "eval_metrics.json"); return em


if __name__ == "__main__":
    results = []
    for cfg, mode in [("det_only", "det_only_unf"), ("dpo", "dpo")]:
        for s in SEEDS: results.append(run_one(cfg, mode, s))
    print("\n## 2.108 DPO Pairwise Best-vs-Worst")
    for r in results:
        bh = max(r["history"], key=lambda x: x["val_ap75"])
        print(f"  {r['config']:<10s} s{r['seed']} AP75={r['best_ap75']:.4f} AP50={bh['val_ap50']:.4f}")
    for cfg in ["det_only", "dpo"]:
        vals = [r for r in results if r["config"] == cfg]
        if vals: print(f"  {cfg}: bestAP75={np.mean([v['best_ap75'] for v in vals]):.4f} +/- {np.std([v['best_ap75'] for v in vals]):.4f}")
