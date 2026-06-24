"""Plan 2.101: NWPU RLVR — IoU-calibrated classifier confidence (round301 on NWPU).

Hypothesis: PennFudan (170 img, 2 cls) is too small for RLVR to matter.
NWPU (800 img, 10 cls, ~3700 instances) should have enough proposal diversity
for cross-proposal GRPO to learn meaningful confidence calibration.

Framework: round301's IoU × confidence reward + cross-proposal GRPO.
Adapted for multi-class (11) and larger images (max_size=480).
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.experiments.runner_utils import (
    decode_boxes,
    gaussian_log_prob,
    unfreeze_rlvr,
)
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import set_seed

DEV = "cuda"
SEED = 42
BATCH = 2
EPOCHS = 8
MAX_SIZE = 480
NUM_CLASSES = 11
DATA = Path("data/NWPU VHR-10 dataset")
ANNOT = Path("data/NWPU_VHR10_coco.json")
CKPT_PATH = "runs/round2100_nwpu_baseline/checkpoint_best.pth"

# RLVR hyperparams
G_SAMPLES = 4
SEEDS = [42, 123, 456]
RL_WEIGHT = 0.00005
KL_WEIGHT = 0.03
HEAD_LR = 0.001
BODY_LR = 0.0001
SIGMA = 0.1


class NWPUDataset(Dataset):
    def __init__(self, root, coco_json, img_ids, max_size):
        self.root = Path(root)
        self.max_size = max_size
        self.coco = json.loads(Path(coco_json).read_text())
        self.img_infos = {img["id"]: img for img in self.coco["images"] if img["id"] in img_ids}
        self.img_ids = list(self.img_infos.keys())
        anns = {}
        for ann in self.coco["annotations"]:
            if ann["image_id"] in img_ids:
                anns.setdefault(ann["image_id"], []).append(ann)
        self.anns = anns

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.img_infos[img_id]
        img_path = self.root / "positive image set" / info["file_name"]
        if not img_path.exists():
            img_path = self.root / "negative image set" / info["file_name"]
        img = Image.open(str(img_path)).convert("RGB")
        img_t = TF.to_tensor(img)
        boxes, labels = [], []
        for ann in self.anns.get(img_id, []):
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }
        _, H, W = img_t.shape
        if max(H, W) > self.max_size:
            scale = self.max_size / max(H, W)
            new_h, new_w = int(H * scale), int(W * scale)
            img_t = F.interpolate(img_t.unsqueeze(0), size=(new_h, new_w), mode="bilinear").squeeze(0)
            target["boxes"] = target["boxes"] * scale
        return img_t, target


def collate(batch):
    return tuple(zip(*batch))


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


def compute_iou_multi(sp_raw, bf, cls_logits, box_predictor, tgts_t):
    """Compute max IoU per proposal using predicted-class deltas.
    cls_logits: (N, num_classes) or (N, G, num_classes) for per-sample IoU.
    Returns iou with same leading dims as cls_logits (minus class dim).
    """
    sp_cat = torch.cat(sp_raw, dim=0)
    N = sp_cat.shape[0]
    if cls_logits.dim() == 3:
        # (N, G, C) — per-sample
        N_g, G, C = cls_logits.shape
        reg_out = box_predictor.bbox_pred(bf[:N_g])  # (N, 44)
        pred_class = cls_logits[:, :, 1:].argmax(dim=-1)  # (N, G)
        offsets = ((pred_class + 1) * 4).unsqueeze(-1) + torch.arange(4, device=DEV)  # (N, G, 4) +1 skip bg
        # Expand reg_out: (N, 44) -> (N, G, 44)
        reg_exp = reg_out.unsqueeze(1).expand(-1, G, -1)  # (N, G, 44)
        deltas = reg_exp.gather(-1, offsets)  # (N, G, 4)
        iou_out = torch.zeros(N_g, G, device=DEV)
        off = 0
        for i_img, p_img in enumerate(sp_raw):
            n_p = p_img.shape[0]
            if n_p == 0:
                continue
            gt = tgts_t[i_img]["boxes"]
            if len(gt) > 0:
                for g in range(G):
                    decoded_g = decode_boxes(sp_cat[off:off+n_p], deltas[off:off+n_p, g, :])
                    iou_out[off:off+n_p, g] = box_iou(decoded_g, gt).max(dim=1).values
            off += n_p
    else:
        # (N, C) — single
        reg_out = box_predictor.bbox_pred(bf[:N])  # (N, 44)
        pred_class = cls_logits[:, 1:].argmax(dim=-1)  # (N,)
        offsets = ((pred_class + 1) * 4).unsqueeze(1) + torch.arange(4, device=DEV).unsqueeze(0)  # (N, 4) +1 skip bg
        deltas = reg_out.gather(1, offsets)  # (N, 4)
        iou_out = torch.zeros(N, device=DEV)
        off = 0
        for i_img, p_img in enumerate(sp_raw):
            n_p = p_img.shape[0]
            if n_p == 0:
                continue
            gt = tgts_t[i_img]["boxes"]
            if len(gt) > 0:
                iou_out[off:off+n_p] = box_iou(decode_boxes(sp_cat[off:off+n_p], deltas[off:off+n_p]), gt).max(dim=1).values
            off += n_p
    return iou_out


def run_one(cfg_name, mode, seed):
    run_name = f"round2101_{cfg_name}_s{seed}"
    set_seed(seed)

    # Build data
    coco = json.loads(ANNOT.read_text())
    all_ids = list(
        set(img["id"] for img in coco["images"] if Path(DATA / "positive image set" / img["file_name"]).exists())
    )
    np.random.seed(42)
    np.random.shuffle(all_ids)
    n_train = int(0.7 * len(all_ids))
    train_ids = set(all_ids[:n_train])
    val_ids = set(all_ids[n_train:])
    train_ds = NWPUDataset(DATA, ANNOT, train_ids, MAX_SIZE)
    val_ds = NWPUDataset(DATA, ANNOT, val_ids, MAX_SIZE)
    tl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=0)
    vl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)

    model = build_detector(
        {"model": {"name": "fasterrcnn_mobilenet_v3_large_fpn", "model_name": "fasterrcnn_mobilenet_v3_large_fpn",
                    "pretrained": True, "num_classes": NUM_CLASSES, "min_size": MAX_SIZE, "max_size": MAX_SIZE}}
    ).to(DEV)
    ckpt = torch.load(CKPT_PATH, map_location=DEV)
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

    run_dir = ensure_run_dir(run_name)
    is_det = mode == "det_only_unf"
    is_rlvr = mode == "rlvr_iou_cls"
    opt = build_opt(model)

    h = []
    best_ap50 = -1.0
    diag = {"adv_std": [], "reward_raw_std": [], "conf_shift": [], "mean_iou": []}

    for ep in range(1, EPOCHS + 1):
        model.train()
        td, trl, tkl = 0.0, 0.0, 0.0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}"):
            imgs_d = [i.to(DEV) for i in imgs]
            tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear()
            box_head_in.clear()

            ld = model(imgs_d, tgts_t)
            det = sum(ld.values()) if isinstance(ld, dict) else sum(sum(d.values()) for d in ld if isinstance(d, dict))
            rf = box_head_in.get("x")
            sp_raw = sampled_props.get("p")
            rl = torch.tensor(0.0, device=DEV)
            kl_loss = torch.tensor(0.0, device=DEV)

            if not is_det and rf is not None and sp_raw is not None and rf.shape[0] > 0:
                bf = model.roi_heads.box_head(rf)
                cls_logits = model.roi_heads.box_predictor.cls_score(bf)  # (N, 11)

                with torch.no_grad():
                    baseline_bf = baseline_model.roi_heads.box_head(rf)
                    baseline_cls = F.softmax(
                        baseline_model.roi_heads.box_predictor.cls_score(baseline_bf), dim=-1
                    )

                # Off-policy: sample from BASELINE, evaluate log_prob on CURRENT
                with torch.no_grad():
                    baseline_logits = baseline_model.roi_heads.box_predictor.cls_score(baseline_bf)
                    s_base = torch.full_like(baseline_logits, SIGMA)
                    perturbed_logits = baseline_logits.unsqueeze(1) + s_base.unsqueeze(1) * torch.randn(
                        baseline_logits.shape[0], G_SAMPLES, baseline_logits.shape[1], device=DEV
                    )
                perturbed_cls = F.softmax(perturbed_logits, dim=-1)  # (N, G, 11)
                perturbed_conf = perturbed_cls[:, :, 1:].max(dim=-1).values  # (N, G)
                baseline_conf = baseline_cls[:, 1:].max(dim=-1).values  # (N,)

                s_cls = torch.full_like(cls_logits, SIGMA)
                log_probs = gaussian_log_prob(perturbed_logits, cls_logits, s_cls)

                # IoU per (proposal, sample) using baseline model and per-sample classes
                iou_p = compute_iou_multi(
                    sp_raw, baseline_bf, perturbed_logits,
                    baseline_model.roi_heads.box_predictor, tgts_t
                )  # (N, G)

                # Reward: R = max_fg_conf * (2*IoU - 1)
                N = min(cls_logits.shape[0], iou_p.shape[0])
                quality = (2 * iou_p[:N] - 1)  # (N, G)
                reward_img = perturbed_conf[:N] * quality  # (N, G)

                reward_flat = reward_img.reshape(-1)
                n_props_per_img = [p.shape[0] * G_SAMPLES for p in sp_raw]
                adv = cross_proposal_grpo(reward_flat, n_props_per_img).view(N, G_SAMPLES)

                diag["adv_std"].append(adv.std().item())
                raw_std_per_img = []
                off_r = 0
                for p in sp_raw:
                    n_r = p.shape[0] * G_SAMPLES
                    if n_r > 0:
                        raw_std_per_img.append(reward_flat[off_r:off_r+n_r].std().item())
                    off_r += n_r
                diag["reward_raw_std"].append(np.mean(raw_std_per_img) if raw_std_per_img else 0.0)
                rl = -(adv.detach() * log_probs[:N]).mean()

                kl_loss = KL_WEIGHT * (perturbed_conf[:N] - baseline_conf[:N].unsqueeze(1)).pow(2).mean()

                diag["conf_shift"].append((perturbed_conf[:N].mean() - baseline_conf[:N].mean()).item())
                diag["mean_iou"].append(iou_p[:N].mean().item())

            loss = det + RL_WEIGHT * rl + kl_loss
            opt.zero_grad(set_to_none=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            loss.backward()
            opt.step()

            td += det.item(); trl += rl.item(); tkl += kl_loss.item()

        model.eval()
        ps, ts = [], []
        for img, tgt in vl:
            with torch.no_grad():
                pred = model([img[0].to(DEV)])[0]
            ps.append({k: v.cpu() for k, v in pred.items()})
            ts.append({k: v.cpu() for k, v in tgt[0].items()})
        em = evaluate_detection_predictions(ps, ts, iou_threshold=0.5, score_threshold=0.05)

        rs_m = np.mean(diag["adv_std"]) if diag["adv_std"] else 0.0
        rr_m = np.mean(diag["reward_raw_std"]) if diag["reward_raw_std"] else 0.0
        cs_m = np.mean(diag["conf_shift"]) if diag["conf_shift"] else 0.0
        mi_m = np.mean(diag["mean_iou"]) if diag["mean_iou"] else 0.0

        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "ece": em.get("ece", 0), "adv_std": float(rs_m), "reward_raw_std": float(rr_m),
               "conf_shift": float(cs_m), "mean_iou": float(mi_m),
               "det_loss": td, "rl_loss": trl, "kl_loss": tkl}
        h.append(row)
        print(f"  e{ep}: AP50={em['ap50']:.4f} AP75={em['ap75']:.4f} "
              f"adv_std={rs_m:.4f} raw_r_std={rr_m:.4f} conf_shift={cs_m:.4f} mean_iou={mi_m:.4f}")
        if em["ap50"] > best_ap50:
            best_ap50 = em["ap50"]
        for k in diag:
            diag[k].clear()

    best_h = max(h, key=lambda r: r["val_ap50"])
    em.update({"run_name": run_name, "config": cfg_name, "mode": mode, "seed": seed,
               "epochs": len(h), "best_ap50": best_ap50, "best_ap75": best_h["val_ap75"],
               "history": h})
    save_json(em, run_dir / "eval_metrics.json")
    return em


if __name__ == "__main__":
    all_results = []
    for cfg, mode in [("det_only_unf", "det_only_unf"), ("rlvr_iou_cls", "rlvr_iou_cls")]:
        for s in SEEDS:
            r = run_one(cfg, mode, s)
            all_results.append(r)

    print("\n## Plan 2.101 RLVR on NWPU — IoU-Calibrated Classifier Confidence")
    for r in all_results:
        bh = max(r["history"], key=lambda x: x["val_ap50"])
        print(f"  {r['config']:<18s} s{r['seed']} AP50={r['best_ap50']:.4f} bestAP75={bh['val_ap75']:.4f}")
    for cfg in ["det_only_unf", "rlvr_iou_cls"]:
        vals = [r for r in all_results if r["config"] == cfg]
        if vals:
            print(f"  {cfg}: bestAP50={np.mean([v['best_ap50'] for v in vals]):.4f} +/- {np.std([v['best_ap50'] for v in vals]):.4f}")
