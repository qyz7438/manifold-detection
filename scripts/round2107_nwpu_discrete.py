"""Plan 2.107: NWPU discrete RLVR — Bernoulli keep/reject + NMS hit reward.

Tests cross-proposal duplicate suppression on larger dataset (800 img, 10 cls).
Discrete action (keep/reject per proposal) + image-level reward (NMS hit count).
LLM-style RLVR — the reward depends on NMS outcome, which is non-differentiable.
"""
import copy, json, sys, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou, nms
from torchvision.transforms import functional as TF
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.experiments.runner_utils import decode_boxes, unfreeze_rlvr
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import set_seed

DEV, SEED = "cuda", 42
BATCH, EPOCHS, MAX_SIZE, NUM_CLASSES = 2, 8, 480, 11
G_SAMPLES, RL_WEIGHT, KL_WEIGHT = 8, 0.005, 0.01
DATA = Path("data/NWPU VHR-10 dataset")
ANNOT = Path("data/NWPU_VHR10_coco.json")
CKPT_PATH = "runs/round2100_nwpu_baseline/checkpoint_best.pth"


class NWPUDataset(Dataset):
    def __init__(self, root, coco_json, img_ids, max_size):
        self.root = Path(root); self.max_size = max_size
        self.coco = json.loads(Path(coco_json).read_text())
        self.img_infos = {img["id"]: img for img in self.coco["images"] if img["id"] in img_ids}
        self.img_ids = list(self.img_infos.keys())
        anns = {}
        for ann in self.coco["annotations"]:
            if ann["image_id"] in img_ids:
                anns.setdefault(ann["image_id"], []).append(ann)
        self.anns = anns

    def __len__(self): return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]; info = self.img_infos[img_id]
        img_path = self.root / "positive image set" / info["file_name"]
        if not img_path.exists(): img_path = self.root / "negative image set" / info["file_name"]
        img = Image.open(str(img_path)).convert("RGB"); img_t = TF.to_tensor(img)
        boxes, labels = [], []
        for ann in self.anns.get(img_id, []):
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h]); labels.append(ann["category_id"])
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32),
                  "labels": torch.tensor(labels, dtype=torch.int64),
                  "image_id": torch.tensor([img_id])}
        _, H, W = img_t.shape
        if max(H, W) > self.max_size:
            scale = self.max_size / max(H, W); nh, nw = int(H * scale), int(W * scale)
            img_t = F.interpolate(img_t.unsqueeze(0), size=(nh, nw), mode="bilinear").squeeze(0)
            target["boxes"] = target["boxes"] * scale
        return img_t, target


def collate(batch): return tuple(zip(*batch))


def build_opt(model):
    body, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (head if "box_head" in n or "box_predictor" in n else body).append(p)
    return torch.optim.SGD([{"params": body, "lr": 0.0001}, {"params": head, "lr": 0.001}],
                           lr=0.001, momentum=0.9, weight_decay=0.0005)


def discrete_nms_reward(sp_raw, kept, conf, pred_class, box_predictor, bf, tgts_t):
    """NMS hit count for each of G keep/reject patterns. Returns (G,).
    pred_class: (N,) — argmax class per proposal (0=bg, 1-10=fg)
    """
    G = kept.shape[1]; sp_cat = torch.cat(sp_raw, dim=0)
    reg_out = box_predictor.bbox_pred(bf)  # (N, 44)
    hits = torch.zeros(G, device=DEV); offset = 0
    for i_img, p_img in enumerate(sp_raw):
        n_p = p_img.shape[0]
        if n_p == 0: continue
        gt = tgts_t[i_img]["boxes"]
        if len(gt) == 0: offset += n_p; continue
        # Class-aware delta: class c (1-10) → indices [c*4 : c*4+4]
        pc = pred_class[offset:offset + n_p]  # (n_p,) 0-10
        off_c = (pc * 4).unsqueeze(1) + torch.arange(4, device=DEV)  # (n_p, 4)
        deltas = reg_out[offset:offset + n_p].gather(1, off_c)  # (n_p, 4)
        all_decoded = decode_boxes(sp_cat[offset:offset + n_p], deltas)
        for g in range(G):
            kidx = kept[offset:offset + n_p, g].bool()
            if not kidx.any(): continue
            k_conf = conf[offset:offset + n_p, g][kidx]
            k_boxes = all_decoded[kidx]
            # NMS on kept boxes
            keep_nms = nms(k_boxes, k_conf, 0.5)
            if len(keep_nms) == 0: continue
            surv_boxes = k_boxes[keep_nms]
            ious = box_iou(surv_boxes, gt)
            matched = torch.zeros(len(gt), dtype=torch.bool, device=DEV)
            # Greedy: sort by confidence, match best
            _, sidx = k_conf[keep_nms].sort(descending=True)
            for si in sidx:
                vi = ious[si] * (~matched).float()
                if vi.max() > 0.5:
                    bi = vi.argmax(); matched[bi] = True
                    if ious[si, bi] >= 0.75: hits[g] += 1
        offset += n_p
    return hits


def run_one(cfg_name, mode, seed):
    run_name = f"round2107_{cfg_name}_s{seed}"; set_seed(seed)
    coco = json.loads(ANNOT.read_text())
    all_ids = list(set(img["id"] for img in coco["images"] if Path(DATA / "positive image set" / img["file_name"]).exists()))
    np.random.seed(42); np.random.shuffle(all_ids)
    nt = int(0.7 * len(all_ids))
    train_ids, val_ids = set(all_ids[:nt]), set(all_ids[nt:])
    tl = DataLoader(NWPUDataset(DATA, ANNOT, train_ids, MAX_SIZE), batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=0)
    vl = DataLoader(NWPUDataset(DATA, ANNOT, val_ids, MAX_SIZE), batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)

    model = build_detector({"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn", "pretrained": True,
        "num_classes": NUM_CLASSES, "min_size": MAX_SIZE, "max_size": MAX_SIZE}}).to(DEV)
    ckpt = torch.load(CKPT_PATH, map_location=DEV); model.load_state_dict(ckpt["model"])
    unfreeze_rlvr(model)
    baseline_model = copy.deepcopy(model); baseline_model.eval()
    for p in baseline_model.parameters(): p.requires_grad = False

    sampled_props, box_head_in = {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(lambda m, args: box_head_in.update({"x": args[0]}))

    run_dir = ensure_run_dir(run_name); opt = build_opt(model)
    is_det = mode == "det_only_unf"; is_rlvr = mode == "rlvr_discrete"
    baseline_bp = baseline_model.roi_heads.box_predictor
    h, best_ap50 = [], -1.0; diag = {"mean_hit": [], "mean_kept": []}

    for ep in range(1, EPOCHS + 1):
        model.train(); td, trl, tkl = 0.0, 0.0, 0.0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}", leave=False):
            imgs_d = [i.to(DEV) for i in imgs]; tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear(); box_head_in.clear()
            ld = model(imgs_d, tgts_t); det = sum(ld.values())
            rf = box_head_in.get("x"); sp_raw = sampled_props.get("p")
            rl = torch.tensor(0.0, device=DEV); kl = torch.tensor(0.0, device=DEV)

            if not is_det and rf is not None and sp_raw is not None and rf.shape[0] > 0:
                bf = model.roi_heads.box_head(rf); cls_logits = model.roi_heads.box_predictor.cls_score(bf)
                cur_conf = F.softmax(cls_logits, dim=-1)[:, 1:].max(dim=-1).values

                with torch.no_grad():
                    baseline_bf = baseline_model.roi_heads.box_head(rf)
                    baseline_logits = baseline_bp.cls_score(baseline_bf)
                    baseline_conf = F.softmax(baseline_logits, dim=-1)[:, 1:].max(dim=-1).values

                # Discrete Bernoulli: sample keep/reject from baseline
                with torch.no_grad():
                    kept = torch.bernoulli(baseline_conf.unsqueeze(1).expand(-1, G_SAMPLES))

                # Log-prob under CURRENT model (off-policy)
                eps = 1e-8; cur_g = cur_conf.unsqueeze(1).expand(-1, G_SAMPLES)
                lp = kept * torch.log(cur_g + eps) + (1 - kept) * torch.log(1 - cur_g + eps)
                log_probs_img = lp.sum(dim=0)  # (G,) image-level sum

                # Reward: NMS hit count (with predicted class for multi-class decoding)
                with torch.no_grad():
                    pred_class = baseline_logits.argmax(dim=-1)  # (N,) 0-10
                hits = discrete_nms_reward(sp_raw, kept, cur_g, pred_class, baseline_bp, baseline_bf, tgts_t)

                # GRPO across G samples
                r_mean, r_std = hits.mean(), hits.std().clamp_min(1e-6)
                adv = (hits - r_mean) / r_std
                rl = -(adv.detach() * log_probs_img).mean()
                diag["mean_hit"].append(hits.mean().item())
                diag["mean_kept"].append(kept.float().mean().item())

                kl = KL_WEIGHT * (cur_conf - baseline_conf).pow(2).mean()

            loss = det + RL_WEIGHT * rl + kl
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            td += det.item(); trl += rl.item(); tkl += kl.item()

        model.eval(); ps, ts = [], []
        for img, tgt in vl:
            with torch.no_grad(): pred = model([img[0].to(DEV)])[0]
            ps.append({k: v.cpu() for k, v in pred.items()})
            ts.append({k: v.cpu() for k, v in tgt[0].items()})
        em = evaluate_detection_predictions(ps, ts, iou_threshold=0.5, score_threshold=0.05)
        mh = np.mean(diag["mean_hit"]) if diag["mean_hit"] else 0
        mk = np.mean(diag["mean_kept"]) if diag["mean_kept"] else 0
        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "ece": em.get("ece", 0), "mean_hit": float(mh), "mean_kept": float(mk),
               "det": td, "rl": trl, "kl": tkl}
        h.append(row); print(f"  e{ep}: AP50={em['ap50']:.4f} AP75={em['ap75']:.4f} hit={mh:.1f} kept={mk:.2f}")
        if em["ap50"] > best_ap50: best_ap50 = em["ap50"]
        for k in diag: diag[k].clear()

    best_h = max(h, key=lambda r: r["val_ap50"])
    em.update({"run_name": run_name, "config": cfg_name, "mode": mode, "seed": seed,
               "epochs": len(h), "best_ap50": best_ap50, "best_ap75": best_h["val_ap75"], "history": h})
    save_json(em, run_dir / "eval_metrics.json"); return em


if __name__ == "__main__":
    results = []
    for cfg, mode in [("det_only", "det_only_unf"), ("discrete", "rlvr_discrete")]:
        r = run_one(cfg, mode, 42); results.append(r)
    print("\n## 2.107 NWPU Discrete RLVR")
    for r in results:
        bh = max(r["history"], key=lambda x: x["val_ap50"])
        print(f"  {r['config']:<12s} s{r['seed']} AP50={r['best_ap50']:.4f} AP75={bh['val_ap75']:.4f}")
