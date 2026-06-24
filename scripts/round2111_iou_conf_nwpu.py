"""Plan 2.111: NWPU IoU×conf RLVR — round2102 proven flow adapted for NWPU.

Hypothesis: if Δ > +0.011 (PennFudan), data scale is the bottleneck.
"""
import copy, json, sys, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.experiments.runner_utils import decode_boxes, gaussian_log_prob, unfreeze_rlvr
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import set_seed

DEV, SEED = "cuda", 42; SEEDS = [42, 123, 456]
BATCH, EPOCHS, MAX_SIZE, NUM_CLASSES = 2, 8, 480, 11
RL_WEIGHT, KL_WEIGHT = 0.0005, 0.01; G_SAMPLES = 4
DATA = Path("data/NWPU VHR-10 dataset"); ANNOT = Path("data/NWPU_VHR10_coco.json")
CKPT_PATH = "runs/round2100_nwpu_baseline/checkpoint_best.pth"

def cross_proposal_grpo(reward, n_props):
    adv = torch.zeros_like(reward); off = 0
    for n_p in n_props:
        if n_p <= 0: continue
        if n_p == 1: adv[off] = 0.0; off += n_p; continue
        r = reward[off:off+n_p]; m, s = r.mean(), r.std().clamp_min(1e-6)
        adv[off:off+n_p] = (r - m) / s; off += n_p
    return adv

class NWPUDataset(Dataset):
    def __init__(self, root, coco_json, img_ids, max_size):
        self.root = Path(root); self.max_size = max_size
        self.coco = json.loads(Path(coco_json).read_text())
        self.img_infos = {img["id"]: img for img in self.coco["images"] if img["id"] in img_ids}
        self.img_ids = list(self.img_infos.keys())
        anns = {}
        for ann in self.coco["annotations"]:
            if ann["image_id"] in img_ids: anns.setdefault(ann["image_id"], []).append(ann)
        self.anns = anns
    def __len__(self): return len(self.img_ids)
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]; info = self.img_infos[img_id]
        img_path = self.root / "positive image set" / info["file_name"]
        if not img_path.exists(): img_path = self.root / "negative image set" / info["file_name"]
        img = Image.open(str(img_path)).convert("RGB"); img_t = TF.to_tensor(img)
        boxes, labels = [], []
        for ann in self.anns.get(img_id, []):
            x, y, w, h = ann["bbox"]; boxes.append([x, y, x + w, y + h]); labels.append(ann["category_id"])
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32),
                  "labels": torch.tensor(labels, dtype=torch.int64), "image_id": torch.tensor([img_id])}
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

def compute_stable_iou(sp_raw, bf, box_predictor, tgts_t):
    """Class-agnostic max IoU using baseline box_predictor (stable, like round2102)."""
    sp_cat = torch.cat(sp_raw, dim=0); N = sp_cat.shape[0]
    with torch.no_grad():
        reg = box_predictor.bbox_pred(bf[:N]); deltas = reg[:, 4:8]  # class 1 (airplane)
        decoded = decode_boxes(sp_cat, deltas)
    iou = torch.zeros(N, device=DEV); off = 0
    for i_img, p_img in enumerate(sp_raw):
        n_p = p_img.shape[0]
        if n_p > 0:
            gt = tgts_t[i_img]["boxes"]
            if len(gt) > 0: iou[off:off+n_p] = box_iou(decoded[off:off+n_p], gt).max(dim=1).values
        off += n_p
    return iou

def run_one(cfg_name, mode, seed):
    run_name = f"round2111_{cfg_name}_s{seed}"; set_seed(seed)
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

    opt = build_opt(model); baseline_bp = baseline_model.roi_heads.box_predictor
    run_dir = ensure_run_dir(run_name)
    is_det = mode == "det_only"; is_rlvr = mode == "rlvr"
    h, best_ap50 = [], -1.0

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
                with torch.no_grad():
                    baseline_bf = baseline_model.roi_heads.box_head(rf)
                    baseline_cls_conf = F.softmax(baseline_bp.cls_score(baseline_bf), dim=-1)[:, 1:].max(dim=-1).values

                # Off-policy: baseline sampling + adaptive sigma
                with torch.no_grad():
                    sp_sigma = 0.05 + 0.2 * (1.0 - baseline_cls_conf)
                    s_base = sp_sigma.unsqueeze(1).expand(-1, cls_logits.shape[1])
                    bl_logits = baseline_bp.cls_score(baseline_bf)
                    perturbed = bl_logits.unsqueeze(1) + s_base.unsqueeze(1) * torch.randn(
                        bl_logits.shape[0], G_SAMPLES, bl_logits.shape[1], device=DEV)
                pert_conf = F.softmax(perturbed, dim=-1)[:, :, 1:].max(dim=-1).values
                s_cls = sp_sigma.unsqueeze(1).expand(-1, cls_logits.shape[1])
                log_probs = gaussian_log_prob(perturbed, cls_logits, s_cls)

                # Stable IoU (from baseline, like round2102)
                iou_p = compute_stable_iou(sp_raw, baseline_bf, baseline_bp, tgts_t)
                N = min(cls_logits.shape[0], iou_p.shape[0])

                # IoU×conf reward
                quality = (2 * iou_p[:N] - 1).unsqueeze(1)
                reward = pert_conf[:N] * quality
                reward_flat = reward.reshape(-1)
                npp = [p.shape[0] * G_SAMPLES for p in sp_raw]
                adv = cross_proposal_grpo(reward_flat, npp).view(N, G_SAMPLES)
                rl = -(adv.detach() * log_probs[:N]).mean()
                kl = KL_WEIGHT * (pert_conf[:N] - baseline_cls_conf[:N].unsqueeze(1)).pow(2).mean()

            loss = det + RL_WEIGHT * rl + kl
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            td += det.item(); trl += rl.item(); tkl += kl.item()

        model.eval(); ps, ts = [], []
        for img, tgt in vl:
            with torch.no_grad(): pred = model([img[0].to(DEV)])[0]
            ps.append({k: v.cpu() for k, v in pred.items()}); ts.append({k: v.cpu() for k, v in tgt[0].items()})
        em = evaluate_detection_predictions(ps, ts, iou_threshold=0.5, score_threshold=0.05)
        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "ece": em.get("ece", 0), "det": td, "rl": trl, "kl": tkl}
        h.append(row); print(f"  e{ep}: AP50={em['ap50']:.4f} AP75={em['ap75']:.4f}")
        if em["ap50"] > best_ap50: best_ap50 = em["ap50"]

    best_h = max(h, key=lambda r: r["val_ap50"])
    em.update({"run_name": run_name, "config": cfg_name, "mode": mode, "seed": seed,
               "epochs": len(h), "best_ap50": best_ap50, "best_ap75": best_h["val_ap75"], "history": h})
    save_json(em, run_dir / "eval_metrics.json"); return em

if __name__ == "__main__":
    results = []
    for cfg, mode in [("det_only", "det_only"), ("rlvr", "rlvr")]:
        for s in SEEDS: results.append(run_one(cfg, mode, s))
    print("\n## 2.111 NWPU IoU×conf RLVR")
    for r in results:
        bh = max(r["history"], key=lambda x: x["val_ap50"])
        print(f"  {r['config']:<10s} s{r['seed']} AP50={r['best_ap50']:.4f} AP75={bh['val_ap75']:.4f}")
    for cfg in ["det_only", "rlvr"]:
        vals = [r for r in results if r["config"] == cfg]
        if vals: print(f"  {cfg}: bestAP50={np.mean([v['best_ap50'] for v in vals]):.4f} +/- {np.std([v['best_ap50'] for v in vals]):.4f}")
