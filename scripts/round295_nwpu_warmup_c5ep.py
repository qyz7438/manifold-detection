"""NWPU VHR-10: mid06 warmup fine-tune then C_5ep continuation.

Phase 1: load baseline, insert AFM (mplseg_mid), full-model fine-tune 2 epochs (lr=1e-3).
Phase 2: freeze all except AFM, feature-constraint MSE*0.05, 5 epochs (lr=1e-4).
"""
import sys, json
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm
import numpy as np

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.utils.seed import set_seed
from spectral_detection_posttrain.utils.io import save_checkpoint

DEV = "cuda"; BATCH = 4; CKPT_PATH = "runs/nwpu_baseline_best.pth"
DATA = Path("data/NWPU VHR-10 dataset"); ANNOT = Path("data/NWPU_VHR10_coco.json")
MAX_SIZE = 480; WARMUP_EPOCHS = 2; C_EPOCHS = 5; SEEDS = [42, 123, 456]

class NWPUDataset(Dataset):
    def __init__(self, root, coco_json, img_ids, max_size):
        self.root = Path(root); self.max_size = max_size
        self.coco = json.loads(Path(coco_json).read_text())
        self.ims = [img for img in self.coco["images"] if img["id"] in img_ids]
        self.anns = {}
        for a in self.coco["annotations"]:
            if a["image_id"] in img_ids: self.anns.setdefault(a["image_id"], []).append(a)
    def __len__(self): return len(self.ims)
    def __getitem__(self, idx):
        info = self.ims[idx]; img_id = info["id"]
        p = self.root / "positive image set" / info["file_name"]
        if not p.exists(): p = self.root / "negative image set" / info["file_name"]
        img = TF.to_tensor(Image.open(str(p)).convert("RGB"))
        boxes, labels = [], []
        for a in self.anns.get(img_id, []):
            x, y, w, h = a["bbox"]; boxes.append([x, y, x+w, y+h]); labels.append(a["category_id"])
        tgt = {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.int64)}
        _, H, W = img.shape
        if max(H, W) > self.max_size:
            scale = self.max_size / max(H, W); nh, nw = int(H*scale), int(W*scale)
            img = nn.functional.interpolate(img.unsqueeze(0), size=(nh, nw), mode="bilinear").squeeze(0)
            tgt["boxes"] = tgt["boxes"] * scale
        return img, tgt
def collate(batch): return tuple(zip(*batch))

coco = json.loads(ANNOT.read_text())
all_ids = list(set(img["id"] for img in coco["images"] if Path(DATA/"positive image set"/img["file_name"]).exists()))
np.random.seed(42); np.random.shuffle(all_ids)
n_train = int(0.7 * len(all_ids))
train_ids = set(all_ids[:n_train]); val_ids = set(all_ids[n_train:])

num_classes = 11

def make_model():
    model = build_detector({"model": {
        "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "pretrained": True, "num_classes": num_classes,
        "min_size": MAX_SIZE, "max_size": MAX_SIZE,
        "afm_type": "mplseg_mid",
    }}).to(DEV)
    ckpt = torch.load(CKPT_PATH, map_location=DEV)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.roi_heads.box_head.afm.gate_strength = 0.6
    model.roi_heads.box_head.afm.residual_scale.data.fill_(1.0)
    return model

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval(); ps, ts = [], []
    for img, tgt in val_loader:
        pred = model([img[0].to(DEV)])[0]
        ps.append({k: v.cpu() for k, v in pred.items()}); ts.append({k: v.cpu() for k, v in tgt[0].items()})
    return evaluate_detection_predictions(ps, ts, iou_threshold=0.5, score_threshold=0.05)

def train_phase(model, epochs, lr, use_constraint, train_loader, val_loader, desc, save_path):
    afm = model.roi_heads.box_head.afm
    afm_in = {}
    if use_constraint:
        def pre_hook(m, inp):
            afm_in["x"] = inp[0].detach()
        def fwd_hook(m, inp, out):
            afm_in["y"] = out
        h1 = afm.register_forward_pre_hook(pre_hook)
        h2 = afm.register_forward_hook(fwd_hook)
    else:
        h1 = h2 = None

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=lr, momentum=0.9, weight_decay=0.0005)
    best = -1
    for ep in range(1, epochs + 1):
        model.train(); total_loss = 0
        for imgs, tgts in tqdm(train_loader, desc=f"{desc} e{ep}"):
            imgs = [i.to(DEV) for i in imgs]; tgts = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            if use_constraint: afm_in.clear()
            ld = model(imgs, tgts)
            loss = sum(v for v in ld.values())
            if use_constraint:
                x = afm_in.get("x"); y = afm_in.get("y")
                if x is not None and y is not None:
                    loss = loss + 0.05 * nn.functional.mse_loss(y, x)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        em = evaluate(model, val_loader)
        print(f"  {desc} e{ep}: AP50={em['ap50']:.4f} AP75={em['ap75']:.4f} loss={total_loss/len(train_loader):.4f}")
        if em['ap50'] > best:
            best = em['ap50']
            torch.save({"model": model.state_dict(), "ap50": best, "ap75": em['ap75']}, save_path)
    if h1: h1.remove()
    if h2: h2.remove()
    return best

for seed in SEEDS:
    set_seed(seed)
    print(f"\n=== seed={seed} ===")
    train_ds = NWPUDataset(DATA, ANNOT, train_ids, MAX_SIZE)
    val_ds = NWPUDataset(DATA, ANNOT, val_ids, MAX_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)

    # Phase 1: warmup full-model fine-tune
    model = make_model().to(DEV)
    train_phase(model, WARMUP_EPOCHS, 1e-3, False, train_loader, val_loader,
                f"warmup s{seed}", f"runs/nwpu_warmup_s{seed}_best.pth")

    # Phase 2: C_5ep with frozen backbone/rpn/box_head
    for p in model.parameters(): p.requires_grad = False
    for p in model.roi_heads.box_head.afm.parameters(): p.requires_grad = True
    train_phase(model, C_EPOCHS, 1e-4, True, train_loader, val_loader,
                f"C_5ep_lr1e4 s{seed}", f"runs/nwpu_C_5ep_lr1e4_s{seed}_best.pth")
