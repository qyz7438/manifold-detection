"""NWPU VHR-10 baseline training + PG post-training test."""
import sys, json, math, copy
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm
import numpy as np
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
from spectral_detection_posttrain.utils.io import save_json, ensure_run_dir

DEV = "cuda"; SEED = 42; BATCH = 4; EPOCHS = 30; LR = 0.005; MAX_SIZE = 480
DATA = Path("data/NWPU VHR-10 dataset")
ANNOT = Path("data/NWPU_VHR10_coco.json")
set_seed(SEED)

# ---------- NWPU Dataset ----------
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

    def __len__(self): return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.img_infos[img_id]
        img_path = self.root / "positive image set" / info["file_name"]
        if not img_path.exists():
            img_path = self.root / "negative image set" / info["file_name"]

        img = Image.open(str(img_path)).convert("RGB")
        img_t = TF.to_tensor(img)

        boxes = []; labels = []
        for ann in self.anns.get(img_id, []):
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x+w, y+h])
            labels.append(ann["category_id"])

        target = {"boxes": torch.tensor(boxes, dtype=torch.float32),
                  "labels": torch.tensor(labels, dtype=torch.int64),
                  "image_id": torch.tensor([img_id])}

        # Resize
        _, H, W = img_t.shape
        if max(H, W) > self.max_size:
            scale = self.max_size / max(H, W)
            new_h, new_w = int(H*scale), int(W*scale)
            img_t = F.interpolate(img_t.unsqueeze(0), size=(new_h, new_w), mode="bilinear").squeeze(0)
            target["boxes"] = target["boxes"] * scale

        return img_t, target

def collate(batch):
    return tuple(zip(*batch))

# ---------- Build data ----------
coco = json.loads(ANNOT.read_text())
all_ids = list(set(img["id"] for img in coco["images"] if Path(DATA/"positive image set"/img["file_name"]).exists()))
np.random.seed(42); np.random.shuffle(all_ids)
n_train = int(0.7 * len(all_ids))
train_ids = set(all_ids[:n_train]); val_ids = set(all_ids[n_train:])
print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

train_ds = NWPUDataset(DATA, ANNOT, train_ids, MAX_SIZE)
val_ds = NWPUDataset(DATA, ANNOT, val_ids, MAX_SIZE)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)

# ---------- Model ----------
num_classes = 11  # 10 NWPU classes + background
model = build_detector({"model":{"name":"fasterrcnn_mobilenet_v3_large_fpn","model_name":"fasterrcnn_mobilenet_v3_large_fpn","pretrained":True,"num_classes":num_classes,"min_size":MAX_SIZE,"max_size":MAX_SIZE}}).to(DEV)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

best_ap50 = -1; history = []

for ep in range(1, EPOCHS+1):
    model.train(); total_loss = 0
    for imgs, tgts in tqdm(train_loader, desc=f"NWPU e{ep}"):
        imgs = [i.to(DEV) for i in imgs]; tgts = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
        loss_dict = model(imgs, tgts)
        loss = sum(v for v in loss_dict.values())
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    lr_scheduler.step()

    # Eval
    model.eval(); ps, ts = [], []
    for img, tgt in val_loader:
        with torch.no_grad():
            pred = model([img[0].to(DEV)])[0]
        ps.append({k: v.cpu() for k, v in pred.items()})
        ts.append({k: v.cpu() for k, v in tgt[0].items()})

    # Compute mAP
    from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
    em = evaluate_detection_predictions(ps, ts, iou_threshold=0.5, score_threshold=0.05)

    print(f"e{ep}: loss={total_loss:.1f} AP50={em['ap50']:.4f} AP75={em['ap75']:.4f}")

    if em['ap50'] > best_ap50:
        best_ap50 = em['ap50']
        torch.save({"model": model.state_dict(), "epoch": ep, "ap50": best_ap50}, "runs/nwpu_baseline_best.pth")

    history.append({"epoch": ep, "loss": total_loss, "ap50": em['ap50'], "ap75": em['ap75']})

print(f"\nBest AP50: {best_ap50:.4f}")
ap_hist = [(h["epoch"], "{:.3f}".format(h["ap50"])) for h in history]
print(f"History: {ap_hist}")

# ---------- Check borderline proposals on trained model ----------
print("\n=== Borderline proposal check (trained model) ===")
model.load_state_dict(torch.load("runs/nwpu_baseline_best.pth")["model"])
model.eval()

all_ious = []
for img, tgt in val_loader:
    with torch.no_grad():
        pred = model([img[0].to(DEV)])[0]
    pboxes = pred["boxes"].cpu()
    gboxes = tgt[0]["boxes"]
    if len(pboxes) > 0 and len(gboxes) > 0:
        ious = box_iou(pboxes, gboxes).max(dim=1).values.cpu().numpy()
        all_ious.extend(ious.tolist())

if all_ious:
    ious = np.array(all_ious)
    for lo, hi in [(0,0.3),(0.3,0.5),(0.5,0.7),(0.7,0.9),(0.9,1.0)]:
        cnt = sum(1 for i in ious if lo <= i < hi)
        print(f"  [{lo:.1f},{hi:.1f}): {cnt:4d} ({100*cnt/len(ious):5.1f}%)")
    bl = sum(1 for i in ious if 0.3 <= i < 0.7)
    print(f"  Borderline (0.3-0.7): {bl}/{len(ious)} ({100*bl/len(ious):.1f}%)")
