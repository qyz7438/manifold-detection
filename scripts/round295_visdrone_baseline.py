"""VisDrone baseline training — 6,471 train / 548 val, 10 classes, dense small objects."""
import sys, math
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm
import numpy as np
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed

DEV = "cuda"; SEED = 42; BATCH = 2; EPOCHS = 30; LR = 0.005; MAX_SIZE = 1000
TRAIN_IMG = Path("data/VisDrone/VisDrone2019-DET-train/images")
TRAIN_ANN = Path("data/VisDrone/VisDrone2019-DET-train/annotations")
VAL_IMG   = Path("data/VisDrone/VisDrone2019-DET-val/images")
VAL_ANN   = Path("data/VisDrone/VisDrone2019-DET-val/annotations")
set_seed(SEED)

class VisDroneDataset(Dataset):
    """VisDrone detection: 1 txt per image, format x1,y1,w,h,score,cat,trunc,occl."""
    def __init__(self, img_dir, ann_dir, max_size):
        self.img_dir = Path(img_dir); self.ann_dir = Path(ann_dir)
        self.max_size = max_size
        self.samples = sorted(p.stem for p in self.img_dir.glob("*.jpg"))
        # filter to samples where annotation exists
        self.samples = [s for s in self.samples if (self.ann_dir / f"{s}.txt").exists()]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        img = Image.open(str(self.img_dir / f"{name}.jpg")).convert("RGB")
        img_t = TF.to_tensor(img)

        boxes, labels = [], []
        for line in (self.ann_dir / f"{name}.txt").read_text().strip().splitlines():
            if not line.strip(): continue
            parts = line.strip().split(",")
            if len(parts) < 6: continue
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            score = int(parts[4]); cat = int(parts[5])
            if w <= 0 or h <= 0: continue
            # Skip ignored regions: (cat=0,score=0) and (cat=11,score=0)
            if score == 0 and cat in (0, 11): continue
            if cat < 1 or cat > 10: continue  # safety
            boxes.append([float(x), float(y), float(x+w), float(y+h)])
            # VisDrone categories: 0/11=ignored(skipped), 1-10=valid classes
            labels.append(cat)

        if len(boxes) == 0:
            # Return empty tensor — will be filtered by collate
            return img_t, {"boxes": torch.zeros((0,4), dtype=torch.float32),
                           "labels": torch.zeros((0,), dtype=torch.int64)}

        _, H, W = img_t.shape
        if max(H, W) > self.max_size:
            scale = self.max_size / max(H, W)
            new_h, new_w = int(H*scale), int(W*scale)
            img_t = F.interpolate(img_t.unsqueeze(0), size=(new_h, new_w), mode="bilinear").squeeze(0)
            boxes = [[x*scale for x in b] for b in boxes]

        target = {"boxes": torch.tensor(boxes, dtype=torch.float32),
                  "labels": torch.tensor(labels, dtype=torch.int64)}
        return img_t, target

def collate(batch):
    # Filter out samples with no boxes
    valid = [(img, tgt) for img, tgt in batch if tgt["boxes"].shape[0] > 0]
    if len(valid) == 0: return [], []
    return tuple(zip(*valid))

if __name__ == "__main__":
    print(f"Loading VisDrone: {len(list(TRAIN_IMG.glob('*.jpg')))} train, {len(list(VAL_IMG.glob('*.jpg')))} val")

    train_ds = VisDroneDataset(TRAIN_IMG, TRAIN_ANN, MAX_SIZE)
    val_ds   = VisDroneDataset(VAL_IMG, VAL_ANN, MAX_SIZE)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)

    num_classes = 11  # 10 VisDrone classes + background
    model = build_detector({"model":{
    "name":"fasterrcnn_mobilenet_v3_large_fpn",
    "model_name":"fasterrcnn_mobilenet_v3_large_fpn",
    "pretrained":True, "num_classes":num_classes,
    "min_size":MAX_SIZE, "max_size":MAX_SIZE,
    }}).to(DEV)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

    best_ap50 = -1; history = []

    for ep in range(1, EPOCHS+1):
    model.train(); total_loss = 0
    pbar = tqdm(train_loader, desc=f"VisDrone e{ep}")
    for imgs, tgts in pbar:
        if len(imgs) == 0: continue
        imgs = [i.to(DEV) for i in imgs]; tgts = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
        loss_dict = model(imgs, tgts)
        loss = sum(v for v in loss_dict.values())
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.2f}")
    lr_scheduler.step()

    # Eval
    model.eval(); ps, ts = [], []
    for img, tgt in val_loader:
        if len(img) == 0: continue
        with torch.no_grad():
            pred = model([img[0].to(DEV)])[0]
        ps.append({k: v.cpu() for k, v in pred.items()})
        ts.append({k: v.cpu() for k, v in tgt[0].items()})

    from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
    em = evaluate_detection_predictions(ps, ts, iou_threshold=0.5, score_threshold=0.05)

    print(f"e{ep:2d}: loss={total_loss:.1f} AP50={em['ap50']:.4f} AP75={em['ap75']:.4f}")

    if em['ap50'] > best_ap50:
        best_ap50 = em['ap50']
        torch.save({"model": model.state_dict(), "epoch": ep, "ap50": best_ap50, "ap75": em['ap75']},
                   "runs/visdrone_baseline_best.pth")

    history.append({"epoch": ep, "loss": total_loss, "ap50": em['ap50'], "ap75": em['ap75']})

    print(f"\nBest AP50: {best_ap50:.4f}")
    for h in history:
    print(f"  e{h['epoch']:2d}: AP50={h['ap50']:.4f} AP75={h['ap75']:.4f}")
