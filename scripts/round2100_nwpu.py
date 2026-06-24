"""Plan 2.100: NWPU VHR-10 baseline training.

NWPU VHR-10: 800 aerial images, 10 classes, ~3700 instances.
Train a Faster R-CNN MobileNetV3 baseline from scratch (30 epoch).
Saves checkpoint for RLVR post-training in round2101.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import set_seed

DEV = "cuda"
SEED = 42
BATCH = 4
EPOCHS = 30
LR = 0.005
MAX_SIZE = 480
NUM_CLASSES = 11
DATA = Path("data/NWPU VHR-10 dataset")
ANNOT = Path("data/NWPU_VHR10_coco.json")
RUN_DIR = Path("runs/round2100_nwpu_baseline")
CKPT_PATH = RUN_DIR / "checkpoint_best.pth"

set_seed(SEED)


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


if __name__ == "__main__":
    ensure_run_dir("round2100_nwpu_baseline")

    coco = json.loads(ANNOT.read_text())
    all_ids = list(
        set(
            img["id"]
            for img in coco["images"]
            if Path(DATA / "positive image set" / img["file_name"]).exists()
        )
    )
    np.random.seed(42)
    np.random.shuffle(all_ids)
    n_train = int(0.7 * len(all_ids))
    train_ids = set(all_ids[:n_train])
    val_ids = set(all_ids[n_train:])
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    train_ds = NWPUDataset(DATA, ANNOT, train_ids, MAX_SIZE)
    val_ds = NWPUDataset(DATA, ANNOT, val_ids, MAX_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)

    model = build_detector(
        {
            "model": {
                "name": "fasterrcnn_mobilenet_v3_large_fpn",
                "model_name": "fasterrcnn_mobilenet_v3_large_fpn",
                "pretrained": True,
                "num_classes": NUM_CLASSES,
                "min_size": MAX_SIZE,
                "max_size": MAX_SIZE,
            }
        }
    ).to(DEV)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

    best_ap50 = -1.0
    history = []

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for imgs, tgts in tqdm(train_loader, desc=f"NWPU baseline e{ep}"):
            imgs = [i.to(DEV) for i in imgs]
            tgts = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            loss_dict = model(imgs, tgts)
            loss = sum(v for v in loss_dict.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        lr_scheduler.step()

        model.eval()
        ps, ts = [], []
        for img, tgt in val_loader:
            with torch.no_grad():
                pred = model([img[0].to(DEV)])[0]
            ps.append({k: v.cpu() for k, v in pred.items()})
            ts.append({k: v.cpu() for k, v in tgt[0].items()})

        em = evaluate_detection_predictions(ps, ts, iou_threshold=0.5, score_threshold=0.05)
        print(f"  e{ep}: loss={total_loss:.1f} AP50={em['ap50']:.4f} AP75={em['ap75']:.4f}")

        if em["ap50"] > best_ap50:
            best_ap50 = em["ap50"]
            ensure_run_dir("round2100_nwpu_baseline")
            torch.save({"model": model.state_dict(), "epoch": ep, "ap50": best_ap50}, CKPT_PATH)
            print(f"  -> saved best checkpoint (AP50={best_ap50:.4f})")

        history.append({"epoch": ep, "loss": total_loss, "ap50": em["ap50"], "ap75": em["ap75"]})

    em.update({"run_name": "round2100_nwpu_baseline", "epochs": EPOCHS, "best_ap50": best_ap50, "history": history})
    save_json(em, RUN_DIR / "eval_metrics.json")
    print(f"\nBest AP50: {best_ap50:.4f}  AP75: {em['ap75']:.4f}")
