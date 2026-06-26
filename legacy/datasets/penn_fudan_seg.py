"""Penn-Fudan Pedestrian segmentation dataset and loaders.

Reusable module extracted from scripts/round4x_seg_runner.py for Plan C.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image


class PennFudanSegDataset(Dataset):
    """Penn-Fudan Pedestrian segmentation dataset.

    Masks are encoded as: 0=background, 1=pedestrian, 2=border. Border pixels
    are merged into the pedestrian class (1).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        fraction: float = 0.8,
        max_size: int = 320,
        limit: int | None = None,
    ):
        self.root = Path(root)
        self.max_size = max_size
        imgs = sorted((self.root / "PNGImages").glob("*.png"))
        n_train = int(len(imgs) * fraction)
        self.images = imgs[:n_train] if split == "train" else imgs[n_train:]
        self.masks = [self.root / "PedMasks" / (p.stem + "_mask.png") for p in self.images]
        if limit is not None:
            self.images = self.images[:limit]
            self.masks = self.masks[:limit]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        # resize
        w, h = img.size
        scale = min(self.max_size / max(w, h), 1.0)
        nw, nh = int(w * scale), int(h * scale)
        img = TF.resize(img, [nh, nw])
        mask = TF.resize(mask, [nh, nw], interpolation=TF.InterpolationMode.NEAREST)
        img_t = TF.to_tensor(img)
        mask_t = torch.as_tensor(np.array(mask), dtype=torch.long)
        # Penn-Fudan PedMasks use 0=background, positive integers=instances/border.
        # Binarize to {0=background, 1=pedestrian}.
        mask_t[mask_t > 0] = 1
        return img_t, mask_t


def _collate(batch):
    return tuple(zip(*batch))


def build_seg_loaders(
    root: str,
    max_size: int = 320,
    batch_size: int = 1,
    num_workers: int = 0,
    limit_train: int | None = None,
    limit_val: int | None = None,
):
    """Build train and validation dataloaders for Penn-Fudan segmentation."""
    train_ds = PennFudanSegDataset(root, "train", max_size=max_size, limit=limit_train)
    val_ds = PennFudanSegDataset(root, "val", max_size=max_size, limit=limit_val)
    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=_collate,
            pin_memory=torch.cuda.is_available(),
        ),
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate,
            pin_memory=torch.cuda.is_available(),
        ),
    )
