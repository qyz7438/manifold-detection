from __future__ import annotations

import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import functional as F


PENN_FUDAN_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"


def _download_penn_fudan(root: Path) -> None:
    dataset_dir = root / "PennFudanPed"
    if dataset_dir.exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "PennFudanPed.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(PENN_FUDAN_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)


class PennFudanDetectionDataset(Dataset):
    def __init__(self, root: str | Path, download: bool = True, max_size: int | None = None) -> None:
        self.root = Path(root)
        self.max_size = max_size
        if download:
            _download_penn_fudan(self.root)
        self.dataset_dir = self.root / "PennFudanPed"
        self.image_dir = self.dataset_dir / "PNGImages"
        self.mask_dir = self.dataset_dir / "PedMasks"
        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(f"Penn-Fudan dataset not found under {self.dataset_dir}")
        self.images = sorted(self.image_dir.glob("*.png"))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        mask_path = self.mask_dir / image_path.name.replace(".png", "_mask.png")
        image = Image.open(image_path).convert("RGB")
        mask = np.array(Image.open(mask_path))

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]
        boxes = []
        for obj_id in obj_ids:
            ys, xs = np.where(mask == obj_id)
            if len(xs) == 0 or len(ys) == 0:
                continue
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
        target = {
            "boxes": boxes_tensor,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        image_tensor = F.to_tensor(image)
        if self.max_size:
            image_tensor, target = _resize_image_and_target(image_tensor, target, int(self.max_size))
        return image_tensor, target


def _resize_image_and_target(image: torch.Tensor, target: dict, max_size: int) -> tuple[torch.Tensor, dict]:
    _, height, width = image.shape
    largest_side = max(height, width)
    if largest_side <= max_size:
        return image, target
    scale = max_size / float(largest_side)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    resized = torch_f.interpolate(
        image.unsqueeze(0),
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    scaled_target = dict(target)
    boxes = target["boxes"].clone()
    boxes[:, [0, 2]] *= new_width / float(width)
    boxes[:, [1, 3]] *= new_height / float(height)
    scaled_target["boxes"] = boxes
    scaled_target["area"] = (boxes[:, 3] - boxes[:, 1]).clamp_min(0) * (boxes[:, 2] - boxes[:, 0]).clamp_min(0)
    return resized, scaled_target


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def _subset(dataset: Dataset, indices: list[int], limit: int | None) -> Dataset:
    if limit is not None:
        indices = indices[: min(limit, len(indices))]
    return Subset(dataset, indices)


def build_penn_fudan_loaders(
    config: dict,
    limit_train: int | None = None,
    limit_val: int | None = None,
    batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    dataset = PennFudanDetectionDataset(
        data_cfg.get("root", "./data"),
        download=bool(data_cfg.get("download", True)),
        max_size=data_cfg.get("max_size"),
    )
    indices = list(range(len(dataset)))
    rng = random.Random(int(config.get("seed", 42)))
    rng.shuffle(indices)
    split = int(len(indices) * float(data_cfg.get("train_fraction", 0.8)))
    train_indices = indices[:split]
    val_indices = indices[split:]
    train_set = _subset(dataset, train_indices, limit_train)
    val_set = _subset(dataset, val_indices, limit_val)
    bs = int(batch_size or config["train"].get("batch_size", 2))
    num_workers = int(data_cfg.get("num_workers", 0))
    return (
        DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=num_workers, collate_fn=detection_collate),
        DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=num_workers, collate_fn=detection_collate),
    )
