"""COCO 2017 detection loader compatible with the project pipeline.

Uses ``torchvision.datasets.CocoDetection`` internally and converts targets to the
same dictionary format as the VOC/Penn-Fudan loaders:
    boxes   -> FloatTensor[N, 4] in (x1, y1, x2, y2)
    labels  -> LongTensor[N]   (1-indexed class labels, 0 = background)
    image_id-> LongTensor[1]
    area    -> FloatTensor[N]
    iscrowd -> LongTensor[N]
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as torch_f
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F


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


def _build_category_map(coco: COCO) -> dict[int, int]:
    """Map original COCO category ids to contiguous 1..80 labels."""
    cats = coco.loadCats(coco.getCatIds())
    ids = sorted([cat["id"] for cat in cats])
    return {cat_id: idx + 1 for idx, cat_id in enumerate(ids)}


class COCODetectionSubset(Dataset):
    """Thin wrapper around ``CocoDetection`` that returns project-compatible targets."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        max_size: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.max_size = max_size
        self.image_dir = self.root / f"{split}2017"
        ann_file = self.root / "annotations" / f"instances_{split}2017.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"COCO image directory not found: {self.image_dir}")
        self.coco_detection = CocoDetection(str(self.image_dir), str(ann_file))
        self.cat_map = _build_category_map(self.coco_detection.coco)
        # Filter out images with no valid annotations.
        self.valid_indices = []
        for idx in range(len(self.coco_detection)):
            _, anns = self.coco_detection[idx]
            if self._target_from_annotations(anns)["boxes"].shape[0] > 0:
                self.valid_indices.append(idx)

    def _target_from_annotations(self, anns: list[dict]) -> dict:
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_map.get(ann["category_id"], -1))
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
        area_tensor = torch.as_tensor(areas, dtype=torch.float32) if areas else torch.empty((0,), dtype=torch.float32)
        iscrowd_tensor = torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.empty((0,), dtype=torch.int64)
        return {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([0]),  # will be overwritten
            "area": area_tensor,
            "iscrowd": iscrowd_tensor,
        }

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        real_idx = self.valid_indices[idx]
        image, anns = self.coco_detection[real_idx]
        image_tensor = F.to_tensor(image)
        target = self._target_from_annotations(anns)
        target["image_id"] = torch.tensor([real_idx])
        if self.max_size:
            image_tensor, target = _resize_image_and_target(image_tensor, target, int(self.max_size))
        return image_tensor, target


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_coco_detection_loaders(
    config: dict,
    limit_train: int | None = None,
    limit_val: int | None = None,
    batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    data_root = str(data_cfg.get("root", "./data"))
    max_size = data_cfg.get("max_size")

    train_set = COCODetectionSubset(data_root, split="train", max_size=max_size)
    val_set = COCODetectionSubset(data_root, split="val", max_size=max_size)

    rng = random.Random(int(config.get("seed", 42)))
    train_indices = list(range(len(train_set)))
    val_indices = list(range(len(val_set)))
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    if limit_train is not None:
        train_indices = train_indices[:limit_train]
    if limit_val is not None:
        val_indices = val_indices[:limit_val]

    bs = int(batch_size or config["train"].get("batch_size", 2))
    num_workers = int(data_cfg.get("num_workers", 0))
    return (
        DataLoader(Subset(train_set, train_indices), batch_size=bs, shuffle=True, num_workers=num_workers, collate_fn=detection_collate),
        DataLoader(Subset(val_set, val_indices), batch_size=bs, shuffle=False, num_workers=num_workers, collate_fn=detection_collate),
    )
