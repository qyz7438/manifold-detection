from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as torch_f
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import functional as F

from spectral_detection_posttrain.datasets.penn_fudan import detection_collate


NWPU_CLASS_TO_LABEL = {
    "airplane": 1,
    "ship": 2,
    "storage_tank": 3,
    "baseball_diamond": 4,
    "tennis_court": 5,
    "basketball_court": 6,
    "ground_track_field": 7,
    "harbor": 8,
    "bridge": 9,
    "vehicle": 10,
}


def _load_coco(coco_json: str | Path) -> dict:
    return json.loads(Path(coco_json).read_text(encoding="utf-8"))


def _default_annotation_path(root: Path) -> Path:
    if root.name == "NWPU VHR-10 dataset":
        return root.parent / "NWPU_VHR10_coco.json"
    return root / "NWPU_VHR10_coco.json"


def _resize_image_and_target(
    image: torch.Tensor, target: dict, max_size: int | None
) -> tuple[torch.Tensor, dict]:
    if max_size is None:
        return image, target
    _, height, width = image.shape
    largest_side = max(height, width)
    if largest_side <= int(max_size):
        return image, target

    scale = int(max_size) / float(largest_side)
    new_height = max(1, int(height * scale))
    new_width = max(1, int(width * scale))
    resized = torch_f.interpolate(
        image.unsqueeze(0),
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    scaled = dict(target)
    boxes = target["boxes"].clone()
    if boxes.numel() > 0:
        boxes *= scale
    scaled["boxes"] = boxes
    scaled["area"] = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp_min(0)
    return resized, scaled


class NWPUVHR10DetectionDataset(Dataset):
    """NWPU VHR-10 detection dataset backed by the local COCO conversion."""

    def __init__(
        self,
        root: str | Path,
        coco_json: str | Path,
        image_ids: Iterable[int],
        max_size: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.max_size = max_size
        self.coco = _load_coco(coco_json)
        selected_ids = {int(image_id) for image_id in image_ids}
        self.img_infos = {
            int(img["id"]): img
            for img in self.coco.get("images", [])
            if int(img["id"]) in selected_ids
        }
        self.img_ids = sorted(self.img_infos.keys())

        anns: dict[int, list[dict]] = {}
        for ann in self.coco.get("annotations", []):
            image_id = int(ann["image_id"])
            if image_id in selected_ids:
                anns.setdefault(image_id, []).append(ann)
        self.anns = anns

    def __len__(self) -> int:
        return len(self.img_ids)

    def _image_path(self, file_name: str) -> Path:
        positive = self.root / "positive image set" / file_name
        if positive.exists():
            return positive
        negative = self.root / "negative image set" / file_name
        if negative.exists():
            return negative
        raise FileNotFoundError(f"NWPU image not found: {file_name}")

    def __getitem__(self, idx: int):
        image_id = self.img_ids[idx]
        info = self.img_infos[image_id]
        image = Image.open(self._image_path(str(info["file_name"]))).convert("RGB")
        image_tensor = F.to_tensor(image)

        boxes = []
        labels = []
        iscrowd = []
        for ann in self.anns.get(image_id, []):
            x, y, width, height = [float(v) for v in ann["bbox"]]
            if width <= 0 or height <= 0:
                continue
            boxes.append([x, y, x + width, y + height])
            labels.append(int(ann["category_id"]))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        if boxes_tensor.numel() == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([image_id]),
            "area": (boxes_tensor[:, 2] - boxes_tensor[:, 0]).clamp_min(0)
            * (boxes_tensor[:, 3] - boxes_tensor[:, 1]).clamp_min(0),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }
        return _resize_image_and_target(image_tensor, target, self.max_size)


def nwpu_positive_image_ids(root: str | Path, coco_json: str | Path) -> list[int]:
    root_path = Path(root)
    coco = _load_coco(coco_json)
    return list(
        set(
            int(img["id"])
            for img in coco.get("images", [])
            if (root_path / "positive image set" / str(img["file_name"])).exists()
        )
    )


def build_nwpu_vhr10_loaders(
    config: dict,
    limit_train: int | None = None,
    limit_val: int | None = None,
    batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    root = Path(data_cfg.get("root", "./data/NWPU VHR-10 dataset"))
    annotation = Path(
        data_cfg.get(
            "annotation",
            data_cfg.get("coco_json", _default_annotation_path(root)),
        )
    )
    max_size = data_cfg.get("max_size")
    ids = nwpu_positive_image_ids(root, annotation)
    rng = np.random.RandomState(int(config.get("seed", 42)))
    rng.shuffle(ids)
    split = int(len(ids) * float(data_cfg.get("train_fraction", 0.7)))
    train_ids = ids[:split]
    val_ids = ids[split:]
    if limit_train is not None:
        train_ids = train_ids[: min(int(limit_train), len(train_ids))]
    if limit_val is not None:
        val_ids = val_ids[: min(int(limit_val), len(val_ids))]

    train_set = NWPUVHR10DetectionDataset(root, annotation, train_ids, max_size=max_size)
    val_set = NWPUVHR10DetectionDataset(root, annotation, val_ids, max_size=max_size)
    bs = int(batch_size or config["train"].get("batch_size", 2))
    num_workers = int(data_cfg.get("num_workers", 0))
    return (
        DataLoader(
            train_set,
            batch_size=bs,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=detection_collate,
        ),
        DataLoader(
            val_set,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=detection_collate,
        ),
    )
