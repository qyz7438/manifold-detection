from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn.functional as torch_f
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import VOCDetection
from torchvision.transforms import functional as F

from spectral_detection_posttrain.datasets.penn_fudan import detection_collate


VOC_CLASS_TO_LABEL = {
    "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5,
    "bus": 6, "car": 7, "cat": 8, "chair": 9, "cow": 10,
    "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14,
    "person": 15, "pottedplant": 16, "sheep": 17, "sofa": 18,
    "train": 19, "tvmonitor": 20,
}


def parse_voc_annotation(xml_path: str | Path, classes: list[str]) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    labels = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in classes:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        if xmax > xmin and ymax > ymin:
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(classes.index(name) + 1)  # 0=bg, 1..N=classes
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)
    if len(boxes_tensor) == 0:
        boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
    area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]).clamp_min(0) * (boxes_tensor[:, 3] - boxes_tensor[:, 1]).clamp_min(0)
    return {
        "boxes": boxes_tensor,
        "labels": labels_tensor,
        "area": area,
        "iscrowd": torch.zeros((len(labels_tensor),), dtype=torch.int64),
    }


class VOCDetectionSubset(Dataset):
    def __init__(
        self,
        root: str | Path,
        year: str = "2007",
        image_set: str = "train",
        classes: list[str] | None = None,
        download: bool = True,
        max_size: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.year = year
        self.image_set = image_set
        self.classes = classes or ["person", "car", "dog"]
        self.max_size = max_size
        if download:
            VOCDetection(str(self.root), year=year, image_set=image_set, download=True)
        self.voc_root = self.root / "VOCdevkit" / f"VOC{year}"
        ids_path = self.voc_root / "ImageSets" / "Main" / f"{image_set}.txt"
        if not ids_path.exists():
            raise FileNotFoundError(f"VOC image set not found: {ids_path}")
        ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.samples = []
        for image_id in ids:
            annotation = self.voc_root / "Annotations" / f"{image_id}.xml"
            target = parse_voc_annotation(annotation, self.classes)
            if len(target["boxes"]) > 0:
                self.samples.append(image_id)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_id = self.samples[idx]
        image = Image.open(self.voc_root / "JPEGImages" / f"{image_id}.jpg").convert("RGB")
        image_tensor = F.to_tensor(image)
        target = parse_voc_annotation(self.voc_root / "Annotations" / f"{image_id}.xml", self.classes)
        target["image_id"] = torch.tensor([idx])
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
    resized = torch_f.interpolate(image.unsqueeze(0), size=(new_height, new_width), mode="bilinear", align_corners=False).squeeze(0)
    scaled = dict(target)
    boxes = target["boxes"].clone()
    boxes[:, [0, 2]] *= new_width / float(width)
    boxes[:, [1, 3]] *= new_height / float(height)
    scaled["boxes"] = boxes
    scaled["area"] = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    return resized, scaled


def build_voc_detection_loaders(config: dict, limit_train: int | None = None, limit_val: int | None = None, batch_size: int | None = None):
    data_cfg = config["data"]
    classes = list(data_cfg.get("classes", ["person", "car", "dog"]))
    year = str(data_cfg.get("year", "2007"))
    data_root = str(data_cfg.get("root", "./data"))
    train_set = VOCDetectionSubset(data_root, year=year, image_set=str(data_cfg.get("train_set", "train")), classes=classes, download=bool(data_cfg.get("download", True)), max_size=data_cfg.get("max_size"))
    val_set = VOCDetectionSubset(data_root, year=year, image_set=str(data_cfg.get("val_set", "val")), classes=classes, download=bool(data_cfg.get("download", True)), max_size=data_cfg.get("max_size"))
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
