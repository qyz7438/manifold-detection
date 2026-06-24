from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from spectral_detection_posttrain.datasets import (
    NWPUVHR10DetectionDataset,
    build_detection_loaders,
)


def _write_tiny_nwpu(
    root: Path,
    *,
    image_count: int = 4,
    image_size: tuple[int, int] = (20, 10),
) -> Path:
    dataset_root = root / "NWPU VHR-10 dataset"
    image_dir = dataset_root / "positive image set"
    image_dir.mkdir(parents=True)
    width, height = image_size

    images = []
    annotations = []
    for idx in range(1, image_count + 1):
        file_name = f"{idx:03d}.jpg"
        Image.new("RGB", (width, height), color=(idx, 0, 0)).save(image_dir / file_name)
        images.append({"id": idx, "file_name": file_name, "width": width, "height": height})
        annotations.append(
            {
                "id": idx,
                "image_id": idx,
                "category_id": 3,
                "bbox": [2.0, 1.0, 10.0, 5.0],
                "area": 50.0,
                "iscrowd": 0,
            }
        )

    annotation_path = root / "NWPU_VHR10_coco.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 3, "name": "storage_tank"}],
            }
        ),
        encoding="utf-8",
    )
    return annotation_path


def test_nwpu_dataset_reads_coco_boxes_and_resizes(tmp_path: Path) -> None:
    annotation_path = _write_tiny_nwpu(tmp_path, image_count=1)
    dataset = NWPUVHR10DetectionDataset(
        tmp_path / "NWPU VHR-10 dataset",
        annotation_path,
        image_ids={1},
        max_size=10,
    )

    image, target = dataset[0]

    assert tuple(image.shape) == (3, 5, 10)
    assert target["labels"].tolist() == [3]
    assert target["boxes"].tolist() == [[1.0, 0.5, 6.0, 3.0]]
    assert target["area"].tolist() == [12.5]
    assert target["iscrowd"].tolist() == [0]


def test_nwpu_resize_matches_legacy_floor_and_uniform_box_scale(tmp_path: Path) -> None:
    annotation_path = _write_tiny_nwpu(tmp_path, image_count=1, image_size=(21, 10))
    dataset = NWPUVHR10DetectionDataset(
        tmp_path / "NWPU VHR-10 dataset",
        annotation_path,
        image_ids={1},
        max_size=10,
    )

    image, target = dataset[0]
    scale = 10.0 / 21.0

    assert tuple(image.shape) == (3, 4, 10)
    expected_box = [2.0 * scale, 1.0 * scale, 12.0 * scale, 6.0 * scale]
    assert target["boxes"][0].tolist() == pytest.approx(expected_box)


def test_build_detection_loaders_dispatches_to_nwpu(tmp_path: Path) -> None:
    annotation_path = _write_tiny_nwpu(tmp_path, image_count=4)
    config = {
        "seed": 42,
        "data": {
            "dataset": "nwpu_vhr10",
            "root": str(tmp_path / "NWPU VHR-10 dataset"),
            "annotation": str(annotation_path),
            "train_fraction": 0.5,
            "max_size": 10,
            "num_workers": 0,
        },
        "train": {"batch_size": 2},
    }

    train_loader, val_loader = build_detection_loaders(config)

    train_images, train_targets = next(iter(train_loader))
    val_images, val_targets = next(iter(val_loader))

    assert len(train_images) == 2
    assert len(train_targets) == 2
    assert len(val_images) == 2
    assert len(val_targets) == 2


def test_nwpu_loader_uses_legacy_round2129_split_membership(tmp_path: Path) -> None:
    annotation_path = _write_tiny_nwpu(tmp_path, image_count=10)
    config = {
        "seed": 42,
        "data": {
            "dataset": "nwpu_vhr10",
            "root": str(tmp_path / "NWPU VHR-10 dataset"),
            "annotation": str(annotation_path),
            "train_fraction": 0.7,
            "max_size": 10,
            "num_workers": 0,
        },
        "train": {"batch_size": 1},
    }

    train_loader, val_loader = build_detection_loaders(config)

    assert set(train_loader.dataset.img_ids) == {9, 2, 6, 1, 8, 3, 10}
    assert set(val_loader.dataset.img_ids) == {5, 4, 7}
