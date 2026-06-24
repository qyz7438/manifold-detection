from .penn_fudan import PennFudanDetectionDataset, build_penn_fudan_loaders
from .nwpu_vhr10 import (
    NWPU_CLASS_TO_LABEL,
    NWPUVHR10DetectionDataset,
    build_nwpu_vhr10_loaders,
)
from .voc_detection import VOC_CLASS_TO_LABEL, VOCDetectionSubset, build_voc_detection_loaders


def build_detection_loaders(
    config: dict,
    limit_train: int | None = None,
    limit_val: int | None = None,
    batch_size: int | None = None,
):
    dataset = str(config.get("data", {}).get("dataset", "penn_fudan")).lower()
    if dataset in {"penn_fudan", "penn-fudan", "pennfudan"}:
        return build_penn_fudan_loaders(config, limit_train, limit_val, batch_size)
    if dataset in {"nwpu", "nwpu_vhr10", "nwpu-vhr10"}:
        return build_nwpu_vhr10_loaders(config, limit_train, limit_val, batch_size)
    if dataset in {"voc", "voc_detection", "pascal_voc"}:
        return build_voc_detection_loaders(config, limit_train, limit_val, batch_size)
    raise ValueError(f"Unknown detection dataset: {dataset}")


__all__ = [
    "PennFudanDetectionDataset",
    "build_penn_fudan_loaders",
    "NWPU_CLASS_TO_LABEL",
    "NWPUVHR10DetectionDataset",
    "build_nwpu_vhr10_loaders",
    "VOC_CLASS_TO_LABEL",
    "VOCDetectionSubset",
    "build_voc_detection_loaders",
    "build_detection_loaders",
]
