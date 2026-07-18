"""CLI entry point to build the re-ROI counterfactual evidence cache.

This script is phase 1 of the protocol: it generates (S, a) records and teacher
utilities, but does not train any model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from spectral_detection_posttrain.datasets.nwpu_vhr10 import (
    NWPUVHR10DetectionDataset,
    detection_collate,
    nwpu_positive_image_ids,
)
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import load_checkpoint
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

from scripts.experiments.re_roi_counterfactual.action_family import action_family_hash, get_action_family
from scripts.experiments.re_roi_counterfactual.cache_builder import CacheConfig, build_image_cache_record


CONFIG_SCHEMA = {
    "model": {
        "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "num_classes": 11,
        "pretrained": False,
        "min_size": 480,
        "max_size": 480,
    },
    "cache": {
        "score_threshold": 0.05,
        "nms_threshold": 0.50,
        "detections_per_img": 100,
        "top_k_candidates": 3,
    },
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_lf(path: str | Path) -> str:
    normalized = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def clean_git_commit() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError("re-ROI cache generation requires a clean Git worktree")
    return commit


def _resolve_requested_device(requested: str) -> torch.device:
    return resolve_device({"device": requested})


def _validate_split_name(split_name: str) -> str:
    allowed = {"fit", "tune", "calibration"}
    if split_name not in allowed:
        raise ValueError(
            f"split {split_name!r} is not cacheable before outer_heldout confirmation"
        )
    return split_name


def load_split_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version_id") != "det.energy.re_roi_counterfactual.split.001":
        raise ValueError("unsupported re-ROI split manifest version")
    return payload


def build_loader(
    data_root: Path,
    annotation: Path,
    image_ids: list[int],
    config: dict[str, Any],
    batch_size: int = 1,
) -> DataLoader:
    dataset = NWPUVHR10DetectionDataset(
        root=data_root,
        coco_json=annotation,
        image_ids=image_ids,
        max_size=int(config["model"]["max_size"]),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate,
    )


def build_cache(
    data_root: Path,
    annotation: Path,
    split_manifest: Path,
    checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    split_name: str = "fit",
) -> dict[str, Any]:
    split_name = _validate_split_name(split_name)
    if device.type == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError("re-ROI GPU cache generation requires CUDA_VISIBLE_DEVICES=2")
    git_commit = clean_git_commit()
    manifest = load_split_manifest(split_manifest)
    image_ids = [int(value) for value in manifest["splits"][split_name]["image_ids"]]

    config = CONFIG_SCHEMA.copy()
    model = build_detector(config)
    load_checkpoint(model, checkpoint, device)
    model.to(device)
    model.eval()

    loader = build_loader(data_root, annotation, image_ids, config)
    actions = get_action_family()
    cache_config = CacheConfig(**config["cache"])

    records: list[dict[str, Any]] = []
    metadata = {
        "format": "re_roi_counterfactual_v1",
        "split_name": split_name,
        "split_manifest": str(split_manifest.resolve()),
        "split_manifest_sha256": sha256_text_lf(split_manifest),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "annotation_sha256": sha256_file(annotation),
        "action_family_hash": action_family_hash(),
        "utility_definition": "raw_locked_scalarization_v1",
        "coordinate_space": "detector_transform",
        "model_config": config["model"],
        "cache_config": config["cache"],
        "device": str(device),
        "git_commit": git_commit,
    }

    for batch_index, (images, targets) in enumerate(loader, start=1):
        image = images[0].to(device)
        target = {
            "boxes": targets[0]["boxes"].to(device),
            "labels": targets[0]["labels"].to(device),
        }
        image_id = int(targets[0]["image_id"].flatten()[0].item())
        record = build_image_cache_record(
            model=model,
            image=image,
            target=target,
            image_id=image_id,
            actions=actions,
            config=cache_config,
        )
        records.append(record)
        if batch_index % 10 == 0:
            print(f"processed {batch_index}/{len(loader)} images", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"re_roi_cache_{split_name}.pt"
    torch.save({"metadata": metadata, "records": records}, output_path)
    output_sha256 = sha256_file(output_path)
    artifact_manifest_path = output_dir / f"re_roi_cache_{split_name}.manifest.json"
    artifact_manifest = {
        "artifact": str(output_path.resolve()),
        "artifact_sha256": output_sha256,
        "metadata": metadata,
        "record_count": len(records),
    }
    artifact_manifest_path.write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "output_path": str(output_path),
        "output_sha256": output_sha256,
        "artifact_manifest_path": str(artifact_manifest_path),
        "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
        "num_records": len(records),
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", default="fit")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = _resolve_requested_device(args.device)
    summary = build_cache(
        data_root=args.data_root,
        annotation=args.annotation,
        split_manifest=args.split_manifest,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=device,
        split_name=args.split_name,
    )
    print(json.dumps({"output_path": summary["output_path"], "sha256": summary["output_sha256"], "num_records": summary["num_records"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
