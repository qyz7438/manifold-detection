"""Audit fixed dense teacher-energy components on locked NWPU inner-fit images."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.dense_teacher_stats.001.json"
)
CONFIG_SHA256 = "c8ff0e29a3d191b2aed0db2883830171c8f60ce379df1a44493342c5b9133496"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_ids_hash(image_ids: Sequence[int]) -> str:
    payload = json.dumps(sorted(int(value) for value in image_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_config() -> dict[str, Any]:
    if sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise ValueError("canonical dense teacher stats config SHA256 mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("version_id") != "det.energy.dense_teacher_stats.001":
        raise ValueError("dense teacher stats version mismatch")
    dataset = config.get("dataset", {})
    if dataset.get("split") != "inner_fit":
        raise ValueError("dense teacher stats may read only inner_fit")
    if dataset.get("forbidden_splits") != [
        "inner_tune",
        "outer_train_heldout",
        "detector_validation",
    ]:
        raise ValueError("dense teacher forbidden split contract drift")
    if config.get("teacher", {}).get("component_weights") != "not_fit_in_this_audit":
        raise ValueError("component fitting is forbidden in the statistics audit")
    return config


def _pearson_matrix(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> dict[str, dict[str, float | None]]:
    values = torch.tensor(
        [[float(row[field]) for field in fields] for row in rows], dtype=torch.float64
    )
    centered = values - values.mean(dim=0, keepdim=True)
    scale = centered.square().sum(dim=0).sqrt()
    output: dict[str, dict[str, float | None]] = {}
    for left_index, left in enumerate(fields):
        output[left] = {}
        for right_index, right in enumerate(fields):
            denominator = float((scale[left_index] * scale[right_index]).item())
            output[left][right] = (
                None
                if denominator == 0.0
                else float((centered[:, left_index] * centered[:, right_index]).sum().item() / denominator)
            )
    return output


def summarize_rows(rows: Sequence[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    from spectral_detection_posttrain.methods.energy_transport.dense_set_energy import robust_scalar_summary

    if not rows:
        raise ValueError("dense teacher statistics require at least one image")
    fields = list(config["statistics"]["components"])
    min_iqr = float(config["statistics"]["min_iqr"])
    values = [[float(row[field]) for field in fields] for row in rows]
    all_finite = all(math.isfinite(value) for image_values in values for value in image_values)
    component_summary = {
        field: robust_scalar_summary([float(row[field]) for row in rows], min_iqr=min_iqr)
        for field in fields
    }
    required = list(config["statistics"]["required_non_degenerate"])
    required_non_degenerate = all(not bool(component_summary[field]["degenerate"]) for field in required)
    images_with_predictions = sum(int(row["prediction_count"]) > 0 for row in rows)
    images_with_duplicate_edges = sum(int(row["duplicate_edge_count"]) > 0 for row in rows)
    gates = {
        "exact_image_count": len(rows) == int(config["gates"]["exact_image_count"]),
        "prediction_support": images_with_predictions >= int(config["gates"]["min_images_with_predictions"]),
        "all_values_finite": all_finite,
        "required_components_non_degenerate": required_non_degenerate,
        "duplicate_support": images_with_duplicate_edges
        >= int(config["gates"]["min_images_with_duplicate_edges"]),
    }
    core_passed = all(gates[key] for key in (
        "exact_image_count",
        "prediction_support",
        "all_values_finite",
        "required_components_non_degenerate",
    ))
    if not core_passed:
        decision = "freeze_dense_teacher_formulation"
    elif not gates["duplicate_support"]:
        decision = "keep_unary_move_duplicate_to_prenms"
    else:
        decision = "dense_teacher_components_supported"
    return {
        "image_count": len(rows),
        "images_with_predictions": images_with_predictions,
        "images_with_duplicate_edges": images_with_duplicate_edges,
        "total_predictions": sum(int(row["prediction_count"]) for row in rows),
        "total_ground_truth": sum(int(row["ground_truth_count"]) for row in rows),
        "total_duplicate_edges": sum(int(row["duplicate_edge_count"]) for row in rows),
        "components": component_summary,
        "pearson_correlation": _pearson_matrix(rows, fields),
        "gates": gates,
        "all_core_gates_passed": core_passed,
        "all_gates_passed": all(gates.values()),
        "decision": decision,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "nwpu_dense_teacher_stats_s42_innerfit")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    from spectral_detection_posttrain.datasets.nwpu_vhr10 import NWPUVHR10DetectionDataset
    from spectral_detection_posttrain.datasets.penn_fudan import detection_collate
    from spectral_detection_posttrain.methods.energy_transport.dense_set_energy import (
        DenseTeacherConfig,
        dense_teacher_components,
    )
    from spectral_detection_posttrain.models import build_detector
    from spectral_detection_posttrain.utils.io import load_checkpoint
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

    config = load_config()
    manifest_path = (ROOT / config["dataset"]["split_manifest"]).resolve()
    if sha256_file(manifest_path) != config["dataset"]["split_manifest_sha256"]:
        raise ValueError("dense endpoint split manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = manifest["splits"]["inner_fit"]
    image_ids = [int(value) for value in split["image_ids"]]
    if len(image_ids) != int(config["dataset"]["image_count"]):
        raise ValueError("inner_fit image count mismatch")
    if image_ids_hash(image_ids) != config["dataset"]["image_ids_sha256"]:
        raise ValueError("inner_fit image ID hash mismatch")

    checkpoint = (args.checkpoint or ROOT / config["detector"]["checkpoint"]).resolve()
    annotation = (args.annotation or ROOT / "data" / "NWPU_VHR10_coco.json").resolve()
    data_root = (args.data_root or ROOT / "data" / "NWPU VHR-10 dataset").resolve()
    if sha256_file(checkpoint) != config["detector"]["checkpoint_sha256"]:
        raise ValueError("dense teacher detector checkpoint SHA256 mismatch")
    if sha256_file(annotation) != config["dataset"]["annotation_sha256"]:
        raise ValueError("dense teacher annotation SHA256 mismatch")
    git_dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    if args.require_clean_git and git_dirty:
        raise RuntimeError("--require-clean-git requested but repository is dirty")

    set_seed(42)
    device = resolve_device({"device": args.device or ("cuda" if torch.cuda.is_available() else "cpu")})
    detector_config = {
        "seed": 42,
        "data_seed": 42,
        "data": {
            "root": str(data_root),
            "annotation": str(annotation),
            "max_size": int(config["detector"]["max_size"]),
            "train_fraction": 0.7,
            "num_workers": int(config["resource"]["num_workers"]),
        },
        "model": {
            "name": config["detector"]["model_name"],
            "model_name": config["detector"]["model_name"],
            "pretrained": False,
            "num_classes": int(config["detector"]["num_classes"]),
            "min_size": int(config["detector"]["min_size"]),
            "max_size": int(config["detector"]["max_size"]),
        },
        "train": {"batch_size": 1},
        "eval": {"batch_size": 1},
    }
    detector = build_detector(detector_config).to(device)
    load_checkpoint(detector, checkpoint, device)
    detector.roi_heads.score_thresh = float(config["detector"]["score_threshold"])
    detector.roi_heads.nms_thresh = float(config["detector"]["nms_threshold"])
    detector.roi_heads.detections_per_img = int(config["detector"]["detections_per_image"])
    detector.eval()
    dataset = NWPUVHR10DetectionDataset(
        data_root,
        annotation,
        image_ids,
        max_size=int(config["detector"]["max_size"]),
    )
    if dataset.img_ids != sorted(image_ids):
        raise RuntimeError("dataset image order does not match locked inner_fit IDs")
    loader = DataLoader(
        dataset,
        batch_size=int(config["resource"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["resource"]["num_workers"]),
        collate_fn=detection_collate,
    )
    # Non-constructor protocol fields are deliberately excluded.
    teacher_config = DenseTeacherConfig(
        **{key: config["teacher"][key] for key in DenseTeacherConfig.__dataclass_fields__}
    )
    rows: list[dict[str, Any]] = []
    for index, (images, targets) in enumerate(loader, start=1):
        image = images[0].to(device)
        prediction = detector([image])[0]
        target = {
            "boxes": targets[0]["boxes"].to(device),
            "labels": targets[0]["labels"].to(device),
        }
        components = dense_teacher_components(prediction, target, teacher_config)
        row = {
            "image_id": int(targets[0]["image_id"].flatten()[0].item()),
            **components.detached_dict(),
        }
        rows.append(row)
        if index % 25 == 0 or index == len(dataset):
            print(json.dumps({"progress_images": index, "total_images": len(dataset)}), flush=True)
    observed_ids = [int(row["image_id"]) for row in rows]
    if image_ids_hash(observed_ids) != config["dataset"]["image_ids_sha256"]:
        raise RuntimeError("observed inner_fit image ID hash mismatch")
    summary = summarize_rows(rows, config)
    result = {
        "completed": True,
        "scientific_status": summary["decision"],
        "version_id": config["version_id"],
        "experiment_scope": "nwpu_train_inner_fit_only_native_post_nms_dense_teacher_statistics",
        "config_sha256": CONFIG_SHA256,
        "split_manifest_sha256": config["dataset"]["split_manifest_sha256"],
        "image_ids_sha256": config["dataset"]["image_ids_sha256"],
        "checkpoint_sha256": config["detector"]["checkpoint_sha256"],
        "annotation_sha256": config["dataset"]["annotation_sha256"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": git_dirty,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "teacher": config["teacher"],
        "summary": summary,
        "rows": rows,
        "forbidden_splits_read": [],
        "decision_rule": config["decision_rule"],
        "claim_boundary": config["claim_boundary"],
        "validated_claim": False,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output = args.run_dir / "eval_metrics.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    result = run(_parse_args())
    print(json.dumps({"status": result["scientific_status"], "summary": result["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
