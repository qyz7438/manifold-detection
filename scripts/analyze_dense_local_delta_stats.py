"""Audit local reduced-teacher Delta-Q density on train-only native sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader
from torchvision.ops import clip_boxes_to_image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.dense_local_delta_stats.001.json"
)
CONFIG_SHA256 = "4bbb5d22584667434c7d1b6181110d5cee28db54b1f878d1ee4644d4088e9010"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_ids_hash(image_ids: Sequence[int]) -> str:
    encoded = json.dumps(sorted(int(value) for value in image_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def load_config() -> dict[str, Any]:
    if sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise ValueError("canonical dense local Delta-Q config SHA256 mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("version_id") != "det.energy.dense_local_delta_stats.001":
        raise ValueError("dense local Delta-Q version mismatch")
    if config.get("dataset", {}).get("detector_validation_forbidden") is not True:
        raise ValueError("local Delta-Q audit must forbid detector validation")
    return config


def apply_set_perturbation(
    prediction: dict[str, torch.Tensor],
    family: str,
    detection_index: int,
    image_size: tuple[int, int],
    *,
    score_step: float,
    box_step: float,
) -> dict[str, torch.Tensor]:
    count = int(prediction["boxes"].shape[0])
    if family == "identity_permutation":
        order = torch.arange(count - 1, -1, -1, device=prediction["boxes"].device)
        return {key: value[order].clone() for key, value in prediction.items()}
    if detection_index < 0 or detection_index >= count:
        raise IndexError("detection_index out of range")
    output = {key: value.clone() for key, value in prediction.items()}
    if family == "drop":
        keep = torch.ones(count, dtype=torch.bool, device=prediction["boxes"].device)
        keep[detection_index] = False
        return {key: value[keep] for key, value in output.items()}
    if family in {"score_down", "score_up"}:
        direction = -1.0 if family == "score_down" else 1.0
        output["scores"][detection_index] = (
            output["scores"][detection_index] + direction * float(score_step)
        ).clamp(0.0, 1.0)
        return output
    box = output["boxes"][detection_index]
    width = (box[2] - box[0]).clamp_min(1e-6)
    height = (box[3] - box[1]).clamp_min(1e-6)
    if family in {"translate_left", "translate_right"}:
        direction = -1.0 if family == "translate_left" else 1.0
        box[[0, 2]] += direction * float(box_step) * width
    elif family in {"translate_up", "translate_down"}:
        direction = -1.0 if family == "translate_up" else 1.0
        box[[1, 3]] += direction * float(box_step) * height
    elif family in {"scale_down", "scale_up"}:
        direction = -1.0 if family == "scale_down" else 1.0
        factor = 1.0 + direction * float(box_step)
        center_x = 0.5 * (box[0] + box[2])
        center_y = 0.5 * (box[1] + box[3])
        half_width = 0.5 * width * factor
        half_height = 0.5 * height * factor
        box[:] = torch.stack(
            (
                center_x - half_width,
                center_y - half_height,
                center_x + half_width,
                center_y + half_height,
            )
        )
    else:
        raise ValueError(f"unknown local perturbation family {family!r}")
    output["boxes"] = clip_boxes_to_image(output["boxes"], image_size)
    return output


def _summary(values: torch.Tensor) -> dict[str, float | int]:
    x = torch.as_tensor(values, dtype=torch.float64).flatten()
    q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75], dtype=x.dtype))
    return {
        "count": int(x.numel()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "q25": float(q[0].item()),
        "median": float(q[1].item()),
        "q75": float(q[2].item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "positive_fraction": float(x.gt(0).double().mean().item()),
        "negative_fraction": float(x.lt(0).double().mean().item()),
        "nonzero_fraction": float(x.ne(0).double().mean().item()),
        "median_abs": float(x.abs().median().item()),
        "unique_rounded_1e6": int(torch.unique(torch.round(x * 1e6)).numel()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "runs" / "nwpu_dense_local_delta_stats_s42_train64",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    from spectral_detection_posttrain.datasets.nwpu_vhr10 import NWPUVHR10DetectionDataset
    from spectral_detection_posttrain.datasets.penn_fudan import detection_collate
    from spectral_detection_posttrain.methods.energy_transport.dense_endpoint import (
        RobustTeacherStats,
        reduced_teacher_values,
    )
    from spectral_detection_posttrain.methods.energy_transport.dense_set_energy import (
        DenseTeacherConfig,
        dense_teacher_components,
    )
    from spectral_detection_posttrain.models import build_detector
    from spectral_detection_posttrain.utils.io import load_checkpoint
    import scripts.train_dense_endpoint_geometry_control as geometry

    config = load_config()
    shift_path = (ROOT / config["sources"]["shift_audit_result"]).resolve()
    teacher_checkpoint = (ROOT / config["sources"]["teacher_checkpoint"]).resolve()
    if sha256_file(shift_path) != config["sources"]["shift_audit_sha256"]:
        raise ValueError("shift audit SHA256 mismatch")
    if sha256_file(teacher_checkpoint) != config["sources"]["teacher_checkpoint_sha256"]:
        raise ValueError("teacher checkpoint SHA256 mismatch")
    shift = json.loads(shift_path.read_text(encoding="utf-8"))
    if shift.get("scientific_status") != config["sources"]["required_status"]:
        raise ValueError("shift diagnosis mismatch")
    checkpoint_payload = torch.load(teacher_checkpoint, map_location="cpu")
    teacher_stats = RobustTeacherStats(
        median=checkpoint_payload["teacher_median"].float(),
        iqr=checkpoint_payload["teacher_iqr"].float(),
    )
    nested = json.loads(
        (ROOT / "spectral_detection_posttrain/configs/splits/nwpu_dense_endpoint_s42_nested.json").read_text(
            encoding="utf-8"
        )
    )
    source_ids = [int(value) for value in nested["splits"]["inner_fit"]["image_ids"]]
    _, image_ids = geometry.resplit_image_ids(source_ids, geometry.load_config())
    if (
        len(image_ids) != int(config["dataset"]["image_count"])
        or image_ids_hash(image_ids) != config["dataset"]["image_ids_sha256"]
    ):
        raise ValueError("local Delta-Q train image manifest mismatch")

    detector_checkpoint = (args.checkpoint or ROOT / config["detector"]["checkpoint"]).resolve()
    annotation = (args.annotation or ROOT / "data" / "NWPU_VHR10_coco.json").resolve()
    data_root = (args.data_root or ROOT / "data" / "NWPU VHR-10 dataset").resolve()
    if sha256_file(detector_checkpoint) != config["detector"]["checkpoint_sha256"]:
        raise ValueError("detector checkpoint SHA256 mismatch")
    if sha256_file(annotation) != config["dataset"]["annotation_sha256"]:
        raise ValueError("annotation SHA256 mismatch")
    git_dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    if args.require_clean_git and git_dirty:
        raise RuntimeError("--require-clean-git requested but repository is dirty")
    device = torch.device(args.device)
    detector_config = {
        "seed": 42,
        "data_seed": 42,
        "data": {
            "root": str(data_root),
            "annotation": str(annotation),
            "max_size": int(config["detector"]["max_size"]),
            "train_fraction": 0.7,
            "num_workers": 0,
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
    load_checkpoint(detector, detector_checkpoint, device)
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
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=detection_collate)
    teacher_config = DenseTeacherConfig(
        **{key: config["teacher"][key] for key in DenseTeacherConfig.__dataclass_fields__}
    )
    families = list(config["perturbations"]["families"])
    nonidentity_families = [family for family in families if family != "identity_permutation"]
    rows = []
    for image_number, (images, targets) in enumerate(loader, start=1):
        image = images[0].to(device)
        prediction = detector([image])[0]
        target = {
            "boxes": targets[0]["boxes"].to(device),
            "labels": targets[0]["labels"].to(device),
        }
        image_size = tuple(int(value) for value in image.shape[-2:])
        base_components = dense_teacher_components(prediction, target, teacher_config)
        base_quality = teacher_stats.quality(reduced_teacher_values(base_components).cpu()).item()
        identity = apply_set_perturbation(
            prediction,
            "identity_permutation",
            0,
            image_size,
            score_step=float(config["perturbations"]["score_step"]),
            box_step=float(config["perturbations"]["box_step_fraction"]),
        )
        identity_quality = teacher_stats.quality(
            reduced_teacher_values(dense_teacher_components(identity, target, teacher_config)).cpu()
        ).item()
        image_id = int(targets[0]["image_id"].flatten()[0].item())
        rows.append(
            {
                "image_id": image_id,
                "family": "identity_permutation",
                "detection_rank": -1,
                "delta_q": identity_quality - base_quality,
            }
        )
        top = torch.argsort(prediction["scores"], descending=True)[: int(config["perturbations"]["top_detections_per_image"])]
        for rank, detection_index in enumerate(top.tolist()):
            for family in nonidentity_families:
                perturbed = apply_set_perturbation(
                    prediction,
                    family,
                    int(detection_index),
                    image_size,
                    score_step=float(config["perturbations"]["score_step"]),
                    box_step=float(config["perturbations"]["box_step_fraction"]),
                )
                current = dense_teacher_components(perturbed, target, teacher_config)
                quality = teacher_stats.quality(reduced_teacher_values(current).cpu()).item()
                rows.append(
                    {
                        "image_id": image_id,
                        "family": family,
                        "detection_rank": rank,
                        "delta_q": quality - base_quality,
                    }
                )
        if image_number % 16 == 0 or image_number == len(dataset):
            print(json.dumps({"progress_images": image_number, "total_images": len(dataset)}), flush=True)
    identity_values = torch.tensor(
        [row["delta_q"] for row in rows if row["family"] == "identity_permutation"]
    )
    action_values = torch.tensor(
        [row["delta_q"] for row in rows if row["family"] != "identity_permutation"]
    )
    overall = _summary(action_values)
    family_summary = {
        family: _summary(torch.tensor([row["delta_q"] for row in rows if row["family"] == family]))
        for family in families
    }
    gates = {
        "image_support": len(image_ids) >= int(config["gates"]["min_images"]),
        "identity": float(identity_values.abs().max().item())
        <= float(config["gates"]["identity_max_abs_delta"]),
        "nonzero": float(overall["nonzero_fraction"])
        >= float(config["gates"]["nonzero_fraction_min"]),
        "positive": float(overall["positive_fraction"])
        >= float(config["gates"]["positive_fraction_min"]),
        "negative": float(overall["negative_fraction"])
        >= float(config["gates"]["negative_fraction_min"]),
        "unique": int(overall["unique_rounded_1e6"])
        >= int(config["gates"]["unique_rounded_1e6_min"]),
        "magnitude": float(overall["median_abs"])
        >= float(config["gates"]["median_abs_delta_min"]),
    }
    all_passed = all(gates.values())
    result = {
        "completed": True,
        "scientific_status": (
            "shift_invariant_local_delta_learner_warranted"
            if all_passed
            else "local_dense_difference_formulation_frozen"
        ),
        "version_id": config["version_id"],
        "experiment_scope": "train_only_native_set_local_teacher_delta_density",
        "config_sha256": CONFIG_SHA256,
        "source_shift_audit_sha256": config["sources"]["shift_audit_sha256"],
        "teacher_checkpoint_sha256": config["sources"]["teacher_checkpoint_sha256"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": git_dirty,
        "detector_validation_read": False,
        "model_training_run": False,
        "native_nms_rerun_after_perturbation": False,
        "image_manifest": {
            "count": len(image_ids),
            "sha256": config["dataset"]["image_ids_sha256"],
        },
        "identity_max_abs_delta_q": float(identity_values.abs().max().item()),
        "overall_nonidentity": overall,
        "family_summary": family_summary,
        "rows": rows,
        "gates": {"all_passed": all_passed, "gates": gates},
        "decision_rule": config["decision_rule"],
        "claim_boundary": config["claim_boundary"],
        "validated_claim": False,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "eval_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    result = run(_parse_args())
    print(
        json.dumps(
            {
                "status": result["scientific_status"],
                "overall": result["overall_nonidentity"],
                "gates": result["gates"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
