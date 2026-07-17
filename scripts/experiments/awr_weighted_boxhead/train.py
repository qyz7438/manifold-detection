"""Run one locked oracle-utility-weighted NWPU box-head arm and seed."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from spectral_detection_posttrain.datasets.nwpu_vhr10 import (  # noqa: E402
    NWPUVHR10DetectionDataset,
    detection_collate,
)
from spectral_detection_posttrain.eval.detection_metrics import (  # noqa: E402
    evaluate_detection_predictions,
)
from spectral_detection_posttrain.methods.energy_transport.endpoint.awr_weighting import (  # noqa: E402
    BudgetCounters,
    build_oracle_utility_weights,
    derive_oracle_utilities,
    deterministic_weight_permutation,
)
from spectral_detection_posttrain.models import build_detector  # noqa: E402
from spectral_detection_posttrain.trainers.detection.awr_boxhead import (  # noqa: E402
    assert_frozen_state_unchanged,
    build_epoch_image_schedule,
    configure_box_head_only,
    configure_native_postprocess,
    frozen_state_hashes,
    validate_re_roi_cache,
    weighted_box_loss,
)
from spectral_detection_posttrain.utils.io import (  # noqa: E402
    load_checkpoint,
    save_checkpoint,
    save_json,
)
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed  # noqa: E402


LOCKED_SEEDS = (42, 2024, 999)
LOCKED_EPOCHS = 10
LOCKED_LOGICAL_BATCH_SIZE = 8
LOCKED_LR = 0.001
LOCKED_MOMENTUM = 0.9
LOCKED_WEIGHT_DECAY = 0.0005


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_split_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version_id") != "det.energy.re_roi_counterfactual.split.001":
        raise ValueError("unsupported nested split manifest")
    expected = {"fit": 250, "tune": 68, "calibration": 68, "outer_heldout": 68}
    for name, count in expected.items():
        image_ids = payload.get("splits", {}).get(name, {}).get("image_ids")
        if not isinstance(image_ids, list) or len(image_ids) != count:
            raise ValueError(f"nested split {name} must contain exactly {count} image IDs")
    all_ids = [int(value) for name in expected for value in payload["splits"][name]["image_ids"]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("nested split image IDs must be pairwise disjoint")
    return payload


def _assert_clean_git() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError("locked AWR runs require a clean Git worktree")
    return commit


def _resolve_device(requested: str) -> torch.device:
    device = resolve_device({"device": requested})
    if device.type == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError("AWR GPU runs require CUDA_VISIBLE_DEVICES=2")
    return device


def _model_config(seed: int, device: torch.device) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "device": str(device),
        "data": {"dataset": "nwpu", "max_size": 480},
        "model": {
            "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "num_classes": 11,
            "pretrained": False,
            "min_size": 480,
            "max_size": 480,
        },
        "train": {
            "batch_size": LOCKED_LOGICAL_BATCH_SIZE,
            "lr": LOCKED_LR,
            "momentum": LOCKED_MOMENTUM,
            "weight_decay": LOCKED_WEIGHT_DECAY,
            "epochs": LOCKED_EPOCHS,
        },
        "matching": {"iou_threshold": 0.5, "score_threshold": 0.05},
        "eval": {"batch_size": 8, "high_conf_threshold": 0.7},
    }


def _to_device_target(target: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in target.items()
    }


def _evaluate(
    model: torch.nn.Module,
    dataset: NWPUVHR10DetectionDataset,
    device: torch.device,
) -> tuple[dict[str, Any], dict[int, dict[str, torch.Tensor]], dict[int, dict[str, torch.Tensor]]]:
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate,
    )
    model.eval()
    predictions: dict[int, dict[str, torch.Tensor]] = {}
    targets: dict[int, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for images, batch_targets in loader:
            outputs = model([image.to(device) for image in images])
            for output, target in zip(outputs, batch_targets, strict=True):
                image_id = int(target["image_id"].flatten()[0].item())
                predictions[image_id] = {
                    key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in output.items()
                }
                targets[image_id] = {
                    key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in target.items()
                }
    image_ids = list(dataset.img_ids)
    metrics = evaluate_detection_predictions(
        [predictions[image_id] for image_id in image_ids],
        [targets[image_id] for image_id in image_ids],
        iou_threshold=0.5,
        score_threshold=0.05,
        high_conf_threshold=0.7,
    )
    return metrics, predictions, targets


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed not in LOCKED_SEEDS:
        raise ValueError(f"seed must be one of {LOCKED_SEEDS}")
    if args.arm not in {"Z", "U", "W", "F", "S"}:
        raise ValueError("arm must be one of Z/U/W/F/S")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse run directory: {args.output_dir}")

    commit = _assert_clean_git()
    set_seed(args.seed)
    device = _resolve_device(args.device)
    split = load_split_manifest(args.split_manifest)
    fit_ids = [int(value) for value in split["splits"]["fit"]["image_ids"]]
    tune_ids = [int(value) for value in split["splits"]["tune"]["image_ids"]]
    split_sha = sha256_file(args.split_manifest)
    checkpoint_sha = sha256_file(args.checkpoint)
    annotation_sha = sha256_file(args.annotation)

    cache_payload = torch.load(args.cache, map_location="cpu")
    records = validate_re_roi_cache(
        cache_payload,
        fit_image_ids=fit_ids,
        split_manifest_sha256=split_sha,
        checkpoint_sha256=checkpoint_sha,
        annotation_sha256=annotation_sha,
    )
    utilities = derive_oracle_utilities(fit_ids, records)
    weight_bundle = build_oracle_utility_weights(utilities.image_utilities)
    support_fraction = utilities.positive_image_count / len(fit_ids)
    diagnostics = weight_bundle.diagnostics
    if support_fraction < 0.15 or utilities.positive_candidate_count < 500:
        raise RuntimeError("support gate failed before training")
    if diagnostics.saturation_fraction > 0.10 or diagnostics.effective_sample_size < 0.50 * len(fit_ids):
        raise RuntimeError("weight-health gate failed before training")
    if not math.isclose(diagnostics.normalized_mean, 1.0, abs_tol=1e-7):
        raise RuntimeError("normalized fit weight mean is not one")

    if args.arm == "W":
        arm_weights = weight_bundle.weights
    elif args.arm == "S":
        arm_weights = deterministic_weight_permutation(weight_bundle.weights, seed=args.seed)
    else:
        arm_weights = {image_id: 1.0 for image_id in fit_ids}
    positive_ids = [image_id for image_id, value in utilities.image_utilities.items() if value > 0.0]

    config = _model_config(args.seed, device)
    run_manifest = {
        "version_id": "det.energy.oracle_utility_boxhead.train_only.001",
        "arm": args.arm,
        "seed": args.seed,
        "git_commit": commit,
        "split_manifest_sha256": split_sha,
        "checkpoint_sha256": checkpoint_sha,
        "annotation_sha256": annotation_sha,
        "cache_sha256": sha256_file(args.cache),
        "config": config,
        "config_sha256": sha256_json(config),
        "utility_support": {
            "positive_image_count": utilities.positive_image_count,
            "positive_candidate_count": utilities.positive_candidate_count,
            "support_fraction": support_fraction,
        },
        "weight_diagnostics": asdict(diagnostics),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_manifest, args.output_dir / "manifest.json")

    fit_dataset = NWPUVHR10DetectionDataset(
        args.data_root, args.annotation, fit_ids, max_size=480
    )
    tune_dataset = NWPUVHR10DetectionDataset(
        args.data_root, args.annotation, tune_ids, max_size=480
    )
    fit_index = {image_id: index for index, image_id in enumerate(fit_dataset.img_ids)}
    if set(fit_index) != set(fit_ids):
        raise RuntimeError("fit dataset IDs do not match locked manifest")

    model = build_detector(config).to(device)
    load_checkpoint(model, args.checkpoint, device)
    configure_native_postprocess(model)
    frozen_before = frozen_state_hashes(model)
    history: list[dict[str, Any]] = []
    counters = BudgetCounters(0, 0, 0, 0)

    if args.arm != "Z":
        trainable = configure_box_head_only(model)
        optimizer = torch.optim.SGD(
            trainable,
            lr=LOCKED_LR,
            momentum=LOCKED_MOMENTUM,
            weight_decay=LOCKED_WEIGHT_DECAY,
        )
        for epoch in range(1, LOCKED_EPOCHS + 1):
            model.train()
            schedule = build_epoch_image_schedule(
                fit_ids,
                arm=args.arm,
                seed=args.seed,
                epoch=epoch,
                positive_image_ids=positive_ids if args.arm == "F" else None,
            )
            total_weighted_loss = 0.0
            logical_batches = 0
            for start in range(0, len(schedule), LOCKED_LOGICAL_BATCH_SIZE):
                batch_ids = schedule[start : start + LOCKED_LOGICAL_BATCH_SIZE]
                batch = [fit_dataset[fit_index[image_id]] for image_id in batch_ids]
                images = [image.to(device) for image, _ in batch]
                targets = [_to_device_target(target, device) for _, target in batch]
                weights = [arm_weights[image_id] for image_id in batch_ids]
                optimizer.zero_grad(set_to_none=True)
                loss = weighted_box_loss(model, images, targets, weights=weights)
                loss.backward()
                optimizer.step()
                total_weighted_loss += float(loss.detach().item())
                logical_batches += 1
                counters = BudgetCounters(
                    image_exposures=counters.image_exposures + len(batch_ids),
                    logical_batches=counters.logical_batches + 1,
                    optimizer_steps=counters.optimizer_steps + 1,
                    scheduler_steps=0,
                )
            history.append(
                {
                    "epoch": epoch,
                    "weighted_box_loss": total_weighted_loss / logical_batches,
                    "image_exposures": len(schedule),
                    "logical_batches": logical_batches,
                }
            )

        expected = BudgetCounters(
            image_exposures=LOCKED_EPOCHS * len(fit_ids),
            logical_batches=LOCKED_EPOCHS * math.ceil(len(fit_ids) / LOCKED_LOGICAL_BATCH_SIZE),
            optimizer_steps=LOCKED_EPOCHS * math.ceil(len(fit_ids) / LOCKED_LOGICAL_BATCH_SIZE),
            scheduler_steps=0,
        )
        if counters != expected:
            raise AssertionError(f"runtime budget mismatch: expected={expected}, actual={counters}")
        assert_frozen_state_unchanged(frozen_before, frozen_state_hashes(model))
        save_checkpoint(model, args.output_dir / "checkpoint_final.pth", {"epoch": LOCKED_EPOCHS})

    fit_metrics, _, _ = _evaluate(model, fit_dataset, device)
    metrics, predictions, targets = _evaluate(model, tune_dataset, device)
    prediction_path = args.output_dir / "tune_predictions.pt"
    torch.save({"predictions": predictions, "targets": targets}, prediction_path)
    result = {
        **metrics,
        "arm": args.arm,
        "seed": args.seed,
        "evaluation_scope": "nwpu_train_only_tune_68",
        "checkpoint_selection": "fixed_final_epoch" if args.arm != "Z" else "source_checkpoint",
        "fit_metrics": fit_metrics,
        "history": history,
        "budget_counters": asdict(counters),
        "prediction_artifact": str(prediction_path),
        "prediction_artifact_sha256": sha256_file(prediction_path),
    }
    checkpoint_final = args.output_dir / "checkpoint_final.pth"
    if checkpoint_final.exists():
        result["checkpoint_final_sha256"] = sha256_file(checkpoint_final)
    save_json(result, args.output_dir / "eval_metrics.json")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("Z", "U", "W", "F", "S"))
    parser.add_argument("--seed", required=True, type=int, choices=LOCKED_SEEDS)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps({key: result[key] for key in ("arm", "seed", "ap50", "ap75")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
