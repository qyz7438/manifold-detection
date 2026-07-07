"""Analyze ROI intra/inter dual energy from cached candidate features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_detection_posttrain.methods.energy_transport import (  # noqa: E402
    inter_class_relation_energy,
    roi_basin_energy,
    roi_compactness_energy,
)
from spectral_detection_posttrain.methods.energy_transport.cone_projection import (  # noqa: E402
    compute_class_prototypes,
)


def _labels_zero_based(class_ids: np.ndarray) -> np.ndarray:
    class_ids = class_ids.astype(np.int64)
    if class_ids.size and class_ids.min() >= 1:
        return class_ids - 1
    return class_ids


def _group_masks(prefix: str, data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    labels = data[f"{prefix}_labels"].astype(bool)
    iou = data[f"{prefix}_best_iou"].astype(np.float64)
    probs = data[f"{prefix}_label_probs"].astype(np.float64)
    return {
        "all": np.ones_like(labels, dtype=bool),
        "ap75_positive": labels,
        "ap75_negative": ~labels,
        "high_iou_ge_050": iou >= 0.50,
        "low_iou_lt_030": iou < 0.30,
        "low_conf_high_iou": (iou >= 0.50) & (probs < 0.30),
        "high_conf_low_iou": (iou < 0.30) & (probs >= 0.50),
    }


def _build_reference(
    data: np.lib.npyio.NpzFile,
    *,
    feature_key: str,
    reference_group: str,
) -> tuple[torch.Tensor, np.ndarray]:
    train_features = data[f"train_{feature_key}"].astype(np.float32)
    train_labels = _labels_zero_based(data["train_class_ids"])
    masks = _group_masks("train", data)
    if reference_group not in masks:
        raise ValueError(f"Unknown reference group {reference_group!r}; choose from {sorted(masks)}")
    mask = masks[reference_group]
    if not mask.any():
        raise ValueError(f"Reference group {reference_group!r} is empty")
    num_classes = int(train_labels.max()) + 1
    features = torch.from_numpy(train_features[mask]).float()
    labels = torch.from_numpy(train_labels[mask]).long()
    prototypes = compute_class_prototypes(features, labels, num_classes=num_classes)
    present = np.flatnonzero(np.bincount(train_labels[mask], minlength=num_classes) > 0)
    return prototypes, present


def _compute_group(
    data: np.lib.npyio.NpzFile,
    *,
    split: str,
    group_name: str,
    mask: np.ndarray,
    feature_key: str,
    reference_prototypes: torch.Tensor,
    reference_present: np.ndarray,
    anchor_weight: float,
    separation_weight: float,
) -> dict:
    features_np = data[f"{split}_{feature_key}"].astype(np.float32)
    labels_np = _labels_zero_based(data[f"{split}_class_ids"])
    iou_np = data[f"{split}_best_iou"].astype(np.float64)
    prob_np = data[f"{split}_label_probs"].astype(np.float64)
    ap75_np = data[f"{split}_labels"].astype(bool)
    mask = mask & np.isin(labels_np, reference_present)
    count = int(mask.sum())
    if count == 0:
        return {"split": split, "group": group_name, "count": 0, "skipped": "empty"}

    present = np.array(sorted(set(labels_np[mask].tolist()) & set(reference_present.tolist())), dtype=np.int64)
    if present.size < 2:
        return {
            "split": split,
            "group": group_name,
            "count": count,
            "present_classes": present.astype(int).tolist(),
            "skipped": "fewer_than_two_classes",
        }

    class_to_local = {int(c): idx for idx, c in enumerate(present.tolist())}
    local_labels = np.array([class_to_local[int(c)] for c in labels_np[mask]], dtype=np.int64)
    features = torch.from_numpy(features_np[mask]).float()
    labels = torch.from_numpy(local_labels).long()
    ref = reference_prototypes[present].float()
    current = compute_class_prototypes(features, labels, num_classes=int(present.size))

    compact = roi_compactness_energy(features, labels, ref)
    basin = roi_basin_energy(
        features,
        labels,
        ref,
        perturb_radius=0.0,
        num_perturbations=0,
    )
    intra = 0.5 * (compact + basin)
    inter, inter_components = inter_class_relation_energy(
        current,
        reference_prototypes=ref,
        anchor_weight=anchor_weight,
        separation_weight=separation_weight,
    )
    dual = 0.5 * (intra + inter)

    result = {
        "split": split,
        "group": group_name,
        "count": count,
        "present_classes": (present + 1).astype(int).tolist(),
        "ap75_positive_count": int(ap75_np[mask].sum()),
        "mean_iou": float(iou_np[mask].mean()),
        "mean_label_prob": float(prob_np[mask].mean()),
        "e_compact": float(compact.detach().item()),
        "e_basin": float(basin.detach().item()),
        "e_intra": float(intra.detach().item()),
        "e_inter": float(inter.detach().item()),
        "dual_energy": float(dual.detach().item()),
    }
    for key, value in inter_components.items():
        result[key] = float(value.detach().item())
    return result


def analyze_cache(
    cache: Path,
    *,
    feature_key: str,
    reference_group: str,
    anchor_weight: float,
    separation_weight: float,
) -> dict:
    data = np.load(cache, allow_pickle=False)
    if f"train_{feature_key}" not in data.files or f"val_{feature_key}" not in data.files:
        raise ValueError(f"Cache does not contain train/val arrays for feature key {feature_key!r}")
    reference, reference_present = _build_reference(
        data,
        feature_key=feature_key,
        reference_group=reference_group,
    )
    rows = []
    for split in ("train", "val"):
        for name, mask in _group_masks(split, data).items():
            rows.append(
                _compute_group(
                    data,
                    split=split,
                    group_name=name,
                    mask=mask,
                    feature_key=feature_key,
                    reference_prototypes=reference,
                    reference_present=reference_present,
                    anchor_weight=anchor_weight,
                    separation_weight=separation_weight,
                )
            )
    return {
        "cache": str(cache),
        "feature_key": feature_key,
        "reference_group": reference_group,
        "reference_present_classes": (reference_present + 1).astype(int).tolist(),
        "anchor_weight": anchor_weight,
        "separation_weight": separation_weight,
        "rows": rows,
    }


def print_rows(rows: list[dict]) -> None:
    columns = ["split", "group", "count", "e_intra", "e_inter", "dual_energy", "mean_iou", "mean_label_prob"]
    print(" | ".join(c.rjust(18) for c in columns))
    print("-" * (21 * len(columns)))
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:18.4f}")
            else:
                values.append(str(value).rjust(18))
        print(" | ".join(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--feature-key", default="features_l2")
    parser.add_argument("--reference-group", default="all")
    parser.add_argument("--anchor-weight", type=float, default=0.25)
    parser.add_argument("--separation-weight", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("output/roi_dual_energy_cache_analysis.json"))
    args = parser.parse_args()

    result = analyze_cache(
        args.cache,
        feature_key=args.feature_key,
        reference_group=args.reference_group,
        anchor_weight=args.anchor_weight,
        separation_weight=args.separation_weight,
    )
    print_rows(result["rows"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
