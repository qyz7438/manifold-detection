"""Build the locked NWPU train-only nested split for dense endpoint validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


EXPECTED_FULL_TRAIN_HASH = "7abe3c8370985f49698dcc3c42ca5917f1e17941fa024643147a58479c7cd9bd"
EXPECTED_FULL_TRAIN_COUNT = 454
DATA_SEED = 42
NESTED_SEED = 42042


def manifest_hash(image_ids: Sequence[int]) -> str:
    encoded = json.dumps(sorted(int(value) for value in image_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def partition_train_ids(
    image_ids: Sequence[int],
    seed: int = NESTED_SEED,
    image_classes: dict[int, set[int]] | None = None,
) -> dict[str, list[int]]:
    ordered = sorted(
        (int(value) for value in image_ids),
        key=lambda image_id: hashlib.sha256(f"dense-endpoint:{seed}:{image_id}".encode("ascii")).digest(),
    )
    fit_count = int(round(0.70 * len(ordered)))
    remaining = len(ordered) - fit_count
    tune_count = remaining // 2
    capacities = {
        "inner_fit": fit_count,
        "inner_tune": tune_count,
        "outer_train_heldout": len(ordered) - fit_count - tune_count,
    }
    if image_classes is None:
        return {
            "inner_fit": sorted(ordered[:fit_count]),
            "inner_tune": sorted(ordered[fit_count : fit_count + tune_count]),
            "outer_train_heldout": sorted(ordered[fit_count + tune_count :]),
        }
    class_totals: Counter[int] = Counter()
    for image_id in ordered:
        class_totals.update(image_classes.get(image_id, set()))
    ordered.sort(
        key=lambda image_id: (
            min((class_totals[label] for label in image_classes.get(image_id, set())), default=10**9),
            -len(image_classes.get(image_id, set())),
            hashlib.sha256(f"dense-endpoint:{seed}:{image_id}".encode("ascii")).digest(),
        )
    )
    assigned = {name: [] for name in capacities}
    class_counts = {name: Counter() for name in capacities}
    total = len(ordered)
    for image_id in ordered:
        labels = image_classes.get(image_id, set())
        options = [name for name, capacity in capacities.items() if len(assigned[name]) < capacity]

        def assignment_cost(name: str) -> tuple[float, str]:
            size_ratio = (len(assigned[name]) + 1) / capacities[name]
            class_ratio = 0.0
            for label in labels:
                target = class_totals[label] * capacities[name] / total
                class_ratio += (class_counts[name][label] + 1) / max(target, 1e-8)
            return (class_ratio + 0.5 * size_ratio, name)

        selected = min(options, key=assignment_cost)
        assigned[selected].append(image_id)
        class_counts[selected].update(labels)
    return {name: sorted(values) for name, values in assigned.items()}


def summarize_split(image_ids: Sequence[int], annotations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    selected = {int(value) for value in image_ids}
    instance_support: Counter[int] = Counter()
    image_classes: dict[int, set[int]] = defaultdict(set)
    objects_per_image: Counter[int] = Counter()
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        if image_id not in selected or int(annotation.get("iscrowd", 0)) != 0:
            continue
        category = int(annotation["category_id"])
        instance_support[category] += 1
        image_classes[image_id].add(category)
        objects_per_image[image_id] += 1
    image_support: Counter[int] = Counter()
    for classes in image_classes.values():
        image_support.update(classes)
    counts = [objects_per_image[image_id] for image_id in selected]
    return {
        "count": len(selected),
        "image_ids_sha256": manifest_hash(image_ids),
        "objects": int(sum(counts)),
        "mean_objects_per_image": float(sum(counts) / max(1, len(counts))),
        "class_image_support": {str(key): int(image_support[key]) for key in sorted(image_support)},
        "class_instance_support": {str(key): int(instance_support[key]) for key in sorted(instance_support)},
    }


def build_manifest(root: Path, annotation_path: Path) -> dict[str, Any]:
    from spectral_detection_posttrain.datasets.nwpu_vhr10 import nwpu_positive_image_ids

    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    positive_ids = nwpu_positive_image_ids(root, annotation_path)
    rng = np.random.RandomState(DATA_SEED)
    rng.shuffle(positive_ids)
    train_count = int(len(positive_ids) * 0.70)
    train_ids = [int(value) for value in positive_ids[:train_count]]
    source_hash = manifest_hash(train_ids)
    if len(train_ids) != EXPECTED_FULL_TRAIN_COUNT or source_hash != EXPECTED_FULL_TRAIN_HASH:
        raise RuntimeError(
            f"NWPU full-train source drift: count={len(train_ids)} hash={source_hash}"
        )
    train_set = set(train_ids)
    image_classes: dict[int, set[int]] = defaultdict(set)
    for annotation in payload.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id in train_set and int(annotation.get("iscrowd", 0)) == 0:
            image_classes[image_id].add(int(annotation["category_id"]))
    partitions = partition_train_ids(train_ids, image_classes=image_classes)
    union = set().union(*(set(values) for values in partitions.values()))
    if len(union) != len(train_ids) or any(
        set(left) & set(right)
        for index, left in enumerate(partitions.values())
        for right in list(partitions.values())[index + 1 :]
    ):
        raise RuntimeError("nested split overlap or coverage failure")
    split_payload = {
        name: {
            "image_ids": values,
            **summarize_split(values, payload.get("annotations", [])),
        }
        for name, values in partitions.items()
    }
    for name, row in split_payload.items():
        supported = set(int(key) for key in row["class_image_support"])
        if supported != set(range(1, 11)):
            raise RuntimeError(f"{name} lacks NWPU class coverage: {sorted(supported)}")
        if name != "inner_fit" and min(row["class_image_support"].values()) < 3:
            raise RuntimeError(f"{name} has fewer than three images for a class")
    return {
        "version_id": "det.energy.dense_endpoint.split.001",
        "dataset": "nwpu_vhr10",
        "data_seed": DATA_SEED,
        "nested_seed": NESTED_SEED,
        "source": {
            "scope": "full_train_only_never_detector_validation",
            "count": len(train_ids),
            "image_ids_sha256": source_hash,
        },
        "splits": split_payload,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args.data_root, args.annotation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in row.items() if key != "image_ids"} for name, row in manifest["splits"].items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
