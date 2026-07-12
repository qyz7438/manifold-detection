"""Build the one-shot fresh train-only split for the local Delta-Q residual protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NESTED_MANIFEST = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "splits"
    / "nwpu_dense_endpoint_s42_nested.json"
)
NESTED_MANIFEST_SHA256 = "ce19316aeaef1cbdf85f2c9668c5ae2c8443da8e848de2d22ed0f0f3a687e080"
ANNOTATION_SHA256 = "dde27d9362d9c6aa358a1cab160c9e075e57405d3fe876de049973c6e6150f0e"
INNER_FIT_SHA256 = "3314b7d59dccf8df83a11662883b822c874757f7dd6b62356cb91335b434e54f"
EXCLUDED_OLD_LOCAL_SHA256 = "cde545e8fb7b29adad9c3901b8ea1590b50ec8dcb1210e41101685f5fdeceefe"
CANDIDATE_POOL_SHA256 = "468ce6127ea87ca9662039824a8d6e4ded6b5fb4fadcf9c3bd9a848da45abee7"
SPLIT_SEED = 52042
FIT_COUNT = 96
TUNE_COUNT = 32


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def partition_candidate_ids(
    image_ids: Sequence[int],
    *,
    image_classes: dict[int, set[int]],
    seed: int = SPLIT_SEED,
) -> dict[str, list[int]]:
    ordered = sorted({int(value) for value in image_ids})
    if len(ordered) != 254:
        raise ValueError("residual candidate pool must contain exactly 254 unique images")
    capacities = {"fit": FIT_COUNT, "tune": TUNE_COUNT, "reserve": len(ordered) - FIT_COUNT - TUNE_COUNT}
    class_totals: Counter[int] = Counter()
    for image_id in ordered:
        class_totals.update(image_classes.get(image_id, set()))
    ordered.sort(
        key=lambda image_id: (
            min((class_totals[label] for label in image_classes.get(image_id, set())), default=10**9),
            -len(image_classes.get(image_id, set())),
            hashlib.sha256(f"dense-local-residual:{seed}:{image_id}".encode("ascii")).digest(),
        )
    )
    assigned = {name: [] for name in capacities}
    class_counts = {name: Counter() for name in capacities}
    total = len(ordered)
    for image_id in ordered:
        labels = image_classes.get(image_id, set())
        options = [name for name in capacities if len(assigned[name]) < capacities[name]]

        def cost(name: str) -> tuple[float, str]:
            size_ratio = (len(assigned[name]) + 1) / capacities[name]
            class_ratio = sum(
                (class_counts[name][label] + 1)
                / max(class_totals[label] * capacities[name] / total, 1e-8)
                for label in labels
            )
            return class_ratio + 0.5 * size_ratio, name

        selected = min(options, key=cost)
        assigned[selected].append(image_id)
        class_counts[selected].update(labels)
    return {name: sorted(values) for name, values in assigned.items()}


def _density_bin(count: int) -> int:
    return 0 if count <= 2 else 1 if count <= 5 else 2 if count <= 10 else 3


def _density_support(image_ids: Sequence[int], object_counts: Counter[int]) -> dict[str, int]:
    counts = Counter(_density_bin(object_counts[int(image_id)]) for image_id in image_ids)
    return {str(index): int(counts[index]) for index in range(4)}


def build_manifest(annotation_path: Path) -> dict[str, Any]:
    from scripts.build_dense_endpoint_nested_manifest import manifest_hash, summarize_split
    from scripts.train_dense_endpoint_geometry_control import load_config as load_geometry_config
    from scripts.train_dense_endpoint_geometry_control import resplit_image_ids

    if sha256_file(annotation_path) != ANNOTATION_SHA256:
        raise ValueError("NWPU annotation SHA256 mismatch")
    if sha256_file(NESTED_MANIFEST) != NESTED_MANIFEST_SHA256:
        raise ValueError("dense endpoint nested manifest SHA256 mismatch")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    nested = json.loads(NESTED_MANIFEST.read_text(encoding="utf-8"))
    inner_fit = [int(value) for value in nested["splits"]["inner_fit"]["image_ids"]]
    if manifest_hash(inner_fit) != INNER_FIT_SHA256:
        raise ValueError("inner-fit source hash mismatch")
    candidate_pool, excluded_old_local = resplit_image_ids(inner_fit, load_geometry_config())
    if manifest_hash(candidate_pool) != CANDIDATE_POOL_SHA256:
        raise ValueError("fresh residual candidate-pool hash mismatch")
    if manifest_hash(excluded_old_local) != EXCLUDED_OLD_LOCAL_SHA256:
        raise ValueError("old local learner exclusion hash mismatch")
    pool_set = set(candidate_pool)
    image_classes: dict[int, set[int]] = defaultdict(set)
    object_counts: Counter[int] = Counter()
    annotations = annotation.get("annotations", [])
    for row in annotations:
        image_id = int(row["image_id"])
        if image_id in pool_set and int(row.get("iscrowd", 0)) == 0:
            image_classes[image_id].add(int(row["category_id"]))
            object_counts[image_id] += 1
    for image_id in candidate_pool:
        image_classes[image_id].add(100 + _density_bin(object_counts[image_id]))
    partitions = partition_candidate_ids(candidate_pool, image_classes=image_classes)
    id_sets = [set(partitions[name]) for name in ("fit", "tune", "reserve")]
    if any(id_sets[left] & id_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise RuntimeError("residual split overlap")
    if set().union(*id_sets) != pool_set:
        raise RuntimeError("residual split coverage failure")
    split_payload = {}
    for name, image_ids in partitions.items():
        summary = summarize_split(image_ids, annotations)
        summary["image_ids"] = image_ids
        summary["density_image_support"] = _density_support(image_ids, object_counts)
        split_payload[name] = summary
    for name in ("fit", "tune"):
        support = split_payload[name]["class_image_support"]
        if set(support) != {str(value) for value in range(1, 11)}:
            raise RuntimeError(f"{name} lacks complete NWPU class support")
        if name == "tune" and min(support.values()) < 3:
            raise RuntimeError("tune has fewer than three images for an NWPU class")
        if min(split_payload[name]["density_image_support"].values()) < 1:
            raise RuntimeError(f"{name} lacks an object-density bin")
    return {
        "version_id": "det.energy.dense_local_delta_residual.split.001",
        "dataset": "nwpu_vhr10",
        "status": "locked_before_local_delta_cache",
        "algorithm": "rare_first_greedy_multilabel_class_density_capacity_v1",
        "seed": SPLIT_SEED,
        "density_bins": {"0": "objects<=2", "1": "3<=objects<=5", "2": "6<=objects<=10", "3": "objects>10"},
        "tie_break": "sha256_dense-local-residual_seed_image-id_then_partition-name",
        "source": {
            "inner_fit_count": len(inner_fit),
            "inner_fit_sha256": manifest_hash(inner_fit),
            "excluded_old_local_count": len(excluded_old_local),
            "excluded_old_local_sha256": manifest_hash(excluded_old_local),
            "candidate_pool_count": len(candidate_pool),
            "candidate_pool_sha256": manifest_hash(candidate_pool),
            "annotation_sha256": ANNOTATION_SHA256,
            "old_inner_tune_outer_and_detector_validation_forbidden": True,
        },
        "splits": split_payload,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = build_manifest(args.annotation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in row.items() if key != "image_ids"} for name, row in manifest["splits"].items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
