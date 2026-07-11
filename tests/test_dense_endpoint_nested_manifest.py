from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_dense_endpoint_nested_manifest import manifest_hash, partition_train_ids, summarize_split


ROOT = Path(__file__).resolve().parents[1]
LOCKED_MANIFEST = ROOT / "spectral_detection_posttrain" / "configs" / "splits" / "nwpu_dense_endpoint_s42_nested.json"
LOCKED_MANIFEST_SHA256 = "ce19316aeaef1cbdf85f2c9668c5ae2c8443da8e848de2d22ed0f0f3a687e080"


def test_manifest_script_adds_repository_root_for_direct_cli() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "build_dense_endpoint_nested_manifest.py").read_text(encoding="utf-8")
    assert "sys.path.insert(0, str(ROOT))" in source


def test_nested_partition_is_deterministic_disjoint_and_complete() -> None:
    image_ids = list(range(1, 101))
    first = partition_train_ids(image_ids, seed=7)
    second = partition_train_ids(list(reversed(image_ids)), seed=7)
    assert first == second
    assert [len(first[name]) for name in ("inner_fit", "inner_tune", "outer_train_heldout")] == [70, 15, 15]
    sets = [set(values) for values in first.values()]
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]
    assert set().union(*sets) == set(image_ids)
    assert manifest_hash(image_ids) == manifest_hash(list(reversed(image_ids)))


def test_split_summary_reports_image_and_instance_class_support() -> None:
    annotations = [
        {"image_id": 1, "category_id": 1},
        {"image_id": 1, "category_id": 1},
        {"image_id": 2, "category_id": 2},
        {"image_id": 3, "category_id": 2, "iscrowd": 1},
    ]
    summary = summarize_split([1, 2, 3], annotations)
    assert summary["objects"] == 3
    assert summary["class_image_support"] == {"1": 1, "2": 1}
    assert summary["class_instance_support"] == {"1": 2, "2": 1}


def test_multilabel_partition_balances_rare_class_across_tune_and_outer() -> None:
    image_ids = list(range(1, 101))
    image_classes = {image_id: {1} for image_id in image_ids}
    for image_id in range(1, 11):
        image_classes[image_id].add(2)
    split = partition_train_ids(image_ids, seed=11, image_classes=image_classes)
    rare_counts = {
        name: sum(2 in image_classes[image_id] for image_id in values)
        for name, values in split.items()
    }
    assert rare_counts["inner_fit"] >= 6
    assert rare_counts["inner_tune"] >= 1
    assert rare_counts["outer_train_heldout"] >= 1


def test_multilabel_partition_balances_density_tags() -> None:
    image_ids = list(range(1, 121))
    image_classes = {image_id: {1, 100 + (image_id % 4)} for image_id in image_ids}
    split = partition_train_ids(image_ids, seed=13, image_classes=image_classes)
    for values in split.values():
        supported_bins = {next(tag for tag in image_classes[image_id] if tag >= 100) for image_id in values}
        assert supported_bins == {100, 101, 102, 103}


def test_locked_dense_endpoint_manifest_is_train_only_disjoint_and_supported() -> None:
    assert hashlib.sha256(LOCKED_MANIFEST.read_bytes()).hexdigest() == LOCKED_MANIFEST_SHA256
    payload = json.loads(LOCKED_MANIFEST.read_text(encoding="utf-8"))
    assert payload["source"] == {
        "scope": "full_train_only_never_detector_validation",
        "count": 454,
        "image_ids_sha256": "7abe3c8370985f49698dcc3c42ca5917f1e17941fa024643147a58479c7cd9bd",
    }
    splits = payload["splits"]
    assert [splits[name]["count"] for name in ("inner_fit", "inner_tune", "outer_train_heldout")] == [318, 68, 68]
    id_sets = [set(splits[name]["image_ids"]) for name in ("inner_fit", "inner_tune", "outer_train_heldout")]
    assert not id_sets[0] & id_sets[1] and not id_sets[0] & id_sets[2] and not id_sets[1] & id_sets[2]
    assert len(set().union(*id_sets)) == 454
    for name in ("inner_tune", "outer_train_heldout"):
        assert set(splits[name]["class_image_support"]) == {str(value) for value in range(1, 11)}
        assert min(splits[name]["class_image_support"].values()) >= 3
