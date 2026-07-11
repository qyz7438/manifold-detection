from __future__ import annotations

from scripts.build_dense_endpoint_nested_manifest import manifest_hash, partition_train_ids, summarize_split


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
