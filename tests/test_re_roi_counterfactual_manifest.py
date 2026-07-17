from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_re_roi_counterfactual_manifest import manifest_hash, partition_train_ids, summarize_split


ROOT = Path(__file__).resolve().parents[1]
LOCKED_MANIFEST = ROOT / "spectral_detection_posttrain" / "configs" / "splits" / "nwpu_re_roi_counterfactual_s42_nested.json"
DENSE_ENDPOINT_MANIFEST = ROOT / "spectral_detection_posttrain" / "configs" / "splits" / "nwpu_dense_endpoint_s42_nested.json"
EXPECTED_SOURCE_HASH = "7abe3c8370985f49698dcc3c42ca5917f1e17941fa024643147a58479c7cd9bd"


def test_manifest_script_adds_repository_root_for_direct_cli() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "build_re_roi_counterfactual_manifest.py").read_text(encoding="utf-8")
    assert "sys.path.insert(0, str(ROOT))" in source


def test_partition_is_deterministic_disjoint_and_complete() -> None:
    image_ids = list(range(1, 101))
    image_classes = {image_id: {1, 100 + (image_id % 4)} for image_id in image_ids}
    first = partition_train_ids(image_ids, seed=7, image_classes=image_classes)
    second = partition_train_ids(list(reversed(image_ids)), seed=7, image_classes=image_classes)
    assert first == second
    assert [len(first[name]) for name in ("fit", "tune", "calibration", "outer_heldout")] == [55, 15, 15, 15]
    sets = [set(values) for values in first.values()]
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            assert not sets[left] & sets[right]
    assert set().union(*sets) == set(image_ids)


def test_partition_guarantees_min_class_support_in_non_fit_splits() -> None:
    image_ids = list(range(1, 121))
    image_classes = {image_id: {1} for image_id in image_ids}
    for image_id in range(1, 21):
        image_classes[image_id].add(2)
    split = partition_train_ids(image_ids, seed=11, image_classes=image_classes, min_class_support=3)
    for name in ("tune", "calibration", "outer_heldout"):
        rare_count = sum(2 in image_classes[image_id] for image_id in split[name])
        assert rare_count >= 3


def test_summarize_split_reports_image_and_instance_class_support() -> None:
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


def test_locked_re_roi_manifest_is_train_only_disjoint_and_supported() -> None:
    assert LOCKED_MANIFEST.exists()
    payload = json.loads(LOCKED_MANIFEST.read_text(encoding="utf-8"))
    assert payload["version_id"] == "det.energy.re_roi_counterfactual.split.001"
    assert payload["dataset"] == "nwpu_vhr10"
    assert payload["researcher_adaptive"] is True
    assert payload["source"] == {
        "scope": "full_train_only_never_detector_validation",
        "count": 454,
        "image_ids_sha256": EXPECTED_SOURCE_HASH,
    }
    splits = payload["splits"]
    assert [splits[name]["count"] for name in ("fit", "tune", "calibration", "outer_heldout")] == [250, 68, 68, 68]
    id_sets = [set(splits[name]["image_ids"]) for name in ("fit", "tune", "calibration", "outer_heldout")]
    for left in range(len(id_sets)):
        for right in range(left + 1, len(id_sets)):
            assert not id_sets[left] & id_sets[right]
    assert len(set().union(*id_sets)) == 454
    for name in ("tune", "calibration", "outer_heldout"):
        assert set(splits[name]["class_image_support"]) == {str(value) for value in range(1, 11)}
        assert min(splits[name]["class_image_support"].values()) >= 3


def test_locked_re_roi_manifest_differs_from_dense_endpoint_split() -> None:
    """The new partition must not be identical to any prior split assignment."""
    if not DENSE_ENDPOINT_MANIFEST.exists():
        return
    re_roi = json.loads(LOCKED_MANIFEST.read_text(encoding="utf-8"))
    dense = json.loads(DENSE_ENDPOINT_MANIFEST.read_text(encoding="utf-8"))
    dense_names = ("inner_fit", "inner_tune", "outer_train_heldout")
    for name in dense_names:
        dense_ids = set(dense["splits"][name]["image_ids"])
        for new_name in ("fit", "tune", "calibration", "outer_heldout"):
            new_ids = set(re_roi["splits"][new_name]["image_ids"])
            # Allow overlap (same image pool), but forbid exact equality.
            assert new_ids != dense_ids


def test_manifest_hashes_match_content() -> None:
    payload = json.loads(LOCKED_MANIFEST.read_text(encoding="utf-8"))
    for name, row in payload["splits"].items():
        assert manifest_hash(row["image_ids"]) == row["image_ids_sha256"]
    source_ids = []
    for name in ("fit", "tune", "calibration", "outer_heldout"):
        source_ids.extend(payload["splits"][name]["image_ids"])
    assert manifest_hash(source_ids) == payload["source"]["image_ids_sha256"]


def test_prior_split_context_records_dense_endpoint_hash() -> None:
    if not DENSE_ENDPOINT_MANIFEST.exists():
        return
    payload = json.loads(LOCKED_MANIFEST.read_text(encoding="utf-8"))
    context = payload.get("prior_split_context", {})
    assert "dense_endpoint_nested" in context
    dense_hash = hashlib.sha256(DENSE_ENDPOINT_MANIFEST.read_bytes()).hexdigest()
    assert context["dense_endpoint_nested"]["sha256"] == dense_hash
    assert context["dense_endpoint_nested"]["expected_sha256"] == dense_hash
    assert context["dense_endpoint_nested"]["matches"] is True
