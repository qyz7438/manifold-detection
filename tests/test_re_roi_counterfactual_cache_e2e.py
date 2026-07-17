"""End-to-end CPU smoke test for the re-ROI cache builder.

Uses a randomly-initialized torchvision FasterRCNN so that no checkpoint,
network download, or GPU is required. The test only verifies structural
contracts and the identity no-op property, not detection quality.
"""

from __future__ import annotations

import pytest
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights

from scripts.experiments.re_roi_counterfactual.action_family import get_action_family
from scripts.experiments.re_roi_counterfactual.cache_builder import CacheConfig, build_image_cache_record


@pytest.fixture
def cpu_detector():
    """A small detector with random weights for CPU-only smoke tests."""
    torch.manual_seed(42)
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=91,
    )
    model.eval()
    return model


@pytest.fixture
def dummy_image_and_target():
    """One 320x320 image and two dummy GT boxes in original image coordinates."""
    torch.manual_seed(7)
    image = torch.rand(3, 320, 320)
    target = {
        "boxes": torch.tensor([[40.0, 40.0, 100.0, 100.0], [150.0, 150.0, 220.0, 220.0]]),
        "labels": torch.tensor([1, 2]),
    }
    return image, target


def test_build_image_cache_record_structure(cpu_detector, dummy_image_and_target) -> None:
    image, target = dummy_image_and_target
    actions = get_action_family()
    config = CacheConfig(score_threshold=0.0, top_k_candidates=1, detections_per_img=100)

    with torch.no_grad():
        record = build_image_cache_record(
            model=cpu_detector,
            image=image,
            target=target,
            image_id=999,
            actions=actions,
            config=config,
        )

    assert record["image_id"] == 999
    assert record["image_size"] == (320, 320)
    assert record["source_image_size"] == (320, 320)
    assert set(record.keys()) == {
        "image_id",
        "image_size",
        "source_image_size",
        "baseline",
        "fpn_keys",
        "actions",
    }

    baseline = record["baseline"]
    assert set(baseline.keys()) == {"boxes", "scores", "labels", "proposal_boxes", "roi_features"}
    assert baseline["boxes"].shape[0] == baseline["scores"].shape[0]
    assert baseline["boxes"].shape[0] == baseline["labels"].shape[0]
    assert baseline["roi_features"].shape[0] == baseline["boxes"].shape[0]
    assert baseline["roi_features"].dtype == torch.float16

    assert isinstance(record["fpn_keys"], list)
    assert len(record["fpn_keys"]) > 0

    # With top_k_candidates=1 we expect one row per action.
    assert len(record["actions"]) == len(actions)


def test_identity_action_produces_zero_delta(cpu_detector, dummy_image_and_target) -> None:
    image, target = dummy_image_and_target
    actions = get_action_family()
    config = CacheConfig(score_threshold=0.0, top_k_candidates=1, detections_per_img=100)

    with torch.no_grad():
        record = build_image_cache_record(
            model=cpu_detector,
            image=image,
            target=target,
            image_id=999,
            actions=actions,
            config=config,
        )

    identity_row = next(row for row in record["actions"] if row["family"] == "identity_permutation")
    assert abs(identity_row["q_teacher"]) < 1e-6
    teacher = identity_row["teacher"]
    assert teacher["native_changed"] is False
    assert teacher["delta_tp75"] == 0
    assert teacher["delta_fp75"] == 0
    assert teacher["delta_fp50"] == 0
    assert teacher["delta_duplicate"] == 0


def test_pre_and_post_roi_features_have_consistent_shape(cpu_detector, dummy_image_and_target) -> None:
    image, target = dummy_image_and_target
    actions = get_action_family()
    config = CacheConfig(score_threshold=0.0, top_k_candidates=1, detections_per_img=100)

    with torch.no_grad():
        record = build_image_cache_record(
            model=cpu_detector,
            image=image,
            target=target,
            image_id=999,
            actions=actions,
            config=config,
        )

    for row in record["actions"]:
        assert row["h_pre"].shape == row["h_post"].shape
        assert row["h_pre"].dtype == row["h_post"].dtype
        assert row["h_pre"].dtype == torch.float16
        assert row["candidate_index"] == 0
        if row["family"] == "drop":
            assert row["h_post_present"] is False
            assert torch.count_nonzero(row["h_post"]) == 0
        else:
            assert row["h_post_present"] is True


def test_resized_input_uses_detector_transform_coordinate_space(cpu_detector) -> None:
    image = torch.rand(3, 160, 320)
    target = {
        "boxes": torch.tensor([[20.0, 20.0, 80.0, 80.0]]),
        "labels": torch.tensor([1]),
    }
    config = CacheConfig(score_threshold=0.0, top_k_candidates=1, detections_per_img=20)

    with torch.no_grad():
        record = build_image_cache_record(
            model=cpu_detector,
            image=image,
            target=target,
            image_id=1000,
            actions=get_action_family(),
            config=config,
        )

    assert record["source_image_size"] == (160, 320)
    assert record["image_size"] == (320, 640)
    assert record["baseline"]["boxes"][:, 0::2].max() <= 640
    assert record["baseline"]["boxes"][:, 1::2].max() <= 320
