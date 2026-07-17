from __future__ import annotations

from types import SimpleNamespace
import json
from pathlib import Path

import pytest
import torch

from spectral_detection_posttrain.trainers.detection.awr_boxhead import (
    build_epoch_image_schedule,
    configure_box_head_only,
    configure_native_postprocess,
    frozen_state_hashes,
    validate_re_roi_cache,
    weighted_box_loss,
)
from spectral_detection_posttrain.methods.energy_transport.endpoint.awr_weighting import (
    BudgetCounters,
    assert_equal_budget,
)
from scripts.experiments.awr_weighted_boxhead.train import (
    LOCKED_EPOCHS,
    LOCKED_LOGICAL_BATCH_SIZE,
    LOCKED_SEEDS,
    load_split_manifest,
)


class _ScalarBoxModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(1, 1)
        self.rpn = torch.nn.Linear(1, 1)
        self.roi_heads = SimpleNamespace(
            box_head=torch.nn.Linear(1, 1, bias=False),
            box_predictor=torch.nn.Linear(1, 1, bias=False),
        )
        self.register_module("box_head_module", self.roi_heads.box_head)
        self.register_module("box_predictor_module", self.roi_heads.box_predictor)

    def forward(self, images, targets):
        assert len(images) == 1
        value = images[0].reshape(-1)[0]
        scale = self.roi_heads.box_head.weight.reshape(())
        return {
            "loss_classifier": value * scale,
            "loss_box_reg": value * 0.0,
            "loss_objectness": 100.0 * scale,
            "loss_rpn_box_reg": 100.0 * scale,
        }


def test_weighted_box_loss_uses_per_image_box_losses_only() -> None:
    model = _ScalarBoxModel()
    model.roi_heads.box_head.weight.data.fill_(1.0)
    images = [torch.tensor([1.0]), torch.tensor([3.0])]
    targets = [{}, {}]
    loss = weighted_box_loss(model, images, targets, weights=[0.5, 1.5])
    assert float(loss.detach()) == pytest.approx((0.5 * 1.0 + 1.5 * 3.0) / 2.0)


def test_all_one_weights_equal_mean_per_image_box_loss() -> None:
    model = _ScalarBoxModel()
    model.roi_heads.box_head.weight.data.fill_(1.0)
    images = [torch.tensor([1.0]), torch.tensor([3.0])]
    loss = weighted_box_loss(model, images, [{}, {}], weights=[1.0, 1.0])
    assert float(loss.detach()) == pytest.approx(2.0)


def test_weighted_box_loss_rejects_bad_weight_shape_or_value() -> None:
    model = _ScalarBoxModel()
    with pytest.raises(ValueError, match="one weight per image"):
        weighted_box_loss(model, [torch.tensor([1.0])], [{}], weights=[])
    with pytest.raises(ValueError, match="finite and positive"):
        weighted_box_loss(model, [torch.tensor([1.0])], [{}], weights=[0.0])


def test_configure_box_head_only_freezes_every_other_parameter() -> None:
    model = _ScalarBoxModel()
    configure_box_head_only(model)
    assert all(parameter.requires_grad for parameter in model.roi_heads.box_head.parameters())
    assert all(parameter.requires_grad for parameter in model.roi_heads.box_predictor.parameters())
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert not any(parameter.requires_grad for parameter in model.rpn.parameters())


def test_native_postprocess_is_explicitly_locked() -> None:
    model = SimpleNamespace(
        roi_heads=SimpleNamespace(
            score_thresh=0.99,
            nms_thresh=0.99,
            detections_per_img=1,
        )
    )

    configure_native_postprocess(model)

    assert model.roi_heads.score_thresh == 0.05
    assert model.roi_heads.nms_thresh == 0.50
    assert model.roi_heads.detections_per_img == 100


def test_frozen_state_hash_detects_buffer_or_parameter_change() -> None:
    model = _ScalarBoxModel()
    configure_box_head_only(model)
    before = frozen_state_hashes(model)
    model.backbone.bias.data.add_(1.0)
    after = frozen_state_hashes(model)
    assert before["backbone"] != after["backbone"]
    assert before["rpn"] == after["rpn"]


def test_uniform_weighted_and_shuffle_share_epoch_schedule() -> None:
    image_ids = [4, 1, 3, 2]
    u = build_epoch_image_schedule(image_ids, arm="U", seed=42, epoch=1)
    w = build_epoch_image_schedule(image_ids, arm="W", seed=42, epoch=1)
    s = build_epoch_image_schedule(image_ids, arm="S", seed=42, epoch=1)
    assert u == w == s
    assert sorted(u) == sorted(image_ids)


def test_filter_schedule_uses_positive_pool_with_exact_exposure_count() -> None:
    schedule = build_epoch_image_schedule(
        [1, 2, 3, 4],
        arm="F",
        seed=42,
        epoch=2,
        positive_image_ids=[2, 4],
    )
    assert len(schedule) == 4
    assert set(schedule) <= {2, 4}
    assert schedule == build_epoch_image_schedule(
        [1, 2, 3, 4],
        arm="F",
        seed=42,
        epoch=2,
        positive_image_ids=[2, 4],
    )


def test_budget_counters_compare_all_runtime_dimensions() -> None:
    baseline = BudgetCounters(image_exposures=2500, logical_batches=320, optimizer_steps=320, scheduler_steps=0)
    counters = {arm: baseline for arm in ("U", "W", "F", "S")}
    assert_equal_budget(counters)
    broken = dict(counters)
    broken["F"] = BudgetCounters(image_exposures=2499, logical_batches=320, optimizer_steps=320, scheduler_steps=0)
    with pytest.raises(AssertionError, match="equal-budget mismatch"):
        assert_equal_budget(broken)


def _cache_payload(*, transform_size: int = 480):
    action_families = (
        "identity_permutation",
        "score_down",
        "score_up",
        "translate_left",
        "translate_right",
        "translate_up",
        "translate_down",
        "scale_down",
        "scale_up",
        "drop",
    )

    def complete_record(image_id: int) -> dict:
        return {
            "image_id": image_id,
            "image_size": (480, 480),
            "baseline": {
                "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
                "scores": torch.tensor([0.8]),
                "labels": torch.tensor([1]),
                "proposal_boxes": torch.tensor([[9.0, 9.0, 21.0, 21.0]]),
                "roi_features": torch.ones(1, 4),
            },
            "fpn_keys": ["0"],
            "actions": [
                {
                    "candidate_index": 0,
                    "family": family,
                    "q_teacher": 0.0 if family == "identity_permutation" else 0.1,
                }
                for family in action_families
            ],
        }

    return {
        "metadata": {
            "format": "re_roi_counterfactual_v1",
            "split_name": "fit",
            "split_manifest_sha256": "split-sha",
            "checkpoint_sha256": "checkpoint-sha",
            "annotation_sha256": "annotation-sha",
            "action_family_hash": "re_roi_action_family_v2_9_non_identity_002_step",
            "utility_definition": "raw_locked_scalarization_v1",
            "coordinate_space": "detector_transform",
            "git_commit": "abc123",
            "model_config": {"min_size": transform_size, "max_size": transform_size},
            "cache_config": {
                "score_threshold": 0.05,
                "nms_threshold": 0.5,
                "detections_per_img": 100,
                "top_k_candidates": 3,
            },
        },
        "records": [complete_record(1), complete_record(2)],
    }


def test_cache_provenance_accepts_exact_locked_contract() -> None:
    records = validate_re_roi_cache(
        _cache_payload(),
        fit_image_ids=[1, 2],
        split_manifest_sha256="split-sha",
        checkpoint_sha256="checkpoint-sha",
        annotation_sha256="annotation-sha",
    )
    assert [record["image_id"] for record in records] == [1, 2]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split_manifest_sha256", "wrong", "split manifest"),
        ("checkpoint_sha256", "wrong", "checkpoint"),
        ("annotation_sha256", "wrong", "annotation"),
        ("action_family_hash", "old", "action family"),
    ],
)
def test_cache_provenance_rejects_hash_mismatch(field: str, value: str, message: str) -> None:
    payload = _cache_payload()
    payload["metadata"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_re_roi_cache(
            payload,
            fit_image_ids=[1, 2],
            split_manifest_sha256="split-sha",
            checkpoint_sha256="checkpoint-sha",
            annotation_sha256="annotation-sha",
        )


def test_cache_provenance_rejects_320_transform_or_wrong_fit_records() -> None:
    with pytest.raises(ValueError, match="transform size"):
        validate_re_roi_cache(
            _cache_payload(transform_size=320),
            fit_image_ids=[1, 2],
            split_manifest_sha256="split-sha",
            checkpoint_sha256="checkpoint-sha",
            annotation_sha256="annotation-sha",
        )


def test_cache_provenance_rejects_incomplete_candidate_action_family() -> None:
    payload = _cache_payload()
    payload["records"][0]["actions"].pop()

    with pytest.raises(ValueError, match="complete action family"):
        validate_re_roi_cache(
            payload,
            fit_image_ids=[1, 2],
            split_manifest_sha256="split-sha",
            checkpoint_sha256="checkpoint-sha",
            annotation_sha256="annotation-sha",
        )


def test_cache_provenance_rejects_missing_baseline_even_with_known_image_id() -> None:
    payload = _cache_payload()
    del payload["records"][0]["baseline"]

    with pytest.raises(ValueError, match="baseline"):
        validate_re_roi_cache(
            payload,
            fit_image_ids=[1, 2],
            split_manifest_sha256="split-sha",
            checkpoint_sha256="checkpoint-sha",
            annotation_sha256="annotation-sha",
        )


def test_training_entry_locks_seeds_epochs_and_logical_batch() -> None:
    assert LOCKED_SEEDS == (42, 2024, 999)
    assert LOCKED_EPOCHS == 10
    assert LOCKED_LOGICAL_BATCH_SIZE == 8


def test_training_entry_validates_nested_split_counts_and_disjointness(tmp_path: Path) -> None:
    counts = {"fit": 250, "tune": 68, "calibration": 68, "outer_heldout": 68}
    cursor = 0
    payload = {"version_id": "det.energy.re_roi_counterfactual.split.001", "splits": {}}
    for name, count in counts.items():
        payload["splits"][name] = {"image_ids": list(range(cursor, cursor + count))}
        cursor += count
    path = tmp_path / "split.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert len(load_split_manifest(path)["splits"]["fit"]["image_ids"]) == 250

    payload["splits"]["tune"]["image_ids"][0] = payload["splits"]["fit"]["image_ids"][0]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pairwise disjoint"):
        load_split_manifest(path)
    payload = _cache_payload()
    payload["records"] = [{"image_id": 1, "actions": []}]
    with pytest.raises(ValueError, match="fit image IDs"):
        validate_re_roi_cache(
            payload,
            fit_image_ids=[1, 2],
            split_manifest_sha256="split-sha",
            checkpoint_sha256="checkpoint-sha",
            annotation_sha256="annotation-sha",
        )
