from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest
import torch

from spectral_detection_posttrain.methods.energy_transport import ROIActionState


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_nwpu_m0_set_oracle.py"
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_m0_set_oracle.sh"
LOCKED_CONFIG = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.set_oracle.m0.001.json"
)


def _load_module():
    assert SCRIPT.exists(), "M0 set-oracle runner has not been implemented"
    spec = importlib.util.spec_from_file_location("eval_nwpu_m0_set_oracle", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state() -> ROIActionState:
    return ROIActionState(
        features=torch.zeros((3, 4)),
        boxes=torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 11.0, 11.0],
                [20.0, 20.0, 30.0, 30.0],
            ]
        ),
        scores=torch.tensor([0.9, 0.8, 0.7]),
        labels=torch.tensor([1, 1, 2]),
        image_indices=torch.zeros(3, dtype=torch.long),
        proposal_indices=torch.arange(3),
        logits=torch.tensor(
            [
                [0.0, 3.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        ),
    )


def test_locked_config_is_preregistered_and_matches_m0_constants() -> None:
    module = _load_module()
    config = json.loads(LOCKED_CONFIG.read_text(encoding="utf-8"))

    assert module.sha256_file(LOCKED_CONFIG) == module.LOCKED_CONFIG_SHA256
    assert config["version_id"] == "det.energy.set_oracle.m0.001"
    assert config["status"] == "preregistered"
    assert config["candidate_pool"]["max_candidates_per_image"] == 12
    assert config["search"]["action_budget"] == 4
    assert config["search"]["beam_width"] == 4
    assert config["search"]["bootstrap_repetitions"] == 10000
    assert config["gates"]["G5_detector_headroom"]["min_ap75_delta_vs_identity"] == 0.005


def test_locked_config_rejects_content_drift_and_noncanonical_path(tmp_path: Path) -> None:
    module = _load_module()
    config = json.loads(LOCKED_CONFIG.read_text(encoding="utf-8"))
    changed = json.loads(json.dumps(config))
    changed["set_utility"]["fp75_weight"] = -0.2

    with pytest.raises(ValueError, match="utility"):
        module.validate_locked_config(changed)

    copied = tmp_path / LOCKED_CONFIG.name
    copied.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        module.load_locked_config(copied)


def test_build_local_candidate_pool_keeps_one_best_action_per_proposal() -> None:
    module = _load_module()
    deltas = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0],
        ]
    )
    quality = torch.tensor(
        [
            [0.70, 0.76, 0.77],
            [0.80, 0.805, 0.79],
            [0.74, 0.76, 0.72],
        ]
    )

    pool = module.build_local_candidate_pool(
        deltas,
        quality,
        torch.tensor([True, True, True]),
        min_iou_gain=0.002,
        boundary_iou=0.75,
        boundary_temperature=0.05,
        action_energy_scale=0.04,
        local_energy_weight=0.05,
        local_move_cost=0.02,
        max_candidates=12,
    )

    assert [(item.proposal_index, item.candidate_index) for item in pool] == [(0, 2), (2, 1)]
    assert all(item.local_value > 0.0 for item in pool)
    assert pool[0].local_value > pool[1].local_value


def test_candidate_energy_divides_by_locked_scale_and_keeps_positive_high_iou_move() -> None:
    module = _load_module()
    deltas = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
        ]
    )

    pool = module.build_local_candidate_pool(
        deltas,
        torch.tensor([[0.76, 0.95]]),
        torch.tensor([True]),
        action_energy_scale=0.04,
    )

    assert len(pool) == 1
    assert pool[0].action_energy == pytest.approx(0.25)
    assert pool[0].base_quality == pytest.approx(0.76)


def test_candidate_pool_breaks_equal_local_value_ties_by_lower_energy() -> None:
    module = _load_module()
    pool = module.build_local_candidate_pool(
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0],
            ]
        ),
        torch.tensor([[0.60, 0.80, 0.80]]),
        torch.tensor([True]),
        local_energy_weight=0.0,
        local_move_cost=0.0,
    )

    assert pool[0].candidate_index == 2
    assert pool[0].action_energy == pytest.approx(0.25)


def test_actions_from_selection_changes_only_selected_proposals() -> None:
    module = _load_module()
    state = _state()
    deltas = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
            [0.0, -0.2, 0.0, 0.0],
        ]
    )
    pool = module.build_local_candidate_pool(
        deltas,
        torch.tensor(
            [
                [0.70, 0.76, 0.60],
                [0.70, 0.60, 0.78],
                [0.90, 0.89, 0.88],
            ]
        ),
        torch.ones(3, dtype=torch.bool),
        max_candidates=12,
    )

    actions = module.actions_from_selection(state, pool)

    assert torch.allclose(actions.box_delta[0], deltas[1])
    assert torch.allclose(actions.box_delta[1], deltas[2])
    assert actions.box_delta[2].count_nonzero().item() == 0
    assert actions.feature_delta.count_nonzero().item() == 0
    assert actions.score_delta.count_nonzero().item() == 0


def test_gate_evaluation_requires_every_preregistered_condition() -> None:
    module = _load_module()
    locked = json.loads(LOCKED_CONFIG.read_text(encoding="utf-8"))
    passing = {
        "strict_parity": {
            "passed": True,
            "mismatched_images": 0,
            "max_box_abs_error": 0.0,
            "max_score_abs_error": 0.0,
        },
        "coverage_fraction_two_candidates": 0.5,
        "beam_minus_local": {
            "sum": 6.0,
            "improved_images": 12,
            "bootstrap_ci95": [0.01, 0.2],
        },
        "beam_minus_local_nms_off_sum": 2.0,
        "beam_minus_permuted": {"bootstrap_ci95": [0.01, 0.2]},
        "detector_deltas": {
            "ap75": 0.006,
            "ap50": -0.001,
            "false_positive_rate": 0.01,
            "num_predictions_relative": 0.04,
        },
    }

    result = module.evaluate_preregistered_gates(passing, locked)
    failing = json.loads(json.dumps(passing))
    failing["beam_minus_local"]["bootstrap_ci95"][0] = -0.01
    failed = module.evaluate_preregistered_gates(failing, locked)

    assert result["all_passed"] is True
    assert all(result["gates"].values())
    assert failed["all_passed"] is False
    assert failed["gates"]["G2_coordination"] is False


def test_json_value_serializes_set_outcome_as_components() -> None:
    module = _load_module()
    from spectral_detection_posttrain.methods.energy_transport import SetOutcome

    value = module._json_value(
        SetOutcome(
            tp50=2,
            tp75=1,
            fp50=3,
            fp75=4,
            prediction_count=5,
            action_energy=0.25,
            action_count=1,
        )
    )

    assert value == {
        "tp50": 2,
        "tp75": 1,
        "fp50": 3,
        "fp75": 4,
        "prediction_count": 5,
        "action_energy": 0.25,
        "action_count": 1,
    }


def test_pool_delta_permutation_is_seeded_and_preserves_candidate_identity() -> None:
    module = _load_module()
    deltas = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
        ]
    )
    pool = module.build_local_candidate_pool(
        deltas,
        torch.tensor(
            [
                [0.60, 0.80, 0.70, 0.69, 0.68, 0.67],
                [0.60, 0.70, 0.81, 0.69, 0.68, 0.67],
                [0.60, 0.70, 0.69, 0.82, 0.68, 0.67],
                [0.60, 0.70, 0.69, 0.68, 0.83, 0.67],
                [0.60, 0.70, 0.69, 0.68, 0.67, 0.84],
            ]
        ),
        torch.ones(5, dtype=torch.bool),
        local_energy_weight=0.0,
        local_move_cost=0.0,
    )

    first = module.permute_pool_deltas(pool, seed=31416)
    second = module.permute_pool_deltas(pool, seed=31417)

    assert [item.action_id for item in first] == [item.action_id for item in pool]
    assert sorted(tuple(item.box_delta.tolist()) for item in first) == sorted(
        tuple(item.box_delta.tolist()) for item in pool
    )
    assert [item.box_delta.tolist() for item in first] != [item.box_delta.tolist() for item in second]


def test_selected_pool_items_reconstructs_fixed_native_action_set() -> None:
    module = _load_module()
    deltas = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0],
        ]
    )
    pool = module.build_local_candidate_pool(
        deltas,
        torch.tensor([[0.60, 0.80, 0.70], [0.60, 0.70, 0.81]]),
        torch.ones(2, dtype=torch.bool),
        local_energy_weight=0.0,
        local_move_cost=0.0,
    )

    selected = module.selected_pool_items(pool, [pool[1].action_id, pool[0].action_id])

    assert [item.action_id for item in selected] == [pool[1].action_id, pool[0].action_id]
    with pytest.raises(ValueError, match="unknown action"):
        module.selected_pool_items(pool, ["missing"])


def test_validation_manifest_verifies_locked_image_ids() -> None:
    module = _load_module()

    class Dataset:
        img_ids = [3, 1, 2]

        def __len__(self):
            return len(self.img_ids)

    class Loader:
        dataset = Dataset()

    encoded = json.dumps([1, 2, 3], separators=(",", ":")).encode("ascii")
    expected = hashlib.sha256(encoded).hexdigest()
    locked = {"dataset": {"validation_images": 3, "val_image_ids_sha256": expected}}

    manifest = module.validation_manifest(Loader())
    module.verify_validation_manifest(manifest, locked)

    assert manifest == {"count": 3, "image_ids_sha256": expected}
    with pytest.raises(ValueError, match="validation split"):
        module.verify_validation_manifest(
            {"count": 3, "image_ids_sha256": "wrong"},
            locked,
        )


def test_strict_prediction_comparison_counts_large_box_or_score_error() -> None:
    module = _load_module()
    expected = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.9]),
    }
    shifted = {
        "boxes": torch.tensor([[0.01, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.89]),
    }

    box_error, score_error, mismatch, same = module._compare_predictions(expected, shifted)

    assert box_error == pytest.approx(0.01)
    assert score_error == pytest.approx(0.01)
    assert mismatch >= 1
    assert same is False


def test_candidate_pool_rejects_nonidentity_first_delta() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="identity"):
        module.build_local_candidate_pool(
            torch.tensor([[0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.5, 0.6]]),
            torch.tensor([True]),
        )


def test_launcher_is_remote_gpu2_only_and_requires_strict_memory_gate() -> None:
    assert LAUNCHER.exists(), "M0 remote launcher has not been implemented"
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "/home/ps/lzz/manifold-detection-energy-transport" in text
    assert "/home/ps/lzz/RLimage" not in text
    assert "CUDA_VISIBLE_DEVICES=2" in text
    assert "memory.free" in text
    assert "free_mb <= 8192" in text
    assert "det.energy.set_oracle.m0.001.json" in text
    assert "--require-clean-git" in text
