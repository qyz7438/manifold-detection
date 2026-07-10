from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_nwpu_joint_delta_u_v2.py"
CONFIG = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.joint_probe.002.json"
)
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_joint_delta_u_probe_v2_s42.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("probe_nwpu_joint_delta_u_v2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_config_locks_train_calibration_before_unseen_validation() -> None:
    module = _load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert module.load_locked_config() == config
    assert config["version_id"] == "det.energy.joint_probe.002"
    assert config["status"] == "preregistered"
    assert config["probe"]["scope"] == "train_fit_calibration_then_detector_unseen_validation"
    assert config["probe"]["validation_evaluated_once"] is True
    assert config["probe"]["claim_boundary"].startswith("B=1 oracle-pool-conditioned")
    assert config["source_train_cache"]["sha256"] == (
        "d30a36631d13d2e95d4340c901d0ea53cac37c7b6d4c30d1a97315105994c6cd"
    )
    assert config["split"]["calibration_fraction"] == pytest.approx(0.2)
    assert config["candidate_pool"]["action_budget"] == 1
    assert config["arms"] == [
        "joint_full",
        "joint_no_edges",
        "joint_spatial_shuffle",
        "joint_edge_topology_shuffle",
    ]
    assert config["calibration"]["max_action_image_rate"] <= 0.25
    assert config["bootstrap"]["resamples"] >= 2000


def _paired_record(image_id: int) -> tuple[dict, dict]:
    return {"image_id": image_id}, {"image_id": image_id}


def test_train_fit_and_calibration_split_is_group_disjoint_and_deterministic() -> None:
    module = _load_module()
    records = [_paired_record(image_id) for image_id in range(20)]

    first = module.split_train_records(
        records, calibration_fraction=0.2, seed=20260711
    )
    second = module.split_train_records(
        records, calibration_fraction=0.2, seed=20260711
    )

    fit_ids = {record[0]["image_id"] for record in first[0]}
    calibration_ids = {record[0]["image_id"] for record in first[1]}
    assert len(fit_ids) == 16
    assert len(calibration_ids) == 4
    assert fit_ids.isdisjoint(calibration_ids)
    assert first == second


def _passing_gate_payload() -> dict:
    return {
        "validation": {
            "support": {
                "candidate_count": 700,
                "image_count": 120,
                "topology_edge_change_fraction": 0.80,
            },
            "raw": {
                "joint_full": {"pairwise_accuracy": 0.57},
                "joint_no_edges": {"pairwise_accuracy": 0.52},
                "joint_spatial_shuffle": {"pairwise_accuracy": 0.515},
                "joint_edge_topology_shuffle": {"pairwise_accuracy": 0.51},
            },
            "calibrated": {
                "joint_full": {
                    "selected_count": 24,
                    "mean_delta_u_per_image": 0.03,
                    "positive_precision_lift": 0.12,
                    "action_image_rate": 0.20,
                },
                "joint_no_edges": {"mean_delta_u_per_image": 0.01},
                "joint_spatial_shuffle": {"mean_delta_u_per_image": 0.008},
                "joint_edge_topology_shuffle": {"mean_delta_u_per_image": 0.005},
            },
            "bootstrap": {
                "full_vs_no_edges": {"ci_low": 0.01},
                "full_vs_spatial_shuffle": {"ci_low": 0.015},
                "full_vs_edge_topology_shuffle": {"ci_low": 0.02},
            },
        },
        "calibration": {
            "joint_full": {
                "used_identity_fallback": False,
                "metrics": {"mean_delta_u_lcb": 0.02},
            }
        },
    }


def test_v2_gates_require_ci_calibration_and_validation_utility() -> None:
    module = _load_module()
    config = {
        "gates": {
            "min_validation_candidates": 500,
            "min_validation_candidate_images": 100,
            "min_topology_edge_change_fraction": 0.25,
            "min_pairwise_accuracy": 0.55,
            "min_pairwise_ci_low_vs_controls": 0.0,
            "min_calibration_lcb": 0.0,
            "min_validation_selected_actions": 12,
            "min_validation_mean_delta_u": 0.01,
            "min_validation_positive_precision_lift": 0.05,
            "max_validation_action_image_rate": 0.25,
            "min_validation_gain_over_controls": 0.005,
        }
    }
    payload = _passing_gate_payload()

    passed = module.evaluate_v2_gates(payload, config)

    assert passed["all_passed"] is True
    assert all(passed["gates"].values())

    payload["validation"]["bootstrap"]["full_vs_no_edges"]["ci_low"] = -0.01
    failed = module.evaluate_v2_gates(payload, config)
    assert failed["all_passed"] is False
    assert failed["gates"]["G1_relational_ranking"] is False

    payload = _passing_gate_payload()
    payload["validation"]["bootstrap"]["full_vs_spatial_shuffle"]["ci_low"] = -0.01
    failed_spatial = module.evaluate_v2_gates(payload, config)
    assert failed_spatial["all_passed"] is False
    assert failed_spatial["gates"]["G1_relational_ranking"] is False


def test_train_and_validation_ids_must_be_explicitly_disjoint() -> None:
    module = _load_module()

    module.require_disjoint_image_ids([1, 2, 3], [4, 5])
    with pytest.raises(ValueError, match="overlap"):
        module.require_disjoint_image_ids([1, 2, 3], [3, 4])


def test_full_validation_guard_rejects_limited_run() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="full validation"):
        module.validate_cli_args(
            Namespace(require_full_validation=True, limit_validation=8)
        )


def test_locked_config_loader_rejects_copy(tmp_path: Path) -> None:
    module = _load_module()
    copied = tmp_path / "copied.json"
    copied.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical"):
        module.load_locked_config(copied)


def test_v2_runner_keeps_claims_below_detector_and_ap_improvement() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    config = CONFIG.read_text(encoding="utf-8")

    assert "m1_factorized" not in source
    assert "validation_evaluated_once" in source
    assert "detector_only_action_generation" in config
    assert "AP improvement" in config
    assert "NotImplemented" not in source


def test_v2_launcher_is_clean_gpu2_only_and_hash_locked() -> None:
    module = _load_module()
    source = LAUNCHER.read_text(encoding="utf-8")
    config_hash = module.sha256_file(CONFIG)

    assert "/home/ps/lzz/manifold-detection-energy-transport" in source
    assert "/home/ps/lzz/RLimage" not in source
    assert "rev-parse --is-inside-work-tree" in source
    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in source
    assert "memory.free" in source
    assert "8192" in source
    assert "-le 8192" in source or "<= 8192" in source
    assert "probe_nwpu_joint_delta_u_v2.py" in source
    assert "--require-clean-git" in source
    assert "--require-full-validation" in source
    assert config_hash == module.LOCKED_CONFIG_SHA256
    assert config_hash in source
