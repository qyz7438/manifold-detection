from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_nwpu_joint_delta_u.py"
CONFIG = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.joint_probe.001.json"
)
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_joint_delta_u_probe_s42.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("probe_nwpu_joint_delta_u", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_locked_config_is_train_split_holdout_and_keeps_validation_unseen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert _load_module().load_locked_config() == config

    assert config["version_id"] == "det.energy.joint_probe.001"
    assert config["probe"]["scope"] == "train_split_image_group_holdout"
    assert config["probe"]["uses_validation"] is False
    assert config["probe"]["heldout_fraction"] == pytest.approx(0.2)
    assert config["candidate_pool"]["candidate_count"] == 73
    assert config["candidate_pool"]["max_candidates_per_image"] == 12
    assert config["candidate_pool"]["action_budget"] == 1
    assert config["cache"]["stores_spatial_features"] is True
    assert config["cache"]["pairing"] == "same_detector_forward"
    assert config["arms"] == [
        "m1_factorized",
        "joint_full",
        "joint_no_edges",
        "joint_spatial_shuffle",
    ]


def test_singleton_targets_are_exact_differences_from_identity() -> None:
    module = _load_module()
    pool = [
        {"action_id": "p0001_c04", "proposal_index": 1, "candidate_index": 4, "box_delta": [0.1, 0, 0, 0], "action_energy": 0.25},
        {"action_id": "p0003_c08", "proposal_index": 3, "candidate_index": 8, "box_delta": [0, -0.1, 0, 0], "action_energy": 0.25},
    ]
    trace = [
        {"action_ids": [], "utility": 2.0, "outcome": {}},
        {"action_ids": ["p0001_c04"], "utility": 2.75, "outcome": {}},
        {"action_ids": ["p0003_c08"], "utility": 1.5, "outcome": {}},
    ]

    record = module.build_trace_record(
        image_id=17,
        proposal_count=5,
        candidate_pool=pool,
        evaluation_trace=trace,
    )

    assert record["image_id"] == 17
    assert torch.equal(record["proposal_indices"], torch.tensor([1, 3]))
    assert torch.equal(record["candidate_indices"], torch.tensor([4, 8]))
    assert torch.allclose(record["singleton_delta_u"], torch.tensor([0.75, -0.5]))
    assert not any("gt" in key.lower() for key in record)


def test_trace_record_rejects_missing_singleton_and_invalid_proposal() -> None:
    module = _load_module()
    trace = [{"action_ids": [], "utility": 0.0, "outcome": {}}]
    pool = [{"action_id": "p0002_c01", "proposal_index": 2, "candidate_index": 1, "box_delta": [0, 0, 0, 0], "action_energy": 0.0}]

    with pytest.raises(ValueError, match="singleton"):
        module.build_trace_record(
            image_id=1,
            proposal_count=3,
            candidate_pool=pool,
            evaluation_trace=trace,
        )

    with pytest.raises(ValueError, match="proposal"):
        module.build_trace_record(
            image_id=1,
            proposal_count=2,
            candidate_pool=pool,
            evaluation_trace=trace + [{"action_ids": ["p0002_c01"], "utility": 1.0, "outcome": {}}],
        )


def test_sparse_target_matrix_only_supervises_traced_candidates() -> None:
    module = _load_module()
    record = {
        "proposal_indices": torch.tensor([0, 2]),
        "candidate_indices": torch.tensor([5, 7]),
        "singleton_delta_u": torch.tensor([0.6, -0.4]),
    }

    target, mask = module.sparse_target_matrix(record, proposal_count=3, candidate_count=9)

    assert target.shape == mask.shape == (3, 9)
    assert mask.sum().item() == 2
    assert target[0, 5].item() == pytest.approx(0.6)
    assert target[2, 7].item() == pytest.approx(-0.4)
    assert not mask[:, 0].any()


def test_m1_baseline_scores_same_sparse_candidates_without_a_move_decision() -> None:
    module = _load_module()
    action_logits = torch.tensor([[2.0, 3.5, 0.0], [1.0, 0.0, 2.0]])
    move_logits = torch.tensor([0.25, -0.5])
    record = {
        "proposal_indices": torch.tensor([0, 1]),
        "candidate_indices": torch.tensor([1, 2]),
    }

    scores = module.m1_sparse_candidate_scores(action_logits, move_logits, record)

    assert torch.allclose(scores, torch.tensor([1.75, 0.5]))


def test_probe_gates_require_joint_to_beat_m1_and_both_controls() -> None:
    module = _load_module()
    config = {
        "gates": {
            "min_heldout_candidates": 20,
            "min_heldout_images": 2,
            "min_pairwise_accuracy": 0.55,
            "min_sign_accuracy": 0.55,
            "min_selected_true_delta_mean": 0.01,
            "min_positive_precision_lift": 0.05,
            "min_pairwise_gain_over_no_edges": 0.01,
            "min_pairwise_gain_over_shuffle": 0.01,
        }
    }
    metrics = {
        "m1_factorized": {"pairwise_accuracy": 0.99, "candidate_count": 100, "image_count": 10},
        "joint_full": {
            "pairwise_accuracy": 0.61,
            "sign_accuracy": 0.63,
            "selected_true_delta_mean": 0.08,
            "positive_precision": 0.70,
            "positive_prevalence": 0.50,
            "candidate_count": 100,
            "image_count": 10,
        },
        "joint_no_edges": {"pairwise_accuracy": 0.58, "candidate_count": 100, "image_count": 10},
        "joint_spatial_shuffle": {"pairwise_accuracy": 0.57, "candidate_count": 100, "image_count": 10},
    }

    result = module.evaluate_probe_gates(metrics, config)

    assert result["all_passed"] is True
    assert all(result["gates"].values())

    metrics["joint_full"]["pairwise_accuracy"] = 0.575
    failed = module.evaluate_probe_gates(metrics, config)
    assert failed["all_passed"] is False
    assert failed["gates"]["G2_joint_vs_no_edges"] is False


def test_probe_metrics_rank_candidates_within_images_not_across_images() -> None:
    module = _load_module()
    predicted = torch.tensor([0.8, -0.1, -0.4, 0.7, -0.1])
    target = torch.tensor([1.0, 0.0, -0.5, 0.4, 0.2])
    image_ids = torch.tensor([10, 10, 10, 20, 20])

    metrics = module.probe_metrics(
        predicted,
        target,
        image_ids,
        action_budget=1,
        target_epsilon=1e-3,
    )

    assert metrics["candidate_count"] == 5
    assert metrics["image_count"] == 2
    assert metrics["pairwise_accuracy"] == pytest.approx(1.0)
    assert metrics["oracle_regret_mean"] == pytest.approx(0.0)
    assert metrics["selected_true_delta_mean"] == pytest.approx(0.7)
    assert metrics["positive_precision"] == pytest.approx(1.0)
    assert metrics["positive_recall"] == pytest.approx(2 / 3)
    assert metrics["positive_prevalence"] == pytest.approx(3 / 5)


def test_probe_metrics_treat_identity_as_the_oracle_when_all_actions_are_harmful() -> None:
    module = _load_module()
    metrics = module.probe_metrics(
        torch.tensor([0.9, 0.8]),
        torch.tensor([-0.2, -0.5]),
        torch.tensor([4, 4]),
        action_budget=1,
        target_epsilon=1e-3,
    )

    assert metrics["pairwise_accuracy"] == pytest.approx(1.0)
    assert metrics["oracle_regret_mean"] == pytest.approx(0.2)
    assert metrics["selected_true_delta_mean"] == pytest.approx(-0.2)
    assert metrics["positive_precision"] == pytest.approx(0.0)


def test_probe_cache_round_trip_preserves_same_pass_feature_label_pairing(tmp_path: Path) -> None:
    module = _load_module()
    detector, trace = _tiny_probe_record(1, True)
    path = tmp_path / "probe.pt"

    module.write_probe_cache(path, [(detector, trace)])
    loaded = module.read_probe_cache(path)

    assert len(loaded) == 1
    loaded_detector, loaded_trace = loaded[0]
    assert loaded_detector["image_id"] == loaded_trace["image_id"] == 1
    assert torch.equal(loaded_detector["spatial_features"], detector["spatial_features"])
    assert torch.equal(loaded_trace["singleton_delta_u"], trace["singleton_delta_u"])

    torch.save({"format": "torch_pt", "schema_version": 999, "records": []}, path)
    with pytest.raises(ValueError, match="schema"):
        module.read_probe_cache(path)

    malformed = dict(trace)
    malformed["box_deltas"] = torch.zeros(3, 4)
    with pytest.raises(ValueError, match="shape"):
        module.write_probe_cache(path, [(detector, malformed)])

    wrong_image = dict(trace)
    wrong_image["image_id"] = 2
    with pytest.raises(ValueError, match="image"):
        module.write_probe_cache(path, [(detector, wrong_image)])


def test_full_train_guard_rejects_limited_probe() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="full train"):
        module.validate_cli_args(
            Namespace(require_full_train=True, limit_train=10)
        )


def test_cuda_probe_requires_reproducible_cublas_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    module.validate_deterministic_cuda_environment(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        module.validate_deterministic_cuda_environment(torch.device("cuda"))

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    module.validate_deterministic_cuda_environment(torch.device("cuda"))


def test_locked_config_loader_rejects_noncanonical_path(tmp_path: Path) -> None:
    module = _load_module()
    copied = tmp_path / "copied.json"
    copied.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical"):
        module.load_locked_config(copied)


def _tiny_probe_record(image_id: int, positive_first: bool) -> tuple[dict, dict]:
    detector = {
        "image_id": image_id,
        "spatial_features": torch.randn(2, 4, 3, 3).half(),
        "class_logits": torch.tensor([[0.0, 2.0, -1.0], [0.0, 1.0, 0.5]]),
        "predicted_labels": torch.tensor([1, 1]),
        "scores": torch.tensor([0.8, 0.6]),
        "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0], [2.0, 0.0, 10.0, 8.0]]),
        "image_size": (16, 16),
        "action_targets": torch.tensor([1, 2]),
        "move_targets": torch.tensor([1, 0]),
    }
    values = torch.tensor([0.8, -0.4]) if positive_first else torch.tensor([-0.3, 0.5])
    trace = {
        "image_id": image_id,
        "proposal_count": 2,
        "action_ids": ("p0000_c01", "p0001_c02"),
        "proposal_indices": torch.tensor([0, 1]),
        "candidate_indices": torch.tensor([1, 2]),
        "box_deltas": torch.tensor([[0.1, 0.0, 0.0, 0.0], [-0.1, 0.0, 0.0, 0.0]]),
        "action_energies": torch.tensor([0.25, 0.25]),
        "identity_utility": 0.0,
        "singleton_delta_u": values,
    }
    return detector, trace


def test_tiny_joint_arm_trains_and_evaluates_without_validation_data(tmp_path: Path) -> None:
    module = _load_module()
    records = [
        _tiny_probe_record(10, True),
        _tiny_probe_record(20, False),
    ]
    candidate_deltas = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0], [-0.1, 0.0, 0.0, 0.0]]
    )
    config = {
        "model": {"hidden_dim": 8, "max_neighbors": 4, "same_class_edges": True},
        "training": {"epochs": 1, "lr": 1e-3, "weight_decay": 0.0, "grad_accum_steps": 1},
        "loss": {
            "regression_weight": 1.0,
            "sign_weight": 0.5,
            "ranking_weight": 0.5,
            "target_epsilon": 1e-3,
        },
        "candidate_pool": {"action_budget": 1},
        "dataset": {"seed": 42},
    }

    model, training = module.train_joint_arm(
        records,
        candidate_deltas,
        config,
        torch.device("cpu"),
        tmp_path,
        arm="joint_full",
        num_classes=3,
    )
    metrics = module.evaluate_joint_arm(
        model,
        records,
        candidate_deltas,
        config,
        torch.device("cpu"),
        arm="joint_full",
    )

    assert Path(training["selected_checkpoint"]).is_file()
    assert len(training["history"]) == 1
    assert metrics["candidate_count"] == 4
    assert metrics["image_count"] == 2
    assert 0.0 <= metrics["pairwise_accuracy"] <= 1.0


def test_joint_arm_rejects_trace_delta_that_does_not_match_candidate_grid(tmp_path: Path) -> None:
    module = _load_module()
    detector, trace = _tiny_probe_record(10, True)
    trace["box_deltas"][0, 0] = 0.9
    candidate_deltas = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0], [-0.1, 0.0, 0.0, 0.0]]
    )
    config = {
        "model": {"hidden_dim": 8, "max_neighbors": 4, "same_class_edges": True},
        "training": {"epochs": 1, "lr": 1e-3, "weight_decay": 0.0, "grad_accum_steps": 1},
        "loss": {"regression_weight": 1.0, "sign_weight": 0.5, "ranking_weight": 0.5, "target_epsilon": 1e-3},
        "candidate_pool": {"action_budget": 1},
        "dataset": {"seed": 42},
    }

    with pytest.raises(ValueError, match="candidate grid"):
        module.train_joint_arm(
            [(detector, trace)],
            candidate_deltas,
            config,
            torch.device("cpu"),
            tmp_path,
            arm="joint_full",
            num_classes=3,
        )


def test_runner_has_no_placeholder_implementation() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "NotImplemented" not in source


def test_launcher_is_gpu2_only_strictly_memory_gated_and_uses_correct_root() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "/home/ps/lzz/manifold-detection-energy-transport" in source
    assert "/home/ps/lzz/RLimage" not in source
    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in source
    assert "memory.free" in source
    assert "8192" in source
    assert "-le 8192" in source or "<= 8192" in source
    assert "probe_nwpu_joint_delta_u.py" in source
    assert "--require-clean-git" in source
    assert "--require-full-train" in source
    assert "eval_metrics.json" in source
    assert '"completed"' in source
    assert "code_commit" in source
    assert "a6c30c89acdc3b213da18eb2cfc2b7df9b83583c3c70d60872854cba1aa0e3d7" in source
