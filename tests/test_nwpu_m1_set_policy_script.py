from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_nwpu_m1_set_policy.py"
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_m1_set_policy.sh"
LOCKED_CONFIG = ROOT / "spectral_detection_posttrain" / "configs" / "versions" / "det.energy.set_policy.m1.001.json"


def _load_module():
    assert SCRIPT.exists(), "M1 set-policy runner has not been implemented"
    spec = importlib.util.spec_from_file_location("train_nwpu_m1_set_policy", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_locked_config_is_canonical_and_contains_m1_hash_and_hyperparameter_locks() -> None:
    module = _load_module()
    config = json.loads(LOCKED_CONFIG.read_text(encoding="utf-8"))

    assert module.sha256_file(LOCKED_CONFIG) == module.LOCKED_CONFIG_SHA256
    assert config["version_id"] == "det.energy.set_policy.m1.001"
    assert config["status"] == "preregistered"
    assert config["dataset"]["seed"] == config["dataset"]["data_seed"] == 42
    assert config["training"]["epochs"] == 4
    assert config["policy"]["hidden_dim"] == 96
    assert config["training"]["lr"] == pytest.approx(1e-3)
    assert config["training"]["weight_decay"] == pytest.approx(1e-4)
    assert config["training"]["grad_accum_steps"] == 8
    assert config["candidate_pool"]["candidate_count"] == 73
    assert config["detector"]["checkpoint_sha256"] == "de126708d063a82dd5c721799cd124fc4e52d28866139e103b1cd65427742027"
    assert config["dataset"]["annotation_sha256"] == "dde27d9362d9c6aa358a1cab160c9e075e57405d3fe876de049973c6e6150f0e"
    assert config["dataset"]["train_image_ids_sha256"] == "7abe3c8370985f49698dcc3c42ca5917f1e17941fa024643147a58479c7cd9bd"
    assert config["dataset"]["val_image_ids_sha256"] == "49f05cc9fa82ccaf924ff3be58a8f6376387c225e020ea6138182a4637219684"
    assert config["source_m0"]["result_sha256"] == "372d872ccaebea8418c2ff4428e56a666d4e4440a77b16acbd127f4999869cda"
    assert config["cache"]["format"] == "torch_pt"
    assert config["cache"]["feature_storage_dtype"] == "float16"


def test_locked_config_rejects_drift_and_noncanonical_path(tmp_path: Path) -> None:
    module = _load_module()
    config = json.loads(LOCKED_CONFIG.read_text(encoding="utf-8"))
    changed = json.loads(json.dumps(config))
    changed["training"]["epochs"] = 5
    with pytest.raises(ValueError, match="locked|epoch|config"):
        module.validate_locked_config(changed)

    copied = tmp_path / LOCKED_CONFIG.name
    copied.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        module.load_locked_config(copied)


def test_cache_record_has_detector_observables_and_no_gt_or_quality_fields() -> None:
    module = _load_module()
    record = module.build_cache_record(
        image_id=7,
        spatial_features=[[0.0]],
        class_logits=[[0.1, 0.9]],
        predicted_labels=[1],
        scores=[0.9],
        boxes=[[0.0, 0.0, 10.0, 10.0]],
        image_size=[480, 480],
        action_targets=[2],
        move_targets=[1],
    )
    encoded = " ".join(record).lower()
    assert record["image_id"] == 7
    assert set(record) == {"image_id", "spatial_features", "class_logits", "predicted_labels", "scores", "boxes", "image_size", "action_targets", "move_targets"}
    assert torch.is_tensor(record["spatial_features"])
    assert record["spatial_features"].dtype == torch.float16
    for forbidden in ("gt", "ground_truth", "iou", "candidate_quality", "local_value", "matched"):
        assert forbidden not in encoded


def test_tensor_cache_round_trip_preserves_compact_spatial_features(tmp_path: Path) -> None:
    module = _load_module()
    record = module.build_cache_record(
        image_id=7,
        spatial_features=torch.ones(2, 4, 3, 3, dtype=torch.float16),
        class_logits=torch.ones(2, 3),
        predicted_labels=torch.tensor([1, 2]),
        scores=torch.tensor([0.9, 0.8]),
        boxes=torch.ones(2, 4),
        image_size=(480, 480),
        action_targets=torch.tensor([2, 0]),
        move_targets=torch.tensor([1, 0]),
    )
    path = tmp_path / "oracle_cache.pt"
    module._write_cache(path, [record])
    loaded = module._read_cache(path)

    assert loaded[0]["spatial_features"].dtype == torch.float16
    assert torch.equal(loaded[0]["action_targets"], torch.tensor([2, 0]))

    storage = module.summarize_cache_storage(loaded)
    assert storage["total_proposals"] == 2
    assert storage["spatial_storage_bytes"] == 2 * 4 * 3 * 3 * 2
    assert storage["mean_proposals_per_image"] == pytest.approx(2.0)


def test_cache_metadata_must_match_locked_inputs() -> None:
    module = _load_module()
    expected = {
        "checkpoint_sha256": "checkpoint",
        "annotation_sha256": "annotation",
        "config_sha256": "config",
        "train_manifest": {"count": 2, "image_ids_sha256": "split"},
        "source_m0_sha256": "m0",
        "code_commit": "commit",
        "cache_schema_version": 1,
    }
    module.validate_cache_metadata(dict(expected), expected)
    stale = dict(expected)
    stale["checkpoint_sha256"] = "old"
    with pytest.raises(ValueError, match="cache metadata"):
        module.validate_cache_metadata(stale, expected)


def test_policy_spatial_quantization_matches_cache_and_validation_precision() -> None:
    module = _load_module()
    spatial = torch.tensor([0.123456, -0.987654], dtype=torch.float32)
    quantized = module.quantize_policy_spatial_features(spatial)

    assert quantized.dtype == torch.float32
    assert torch.equal(quantized, spatial.to(torch.float16).to(torch.float32))


def test_experiment_scope_distinguishes_partial_train_from_full_run() -> None:
    module = _load_module()
    assert module.experiment_scopes(None, None) == {
        "experiment_scope": "full",
        "train_scope": "full_train",
        "eval_scope": "full_val",
    }
    assert module.experiment_scopes(8, None)["experiment_scope"] == "smoke"
    assert module.experiment_scopes(8, None)["train_scope"] == "limited_train"
    assert module.experiment_scopes(8, None)["eval_scope"] == "full_val"


def test_training_metric_aggregation_keeps_action_and_gate_diagnostics() -> None:
    module = _load_module()
    summary = module.aggregate_training_metrics(
        [
            {
                "loss_total": 3.0,
                "loss_action": 2.0,
                "loss_move": 1.0,
                "action_accuracy": 0.25,
                "move_accuracy": 0.75,
                "action_positive_accuracy": 0.5,
            },
            {
                "loss_total": 1.0,
                "loss_action": 0.5,
                "loss_move": 0.5,
                "action_accuracy": 0.75,
                "move_accuracy": 0.25,
                "action_positive_accuracy": 1.0,
            },
        ]
    )

    assert summary == {
        "loss_total": pytest.approx(2.0),
        "loss_action": pytest.approx(1.25),
        "loss_move": pytest.approx(0.75),
        "action_accuracy": pytest.approx(0.5),
        "move_accuracy": pytest.approx(0.5),
        "action_positive_accuracy": pytest.approx(0.75),
    }


def test_policy_output_diagnostics_separate_gate_margin_and_budget() -> None:
    module = _load_module()
    output = SimpleNamespace(
        action_logits=torch.tensor([[0.0, 2.0, 1.0], [0.0, -1.0, -2.0], [0.0, 3.0, 1.0]]),
        move_logits=torch.tensor([1.0, 1.0, -1.0]),
    )
    selection = SimpleNamespace(
        selected_mask=torch.tensor([True, False, False]),
        box_delta=torch.tensor([[0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
    )
    summary = module.policy_output_diagnostics(
        output,
        selection,
        observable_mask=torch.tensor([True, True, True]),
        move_threshold=0.0,
        action_energy_scale=0.04,
    )

    assert summary["observable_count"] == 3
    assert summary["move_gate_positive"] == 2
    assert summary["action_margin_positive"] == 2
    assert summary["eligible_before_budget"] == 1
    assert summary["selected_count"] == 1
    assert summary["selected_energy_sum"] == pytest.approx(0.25)
    assert summary["move_logit_mean"] == pytest.approx(1.0 / 3.0)
    assert summary["action_margin_mean"] == pytest.approx(4.0 / 3.0)


def test_beam_labels_map_to_local_candidate_indices_and_identity_is_observable() -> None:
    module = _load_module()
    pool = [
        {"action_id": "p0002_c01", "proposal_index": 2, "candidate_index": 1},
        {"action_id": "p0000_c03", "proposal_index": 0, "candidate_index": 3},
        {"action_id": "p0001_c02", "proposal_index": 1, "candidate_index": 2},
    ]
    result = module.build_oracle_supervision(pool, ["p0002_c01", "p0000_c03"], proposal_count=4)
    assert result["action_targets"] == [3, 2, 1, 0]
    assert result["move_targets"] == [1, 0, 1, 0]
    assert result["candidate_indices"] == [3, 2, 1, 0]


def test_eval_contract_is_gt_free_and_uses_none_targets_for_extraction() -> None:
    _load_module()
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "extract_proposal_action_batch"]
    assert calls, "runner must call extract_proposal_action_batch"
    assert any(any(keyword.arg == "targets" and isinstance(keyword.value, ast.Constant) and keyword.value.value is None for keyword in call.keywords) for call in calls)
    eval_start = source.index("def evaluate_validation")
    eval_source = source[eval_start:]
    assert "targets=None" in eval_source
    assert "target_for_metrics" in eval_source
    assert "select_set_policy_actions" in eval_source


def test_cli_rejects_limited_full_validation_and_full_train() -> None:
    module = _load_module()
    args = module.parse_args(["--limit-train", "2", "--limit-val", "2", "--require-full-validation"])
    with pytest.raises(ValueError, match="full-validation"):
        module.validate_cli_args(args)

    args = module.parse_args(["--limit-train", "2", "--require-full-validation"])
    with pytest.raises(ValueError, match="full-train|limit-train"):
        module.validate_cli_args(args)


def test_launcher_is_gpu2_only_memory_gated_and_uses_m1_root_and_run_name() -> None:
    assert LAUNCHER.exists(), "M1 remote launcher has not been implemented"
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'ROOT="/home/ps/lzz/manifold-detection-energy-transport"' in source
    assert "/home/ps/lzz/RLimage" not in source
    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "memory.free" in source
    assert "free_mb <= 8192" in source
    assert "nwpu_m1_set_policy_s42_fulltrain_fullval" in source
    assert "det.energy.set_policy.m1.001.json" in source
    assert "--require-clean-git" in source


def test_m1_gates_separate_energy_zero_from_structure_gates() -> None:
    module = _load_module()
    summary = {
        "parity": {"passed": True, "mismatched_images": 0, "max_box_abs_error": 0.0, "max_score_abs_error": 0.0},
        "gt_free_eval": True,
        "detector_deltas": {"ap75": 0.003, "ap50": -0.001, "false_positive_rate": 0.01, "num_predictions_relative": 0.04},
        "learned_vs_shuffled": {"ap75": 0.002},
        "learned_vs_energy_zero": {"ap75": -0.01},
    }
    result = module.evaluate_m1_gates(summary, json.loads(LOCKED_CONFIG.read_text(encoding="utf-8")))
    assert result["gates"]["G0_parity"] is True
    assert result["gates"]["G1_gt_free_eval"] is True
    assert result["gates"]["G2_learned_vs_identity"] is True
    assert result["gates"]["G3_learned_vs_shuffled"] is True
    assert result["gates"]["G4_energy"] is False
    assert result["all_passed"] is True
    assert result["all_with_energy_passed"] is False


def test_manifest_hash_helper_matches_locked_train_split_contract() -> None:
    module = _load_module()
    assert module.manifest_hash([3, 1, 2]) == hashlib.sha256(b"[1,2,3]").hexdigest()
