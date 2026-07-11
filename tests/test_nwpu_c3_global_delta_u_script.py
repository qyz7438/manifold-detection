from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import pytest

from spectral_detection_posttrain.methods.energy_transport.set_search import SetOutcome


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_nwpu_c3_global_delta_u.py"
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_c3_global_delta_u_smoke_s42.sh"


def _module():
    spec = importlib.util.spec_from_file_location("c3", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_locks_detector_only_global_endpoint_and_unified_noop() -> None:
    module = _module()
    config = module.load_locked_config()
    assert config["candidate_pool"]["source"] == "detector_visible_proposals_only"
    assert config["candidate_pool"]["candidate_count"] == 9
    assert config["candidate_pool"]["max_actions_per_image"] == 1
    assert config["utility"]["endpoint"] == "native_decode_threshold_class_matching_nms"
    assert config["utility"]["gt_scope"] == "train_utility_only_never_candidate_filter"
    assert config["policy"]["objective"] == "single_image_level_cross_entropy"
    assert config["dataset"]["smoke_train_image_ids_sha256"] == "37912bbb682f370f551c80aeae8e6dd600122fbec6a2450360ec2fdd54a81d69"
    assert config["dataset"]["smoke_val_image_ids_sha256"] == "bb6bb1764892bcc5b496c8ccbfca60c3d1e745a2901349429af3927a5770f85c"


def test_whole_image_utility_uses_locked_config_coefficients() -> None:
    module = _module()
    utility = module.whole_image_utility(
        SetOutcome(tp75=2, fp75=1, fp50=3),
        module.load_locked_config()["utility"],
        action_energy=0.0625,
        action_count=1,
    )
    assert utility == pytest.approx(2.0 - 0.25 - 0.30 - 0.05 * 0.0625 - 0.02)


def test_singleton_enumerator_visits_only_observable_non_noop_actions() -> None:
    module = _module()
    calls = []

    def evaluate(proposal_index: int, candidate_index: int) -> float:
        calls.append((proposal_index, candidate_index))
        return float(10 * proposal_index + candidate_index)

    delta_u = module.enumerate_singleton_delta_u(
        proposal_count=3,
        candidate_count=4,
        observable_mask=torch.tensor([True, False, True]),
        identity_utility=1.0,
        evaluate_utility=evaluate,
    )
    assert delta_u.device == torch.device("cpu")
    assert calls == [(0, 1), (0, 2), (0, 3), (2, 1), (2, 2), (2, 3)]
    assert torch.equal(delta_u[:, 0], torch.zeros(3))
    assert torch.isneginf(delta_u[1, 1:]).all()
    assert delta_u[2, 3].item() == 22.0


def test_controls_shuffle_only_the_intended_observable_alignment() -> None:
    module = _module()
    record = {
        "image_id": 3,
        "spatial_features": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "class_logits": torch.tensor([[0.0, 1.0]] * 3),
        "predicted_labels": torch.ones(3, dtype=torch.long),
        "scores": torch.ones(3),
        "boxes": torch.ones(3, 4),
        "image_size": (480, 480),
        "observable_mask": torch.tensor([True, True, True]),
        "delta_u": torch.tensor([[0.0, 1.0, 2.0], [0.0, 3.0, 4.0], [0.0, 5.0, 6.0]]),
    }
    feature = module.make_control_records([record], "feature_shuffle", seed=142)[0]
    utility = module.make_control_records([record], "utility_shuffle", seed=242)[0]
    assert not torch.equal(feature["spatial_features"], record["spatial_features"])
    assert torch.equal(feature["delta_u"], record["delta_u"])
    assert torch.equal(utility["spatial_features"], record["spatial_features"])
    assert sorted(utility["delta_u"][:, 1:].reshape(-1).tolist()) == sorted(record["delta_u"][:, 1:].reshape(-1).tolist())
    assert torch.equal(utility["delta_u"][:, 0], torch.zeros(3))


def test_cache_reuse_requires_exact_provenance(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "cache.pt"
    metadata = {"config_sha256": "config", "checkpoint_sha256": "checkpoint", "split": {"count": 1}}
    module._write_cache(path, [{"image_id": 1}], metadata)
    assert module._read_cache(path, metadata)[0]["image_id"] == 1
    stale = dict(metadata)
    stale["checkpoint_sha256"] = "other"
    with pytest.raises(ValueError, match="provenance"):
        module._read_cache(path, stale)


def test_source_and_launcher_lock_native_candidate_boundary_and_gpu2() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    cache_source = source[source.index("def build_train_cache") : source.index("def train_arm")]
    assert "targets=None" in cache_source
    assert "action_batch_to_predictions" in cache_source
    assert "set_outcome_from_prediction" in cache_source
    assert "build_candidate_quality_targets" not in cache_source
    assert "CUDA_VISIBLE_DEVICES=2" in launcher
    assert "free_mb <= 8192" in launcher
    assert "--limit-train 32" in launcher
    assert "--limit-val 32" in launcher
    assert "/home/ps/lzz/RLimage" not in launcher
    args = _module().parse_args([])
    assert args.limit_train is None and args.limit_val is None
