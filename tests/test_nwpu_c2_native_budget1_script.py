from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_nwpu_c2_native_budget1.py"
CONFIG = ROOT / "spectral_detection_posttrain" / "configs" / "versions" / "det.energy.native_budget1.c2.001.json"
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_c2_native_budget1_smoke_s42.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("train_nwpu_c2_native_budget1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_locked_config_uses_c1_actions_budget_one_and_three_controls() -> None:
    module = _load_module()
    config = module.load_locked_config()
    assert config == json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["candidate_pool"]["candidate_count"] == 9
    assert config["candidate_pool"]["action_budget"] == 1
    assert config["arms"] == ["local_full", "feature_shuffle", "utility_shuffle"]
    deltas = module.candidate_deltas(config)
    assert deltas.shape == (9, 4)
    assert torch.equal(deltas[0], torch.zeros(4))
    assert torch.all(deltas[1:].count_nonzero(dim=1) == 1)


def test_budget_one_supervision_has_explicit_noop_and_at_most_one_move() -> None:
    module = _load_module()
    targets = module.build_budget_one_supervision(
        best_indices=torch.tensor([2, 3, 1, 4]),
        oracle_gain=torch.tensor([0.0, 0.03, 0.02, 0.01]),
        observable_mask=torch.tensor([True, True, True, False]),
        min_iou_gain=0.002,
    )
    assert targets["action_targets"].tolist() == [0, 3, 1, 0]
    assert targets["move_targets"].sum().item() == 1
    assert targets["move_targets"].tolist() == [0, 1, 0, 0]


def test_controls_preserve_support_and_break_the_intended_alignment() -> None:
    module = _load_module()
    record = {
        "image_id": 5,
        "spatial_features": torch.arange(3 * 2, dtype=torch.float32).reshape(3, 2),
        "class_logits": torch.tensor([[0.0, 1.0]] * 3),
        "predicted_labels": torch.ones(3, dtype=torch.long),
        "scores": torch.ones(3),
        "boxes": torch.ones(3, 4),
        "image_size": (480, 480),
        "action_targets": torch.tensor([0, 2, 0]),
        "move_targets": torch.tensor([0, 1, 0]),
    }
    feature = module.make_control_cache([record], "feature_shuffle", seed=42)[0]
    utility = module.make_control_cache([record], "utility_shuffle", seed=42)[0]
    assert torch.equal(feature["action_targets"], record["action_targets"])
    assert not torch.equal(feature["spatial_features"], record["spatial_features"])
    assert torch.equal(utility["spatial_features"], record["spatial_features"])
    assert sorted(utility["action_targets"].tolist()) == sorted(record["action_targets"].tolist())
    assert sorted(utility["move_targets"].tolist()) == sorted(record["move_targets"].tolist())


def test_launcher_is_gpu2_memory_gated_and_smoke_scoped() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "free_mb <= 8192" in source
    assert "--limit-train 32" in source
    assert "--limit-val 32" in source
    assert "/home/ps/lzz/RLimage" not in source


def test_training_extracts_detector_candidates_before_gt_matching() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    train_cache = source[source.index("def build_train_cache") : source.index("def cache_support")]
    assert "targets=None" in train_cache
    assert "rematch_roi_action_state" in train_cache
    assert "extract_proposal_action_batch(model, images_device, [train_target]" not in train_cache
