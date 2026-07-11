from pathlib import Path

import importlib.util
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_nwpu_e1_post_nms_suppress.py"
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_e1_post_nms_suppress_smoke_s42.sh"


def _module():
    spec = importlib.util.spec_from_file_location("e1_script_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e1_uses_native_post_nms_kept_set_without_gt_filtering() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "build_post_nms_detection_features" in source
    assert "suppress_detection" in source
    assert '"candidate_filter_uses_gt": False' in source
    assert '"local_full", "feature_shuffle", "utility_shuffle"' in source
    assert "set_outcome_from_prediction" in source


def test_e1_gates_detector_metrics_and_controls() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'delta75 > float(required["min_ap75_delta_vs_identity_exclusive"])' in source
    assert 'fpr_delta <= float(required["max_fpr_delta"])' in source
    assert "post_nms_suppress_signal_detected" in source
    assert "post_nms_suppress_frozen" in source
    assert 'row["accuracy"]' in source
    assert 'row["correct"]' not in source


def test_e1_launcher_enforces_gpu2_reserve_and_provenance() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "estimated_peak_mib=6144" in source
    assert "free_mb - estimated_peak_mib <= 8192" in source
    assert "git_commit" in source
    assert "/home/ps/lzz/RLimage" not in source


def test_e1_controls_break_only_the_intended_alignment() -> None:
    record = {
        "features": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "observable_mask": torch.ones(3, dtype=torch.bool),
        "delta_u": torch.tensor([[0.0, -1.0], [0.0, 2.0], [0.0, 3.0]]),
    }
    feature = _module().make_control_records([record], "feature_shuffle", 142)[0]
    utility = _module().make_control_records([record], "utility_shuffle", 242)[0]
    assert not torch.equal(feature["features"], record["features"])
    assert torch.equal(feature["delta_u"], record["delta_u"])
    assert torch.equal(utility["features"], record["features"])
    assert sorted(utility["delta_u"][:, 1].tolist()) == sorted(record["delta_u"][:, 1].tolist())
