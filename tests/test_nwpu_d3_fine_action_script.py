from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_nwpu_d3_fine_action.py"
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_d3_fine_action_smoke_s42.sh"


def test_d3_changes_only_locked_action_magnitude_and_rebuilds_delta_u() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"step": 0.02' in source
    assert "build_train_cache(" in source
    assert '"local_full", "feature_shuffle", "utility_shuffle"' in source
    assert '"loss_type": "balanced_margin"' in source
    assert "native_decode_threshold_class_matching_nms" in source


def test_d3_rejects_step_sweep_after_failure() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "fine_action_signal_detected" in source
    assert "fixed_grid_action_branch_frozen" in source
    assert 'delta75 > float(required["min_ap75_delta_vs_identity_exclusive"])' in source


def test_d3_launcher_enforces_gpu2_reserve_and_provenance() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "estimated_peak_mib=7680" in source
    assert "free_mb - estimated_peak_mib <= 8192" in source
    assert "git_commit" in source
    assert "/home/ps/lzz/RLimage" not in source
