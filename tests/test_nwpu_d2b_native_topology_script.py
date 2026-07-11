from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_nwpu_d2b_native_topology.py"
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_d2b_native_topology_smoke_s42.sh"


def test_d2b_reuses_locked_delta_u_and_builds_detector_only_native_topology() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"architecture": "native_action_topology"' in source
    assert "global_delta_u_cache.pt" in source
    assert "build_train_cache(" not in source
    assert "targets=None" in source
    assert "native_topology" in source
    assert "cache_alignment" in source
    assert 'max_cache_alignment_abs_error=float(config["gates"]["max_cache_alignment_abs_error"])' in source


def test_d2b_keeps_topology_control_shuffled_during_evaluation() -> None:
    source = (ROOT / "scripts" / "train_nwpu_c3_global_delta_u.py").read_text(encoding="utf-8")
    assert 'arm == "topology_shuffle"' in source
    assert 'forward_kwargs["topology_shuffle_seed"]' in source
    assert "evaluation_image_index" in source


def test_d2b_failure_moves_to_d3_without_capacity_expansion() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "d2b_native_topology_frozen_move_d3" in source
    assert "larger_train_confirmation_allowed" in source
    assert 'delta75 > float(required["min_ap75_delta_vs_identity_exclusive"])' in source


def test_d2b_launcher_enforces_gpu2_reserve() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "estimated_peak_mib=7680" in source
    assert "free_mb - estimated_peak_mib <= 8192" in source
    assert "/home/ps/lzz/RLimage" not in source
