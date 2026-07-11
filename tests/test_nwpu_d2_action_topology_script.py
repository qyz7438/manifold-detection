from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_nwpu_d2_action_topology.py"
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_d2_action_topology_smoke_s42.sh"


def test_d2_uses_action_topology_and_dedicated_shuffle_control() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"architecture": "action_topology"' in source
    assert '"local_full", "topology_shuffle", "utility_shuffle"' in source
    assert "global_delta_u_cache.pt" in source
    assert "build_train_cache(" not in source
    assert "topology_shuffle_seed" in source


def test_d2_gate_failure_moves_to_action_space_not_more_topology_tuning() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'delta75 > float(required["min_ap75_delta_vs_identity_exclusive"])' in source
    assert 'learned["ap75"] > evaluation["metrics"][arm]["ap75"]' in source
    assert "d2_action_topology_frozen_move_d3" in source
    assert "larger_train_confirmation_allowed" in source


def test_d2_launcher_enforces_estimated_gpu2_reserve() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "estimated_peak_mib=7680" in source
    assert "free_mb - estimated_peak_mib <= 8192" in source
    assert "/home/ps/lzz/RLimage" not in source
