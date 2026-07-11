from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_nwpu_c3b_balanced.py"
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_c3b_balanced_smoke_s42.sh"


def test_c3b_reuses_locked_cache_and_runs_three_equal_capacity_arms() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "global_delta_u_cache.pt" in source
    assert '"local_full", "feature_shuffle", "utility_shuffle"' in source
    assert "balanced_margin" in source
    assert "build_train_cache(" not in source
    assert "same_initialization_seed" in source


def test_c3b_gates_are_strict_and_this_is_final_same_cache_attempt() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'delta75 > float(required["min_ap75_delta_vs_identity_exclusive"])' in source
    assert 'learned["ap75"] > evaluation["metrics"][arm]["ap75"]' in source
    assert "current_route_frozen" in source


def test_c3b_launcher_is_gpu2_only_and_memory_gated() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "free_mb <= 8192" in source
    assert "/home/ps/lzz/RLimage" not in source
