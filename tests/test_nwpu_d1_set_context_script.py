from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_nwpu_d1_set_context.py"
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_d1_set_context_smoke_s42.sh"


def test_d1_changes_only_representation_and_reuses_locked_cache() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "global_delta_u_cache.pt" in source
    assert '"architecture": "set_context"' in source
    assert '"local_full", "feature_shuffle", "utility_shuffle"' in source
    assert "build_train_cache(" not in source
    assert "same_initialization_seed" in source


def test_d1_gate_failure_prevents_train_size_expansion() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'delta75 > float(required["min_ap75_delta_vs_identity_exclusive"])' in source
    assert 'learned["ap75"] > evaluation["metrics"][arm]["ap75"]' in source
    assert "d1_set_context_frozen" in source
    assert "larger_train_confirmation_allowed" in source


def test_d1_launcher_is_gpu2_only_and_memory_gated() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "free_mb <= 8192" in source
    assert "/home/ps/lzz/RLimage" not in source
