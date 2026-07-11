from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_nwpu_c3a_forced_top1.py"
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_c3a_forced_top1_smoke_s42.sh"


def test_c3a_reuses_three_checkpoints_without_training_and_forces_one_action() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"local_full", "feature_shuffle", "utility_shuffle"' in source
    assert "allow_noop=False" in source
    assert "train_arm(" not in source
    assert "train_policy(" not in source


def test_c3a_launcher_is_gpu2_only_and_memory_gated() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "free_mb <= 8192" in source
    assert "/home/ps/lzz/RLimage" not in source
