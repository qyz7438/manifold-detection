from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_nwpu_c2a_action_margin.py"


def test_c2a_is_checkpoint_only_three_arm_native_eval() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "include_action_margin_only=True" in source
    assert '"local_full", "feature_shuffle", "utility_shuffle"' in source
    assert "targets=None" not in source  # inherited locked M1 evaluator owns extraction
    assert "train_policy(" not in source
    assert "CUDA_VISIBLE_DEVICES" not in source
