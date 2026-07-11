from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_nwpu_d4_adaptive_consensus.py"
LAUNCHER = ROOT / "scripts" / "experiments" / "run_nwpu_d4_adaptive_consensus_smoke_s42.sh"


def test_d4_builds_detector_only_graph_actions_and_native_delta_u() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"architecture": "adaptive_consensus"' in source
    assert "proposal_graph_consensus_deltas" in source
    assert "targets=None" in source
    assert "action_batch_to_predictions" in source
    assert "whole_image_utility" in source
    assert '"git_dirty": git_dirty' in source
    assert '"local_full", "feature_shuffle", "utility_shuffle"' in source


def test_d4_is_hard_stop_for_bbox_action_line() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "adaptive_consensus_signal_detected" in source
    assert "bbox_action_line_frozen" in source
    assert 'delta75 > float(required["min_ap75_delta_vs_identity_exclusive"])' in source


def test_d4_launcher_locks_gpu2_and_provenance() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "estimated_peak_mib=7680" in source
    assert "free_mb - estimated_peak_mib <= 8192" in source
    assert "git_commit" in source
    assert "/home/ps/lzz/RLimage" not in source
