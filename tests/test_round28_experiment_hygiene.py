from __future__ import annotations

import hashlib
from pathlib import Path

from scripts import round28_train_eval


CONVERGENCE_LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "experiments"
    / "run_nwpu_full_convergence_control.sh"
)


def test_epoch_metric_row_keeps_decision_metrics() -> None:
    assert hasattr(round28_train_eval, "_epoch_metric_row")

    row = round28_train_eval._epoch_metric_row(
        epoch=3,
        train_loss=0.25,
        metrics={
            "ap50": 0.60,
            "ap75": 0.40,
            "precision": 0.70,
            "recall": 0.50,
            "false_positive_rate": 0.30,
            "ece": 0.02,
            "num_predictions": 123,
            "per_size": {"small": {"ap75": 0.10}},
        },
    )

    assert row == {
        "epoch": 3,
        "train_loss": 0.25,
        "val_ap50": 0.60,
        "val_ap75": 0.40,
        "val_precision": 0.70,
        "val_recall": 0.50,
        "val_false_positive_rate": 0.30,
        "val_ece": 0.02,
        "val_num_predictions": 123,
    }


def test_selection_value_supports_ap75() -> None:
    assert hasattr(round28_train_eval, "_selection_value")
    metrics = {"ap50": 0.61, "ap75": 0.43}

    assert round28_train_eval._selection_value(metrics, "ap75") == 0.43


def test_checkpoint_provenance_records_resolved_path_and_hash(tmp_path: Path) -> None:
    assert hasattr(round28_train_eval, "_checkpoint_provenance")
    checkpoint = tmp_path / "source.pth"
    checkpoint.write_bytes(b"source-checkpoint")

    provenance = round28_train_eval._checkpoint_provenance(checkpoint)

    assert provenance == {
        "path": str(checkpoint.resolve()),
        "sha256": hashlib.sha256(b"source-checkpoint").hexdigest(),
    }


def test_convergence_launcher_is_gpu2_and_memory_gated() -> None:
    assert CONVERGENCE_LAUNCHER.exists()
    launcher = CONVERGENCE_LAUNCHER.read_text(encoding="utf-8")

    assert 'GPU_ID="${GPU_ID:-2}"' in launcher
    assert 'MIN_FREE_MB="${MIN_FREE_MB:-8192}"' in launcher
    assert '[[ "${GPU_ID}" != "2" ]]' in launcher
    assert '"${free_mb}" -gt "${MIN_FREE_MB}"' in launcher
    assert "pgrep" not in launcher
    assert '--selection-metric ap75' in launcher
