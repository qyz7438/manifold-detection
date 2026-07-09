from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_strong_native_action_matrix.py"


def _load_module():
    assert SCRIPT.exists()
    spec = importlib.util.spec_from_file_location("analyze_strong_native_action_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(epoch: int, ap50: float, ap75: float, ece: float = 0.05) -> dict:
    return {
        "epoch": epoch,
        "val_ap50": ap50,
        "val_ap75": ap75,
        "val_precision": 0.5,
        "val_recall": 0.6,
        "val_false_positive_rate": 0.5,
        "val_ece": ece,
        "val_num_predictions": 100,
    }


def test_matrix_analysis_uses_strong_selected_checkpoint_as_common_baseline() -> None:
    module = _load_module()
    strong = {
        "completed": True,
        "best_epoch": 2,
        "history": [_row(1, 0.60, 0.30), _row(2, 0.62, 0.34), _row(3, 0.63, 0.32)],
    }
    full = {"completed": True, "history": [_row(1, 0.625, 0.345), _row(2, 0.63, 0.35)]}
    box = {
        "completed": True,
        "default_metrics": {"ap50": 0.62, "ap75": 0.34},
        "history": [_row(1, 0.621, 0.346), _row(2, 0.622, 0.349)],
    }
    preserve = {
        "completed": True,
        "default_metrics": {"ap50": 0.62, "ap75": 0.34},
        "history": [_row(1, 0.623, 0.352), _row(2, 0.624, 0.358)],
    }
    parity = {
        "completed": True,
        "aggregate_zero_action_parity": {"passed": True},
        "strict_zero_action_parity": {"passed": True, "mismatched_images": 0},
    }

    result = module.analyze_matrix(strong, full, box, preserve, parity)

    assert result["baseline"]["epoch"] == 2
    assert result["baseline"]["ap75"] == 0.34
    assert result["runs"]["full_control"]["best_ap75_delta"] == 0.01
    assert result["runs"]["boxonly"]["best_ap75_delta"] == 0.009
    assert result["runs"]["preserve2"]["best_ap75_delta"] == 0.018
    assert result["decision"]["parity_passed"] is True
    assert result["decision"]["preserve_beats_full_by_ap75"] == 0.008
