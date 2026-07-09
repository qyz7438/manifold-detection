from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_nwpu_finetune_curve.py"


def _load_module():
    assert SCRIPT.exists()
    spec = importlib.util.spec_from_file_location("analyze_nwpu_finetune_curve", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(epoch: int, ap50: float, ap75: float) -> dict:
    return {
        "epoch": epoch,
        "val_ap50": ap50,
        "val_ap75": ap75,
        "val_precision": 0.5,
        "val_recall": 0.6,
        "val_false_positive_rate": 0.5,
        "val_ece": 0.03,
        "val_num_predictions": 100,
    }


def test_detect_saturation_after_three_epochs_without_new_best() -> None:
    module = _load_module()
    history = [
        _row(1, 0.55, 0.25),
        _row(2, 0.58, 0.30),
        _row(3, 0.581, 0.302),
        _row(4, 0.579, 0.299),
        _row(5, 0.580, 0.301),
    ]

    status = module.detect_saturation(history, window=3, ap50_gain=0.003, ap75_gain=0.005)

    assert status["saturated"] is True
    assert status["epochs_checked"] == [3, 4, 5]


def test_detect_saturation_rejects_late_material_gain() -> None:
    module = _load_module()
    history = [
        _row(1, 0.55, 0.25),
        _row(2, 0.56, 0.27),
        _row(3, 0.561, 0.271),
        _row(4, 0.562, 0.272),
        _row(5, 0.570, 0.281),
    ]

    status = module.detect_saturation(history, window=3, ap50_gain=0.003, ap75_gain=0.005)

    assert status["saturated"] is False
    assert status["latest_material_gain"] is True


def test_summarize_curve_reports_best_final_and_source_deltas() -> None:
    module = _load_module()
    history = [_row(1, 0.55, 0.25), _row(2, 0.57, 0.31), _row(3, 0.56, 0.29)]
    run = {"completed": True, "history": history}
    baseline = {"ap50": 0.50, "ap75": 0.20}

    summary = module.summarize_curve(run, baseline=baseline)

    assert summary["completed"] is True
    assert summary["epochs_observed"] == 3
    assert summary["best_ap75"]["epoch"] == 2
    assert summary["best_ap75"]["value"] == 0.31
    assert summary["final"]["ap75"] == 0.29
    assert summary["best_final_ap75_gap"] == 0.02
    assert summary["delta_vs_source"]["final_ap75"] == 0.09
    assert summary["delta_vs_source"]["best_ap75"] == 0.11


def test_load_run_state_uses_progress_file_before_completion(tmp_path: Path) -> None:
    module = _load_module()
    progress = {"completed": False, "history": [_row(1, 0.55, 0.25)]}
    (tmp_path / "metrics_history.json").write_text(json.dumps(progress), encoding="utf-8")

    state = module.load_run_state(tmp_path)

    assert state == progress
