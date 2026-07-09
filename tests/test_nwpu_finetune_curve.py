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


def test_detect_saturation_requires_robust_late_plateau() -> None:
    module = _load_module()
    history = []
    for epoch in range(1, 17):
        if epoch <= 4:
            history.append(_row(epoch, 0.54 + 0.01 * epoch, 0.24 + 0.015 * epoch))
        else:
            jitter = (epoch % 3 - 1) * 0.0005
            history.append(_row(epoch, 0.580 + jitter, 0.300 + jitter))

    status = module.detect_saturation(history, window=3, ap50_gain=0.003, ap75_gain=0.005)

    assert status["saturated"] is True
    assert status["status"] == "SATURATED"
    assert status["epochs_checked"] == list(range(5, 17))


def test_detect_saturation_rejects_late_material_gain() -> None:
    module = _load_module()
    history = [_row(epoch, 0.56, 0.28) for epoch in range(1, 16)]
    history.append(_row(16, 0.57, 0.291))

    status = module.detect_saturation(history, window=3, ap50_gain=0.003, ap75_gain=0.005)

    assert status["saturated"] is False
    assert status["status"] == "CONTINUE"
    assert status["latest_material_gain"] is True


def test_detect_saturation_never_stops_before_epoch_16() -> None:
    module = _load_module()
    history = [_row(epoch, 0.56, 0.28) for epoch in range(1, 10)]

    status = module.detect_saturation(history)

    assert status["saturated"] is False
    assert status["status"] == "INSUFFICIENT_HISTORY"


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
    assert summary["robust_best_ap75"] == {"epoch": 2, "value": 0.29, "window": 3}
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
