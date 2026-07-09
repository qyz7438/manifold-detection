from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_action_transport_directions.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("action_direction_analysis", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mode(ap75: float) -> dict:
    return {"metrics": {"ap75": ap75}}


def test_scale_and_action_run_parsing() -> None:
    module = _load_module()

    assert module.parse_scales("1,.1,0,.1") == [0.0, 0.1, 1.0]
    assert module.parse_action_run("preserve=runs/example") == (
        "preserve",
        Path("runs/example"),
    )
    with pytest.raises(ValueError):
        module.parse_scales("0,.5")
    with pytest.raises(ValueError):
        module.parse_action_run("missing-separator")


def test_direction_decision_requires_gain_and_same_scale_shuffle_gap() -> None:
    module = _load_module()
    modes = {
        "learned_s0": _mode(0.30),
        "learned_s0p1": _mode(0.304),
        "permuted_s0p1": _mode(0.303),
        "oracle_accept_s0p1": _mode(0.31),
        "learned_s1": _mode(0.29),
        "permuted_s1": _mode(0.28),
        "oracle_accept_s1": _mode(0.32),
        "oracle_mid_preserve": _mode(0.35),
    }

    result = module.summarize_direction_decision(modes, scales=[0.0, 0.1, 1.0])

    assert result["directional_signal"] is False
    assert result["oracle_mid_headroom"] == pytest.approx(0.05)
    assert result["benefit_gate_has_upper_bound"] is True


def test_direction_decision_accepts_aligned_small_step() -> None:
    module = _load_module()
    modes = {
        "learned_s0": _mode(0.30),
        "learned_s0p1": _mode(0.306),
        "permuted_s0p1": _mode(0.301),
        "oracle_accept_s0p1": _mode(0.312),
        "learned_s1": _mode(0.27),
        "permuted_s1": _mode(0.26),
        "oracle_accept_s1": _mode(0.315),
        "oracle_mid_preserve": _mode(0.34),
    }

    result = module.summarize_direction_decision(modes, scales=[0.0, 0.1, 1.0])

    assert result["directional_signal"] is True
    assert result["best_learned_scale"]["scale"] == 0.1
