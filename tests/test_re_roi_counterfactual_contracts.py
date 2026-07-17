"""Contract tests for the re-ROI counterfactual evidence protocol.

These tests lock down design decisions from
``docs/re_roi_counterfactual_evidence_protocol.md`` before any cache or model
work begins.
"""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "docs" / "re_roi_counterfactual_evidence_protocol.md"


def test_protocol_document_exists_and_states_researcher_adaptive() -> None:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "## Status" in text
    assert "researcher-adaptive" in text.lower()
    assert "No detector validation read" in text


def test_protocol_declares_no_revival_of_frozen_lines() -> None:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "energy_transport.native_actions.c1_e1" in text
    assert "energy_transport.dense_absolute_endpoint" in text
    assert "energy_transport.local_delta_q" in text
    assert "energy_transport.residual_content_protocol" in text


def test_action_family_is_frozen_to_eight_non_identity_primitives() -> None:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    family_section = text.split("## Action Family")[1].split("## ")[0]
    primitives = [
        "score_down",
        "score_up",
        "translate_left",
        "translate_right",
        "translate_up",
        "translate_down",
        "scale_down",
        "scale_up",
        "drop",
    ]
    for primitive in primitives:
        assert primitive in family_section
    assert family_section.count("step:") >= 3
    assert "identity_permutation" in family_section


def test_q_teacher_scalarization_is_locked_before_fitting() -> None:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    section = text.split("## Teacher Utility")[1].split("## ")[0]
    assert "Q_teacher(S, a) =" in section
    assert "delta_tp75" in section
    assert "delta_fp75" in section
    assert "delta_fp50" in section
    assert "delta_duplicate" in section
    assert "action_energy" in section
    assert "no weight may be selected from tune" in section


def test_y_teacher_contains_structured_results_not_ap75_delta() -> None:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    section = text.split("## Teacher Structured Result")[1].split("## ")[0]
    assert "Y_teacher(S, a)" in section
    assert "native_changed" in section
    assert "delta_tp75" in section
    assert "delta_fp75" in section
    assert "delta_duplicate" in section
    # AP75 itself is not a per-action target.
    assert "delta_ap75" not in section


def test_native_changed_is_not_a_learned_target() -> None:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "native_changed" in text
    assert "deterministic" in text
    assert "not as a learned target" in text


def test_primary_information_gain_is_c_versus_b() -> None:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    arms = text.split("## Arms")[1].split("## ")[0]
    assert "A. `family_prior`" in arms
    assert "B. `static_roi`" in arms
    assert "C. `re_roi`" in arms
    assert "D. `re_roi_bundle_shuffle`" in arms
    assert "primary information-gain test is **C versus B**" in arms


def test_re_roi_does_not_rerun_full_backbone() -> None:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    section = text.split("## Re-ROI Evidence")[1].split("## ")[0]
    assert "Do not rerun the full backbone" in section


def test_gates_include_bundle_integrity_and_calibration() -> None:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    section = text.split("## Locked Gates")[1].split("## ")[0]
    assert "re_roi_gain" in section
    assert "bundle_integrity" in section
    assert "calibration" in section
    assert "split-conformal LCB" in section


def test_branch_rule_freezes_on_any_failure() -> None:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    section = text.split("## Branch Rule")[1]
    assert "If any gate fails, freeze" in section
    assert "Do not add more actions" in section
