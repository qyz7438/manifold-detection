from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from spectral_detection_posttrain.experiments.contracts import (
    EvaluationScope,
    ExperimentDefinition,
    ResearchStatus,
    RunnableMode,
    parse_evaluation_scope,
    parse_experiment_definition,
    validate_relative_path,
)
from spectral_detection_posttrain.experiments.research_status import (
    STATUS_TO_RUNNABLE,
    ResearchLine,
    can_authorize_new_run,
    load_research_status_registry,
    validate_runnable_for_status,
    validate_status_transition,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = (
    REPO_ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "registry"
    / "research_lines.json"
)

ACTIVE = ResearchStatus.ACTIVE
QUEUED = ResearchStatus.QUEUED
PAUSED = ResearchStatus.PAUSED
FROZEN = ResearchStatus.FROZEN
BLOCKED = ResearchStatus.BLOCKED
DIAGNOSTIC = ResearchStatus.DIAGNOSTIC
BASELINE = ResearchStatus.BASELINE
HISTORICAL = ResearchStatus.HISTORICAL
INVALIDATED = ResearchStatus.INVALIDATED
VALIDATED = ResearchStatus.VALIDATED

DECISION_POINTER = "docs/autonomous_exploration_report.md"

# Plan Section 3.1: the initial registry statuses are fixed.
EXPECTED_INITIAL_STATUSES = {
    "energy_transport.native_actions.c1_e1": "frozen",
    "energy_transport.dense_absolute_endpoint": "frozen",
    "energy_transport.local_delta_q": "frozen",
    "energy_transport.residual_content_protocol": "blocked",
    "manifold.prototype_attraction": "diagnostic",
    "manifold.intrinsic_dimension": "diagnostic",
    "manifold.dual_energy": "diagnostic",
    "signals.fft_reward": "invalidated",
    "rlvr_grpo_dpo": "historical",
    "architecture.fpn_sm_afm": "baseline",
}


def _valid_line_dict(**overrides: object) -> dict:
    line = {
        "id": "test.line",
        "status": "frozen",
        "hypothesis": "h",
        "claim_ceiling": "c",
        "evidence": [DECISION_POINTER],
        "decision_document": DECISION_POINTER,
    }
    line.update(overrides)
    return line


def _write_registry(tmp_path: Path, lines: list, **top_overrides: object) -> Path:
    payload = {"schema_version": "research_lines.v1", "lines": lines}
    payload.update(top_overrides)
    path = tmp_path / "research_lines.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def test_status_vocabulary_is_exact() -> None:
    assert len(ResearchStatus) == 10
    assert {status.value for status in ResearchStatus} == {
        "active",
        "queued",
        "paused",
        "frozen",
        "blocked",
        "diagnostic",
        "baseline",
        "historical",
        "invalidated",
        "validated",
    }


def test_runnable_vocabulary_is_exact() -> None:
    assert len(RunnableMode) == 4
    assert {mode.value for mode in RunnableMode} == {
        "disabled",
        "diagnostic_only",
        "frozen_reproduction_only",
        "authorized",
    }


def test_contract_types_are_immutable() -> None:
    line = ResearchLine(
        id="x",
        status=ACTIVE,
        hypothesis="h",
        claim_ceiling="c",
        evidence=(DECISION_POINTER,),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        line.id = "y"
    scope = EvaluationScope(kind="smoke", image_count=None, limit_train=None, limit_val=32)
    with pytest.raises(dataclasses.FrozenInstanceError):
        scope.kind = "full_val"


# ---------------------------------------------------------------------------
# Fixed status -> runnable mapping table (plan Section 3.1)
# ---------------------------------------------------------------------------


def test_status_to_runnable_mapping_table_is_fixed() -> None:
    disabled = RunnableMode.DISABLED
    diagnostic_only = RunnableMode.DIAGNOSTIC_ONLY
    frozen_repro = RunnableMode.FROZEN_REPRODUCTION_ONLY
    authorized = RunnableMode.AUTHORIZED
    expected = {
        ACTIVE: frozenset({authorized, diagnostic_only, frozen_repro}),
        QUEUED: frozenset({disabled, diagnostic_only}),
        PAUSED: frozenset({disabled, diagnostic_only}),
        FROZEN: frozenset({disabled, diagnostic_only, frozen_repro}),
        BLOCKED: frozenset({disabled, diagnostic_only}),
        DIAGNOSTIC: frozenset({disabled, diagnostic_only}),
        BASELINE: frozenset({disabled, diagnostic_only, frozen_repro}),
        HISTORICAL: frozenset({disabled, frozen_repro}),
        INVALIDATED: frozenset({disabled, diagnostic_only}),
        VALIDATED: frozenset({disabled, frozen_repro}),
    }
    assert STATUS_TO_RUNNABLE == expected
    assert set(STATUS_TO_RUNNABLE) == set(ResearchStatus)


def test_invalid_status_runnable_combination_rejected() -> None:
    validate_runnable_for_status(ACTIVE, RunnableMode.AUTHORIZED)
    validate_runnable_for_status(FROZEN, RunnableMode.FROZEN_REPRODUCTION_ONLY)
    validate_runnable_for_status(HISTORICAL, RunnableMode.DISABLED)
    with pytest.raises(ValueError):
        validate_runnable_for_status(FROZEN, RunnableMode.AUTHORIZED)
    with pytest.raises(ValueError):
        validate_runnable_for_status(HISTORICAL, RunnableMode.AUTHORIZED)
    with pytest.raises(ValueError):
        validate_runnable_for_status(HISTORICAL, RunnableMode.DIAGNOSTIC_ONLY)
    with pytest.raises(ValueError):
        validate_runnable_for_status(QUEUED, RunnableMode.AUTHORIZED)


# ---------------------------------------------------------------------------
# Transition matrix (plan Section 3.1)
# ---------------------------------------------------------------------------

MATRIX_ALLOWED = {
    ACTIVE: {QUEUED, PAUSED, FROZEN, BLOCKED, DIAGNOSTIC, VALIDATED},
    QUEUED: {ACTIVE, PAUSED, FROZEN, BLOCKED},
    PAUSED: {ACTIVE, QUEUED, FROZEN, BLOCKED},
}

LOCKED_STATUSES = {
    FROZEN,
    BLOCKED,
    INVALIDATED,
    VALIDATED,
    HISTORICAL,
    BASELINE,
    DIAGNOSTIC,
}


def test_transition_matrix_allowed_rows() -> None:
    for old, targets in MATRIX_ALLOWED.items():
        for new in targets:
            validate_status_transition(old, new, "")


def test_transition_matrix_rejects_unlisted_moves() -> None:
    rejected = [
        (ACTIVE, HISTORICAL),
        (ACTIVE, BASELINE),
        (ACTIVE, INVALIDATED),
        (QUEUED, VALIDATED),
        (QUEUED, DIAGNOSTIC),
        (QUEUED, HISTORICAL),
        (PAUSED, VALIDATED),
        (PAUSED, DIAGNOSTIC),
        (PAUSED, BASELINE),
    ]
    for old, new in rejected:
        with pytest.raises(ValueError):
            validate_status_transition(old, new, "")


def test_locked_statuses_require_superseding_decision_document() -> None:
    for old in LOCKED_STATUSES:
        with pytest.raises(ValueError, match="decision_document"):
            validate_status_transition(old, ACTIVE, "")
        validate_status_transition(old, ACTIVE, DECISION_POINTER)


def test_no_automatic_or_noop_status_change() -> None:
    for status in ResearchStatus:
        with pytest.raises(ValueError):
            validate_status_transition(status, status, DECISION_POINTER)


# ---------------------------------------------------------------------------
# ResearchLine invariants
# ---------------------------------------------------------------------------


def test_frozen_line_requires_decision_evidence() -> None:
    with pytest.raises(ValueError, match="decision_document"):
        ResearchLine(id="x", status=ResearchStatus.FROZEN, hypothesis="h")


def test_other_locked_statuses_require_decision_evidence() -> None:
    for status in (BLOCKED, INVALIDATED, VALIDATED):
        with pytest.raises(ValueError, match="decision_document"):
            ResearchLine(id="x", status=status, hypothesis="h", claim_ceiling="c")


def test_research_line_requires_hypothesis_claim_ceiling_and_evidence() -> None:
    with pytest.raises(ValueError, match="hypothesis"):
        ResearchLine(
            id="x",
            status=ACTIVE,
            hypothesis="  ",
            claim_ceiling="c",
            evidence=(DECISION_POINTER,),
        )
    with pytest.raises(ValueError, match="claim_ceiling"):
        ResearchLine(
            id="x",
            status=ACTIVE,
            hypothesis="h",
            claim_ceiling="",
            evidence=(DECISION_POINTER,),
        )
    with pytest.raises(ValueError, match="evidence"):
        ResearchLine(id="x", status=ACTIVE, hypothesis="h", claim_ceiling="c", evidence=())


def test_invalid_relative_path_rejected() -> None:
    for bad in ("../x", "/abs", "docs\\x", "C:/x", "docs/../x", "docs//x"):
        with pytest.raises(ValueError):
            validate_relative_path(bad)
        with pytest.raises(ValueError):
            ResearchLine(
                id="x",
                status=ACTIVE,
                hypothesis="h",
                claim_ceiling="c",
                evidence=(bad,),
            )
    assert validate_relative_path("docs/reports/index.md") == "docs/reports/index.md"


def test_can_authorize_new_run_only_for_active_status() -> None:
    for status in ResearchStatus:
        kwargs: dict = {}
        if status in {FROZEN, BLOCKED, INVALIDATED, VALIDATED}:
            kwargs["decision_document"] = DECISION_POINTER
        line = ResearchLine(
            id=f"test.{status.value}",
            status=status,
            hypothesis="h",
            claim_ceiling="c",
            evidence=(DECISION_POINTER,),
            **kwargs,
        )
        assert can_authorize_new_run(line) is (status is ACTIVE)


# ---------------------------------------------------------------------------
# Registry loader strictness
# ---------------------------------------------------------------------------


def test_registry_loader_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_valid_line_dict()], bogus=1)
    with pytest.raises(ValueError, match="bogus"):
        load_research_status_registry(path)


def test_registry_loader_rejects_unknown_line_key(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_valid_line_dict(bogus=1)])
    with pytest.raises(ValueError, match="bogus"):
        load_research_status_registry(path)


def test_registry_loader_rejects_missing_required_line_key(tmp_path: Path) -> None:
    line = _valid_line_dict()
    del line["claim_ceiling"]
    path = _write_registry(tmp_path, [line])
    with pytest.raises(ValueError, match="claim_ceiling"):
        load_research_status_registry(path)


def test_registry_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_valid_line_dict(), _valid_line_dict()])
    with pytest.raises(ValueError, match="duplicate"):
        load_research_status_registry(path)


def test_registry_loader_rejects_invalid_status_value(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_valid_line_dict(status="bogus")])
    with pytest.raises(ValueError, match="status"):
        load_research_status_registry(path)


def test_registry_loader_rejects_missing_evidence(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_valid_line_dict(evidence=[])])
    with pytest.raises(ValueError, match="evidence"):
        load_research_status_registry(path)


def test_registry_loader_rejects_nonexistent_evidence_file(tmp_path: Path) -> None:
    path = _write_registry(
        tmp_path, [_valid_line_dict(evidence=["docs/definitely_missing_evidence.md"])]
    )
    with pytest.raises(ValueError):
        load_research_status_registry(path)


def test_registry_loader_rejects_invalid_relative_paths(tmp_path: Path) -> None:
    for bad in ("../x", "/abs", "docs\\x"):
        path = _write_registry(tmp_path, [_valid_line_dict(evidence=[bad])])
        with pytest.raises(ValueError):
            load_research_status_registry(path)
        path = _write_registry(tmp_path, [_valid_line_dict(decision_document=bad)])
        with pytest.raises(ValueError):
            load_research_status_registry(path)


# ---------------------------------------------------------------------------
# Shipped initial registry
# ---------------------------------------------------------------------------


def test_initial_registry_matches_section_3_1_statuses() -> None:
    registry = load_research_status_registry(REGISTRY_PATH)
    actual = {line.id: line.status.value for line in registry.lines}
    assert actual == EXPECTED_INITIAL_STATUSES


def test_registry_lines_have_unique_ids_and_complete_fields() -> None:
    registry = load_research_status_registry(REGISTRY_PATH)
    ids = [line.id for line in registry.lines]
    assert len(ids) == len(set(ids))
    for line in registry.lines:
        assert line.hypothesis.strip()
        assert line.claim_ceiling.strip()
        assert line.evidence
        for pointer in line.evidence:
            assert (REPO_ROOT / pointer).is_file(), pointer
        assert line.decision_document.strip()
        assert (REPO_ROOT / line.decision_document).is_file()


def test_initial_registry_authorizes_no_experiment_expansion() -> None:
    registry = load_research_status_registry(REGISTRY_PATH)
    assert not [line for line in registry.lines if line.status in {"active", "queued"}]
    assert not [line for line in registry.lines if line.status == "validated"]


def test_no_registry_line_can_authorize_new_run() -> None:
    registry = load_research_status_registry(REGISTRY_PATH)
    assert not [line for line in registry.lines if can_authorize_new_run(line)]


def test_uncommitted_context_is_never_required_evidence() -> None:
    registry = load_research_status_registry(REGISTRY_PATH)
    untracked = "docs/energy_transport_vs_fpn_sm_analysis.md"
    for line in registry.lines:
        assert untracked not in line.evidence
        assert line.decision_document != untracked
        for pointer in line.uncommitted_context:
            validate_relative_path(pointer)


# ---------------------------------------------------------------------------
# contracts.py strict parsing
# ---------------------------------------------------------------------------


def test_evaluation_scope_kind_vocabulary() -> None:
    kinds = {
        "smoke",
        "limited",
        "train_only",
        "cache_only",
        "detector_unseen",
        "full_val",
        "synthetic",
        "oracle",
        "limited_unknown",
    }
    for kind in kinds:
        raw = {"kind": kind, "image_count": None, "limit_train": None, "limit_val": None}
        if kind in {"smoke", "limited"}:
            # contracts v2: smoke/limited require a positive explicit limit
            raw["limit_val"] = 32
        scope = parse_evaluation_scope(raw)
        assert scope.kind == kind
    with pytest.raises(ValueError):
        parse_evaluation_scope(
            {"kind": "bogus", "image_count": None, "limit_train": None, "limit_val": None}
        )


def test_evaluation_scope_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="bogus"):
        parse_evaluation_scope(
            {
                "kind": "smoke",
                "image_count": None,
                "limit_train": None,
                "limit_val": 32,
                "bogus": 1,
            }
        )


def test_evaluation_scope_rejects_bad_counts() -> None:
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError):
            EvaluationScope(kind="smoke", image_count=bad, limit_train=None, limit_val=32)


def _experiment_definition_dict(**overrides: object) -> dict:
    data = {
        "id": "det.energy.global_delta_u.c3b.balanced.001",
        "research_line": "energy_transport.native_actions.c1_e1",
        "capability": "detection.energy_transport.c3b",
        "config_path": "spectral_detection_posttrain/configs/versions/det.energy.global_delta_u.c3b.balanced.001.json",
        "legacy_entrypoint": "scripts/train_nwpu_c3b_balanced.py",
        "handler": "native_contract",
        "runnable": "frozen_reproduction_only",
        "evaluation_scope": {
            "kind": "smoke",
            "image_count": None,
            "limit_train": None,
            "limit_val": 32,
        },
        "required_inputs": ["checkpoint", "annotation", "split_manifest"],
        "expected_outputs": ["eval_metrics", "launcher_log"],
        "gpu_policy": "remote_gpu2_guarded",
        "decision_document": DECISION_POINTER,
    }
    data.update(overrides)
    return data


def test_experiment_definition_parses_plan_example_shape() -> None:
    definition = parse_experiment_definition(_experiment_definition_dict())
    assert isinstance(definition, ExperimentDefinition)
    assert definition.runnable is RunnableMode.FROZEN_REPRODUCTION_ONLY
    assert definition.evaluation_scope.kind == "smoke"
    assert definition.required_inputs == ("checkpoint", "annotation", "split_manifest")
    with pytest.raises(dataclasses.FrozenInstanceError):
        definition.id = "other"


def test_experiment_definition_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="bogus"):
        parse_experiment_definition(_experiment_definition_dict(bogus=1))


def test_experiment_definition_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        parse_experiment_definition(_experiment_definition_dict(runnable="bogus"))
    with pytest.raises(ValueError):
        parse_experiment_definition(_experiment_definition_dict(config_path="../escape.json"))
    with pytest.raises(ValueError):
        parse_experiment_definition(_experiment_definition_dict(decision_document="/abs.md"))
    with pytest.raises(ValueError):
        parse_experiment_definition(_experiment_definition_dict(required_inputs="checkpoint"))
