from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from spectral_detection_posttrain.experiments.contracts import (
    EvaluationScope,
    RunnableMode,
)
from spectral_detection_posttrain.experiments.registry import (
    EVIDENCE_KINDS,
    HANDLERS,
    ExecutableDefinition,
    ExperimentRegistry,
    HistoricalArtifact,
    load_experiment_registry,
)
from spectral_detection_posttrain.experiments.research_status import (
    STATUS_TO_RUNNABLE,
    ResearchStatus,
    load_research_status_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = REPO_ROOT / "spectral_detection_posttrain" / "configs" / "registry"
REGISTRY_PATH = REGISTRY_DIR / "experiments.json"
RESEARCH_LINES_PATH = REGISTRY_DIR / "research_lines.json"

EXECUTABLE_IDS = (
    "det.energy.native_listwise.c1.001",
    "det.energy.native_budget1.c2.001",
    "det.energy.global_delta_u.c3.001",
    "det.energy.global_delta_u.c3b.balanced.001",
    "det.energy.set_context.d1.001",
    "det.energy.native_topology.d2b.001",
    "det.energy.fine_action.d3.001",
    "det.energy.adaptive_consensus.d4.001",
    "det.energy.post_nms_suppress.e1.001",
    "det.energy.dense_endpoint.absolute.001",
    "det.energy.dense_endpoint.cleanval.001",
    "det.energy.dense_local_delta_learner.001",
    "det.energy.dense_local_delta_family_prior.001",
)

HISTORICAL_IDS = (
    "nwpu_mob_strong_cosine_s42_bs8_36ep",
    "native_zero_parity_baseline",
    "native_zero_parity_fullft",
    "det.energy.set_policy.m1.001",
    "det.energy.dense_endpoint.geometry_control.001",
    "det.energy.dense_endpoint.shift_audit.001",
    "det.energy.dense_local_delta_stats.002",
    "det.energy.dense_local_delta_family_audit.001",
)

BACKFILL_IDS = EXECUTABLE_IDS + HISTORICAL_IDS

EXECUTION_FIELDS = (
    "capability",
    "config_path",
    "legacy_entrypoint",
    "handler",
    "required_inputs",
    "expected_outputs",
    "gpu_policy",
)


@pytest.fixture(scope="module")
def registry() -> ExperimentRegistry:
    return load_experiment_registry(REGISTRY_PATH)


def _scope(kind: str = "smoke") -> dict:
    return {"kind": kind, "image_count": None, "limit_train": None, "limit_val": None}


def _executable_record(**overrides) -> dict:
    record = {
        "record_kind": "executable_definition",
        "id": "det.test.executable.001",
        "research_line": "energy_transport.native_actions.c1_e1",
        "capability": "detection.energy_transport.test",
        "config_path": "spectral_detection_posttrain/configs/versions/det.energy.global_delta_u.c3b.balanced.001.json",
        "legacy_entrypoint": "scripts/train_nwpu_c3b_balanced.py",
        "handler": "native_contract",
        "runnable": "frozen_reproduction_only",
        "evaluation_scope": _scope(),
        "required_inputs": ["checkpoint", "annotation", "split_manifest"],
        "expected_outputs": ["eval_metrics", "launcher_log"],
        "gpu_policy": "remote_gpu2_guarded",
        "decision_document": "docs/autonomous_exploration_report.md",
    }
    record.update(overrides)
    return record


def _historical_record(**overrides) -> dict:
    record = {
        "record_kind": "historical_artifact",
        "id": "det.test.historical.001",
        "research_line": "energy_transport.native_actions.c1_e1",
        "capability": None,
        "config_path": None,
        "legacy_entrypoint": None,
        "handler": None,
        "runnable": "disabled",
        "evaluation_scope": _scope("full_val"),
        "required_inputs": None,
        "expected_outputs": None,
        "gpu_policy": None,
        "evidence_kind": "full_val_run",
        "source_evidence_pointer": "docs/autonomous_exploration_report.md",
        "decision_document": "docs/autonomous_exploration_report.md",
    }
    record.update(overrides)
    return record


def _write_registry(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "experiments.json"
    payload = {"schema_version": "experiments.v1", "records": records}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _load_tmp(tmp_path: Path, records: list[dict]) -> ExperimentRegistry:
    return load_experiment_registry(
        _write_registry(tmp_path, records), research_lines_path=RESEARCH_LINES_PATH
    )


# ---------------------------------------------------------------------------
# Committed registry shape
# ---------------------------------------------------------------------------


def test_registry_loads_and_has_unique_ids(registry: ExperimentRegistry):
    ids = [record.id for record in registry]
    assert len(ids) == len(set(ids)), "duplicate experiment ids"
    assert len(registry) == 21


def test_every_backfill_id_resolves(registry: ExperimentRegistry):
    for experiment_id in BACKFILL_IDS:
        record = registry.require(experiment_id)
        assert record.id == experiment_id
    with pytest.raises(KeyError):
        registry.require("det.energy.does_not_exist.999")


def test_record_kinds_partition(registry: ExperimentRegistry):
    executables = [r for r in registry if isinstance(r, ExecutableDefinition)]
    historicals = [r for r in registry if isinstance(r, HistoricalArtifact)]
    assert sorted(r.id for r in executables) == sorted(EXECUTABLE_IDS)
    assert sorted(r.id for r in historicals) == sorted(HISTORICAL_IDS)


def test_executable_paths_exist_on_disk(registry: ExperimentRegistry):
    for record in registry:
        if not isinstance(record, ExecutableDefinition):
            continue
        assert (REPO_ROOT / record.config_path).is_file(), record.id
        assert (REPO_ROOT / record.legacy_entrypoint).is_file(), record.id
        assert (REPO_ROOT / record.decision_document).is_file(), record.id


def test_historical_evidence_pointers_exist_on_disk(registry: ExperimentRegistry):
    for record in registry:
        if not isinstance(record, HistoricalArtifact):
            continue
        assert (REPO_ROOT / record.source_evidence_pointer).is_file(), record.id
        assert (REPO_ROOT / record.decision_document).is_file(), record.id
        assert "runs/" not in record.source_evidence_pointer, record.id


def test_every_research_line_resolves(registry: ExperimentRegistry):
    lines = load_research_status_registry(RESEARCH_LINES_PATH)
    for record in registry:
        line = lines.get(record.research_line)
        assert line.id == record.research_line


def test_explicit_evaluation_scope_on_every_record(registry: ExperimentRegistry):
    for record in registry:
        assert isinstance(record.evaluation_scope, EvaluationScope), record.id
        assert record.evaluation_scope.kind, record.id


def test_explicit_capability_on_executable_records(registry: ExperimentRegistry):
    for record in registry:
        if isinstance(record, ExecutableDefinition):
            assert isinstance(record.capability, str) and record.capability, record.id


def test_no_authorized_records(registry: ExperimentRegistry):
    offenders = [r.id for r in registry if r.runnable is RunnableMode.AUTHORIZED]
    assert offenders == []


def test_no_records_on_active_or_queued_lines(registry: ExperimentRegistry):
    lines = load_research_status_registry(RESEARCH_LINES_PATH)
    offenders = [
        r.id
        for r in registry
        if lines.get(r.research_line).status
        in {ResearchStatus.ACTIVE, ResearchStatus.QUEUED}
    ]
    assert offenders == []


def test_runnable_allowed_by_line_status(registry: ExperimentRegistry):
    lines = load_research_status_registry(RESEARCH_LINES_PATH)
    for record in registry:
        status = lines.get(record.research_line).status
        assert record.runnable in STATUS_TO_RUNNABLE[status], record.id


def test_handlers_are_static_names(registry: ExperimentRegistry):
    for record in registry:
        if isinstance(record, ExecutableDefinition):
            assert record.handler in HANDLERS, record.id
            assert "." not in record.handler
            assert ":" not in record.handler


def test_historical_evidence_kind_vocabulary(registry: ExperimentRegistry):
    for record in registry:
        if isinstance(record, HistoricalArtifact):
            assert record.evidence_kind in EVIDENCE_KINDS, record.id
            assert record.runnable is RunnableMode.DISABLED


# ---------------------------------------------------------------------------
# Frozen reproduction and dispatch guards
# ---------------------------------------------------------------------------


def test_frozen_experiment_cannot_accept_overrides(registry: ExperimentRegistry):
    definition = registry.require("det.energy.global_delta_u.c3b.balanced.001")
    with pytest.raises(ValueError, match="frozen reproduction"):
        definition.resolve_overrides({"seed": 2024})


def test_frozen_experiment_rejects_any_override_key(registry: ExperimentRegistry):
    definition = registry.require("det.energy.global_delta_u.c3b.balanced.001")
    for key in ("epochs", "split", "gate", "action_space", "evaluation_scope", "lr"):
        with pytest.raises(ValueError, match="frozen reproduction"):
            definition.resolve_overrides({key: 1})


def test_non_authorized_record_rejects_scope_changing_overrides(
    registry: ExperimentRegistry,
):
    # cleanval is diagnostic_only (not frozen), so the restricted-key guard applies.
    definition = registry.require("det.energy.dense_endpoint.cleanval.001")
    assert definition.runnable is not RunnableMode.AUTHORIZED
    for key in ("epochs", "seed", "split", "action_space", "gate", "evaluation_scope"):
        with pytest.raises(ValueError, match="non-authorized"):
            definition.resolve_overrides({key: 1})


def test_can_dispatch_false_for_entire_initial_registry(registry: ExperimentRegistry):
    for record in registry:
        assert registry.can_dispatch(record) is False
        assert registry.can_dispatch(record.id) is False


def test_historical_artifacts_cannot_be_dispatched(registry: ExperimentRegistry):
    for record in registry:
        if isinstance(record, HistoricalArtifact):
            assert registry.can_dispatch(record) is False


# ---------------------------------------------------------------------------
# Loader rejection cases (strict discriminated union)
# ---------------------------------------------------------------------------


def test_duplicate_ids_rejected(tmp_path: Path):
    records = [_executable_record(), _executable_record()]
    with pytest.raises(ValueError, match="duplicate"):
        _load_tmp(tmp_path, records)


def test_unknown_research_line_rejected(tmp_path: Path):
    records = [_executable_record(research_line="no.such.line")]
    with pytest.raises(ValueError, match="unknown research_line"):
        _load_tmp(tmp_path, records)


def test_unknown_record_kind_rejected(tmp_path: Path):
    records = [_executable_record(record_kind="mystery")]
    with pytest.raises(ValueError, match="record_kind"):
        _load_tmp(tmp_path, records)


def test_mixed_record_fields_rejected(tmp_path: Path):
    records = [_executable_record(evidence_kind="smoke_run")]
    with pytest.raises(ValueError, match="unknown|mixed"):
        _load_tmp(tmp_path, records)


def test_executable_missing_required_field_rejected(tmp_path: Path):
    for field in EXECUTION_FIELDS + ("evaluation_scope",):
        record = _executable_record()
        del record[field]
        with pytest.raises(ValueError, match="missing"):
            _load_tmp(tmp_path, [record])


def test_executable_missing_config_path_on_disk_rejected(tmp_path: Path):
    records = [
        _executable_record(
            config_path="spectral_detection_posttrain/configs/versions/nope.001.json"
        )
    ]
    with pytest.raises(ValueError, match="does not exist"):
        _load_tmp(tmp_path, records)


def test_executable_missing_entrypoint_on_disk_rejected(tmp_path: Path):
    records = [_executable_record(legacy_entrypoint="scripts/no_such_script.py")]
    with pytest.raises(ValueError, match="does not exist"):
        _load_tmp(tmp_path, records)


def test_dynamic_module_string_handler_rejected(tmp_path: Path):
    records = [
        _executable_record(handler="spectral_detection_posttrain.trainers.run_c3b")
    ]
    with pytest.raises(ValueError, match="dynamic|handler"):
        _load_tmp(tmp_path, records)


def test_unknown_handler_name_rejected(tmp_path: Path):
    records = [_executable_record(handler="made_up_handler")]
    with pytest.raises(ValueError, match="handler"):
        _load_tmp(tmp_path, records)


def test_historical_with_non_null_execution_field_rejected(tmp_path: Path):
    for field in EXECUTION_FIELDS:
        value = "x" if field not in ("required_inputs", "expected_outputs") else ["x"]
        records = [_historical_record(**{field: value})]
        with pytest.raises(ValueError, match="null"):
            _load_tmp(tmp_path, records)


def test_historical_with_non_disabled_runnable_rejected(tmp_path: Path):
    records = [_historical_record(runnable="diagnostic_only")]
    with pytest.raises(ValueError, match="disabled"):
        _load_tmp(tmp_path, records)


def test_historical_missing_evidence_pointer_rejected(tmp_path: Path):
    record = _historical_record()
    del record["source_evidence_pointer"]
    with pytest.raises(ValueError, match="missing"):
        _load_tmp(tmp_path, [record])


def test_historical_evidence_pointer_must_exist_on_disk(tmp_path: Path):
    records = [_historical_record(source_evidence_pointer="docs/no_such_doc.md")]
    with pytest.raises(ValueError, match="does not exist"):
        _load_tmp(tmp_path, records)


def test_status_to_runnable_enforced_through_line_reference(tmp_path: Path):
    # The c1_e1 line is frozen; authorized is not in its allowed runnable set.
    records = [_executable_record(runnable="authorized")]
    with pytest.raises(ValueError, match="not allowed for status"):
        _load_tmp(tmp_path, records)


def test_unknown_runnable_mode_rejected(tmp_path: Path):
    records = [_executable_record(runnable="warp_speed")]
    with pytest.raises(ValueError, match="runnable"):
        _load_tmp(tmp_path, records)


def test_path_escape_rejected(tmp_path: Path):
    records = [_executable_record(legacy_entrypoint="../outside.py")]
    with pytest.raises(ValueError):
        _load_tmp(tmp_path, records)


def test_raw_json_record_count_and_shape():
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "experiments.v1"
    records = payload["records"]
    assert len(records) == 21
    for raw in records:
        assert raw["record_kind"] in {"executable_definition", "historical_artifact"}
        if raw["record_kind"] == "historical_artifact":
            for field in EXECUTION_FIELDS:
                assert field in raw and raw[field] is None, (raw["id"], field)
