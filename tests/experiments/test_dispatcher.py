"""Contract tests for the explicit experiment dispatcher (refactor Task 7).

The dispatcher is a thin, deterministic, side-effect-free layer over the
experiment registry:

- a static ``HANDLERS`` map keyed by ``ExperimentCapability`` (never dynamic
  import strings);
- guards rejecting unknown experiment IDs, unknown/dynamic handlers, frozen
  records without ``--allow-frozen-reproduction``, and any override on frozen
  records;
- ``run`` refused for every record while the registry authorizes zero
  experiments;
- deterministic ``list``/``status``/``validate``/``dry-run`` rendering that
  never creates a run directory and never imports dataset code.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from spectral_detection_posttrain.experiments.dispatcher import (
    HANDLERS,
    DispatchError,
    DispatchRefusedError,
    dispatch_dry_run,
    dispatch_run,
    dispatch_validate,
    load_dispatcher_registry,
    render_plan,
    render_record_list,
    render_status,
    resolve_handler,
)
from spectral_detection_posttrain.experiments.handlers import (
    DenseEndpointHandler,
    DispatchPlan,
    ExperimentCapability,
    ExperimentHandler,
    NativeContractHandler,
    StandardDetectionHandler,
)
from spectral_detection_posttrain.experiments.registry import (
    ExecutableDefinition,
    ExperimentRegistry,
    HistoricalArtifact,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_EXPERIMENT_PATH = REPO_ROOT / "scripts" / "run_experiment.py"

C1_ID = "det.energy.native_listwise.c1.001"
C3B_ID = "det.energy.global_delta_u.c3b.balanced.001"
DENSE_ABSOLUTE_ID = "det.energy.dense_endpoint.absolute.001"
FAMILY_PRIOR_ID = "det.energy.dense_local_delta_family_prior.001"
HISTORICAL_ID = "native_zero_parity_baseline"

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
    "det.energy.re_roi_counterfactual_evidence.001",
    "det.energy.oracle_utility_boxhead.001",
)

ALL_IDS = EXECUTABLE_IDS + HISTORICAL_IDS


@pytest.fixture(scope="module")
def registry() -> ExperimentRegistry:
    return load_dispatcher_registry()


def load_cli_module():
    assert RUN_EXPERIMENT_PATH.exists(), "dispatcher CLI has not been implemented"
    spec = importlib.util.spec_from_file_location("run_experiment_cli", RUN_EXPERIMENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli():
    return load_cli_module()


# ---------------------------------------------------------------------------
# Static handler map
# ---------------------------------------------------------------------------


def test_handler_map_is_static_and_complete():
    assert set(HANDLERS) == {
        ExperimentCapability.STANDARD_DETECTION,
        ExperimentCapability.NATIVE_CONTRACT,
        ExperimentCapability.DENSE_ENDPOINT,
    }
    for capability, handler in HANDLERS.items():
        assert isinstance(handler, ExperimentHandler)
        assert handler.capability is capability


def test_handler_capability_values_match_registry_handler_names():
    from spectral_detection_posttrain.experiments.registry import HANDLERS as REGISTRY_HANDLERS

    assert {capability.value for capability in HANDLERS} == set(REGISTRY_HANDLERS)


def test_every_registered_executable_handler_resolves(registry):
    for record in registry:
        if isinstance(record, ExecutableDefinition):
            handler = resolve_handler(record)
            assert handler.capability.value == record.handler


def test_dispatcher_uses_no_dynamic_imports():
    import spectral_detection_posttrain.experiments.dispatcher as dispatcher_module
    import spectral_detection_posttrain.experiments.handlers as handlers_package

    sources = [
        inspect.getsource(dispatcher_module),
        inspect.getsource(handlers_package),
    ]
    for handler in HANDLERS.values():
        sources.append(inspect.getsource(type(handler)))
    for source in sources:
        assert "importlib" not in source
        assert "import_module" not in source
        assert "__import__" not in source
        assert "eval(" not in source
        assert "exec(" not in source


def test_dispatcher_does_not_import_dataset_or_training_code():
    import spectral_detection_posttrain.experiments.dispatcher as dispatcher_module
    import spectral_detection_posttrain.experiments.handlers as handlers_package

    sources = [
        inspect.getsource(dispatcher_module),
        inspect.getsource(handlers_package),
    ]
    for handler in HANDLERS.values():
        sources.append(inspect.getsource(type(handler)))
    for source in sources:
        assert "datasets" not in source
        assert "import torch" not in source
        assert "canonical_runner" not in source
        assert "prepare_experiment(" not in source


def test_resolve_handler_rejects_unknown_capability():
    with pytest.raises(DispatchError, match="unknown handler"):
        resolve_handler(SimpleNamespace(handler="does_not_exist"))


def test_resolve_handler_rejects_dynamic_import_strings():
    for dynamic in (
        "spectral_detection_posttrain.experiments.handlers.native_contract.NativeContractHandler",
        "pkg.module:Class",
        "handlers/native_contract",
        "handlers\\native_contract",
    ):
        with pytest.raises(DispatchError, match="dynamic import strings"):
            resolve_handler(SimpleNamespace(handler=dynamic))


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_unknown_experiment_id_rejected(registry):
    with pytest.raises(DispatchError, match="unknown experiment id"):
        dispatch_dry_run(registry, "det.energy.not_registered.999", "probe_run")
    with pytest.raises(DispatchError, match="unknown experiment id"):
        dispatch_validate(registry, "det.energy.not_registered.999")
    with pytest.raises(DispatchError, match="unknown experiment id"):
        dispatch_run(registry, "det.energy.not_registered.999", "probe_run")


def test_historical_artifact_can_never_be_dispatched(registry):
    record = registry.require(HISTORICAL_ID)
    assert isinstance(record, HistoricalArtifact)
    with pytest.raises(DispatchError, match="historical artifact"):
        dispatch_dry_run(registry, HISTORICAL_ID, "probe_run")
    with pytest.raises(DispatchError, match="historical artifact"):
        dispatch_validate(registry, HISTORICAL_ID)
    with pytest.raises(DispatchError, match="historical artifact"):
        dispatch_run(registry, HISTORICAL_ID, "probe_run")


@pytest.mark.parametrize("experiment_id", [C1_ID, C3B_ID, DENSE_ABSOLUTE_ID])
def test_frozen_record_requires_explicit_flag(registry, experiment_id):
    record = registry.require(experiment_id)
    assert record.runnable.value == "frozen_reproduction_only"
    with pytest.raises(DispatchError, match="--allow-frozen-reproduction"):
        dispatch_dry_run(registry, experiment_id, "probe_run")


@pytest.mark.parametrize("experiment_id", [C1_ID, C3B_ID, DENSE_ABSOLUTE_ID])
def test_frozen_record_rejects_any_override(registry, experiment_id):
    with pytest.raises(DispatchError, match="override"):
        dispatch_dry_run(
            registry,
            experiment_id,
            "probe_run",
            allow_frozen_reproduction=True,
            overrides={"seed": 43},
        )
    with pytest.raises(DispatchError, match="override"):
        dispatch_dry_run(
            registry,
            experiment_id,
            "probe_run",
            allow_frozen_reproduction=True,
            overrides={"comment": "even a benign key is rejected on frozen records"},
        )


def test_restricted_override_rejected_on_diagnostic_record(registry):
    record = registry.require(FAMILY_PRIOR_ID)
    assert record.runnable.value == "diagnostic_only"
    with pytest.raises(DispatchError, match="override"):
        dispatch_dry_run(registry, FAMILY_PRIOR_ID, "probe_run", overrides={"epochs": 5})


def test_run_name_validation(registry):
    for bad in ("", "a/b", "a\\b", "..", ".", "a..b"):
        with pytest.raises(DispatchError, match="run name"):
            dispatch_dry_run(registry, FAMILY_PRIOR_ID, bad)


# ---------------------------------------------------------------------------
# run is disabled for the whole registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("experiment_id", ALL_IDS)
def test_run_refused_for_every_record(registry, experiment_id):
    with pytest.raises(DispatchRefusedError) as excinfo:
        dispatch_run(
            registry,
            experiment_id,
            "probe_run",
            allow_frozen_reproduction=True,
        )
    message = str(excinfo.value)
    assert experiment_id in message or "historical artifact" in message


def test_run_refusal_message_is_clear(registry):
    with pytest.raises(DispatchRefusedError) as excinfo:
        dispatch_run(registry, C1_ID, "probe_run", allow_frozen_reproduction=True)
    message = str(excinfo.value)
    assert C1_ID in message
    assert "not authorized" in message
    assert "frozen_reproduction_only" in message
    assert "frozen" in message


# ---------------------------------------------------------------------------
# dry-run rendering
# ---------------------------------------------------------------------------


def test_dry_run_plan_fields_for_c1(registry):
    plan = dispatch_dry_run(
        registry, C1_ID, "probe_run", allow_frozen_reproduction=True
    )
    assert isinstance(plan, DispatchPlan)
    assert plan.experiment_id == C1_ID
    assert plan.research_line == "energy_transport.native_actions.c1_e1"
    assert plan.research_status == "frozen"
    assert plan.capability == "detection.energy_transport.c1"
    assert plan.handler == "native_contract"
    assert plan.runnable == "frozen_reproduction_only"
    assert plan.authorized is False
    expected_hash = hashlib.sha256(
        (REPO_ROOT / plan.config_path).read_bytes()
    ).hexdigest()
    assert plan.config_sha256 == expected_hash
    assert plan.required_inputs == ("checkpoint", "annotation", "split_manifest")
    assert plan.expected_outputs == ("eval_metrics", "launcher_log")
    assert plan.evaluation_scope.kind == "limited_unknown"
    assert plan.gpu_policy == "remote_gpu2_guarded"
    assert plan.run_name == "probe_run"
    assert plan.command == ("python", "scripts/verify_nwpu_native_c1_contract.py")


def test_rendered_dry_run_contains_all_required_lines(registry):
    text = render_plan(
        dispatch_dry_run(
            registry, C3B_ID, "probe_run", allow_frozen_reproduction=True
        )
    )
    assert f"experiment: {C3B_ID}" in text
    assert "research status: frozen" in text
    assert "config sha256: " in text
    assert "required inputs: checkpoint, annotation, split_manifest, action_cache" in text
    assert "evaluation scope: smoke" in text
    assert "image_count=64" in text and "limit_train=32" in text and "limit_val=32" in text
    assert "gpu policy: remote_gpu2_guarded" in text
    assert "expected outputs: eval_metrics, launcher_log" in text
    assert "exact command: python scripts/train_nwpu_c3b_balanced.py" in text
    assert "authorized: no" in text


def test_dry_run_is_deterministic(registry):
    first = render_plan(
        dispatch_dry_run(
            registry, DENSE_ABSOLUTE_ID, "probe_run", allow_frozen_reproduction=True
        )
    )
    second = render_plan(
        dispatch_dry_run(
            registry, DENSE_ABSOLUTE_ID, "probe_run", allow_frozen_reproduction=True
        )
    )
    assert first == second


def test_dry_run_creates_no_run_directory(registry):
    run_name = "pytest_dispatcher_dryrun_probe"
    planned = REPO_ROOT / "runs" / run_name
    assert not planned.exists()
    dispatch_dry_run(registry, FAMILY_PRIOR_ID, run_name)
    assert not planned.exists()
    dispatch_dry_run(registry, C1_ID, run_name, allow_frozen_reproduction=True)
    assert not planned.exists()


def test_dry_run_reports_non_authorized_for_all_four_probe_records(registry):
    probes = (
        (C1_ID, True),
        (C3B_ID, True),
        (DENSE_ABSOLUTE_ID, True),
        (FAMILY_PRIOR_ID, False),
    )
    for experiment_id, frozen in probes:
        plan = dispatch_dry_run(
            registry,
            experiment_id,
            "probe_run",
            allow_frozen_reproduction=frozen,
        )
        assert plan.authorized is False
        assert "authorized: no" in render_plan(plan)


# ---------------------------------------------------------------------------
# validate / list / status
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("experiment_id", EXECUTABLE_IDS)
def test_validate_passes_for_registered_executable_records(registry, experiment_id):
    text = dispatch_validate(registry, experiment_id)
    assert "valid" in text
    assert experiment_id in text


def test_handler_validate_reports_handler_mismatch():
    handler = NativeContractHandler()
    fake = SimpleNamespace(handler="dense_endpoint")
    problems = handler.validate(fake, registry=None)
    assert problems
    assert "dense_endpoint" in problems[0]


def test_list_is_deterministic_and_covers_every_record(registry):
    first = render_record_list(registry)
    second = render_record_list(registry)
    assert first == second
    for experiment_id in ALL_IDS:
        assert experiment_id in first
    assert "23 records" in first
    assert "13 executable_definition" in first
    assert "10 historical_artifact" in first


def test_status_reports_zero_authorized(registry):
    text = render_status(registry)
    assert "active lines: 0" in text
    assert "queued lines: 0" in text
    assert "authorized experiments: 0" in text
    assert "No experiment is currently authorized." in text
    assert render_status(registry) == text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_list_and_status(cli, capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "23 records" in out
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "No experiment is currently authorized." in out


def test_cli_validate(cli, capsys):
    assert cli.main(["validate", "--experiment", C1_ID]) == 0
    assert "valid" in capsys.readouterr().out
    assert cli.main(["validate", "--experiment", "det.energy.nope.999"]) == 1
    assert "unknown experiment id" in capsys.readouterr().err


def test_cli_dry_run_frozen_guard_and_flag(cli, capsys):
    assert cli.main(["dry-run", "--experiment", C1_ID, "--run-name", "probe_run"]) == 1
    assert "--allow-frozen-reproduction" in capsys.readouterr().err
    assert (
        cli.main(
            [
                "dry-run",
                "--experiment",
                C1_ID,
                "--run-name",
                "probe_run",
                "--allow-frozen-reproduction",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert f"experiment: {C1_ID}" in out
    assert "authorized: no" in out
    assert "exact command: python scripts/verify_nwpu_native_c1_contract.py" in out


def test_cli_dry_run_rejects_override_on_frozen(cli, capsys):
    assert (
        cli.main(
            [
                "dry-run",
                "--experiment",
                C3B_ID,
                "--run-name",
                "probe_run",
                "--allow-frozen-reproduction",
                "--override",
                "seed=43",
            ]
        )
        == 1
    )
    assert "override" in capsys.readouterr().err


def test_cli_run_refused(cli, capsys):
    assert (
        cli.main(
            [
                "run",
                "--experiment",
                FAMILY_PRIOR_ID,
                "--run-name",
                "probe_run",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "not authorized" in err
    assert FAMILY_PRIOR_ID in err
