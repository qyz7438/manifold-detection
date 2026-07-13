"""Reusable experiment components for spectral_detection_posttrain.

Shared export surface (contracts v2 integration). The canonical
``EvaluationScope`` exported here is the contracts v2 type from
``experiments.contracts``; ``experiments.artifacts`` keeps its own
self-contained scope type for manifest validation until a future dedup pass.
"""
from spectral_detection_posttrain.experiments.contracts import (
    CONTRACTS_VERSION,
    EVALUATION_SCOPE_KINDS,
    EvaluationScope,
    ExperimentDefinition,
    ResearchStatus,
    RunnableMode,
    normalize_evaluation_scope,
    parse_evaluation_scope,
    parse_experiment_definition,
    validate_relative_path,
)
from spectral_detection_posttrain.experiments.research_status import (
    STATUS_TO_RUNNABLE,
    ResearchLine,
    ResearchLineRegistry,
    can_authorize_new_run,
    load_research_status_registry,
    validate_runnable_for_status,
    validate_status_transition,
)
from spectral_detection_posttrain.experiments.registry import (
    EVIDENCE_KINDS,
    HANDLERS,
    ExecutableDefinition,
    ExperimentRegistry,
    HistoricalArtifact,
    load_experiment_registry,
)
from spectral_detection_posttrain.experiments.artifacts import (
    ArtifactManifest,
    ArtifactRef,
    complete_manifest,
    fail_manifest,
    load_artifact_manifest,
    promote_reviewed_manifest,
    serialize_artifact_manifest,
    start_manifest,
    verify_artifact_ref,
    write_artifact_manifest,
)

__all__ = [
    "CONTRACTS_VERSION",
    "EVALUATION_SCOPE_KINDS",
    "EvaluationScope",
    "ExperimentDefinition",
    "ResearchStatus",
    "RunnableMode",
    "normalize_evaluation_scope",
    "parse_evaluation_scope",
    "parse_experiment_definition",
    "validate_relative_path",
    "STATUS_TO_RUNNABLE",
    "ResearchLine",
    "ResearchLineRegistry",
    "can_authorize_new_run",
    "load_research_status_registry",
    "validate_runnable_for_status",
    "validate_status_transition",
    "EVIDENCE_KINDS",
    "HANDLERS",
    "ExecutableDefinition",
    "ExperimentRegistry",
    "HistoricalArtifact",
    "load_experiment_registry",
    "ArtifactManifest",
    "ArtifactRef",
    "complete_manifest",
    "fail_manifest",
    "load_artifact_manifest",
    "promote_reviewed_manifest",
    "serialize_artifact_manifest",
    "start_manifest",
    "verify_artifact_ref",
    "write_artifact_manifest",
]
