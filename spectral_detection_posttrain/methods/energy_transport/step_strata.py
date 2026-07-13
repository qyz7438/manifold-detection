"""Compatibility shim: transport diagnostics moved to ``energy_transport.diagnostics``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.diagnostics.step_strata``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.diagnostics.step_strata import (
    StepActionRows,
    extract_step_action_rows,
    strata_masks,
    summarize_stratum,
    discover_stable_strata,
    select_primary_stratum,
    rate_matched_uniform_control,
    target_permutation_control,
    evaluate_step_strata_gates,
)

__all__ = [
    "StepActionRows",
    "extract_step_action_rows",
    "strata_masks",
    "summarize_stratum",
    "discover_stable_strata",
    "select_primary_stratum",
    "rate_matched_uniform_control",
    "target_permutation_control",
    "evaluate_step_strata_gates",
]
