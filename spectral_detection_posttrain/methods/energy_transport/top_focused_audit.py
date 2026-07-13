"""Compatibility shim: transport diagnostics moved to ``energy_transport.diagnostics``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.diagnostics.top_focused_audit``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.diagnostics.top_focused_audit import (
    median_sign_metrics,
    top_focused_rank_metrics,
    evaluate_top_focused_gates,
)

__all__ = [
    "median_sign_metrics",
    "top_focused_rank_metrics",
    "evaluate_top_focused_gates",
]
