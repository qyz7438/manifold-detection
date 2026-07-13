"""Compatibility shim: selection policies moved to ``energy_transport.policy``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.policy.adaptive_consensus``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.policy.adaptive_consensus import (
    proposal_graph_consensus_deltas,
)

__all__ = [
    "proposal_graph_consensus_deltas",
]
