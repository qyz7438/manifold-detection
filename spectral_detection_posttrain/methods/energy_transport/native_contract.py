"""Compatibility shim: native contracts moved to ``energy_transport.native``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.native.native_contract``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.native.native_contract import (
    DetectorNativeCandidates,
    build_detector_native_candidates,
    build_native_c1_deltas,
    evaluate_native_contract_gates,
    validate_strict_parity_artifact,
)

__all__ = [
    "DetectorNativeCandidates",
    "build_native_c1_deltas",
    "build_detector_native_candidates",
    "validate_strict_parity_artifact",
    "evaluate_native_contract_gates",
]
