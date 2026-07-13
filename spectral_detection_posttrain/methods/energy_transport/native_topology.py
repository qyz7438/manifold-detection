"""Compatibility shim: native topology moved to ``energy_transport.native``.

The implementation now lives in
``spectral_detection_posttrain.methods.energy_transport.native.native_topology``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).

``apply_box_delta`` is also imported here (but not re-exported in ``__all__``)
because ``tests/test_energy_transport_native_topology.py`` patches
``spectral_detection_posttrain.methods.energy_transport.native_topology.apply_box_delta``
to prove the identity action never transforms boxes; the attribute must exist
on this module for that patch target to keep resolving.
"""

from spectral_detection_posttrain.methods.energy_transport.action.operators import (
    apply_box_delta,
)
from spectral_detection_posttrain.methods.energy_transport.native.native_topology import (
    NATIVE_TOPOLOGY_FEATURE_NAMES,
    native_action_nms_topology,
)

__all__ = [
    "NATIVE_TOPOLOGY_FEATURE_NAMES",
    "native_action_nms_topology",
]
