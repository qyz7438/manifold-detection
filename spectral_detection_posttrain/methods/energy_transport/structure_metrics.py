"""Compatibility shim: transport diagnostics moved to ``energy_transport.diagnostics``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.diagnostics.structure_metrics``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.diagnostics.structure_metrics import (
    PrototypeBasinGeometry,
    ROIStructureSignature,
    ROIDualEnergy,
    roi_compactness_energy,
    roi_basin_energy,
    roi_basin_retention,
    centered_relation_matrix,
    relation_cka,
    inter_class_separation_energy,
    prototype_anchor_energy,
    inter_class_relation_energy,
    simplex_energy,
    class_topk_adjacency,
    graph_jaccard,
    basin_leakage_graph,
    prototype_basin_geometry,
    roi_structure_signature,
    roi_dual_energy,
)

__all__ = [
    "PrototypeBasinGeometry",
    "ROIStructureSignature",
    "ROIDualEnergy",
    "roi_compactness_energy",
    "roi_basin_energy",
    "roi_basin_retention",
    "centered_relation_matrix",
    "relation_cka",
    "inter_class_separation_energy",
    "prototype_anchor_energy",
    "inter_class_relation_energy",
    "simplex_energy",
    "class_topk_adjacency",
    "graph_jaccard",
    "basin_leakage_graph",
    "prototype_basin_geometry",
    "roi_structure_signature",
    "roi_dual_energy",
]
