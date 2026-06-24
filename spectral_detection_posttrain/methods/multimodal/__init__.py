"""Multimodal alignment methods for image-text tasks."""

from spectral_detection_posttrain.methods.multimodal.chord_text_guided import (
    ChordTextGuidedEdit,
)
from spectral_detection_posttrain.methods.multimodal.cross_modal_transport import (
    CrossModalTransport,
)
from spectral_detection_posttrain.methods.multimodal.eval_retrieval import (
    compute_similarity_matrix,
    evaluate_image_text_retrieval,
    recall_at_k,
)
from spectral_detection_posttrain.methods.multimodal.ot_alignment import (
    OTImageTextAlignment,
)

__all__ = [
    "OTImageTextAlignment",
    "ChordTextGuidedEdit",
    "CrossModalTransport",
    "compute_similarity_matrix",
    "recall_at_k",
    "evaluate_image_text_retrieval",
]
