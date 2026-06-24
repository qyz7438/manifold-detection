"""Image classification methods built on spectral manifolds and OT."""

from spectral_detection_posttrain.methods.classification.eval_classification import (
    accuracy,
    evaluate_classifier,
)
from spectral_detection_posttrain.methods.classification.ot_prototype_classifier import (
    OTPrototypeClassifier,
)
from spectral_detection_posttrain.methods.classification.spectral_classifier_head import (
    SpectralClassifierHead,
)
from spectral_detection_posttrain.methods.classification.spectral_mixup import (
    SpectralMixup,
)

__all__ = [
    "SpectralClassifierHead",
    "OTPrototypeClassifier",
    "SpectralMixup",
    "accuracy",
    "evaluate_classifier",
]
