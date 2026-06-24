"""Learnable spectral manifold infrastructure for ChordEdit-style transport."""

from spectral_detection_posttrain.methods.manifold.complex_manifold import (
    ComplexLinear,
    ComplexMLP,
    ComplexSpectralManifold,
)
from spectral_detection_posttrain.methods.manifold.riemannian_metric import (
    AdaptiveRiemannianMetric,
)
from spectral_detection_posttrain.methods.manifold.chord_transport import (
    ChordTransport,
)
from spectral_detection_posttrain.methods.manifold.sinkhorn_ot import (
    SinkhornOT,
)
from spectral_detection_posttrain.methods.manifold.prototype_bank import (
    PrototypeBank,
)
from spectral_detection_posttrain.methods.manifold.sinkhorn_assigner import (
    SinkhornAssigner,
)
from spectral_detection_posttrain.methods.manifold.transport_head import (
    TransportHead,
)
from spectral_detection_posttrain.methods.manifold.correction import (
    ManifoldCorrectionPredictor,
)
from spectral_detection_posttrain.methods.manifold.intrinsic_dim import (
    IntrinsicDimEstimator,
)

__all__ = [
    "ComplexLinear",
    "ComplexMLP",
    "ComplexSpectralManifold",
    "AdaptiveRiemannianMetric",
    "ChordTransport",
    "SinkhornOT",
    "PrototypeBank",
    "SinkhornAssigner",
    "TransportHead",
    "ManifoldCorrectionPredictor",
    "IntrinsicDimEstimator",
]
