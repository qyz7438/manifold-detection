"""Spectral manifold modules for frequency-domain and geometric analysis."""

# Complex / spectral modules
from spectral_detection_posttrain.methods.manifold.complex_manifold import (
    ComplexLinear,
    ComplexMLP,
    ComplexSpectralManifold,
    modReLU,
)
from spectral_detection_posttrain.methods.manifold.image_spectral_manifold import (
    ImageSpectralManifold,
)
from spectral_detection_posttrain.methods.manifold.roi_spectral_manifold import (
    ROISpectralManifold,
)
from spectral_detection_posttrain.methods.manifold.fpn_spectral_manifold import (
    FPNSpectralManifold,
)
from spectral_detection_posttrain.methods.manifold.fpn_real_adapter import (
    FPNRealAdapter,
)
from spectral_detection_posttrain.methods.manifold.fpn_attention_baselines import (
    FPNAttentionWrapper,
)

# Geometric / prototype modules
from spectral_detection_posttrain.methods.manifold.geometry_metrics import (
    compute_effective_rank,
    compute_nc1,
)
from spectral_detection_posttrain.methods.manifold.prototype_bank import (
    PrototypeBank,
    RemoteSensingPrototypeBank,
    compute_class_frequency_weights,
)
from spectral_detection_posttrain.methods.manifold.sinkhorn_assigner import (
    SinkhornAssigner,
)
from spectral_detection_posttrain.methods.manifold.transport_head import (
    TransportHead,
)
from spectral_detection_posttrain.methods.manifold.intrinsic_dim import (
    IntrinsicDimEstimator,
)
from spectral_detection_posttrain.methods.manifold.correction import (
    ManifoldCorrectionPredictor,
)

__all__ = [
    "ComplexLinear",
    "ComplexMLP",
    "ComplexSpectralManifold",
    "modReLU",
    "ImageSpectralManifold",
    "ROISpectralManifold",
    "FPNSpectralManifold",
    "FPNRealAdapter",
    "FPNAttentionWrapper",
    "compute_effective_rank",
    "compute_nc1",
    "PrototypeBank",
    "RemoteSensingPrototypeBank",
    "compute_class_frequency_weights",
    "SinkhornAssigner",
    "TransportHead",
    "IntrinsicDimEstimator",
    "ManifoldCorrectionPredictor",
]
