"""Remote-sensing-specific spectral methods built on Plan A manifold infrastructure."""

from spectral_detection_posttrain.methods.remote_sensing.remote_sensing_afm import (
    RemoteSensingAFM,
)
from spectral_detection_posttrain.methods.remote_sensing.multiscale_spectral_head import (
    MultiScaleSpectralHead,
)
from spectral_detection_posttrain.methods.remote_sensing.rotation_equivariant_fft import (
    RotationEquivariantFFT,
)
from spectral_detection_posttrain.methods.remote_sensing.rs_manifold import (
    RemoteSensingManifold,
)
from spectral_detection_posttrain.methods.remote_sensing.eval_remote_sensing import (
    compute_ap,
    evaluate_remote_sensing_ap,
)

__all__ = [
    "RemoteSensingAFM",
    "MultiScaleSpectralHead",
    "RotationEquivariantFFT",
    "RemoteSensingManifold",
    "compute_ap",
    "evaluate_remote_sensing_ap",
]
