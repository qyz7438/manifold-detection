"""Spectral adversarial defense modules built on Plan A manifold infrastructure."""

from spectral_detection_posttrain.methods.defense.spectral_chord_defense import (
    SpectralChordDefense,
)
from spectral_detection_posttrain.methods.defense.patch_attack import (
    AdversarialPatchAttack,
)
from spectral_detection_posttrain.methods.defense.manifold_natural import (
    NaturalSpectrumModel,
)
from spectral_detection_posttrain.methods.defense.eval_defense import (
    defense_success_rate,
    clean_accuracy_drop,
    robust_accuracy,
)

__all__ = [
    "SpectralChordDefense",
    "AdversarialPatchAttack",
    "NaturalSpectrumModel",
    "defense_success_rate",
    "clean_accuracy_drop",
    "robust_accuracy",
]
