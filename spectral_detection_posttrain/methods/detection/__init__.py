"""Detection structural components: PBG, TAM, PAH."""

from .pbg import PhaseBoundaryGate
from .tam import TaskAlignedManifold
from .pah import PrototypeAwareHead

__all__ = ["PhaseBoundaryGate", "TaskAlignedManifold", "PrototypeAwareHead"]
