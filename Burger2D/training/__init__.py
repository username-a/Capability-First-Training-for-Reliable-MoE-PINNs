from .losses import LossConfig2D, PhysicsLoss2D, l2_relative_error, max_absolute_error
from .staged_burgers2d import StagedBurgers2DConfig
from .trainer import Trainer

__all__ = [
    "LossConfig2D",
    "PhysicsLoss2D",
    "StagedBurgers2DConfig",
    "Trainer",
    "l2_relative_error",
    "max_absolute_error",
]
