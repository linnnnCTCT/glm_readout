"""Losses for JEPA training."""

from .jepa_losses import JEPALoss, covariance_penalty, cosine_regression_loss, variance_floor_penalty

__all__ = [
    "JEPALoss",
    "cosine_regression_loss",
    "variance_floor_penalty",
    "covariance_penalty",
]
