"""JEPA objective and optional anti-collapse regularizers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine regression between student prediction and teacher target."""
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
    pred = F.normalize(pred, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)
    return 1.0 - (pred * target).sum(dim=-1).mean()


def variance_floor_penalty(z_seq: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Penalizes per-dimension std below a floor."""
    if z_seq.ndim != 2:
        raise ValueError(f"Expected z_seq [B,D], got {tuple(z_seq.shape)}")
    if z_seq.shape[0] < 2:
        return z_seq.new_tensor(0.0)
    std = torch.sqrt(z_seq.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def covariance_penalty(z_seq: torch.Tensor) -> torch.Tensor:
    """Penalizes off-diagonal covariance terms."""
    if z_seq.ndim != 2:
        raise ValueError(f"Expected z_seq [B,D], got {tuple(z_seq.shape)}")
    batch_size, dim = z_seq.shape
    if batch_size < 2 or dim < 2:
        return z_seq.new_tensor(0.0)
    centered = z_seq - z_seq.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (batch_size - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag.pow(2).sum()) / dim


class JEPALoss(nn.Module):
    """Combined JEPA loss with optional variance/covariance regularization."""

    def __init__(
        self,
        variance_weight: float = 0.0,
        covariance_weight: float = 0.0,
        variance_floor: float = 1.0,
    ) -> None:
        super().__init__()
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.variance_floor = variance_floor

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, z_seq: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        main_loss = cosine_regression_loss(pred=pred, target=target)
        var_loss = variance_floor_penalty(z_seq, gamma=self.variance_floor)
        cov_loss = covariance_penalty(z_seq)
        total = main_loss
        if self.variance_weight > 0:
            total = total + self.variance_weight * var_loss
        if self.covariance_weight > 0:
            total = total + self.covariance_weight * cov_loss

        return {
            "loss": total,
            "loss_jepa": main_loss,
            "loss_var": var_loss,
            "loss_cov": cov_loss,
        }
