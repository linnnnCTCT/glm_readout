"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    unwrapped_model = model.module if hasattr(model, "module") else model

    payload: dict[str, Any] = {
        "model_state_dict": unwrapped_model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    if hasattr(unwrapped_model, "student_readout"):
        payload["student_readout_state_dict"] = unwrapped_model.student_readout.state_dict()
    if hasattr(unwrapped_model, "teacher_readout"):
        payload["teacher_readout_state_dict"] = unwrapped_model.teacher_readout.state_dict()
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if extra_state is not None:
        payload["extra_state"] = extra_state

    torch.save(payload, path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=map_location)
    if model is not None:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in payload:
        scaler.load_state_dict(payload["scaler_state_dict"])
    return payload
