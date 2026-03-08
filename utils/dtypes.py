"""Torch dtype helpers."""

from __future__ import annotations

from typing import Any

import torch


_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_torch_dtype(dtype: Any) -> torch.dtype | None:
    """Resolves a string / torch dtype into a torch.dtype."""
    if dtype is None or dtype == "":
        return None
    if isinstance(dtype, torch.dtype):
        return dtype

    text = str(dtype).strip().lower()
    if text in {"none", "preserve", "original"}:
        return None
    if text not in _DTYPE_ALIASES:
        raise ValueError(
            f"Unsupported dtype '{dtype}'. Expected one of: {sorted(_DTYPE_ALIASES)} or None."
        )
    return _DTYPE_ALIASES[text]
