"""Token-dimension pre-aggregation utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkPooler(nn.Module):
    """Pools long token sequences into shorter token chunks.

    Supports non-overlapping and strided windows.
    """

    def __init__(self, chunk_size: int = 64, stride: int | None = None, mode: str = "mean") -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.chunk_size = int(chunk_size)
        self.stride = int(stride or chunk_size)
        self.mode = mode
        if self.mode not in {"mean", "max"}:
            raise ValueError(f"Unsupported chunk pooling mode: {self.mode}")

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pools hidden states.

        Args:
            hidden_states: Tensor [B, L, D].
            attention_mask: Tensor [B, L] with True for valid tokens.
        Returns:
            pooled_hidden_states: Tensor [B, L', D].
            pooled_attention_mask: Tensor [B, L'].
        """
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected hidden_states [B,L,D], got {tuple(hidden_states.shape)}")
        batch_size, seq_length, _ = hidden_states.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_length, dtype=torch.bool, device=hidden_states.device
            )
        if attention_mask.shape != (batch_size, seq_length):
            raise ValueError(
                f"Expected attention_mask [B,L], got {tuple(attention_mask.shape)}"
            )

        pad_right = self._compute_right_padding(seq_length)
        if pad_right > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_right))
            attention_mask = F.pad(attention_mask, (0, pad_right), value=False)

        windowed_states = hidden_states.unfold(dimension=1, size=self.chunk_size, step=self.stride)
        windowed_masks = attention_mask.unfold(dimension=1, size=self.chunk_size, step=self.stride)
        # windowed_states: [B, L', D, chunk_size], windowed_masks: [B, L', chunk_size]

        if self.mode == "mean":
            pooled_states = self._masked_window_mean(windowed_states, windowed_masks)
        else:
            pooled_states = self._masked_window_max(windowed_states, windowed_masks)

        pooled_mask = windowed_masks.any(dim=-1)
        return pooled_states, pooled_mask

    def _compute_right_padding(self, seq_length: int) -> int:
        if seq_length <= 0:
            raise ValueError("seq_length must be > 0")
        if seq_length < self.chunk_size:
            return self.chunk_size - seq_length
        remainder = (seq_length - self.chunk_size) % self.stride
        return (self.stride - remainder) % self.stride

    @staticmethod
    def _masked_window_mean(
        windowed_states: torch.Tensor, windowed_masks: torch.Tensor
    ) -> torch.Tensor:
        mask = windowed_masks.unsqueeze(2).to(windowed_states.dtype)
        weighted_sum = (windowed_states * mask).sum(dim=-1)
        counts = mask.sum(dim=-1).clamp(min=1.0)
        return weighted_sum / counts

    @staticmethod
    def _masked_window_max(
        windowed_states: torch.Tensor, windowed_masks: torch.Tensor
    ) -> torch.Tensor:
        masked_states = windowed_states.masked_fill(~windowed_masks.unsqueeze(2), float("-inf"))
        pooled_states = masked_states.max(dim=-1).values
        valid_mask = windowed_masks.any(dim=-1, keepdim=True)
        return torch.where(valid_mask, pooled_states, torch.zeros_like(pooled_states))
