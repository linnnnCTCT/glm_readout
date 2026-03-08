"""Student-view corruption for JEPA training."""

from __future__ import annotations

import torch


def build_student_view(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_ratio: float = 0.15,
    noise_std: float = 0.0,
    zero_replace_prob: float = 1.0,
    forced_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds a masked/corrupted view of hidden states.

    Args:
        hidden_states: Tensor [B, L, D].
        attention_mask: Tensor [B, L], True for valid tokens.
        mask_ratio: Fraction of valid positions to mask.
        noise_std: Gaussian noise std to add on masked tokens.
        zero_replace_prob: Probability to replace masked states with zero vectors.
        forced_mask: Optional boolean tensor [B, L] that must always be masked.

    Returns:
        student_view: Corrupted tensor [B, L, D].
        corruption_mask: Boolean tensor [B, L] where corruption happened.
    """

    if hidden_states.ndim != 3:
        raise ValueError(f"Expected hidden_states [B,L,D], got {tuple(hidden_states.shape)}")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"Mask shape {tuple(attention_mask.shape)} mismatches hidden states {tuple(hidden_states.shape)}"
        )

    valid_mask = attention_mask.bool()
    if forced_mask is not None:
        if forced_mask.shape != valid_mask.shape:
            raise ValueError(
                f"forced_mask shape {tuple(forced_mask.shape)} mismatches valid mask {tuple(valid_mask.shape)}"
            )
        forced_mask = forced_mask.bool() & valid_mask

    random_scores = torch.rand_like(valid_mask, dtype=torch.float32)
    random_mask = (random_scores < mask_ratio) & valid_mask
    if forced_mask is not None:
        random_mask = random_mask & ~forced_mask
        corruption_mask = forced_mask | random_mask
    else:
        corruption_mask = random_mask

    student_view = hidden_states.clone()

    if zero_replace_prob > 0.0:
        zero_decision = torch.rand_like(random_scores) < zero_replace_prob
        zero_mask = corruption_mask & zero_decision
        student_view = student_view.masked_fill(zero_mask.unsqueeze(-1), 0.0)

    if noise_std > 0.0:
        noise = torch.randn_like(student_view) * noise_std
        student_view = student_view + noise * corruption_mask.unsqueeze(-1).to(student_view.dtype)

    return student_view, corruption_mask
