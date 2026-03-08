"""Contiguous target-span sampling for JEPA."""

from __future__ import annotations

import torch


class ContiguousSpanSampler:
    """Samples one contiguous target span per sequence."""

    def __init__(
        self,
        min_tokens: int = 64,
        max_tokens: int = 2048,
        span_ratio: float = 0.1,
    ) -> None:
        if min_tokens <= 0:
            raise ValueError("min_tokens must be > 0")
        if max_tokens < min_tokens:
            raise ValueError("max_tokens must be >= min_tokens")
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.span_ratio = span_ratio

    def sample(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns spans [B, 2] as [start, end) token indices."""
        if attention_mask.ndim != 2:
            raise ValueError(f"Expected attention_mask [B,L], got {tuple(attention_mask.shape)}")

        batch_size, _ = attention_mask.shape
        valid_lengths = attention_mask.sum(dim=1).long()
        spans = torch.zeros(batch_size, 2, dtype=torch.long, device=attention_mask.device)

        for row in range(batch_size):
            length = int(valid_lengths[row].item())
            if length <= 1:
                spans[row] = torch.tensor([0, 1], device=attention_mask.device)
                continue

            ratio_length = int(length * self.span_ratio)
            span_length = max(self.min_tokens, ratio_length)
            span_length = min(span_length, self.max_tokens, length)
            span_length = max(span_length, 1)

            max_start = max(length - span_length, 0)
            if max_start == 0:
                start = 0
            else:
                start = torch.randint(0, max_start + 1, (1,), device=attention_mask.device).item()
            end = start + span_length
            spans[row, 0] = start
            spans[row, 1] = end

        return spans

    @staticmethod
    def spans_to_mask(spans: torch.Tensor, seq_length: int) -> torch.Tensor:
        """Converts [B,2] spans to boolean mask [B,L]."""
        if spans.ndim != 2 or spans.shape[1] != 2:
            raise ValueError(f"Expected spans [B,2], got {tuple(spans.shape)}")
        if seq_length <= 0:
            raise ValueError("seq_length must be > 0")

        positions = torch.arange(seq_length, device=spans.device).unsqueeze(0)
        starts = spans[:, 0].unsqueeze(1)
        ends = spans[:, 1].unsqueeze(1)
        return (positions >= starts) & (positions < ends)
