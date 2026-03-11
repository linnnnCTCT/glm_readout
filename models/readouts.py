"""Readout heads from pretrained decoder hidden states."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .chunk_pooling import ChunkPooler


def masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Computes mask-aware mean over token dimension."""
    if hidden_states.ndim != 3:
        raise ValueError(f"Expected hidden_states [B,L,D], got {tuple(hidden_states.shape)}")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"Mask shape {tuple(attention_mask.shape)} does not match {tuple(hidden_states.shape[:2])}"
        )
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return (hidden_states * mask).sum(dim=1) / counts


def masked_softmax(scores: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Applies softmax while ignoring invalid positions."""
    if scores.shape != attention_mask.shape:
        raise ValueError(
            f"Scores {tuple(scores.shape)} and attention_mask {tuple(attention_mask.shape)} mismatch"
        )
    scores = scores.masked_fill(~attention_mask.bool(), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    return weights


@dataclass
class ReadoutOutput:
    """Unified readout output container."""

    Z_latent: torch.Tensor
    z_seq: torch.Tensor
    token_embeddings: torch.Tensor
    token_mask: torch.Tensor


class BaseReadout(nn.Module):
    """Base class for sequence readout heads."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        chunk_pooler: ChunkPooler | None = None,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.input_proj = nn.Linear(d_in, d_out)
        self.chunk_pooler = chunk_pooler

    def preprocess_tokens(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected hidden_states [B,L,D], got {tuple(hidden_states.shape)}")
        if hidden_states.shape[-1] != self.d_in:
            raise ValueError(
                f"Expected hidden size D={self.d_in}, got D={hidden_states.shape[-1]}"
            )
        batch_size, seq_length, _ = hidden_states.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_length, dtype=torch.bool, device=hidden_states.device
            )
        if attention_mask.shape != (batch_size, seq_length):
            raise ValueError(
                f"Expected attention_mask [B,L], got {tuple(attention_mask.shape)}"
            )

        pooled_states = hidden_states
        pooled_mask = attention_mask.bool()
        if self.chunk_pooler is not None:
            pooled_states, pooled_mask = self.chunk_pooler(pooled_states, pooled_mask)

        empty_rows = ~pooled_mask.any(dim=1)
        if empty_rows.any():
            pooled_states = pooled_states.clone()
            pooled_mask = pooled_mask.clone()
            pooled_states[empty_rows, 0] = 0.0
            pooled_mask[empty_rows, 0] = True

        token_embeddings = self.input_proj(pooled_states)
        return token_embeddings, pooled_mask

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class MeanPoolingReadout(BaseReadout):
    """Mask-aware mean pooling baseline."""

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        token_embeddings, token_mask = self.preprocess_tokens(hidden_states, attention_mask)
        z_seq = masked_mean(token_embeddings, token_mask)
        z_latent = z_seq.unsqueeze(1)
        return {
            "Z_latent": z_latent,
            "z_seq": z_seq,
            "token_embeddings": token_embeddings,
            "token_mask": token_mask,
        }


class LastTokenReadout(BaseReadout):
    """Last valid token baseline."""

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        token_embeddings, token_mask = self.preprocess_tokens(hidden_states, attention_mask)
        lengths = token_mask.sum(dim=1).clamp(min=1) - 1
        batch_index = torch.arange(token_embeddings.shape[0], device=token_embeddings.device)
        z_seq = token_embeddings[batch_index, lengths]
        z_latent = z_seq.unsqueeze(1)
        return {
            "Z_latent": z_latent,
            "z_seq": z_seq,
            "token_embeddings": token_embeddings,
            "token_mask": token_mask,
        }


class AttentionPoolingReadout(BaseReadout):
    """Learned attention pooling baseline."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        chunk_pooler: ChunkPooler | None = None,
    ) -> None:
        super().__init__(d_in=d_in, d_out=d_out, chunk_pooler=chunk_pooler)
        self.score_proj = nn.Linear(d_out, 1)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        token_embeddings, token_mask = self.preprocess_tokens(hidden_states, attention_mask)
        scores = self.score_proj(token_embeddings).squeeze(-1)
        weights = masked_softmax(scores, token_mask)
        z_seq = (weights.unsqueeze(-1) * token_embeddings).sum(dim=1)
        z_latent = z_seq.unsqueeze(1)
        return {
            "Z_latent": z_latent,
            "z_seq": z_seq,
            "token_embeddings": token_embeddings,
            "token_mask": token_mask,
            "pool_weights": weights,
        }


class GatedAttentionPoolingReadout(BaseReadout):
    """Gated attention pooling with a learnable content gate."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        chunk_pooler: ChunkPooler | None = None,
        gate_hidden_dim: int | None = None,
    ) -> None:
        super().__init__(d_in=d_in, d_out=d_out, chunk_pooler=chunk_pooler)
        hidden_dim = gate_hidden_dim or d_out
        self.attn_proj = nn.Linear(d_out, hidden_dim)
        self.gate_proj = nn.Linear(d_out, hidden_dim)
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        token_embeddings, token_mask = self.preprocess_tokens(hidden_states, attention_mask)
        attn_features = torch.tanh(self.attn_proj(token_embeddings))
        gate_features = torch.sigmoid(self.gate_proj(token_embeddings))
        scores = self.score_proj(attn_features * gate_features).squeeze(-1)
        weights = masked_softmax(scores, token_mask)
        z_seq = (weights.unsqueeze(-1) * token_embeddings).sum(dim=1)
        z_latent = z_seq.unsqueeze(1)
        return {
            "Z_latent": z_latent,
            "z_seq": z_seq,
            "token_embeddings": token_embeddings,
            "token_mask": token_mask,
            "pool_weights": weights,
        }


class QFormerBlock(nn.Module):
    """Cross-attention + self-attention + FFN block for latent queries."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_multiplier, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if context_mask.shape != context_tokens.shape[:2]:
            raise ValueError("context_mask shape must match context_tokens [B,L]")

        key_padding_mask = ~context_mask.bool()

        cross_input = self.norm_cross(query_tokens)
        cross_out, _ = self.cross_attn(
            query=cross_input,
            key=context_tokens,
            value=context_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query_tokens = query_tokens + self.dropout(cross_out)

        self_input = self.norm_self(query_tokens)
        self_out, _ = self.self_attn(
            query=self_input,
            key=self_input,
            value=self_input,
            need_weights=False,
        )
        query_tokens = query_tokens + self.dropout(self_out)

        ffn_out = self.ffn(self.norm_ffn(query_tokens))
        query_tokens = query_tokens + self.dropout(ffn_out)
        return query_tokens


class ReadoutQFormer(BaseReadout):
    """Learnable latent-query readout inspired by Q-Former / Perceiver."""

    def __init__(
        self,
        d_in: int = 1024,
        d_out: int = 512,
        num_queries: int = 32,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
        chunk_pooler: ChunkPooler | None = None,
        seq_pool: str = "mean",
    ) -> None:
        super().__init__(d_in=d_in, d_out=d_out, chunk_pooler=chunk_pooler)
        if num_queries <= 0:
            raise ValueError("num_queries must be > 0")
        if seq_pool not in {"mean", "attn"}:
            raise ValueError(f"Unsupported seq_pool: {seq_pool}")

        self.num_queries = num_queries
        self.seq_pool = seq_pool
        self.query_tokens = nn.Parameter(torch.randn(num_queries, d_out) * 0.02)
        self.blocks = nn.ModuleList(
            [
                QFormerBlock(
                    d_model=d_out,
                    num_heads=num_heads,
                    ffn_multiplier=ffn_multiplier,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.query_norm = nn.LayerNorm(d_out)
        self.query_pool = nn.Linear(d_out, 1) if seq_pool == "attn" else None

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        token_embeddings, token_mask = self.preprocess_tokens(hidden_states, attention_mask)
        batch_size = token_embeddings.shape[0]

        query_tokens = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        for block in self.blocks:
            query_tokens = block(
                query_tokens=query_tokens,
                context_tokens=token_embeddings,
                context_mask=token_mask,
            )
        z_latent = self.query_norm(query_tokens)

        if self.seq_pool == "mean":
            z_seq = z_latent.mean(dim=1)
        else:
            scores = self.query_pool(z_latent).squeeze(-1)
            weights = torch.softmax(scores, dim=-1)
            z_seq = (weights.unsqueeze(-1) * z_latent).sum(dim=1)

        return {
            "Z_latent": z_latent,
            "z_seq": z_seq,
            "token_embeddings": token_embeddings,
            "token_mask": token_mask,
        }
