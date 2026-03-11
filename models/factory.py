"""Factory helpers for model construction from config dicts."""

from __future__ import annotations

from typing import Any

from .chunk_pooling import ChunkPooler
from .readouts import (
    AttentionPoolingReadout,
    GatedAttentionPoolingReadout,
    LastTokenReadout,
    MeanPoolingReadout,
    ReadoutQFormer,
)


def build_chunk_pooler(cfg: dict[str, Any]) -> ChunkPooler | None:
    if not cfg.get("enabled", True):
        return None
    return ChunkPooler(
        chunk_size=int(cfg.get("chunk_size", 64)),
        stride=int(cfg.get("stride", cfg.get("chunk_size", 64))),
        mode=str(cfg.get("mode", "mean")),
    )


def build_readout(cfg: dict[str, Any]):
    model_type = str(cfg.get("type", "qformer")).lower()
    d_in = int(cfg.get("d_in", 1024))
    d_out = int(cfg.get("d_out", 512))
    chunk_pooler = build_chunk_pooler(cfg.get("chunk_pooling", {"enabled": True}))

    if model_type == "qformer":
        return ReadoutQFormer(
            d_in=d_in,
            d_out=d_out,
            num_queries=int(cfg.get("num_queries", 32)),
            num_layers=int(cfg.get("num_layers", 4)),
            num_heads=int(cfg.get("num_heads", 8)),
            ffn_multiplier=int(cfg.get("ffn_multiplier", 4)),
            dropout=float(cfg.get("dropout", 0.1)),
            chunk_pooler=chunk_pooler,
            seq_pool=str(cfg.get("seq_pool", "mean")),
        )
    if model_type == "mean":
        return MeanPoolingReadout(d_in=d_in, d_out=d_out, chunk_pooler=chunk_pooler)
    if model_type == "last":
        return LastTokenReadout(d_in=d_in, d_out=d_out, chunk_pooler=chunk_pooler)
    if model_type == "attention":
        return AttentionPoolingReadout(d_in=d_in, d_out=d_out, chunk_pooler=chunk_pooler)
    if model_type == "gated_attention":
        return GatedAttentionPoolingReadout(
            d_in=d_in,
            d_out=d_out,
            chunk_pooler=chunk_pooler,
            gate_hidden_dim=cfg.get("gate_hidden_dim"),
        )

    raise ValueError(f"Unknown readout type '{model_type}'")
