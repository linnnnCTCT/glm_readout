"""Batch collation utilities."""

from __future__ import annotations

from typing import Any

import torch


def hidden_state_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pads variable-length hidden-state samples to a dense mini-batch."""
    if not batch:
        raise ValueError("Empty batch is not allowed.")

    batch_size = len(batch)
    hidden_dim = batch[0]["hidden_states"].shape[-1]
    max_length = max(sample["hidden_states"].shape[0] for sample in batch)
    hidden_dtype = batch[0]["hidden_states"].dtype

    hidden_states = torch.zeros(batch_size, max_length, hidden_dim, dtype=hidden_dtype)
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    sample_ids: list[str] = []
    paths: list[str] = []

    for row, sample in enumerate(batch):
        sample_hidden = sample["hidden_states"]
        sample_mask = sample["attention_mask"]
        seq_len, dim = sample_hidden.shape
        if dim != hidden_dim:
            raise ValueError(f"Hidden dim mismatch at row {row}: {dim} vs {hidden_dim}")
        if sample_mask.shape != (seq_len,):
            raise ValueError(
                f"Mask shape mismatch at row {row}: {tuple(sample_mask.shape)} vs {(seq_len,)}"
            )

        hidden_states[row, :seq_len] = sample_hidden
        attention_mask[row, :seq_len] = sample_mask.bool()
        sample_ids.append(str(sample.get("id", row)))
        paths.append(str(sample.get("path", "")))

    output: dict[str, Any] = {
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "ids": sample_ids,
        "paths": paths,
    }

    labels = [sample.get("label") for sample in batch if "label" in sample]
    if labels and len(labels) == len(batch):
        output["labels"] = torch.stack([torch.as_tensor(label) for label in labels], dim=0)

    return output
