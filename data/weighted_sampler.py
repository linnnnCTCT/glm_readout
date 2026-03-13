"""Bucket-aware weighted batch sampling for mixed-length curriculum training."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class WeightedBucketBatchSampler(Sampler[list[int]]):
    """Yields single-bucket batches using configured bucket weights.

    Each optimizer step consumes one batch from a single bucket, which avoids
    pathological padding when mixing lengths inside a curriculum phase.
    """

    def __init__(
        self,
        group_to_indices: dict[str, list[int]],
        batch_size: int,
        steps_per_epoch: int,
        bucket_weights: dict[str, float],
        bucket_batch_sizes: dict[str, int] | None = None,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be > 0")
        if world_size <= 0:
            raise ValueError("world_size must be > 0")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be in [0, world_size)")

        normalized_groups = {str(name): list(indices) for name, indices in group_to_indices.items()}
        if not normalized_groups:
            raise ValueError("group_to_indices must not be empty")

        self.group_to_indices = normalized_groups
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)

        self.bucket_names = list(bucket_weights)
        if not self.bucket_names:
            raise ValueError("bucket_weights must not be empty")
        missing = [name for name in self.bucket_names if name not in self.group_to_indices]
        if missing:
            raise KeyError(f"bucket_weights references unknown groups: {missing}")

        weights = torch.tensor([float(bucket_weights[name]) for name in self.bucket_names], dtype=torch.float64)
        if torch.any(weights < 0):
            raise ValueError("bucket_weights must be non-negative")
        if float(weights.sum().item()) <= 0:
            raise ValueError("bucket_weights must sum to > 0")
        self.bucket_probs = (weights / weights.sum()).to(dtype=torch.float32)
        self.bucket_batch_sizes = {
            name: int(bucket_batch_sizes.get(name, self.batch_size)) if bucket_batch_sizes else self.batch_size
            for name in self.bucket_names
        }
        for bucket_name, bucket_batch_size in self.bucket_batch_sizes.items():
            if bucket_batch_size <= 0:
                raise ValueError(f"bucket batch size for '{bucket_name}' must be > 0")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        global_batches = self.steps_per_epoch * self.world_size
        bucket_choices = torch.multinomial(
            self.bucket_probs,
            num_samples=global_batches,
            replacement=True,
            generator=generator,
        ).tolist()

        shuffled_indices: dict[str, torch.Tensor] = {}
        offsets = {name: 0 for name in self.bucket_names}
        for bucket_name in self.bucket_names:
            indices = self.group_to_indices[bucket_name]
            if not indices:
                raise ValueError(f"Bucket '{bucket_name}' has no dataset items")
            permutation = torch.randperm(len(indices), generator=generator)
            shuffled_indices[bucket_name] = torch.tensor(indices, dtype=torch.long)[permutation]

        def draw_from_bucket(bucket_name: str) -> list[int]:
            batch_size = self.bucket_batch_sizes[bucket_name]
            current = shuffled_indices[bucket_name]
            offset = offsets[bucket_name]
            if offset + batch_size > current.numel():
                permutation = torch.randperm(len(self.group_to_indices[bucket_name]), generator=generator)
                current = torch.tensor(
                    self.group_to_indices[bucket_name], dtype=torch.long
                )[permutation]
                shuffled_indices[bucket_name] = current
                offset = 0
            batch = current[offset : offset + batch_size].tolist()
            offsets[bucket_name] = offset + batch_size
            return batch

        for batch_index in range(self.rank, global_batches, self.world_size):
            bucket_name = self.bucket_names[bucket_choices[batch_index]]
            yield draw_from_bucket(bucket_name)
