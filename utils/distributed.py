"""Helpers for single-node distributed training."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def init_distributed() -> tuple[bool, int, int, int]:
    """Initializes torch.distributed from environment if requested."""
    if not dist.is_available():
        return False, 0, 0, 1

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1

    if dist.is_initialized():
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        return True, rank, local_rank, dist.get_world_size()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Destroys the default process group if it exists."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0
