"""Train JEPA readout on precomputed Genos-m hidden states."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import (
    ContiguousSpanSampler,
    HiddenStateDataset,
    WeightedBucketBatchSampler,
    hidden_state_collate_fn,
)
from losses import JEPALoss
from models import TeacherStudentJEPA, build_readout
from trainers import JEPATrainer
from utils import (
    cleanup_distributed,
    init_distributed,
    is_main_process,
    load_checkpoint,
    load_config,
    resolve_torch_dtype,
    setup_logging,
    set_seed,
    WandbLogger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Genos-m readout JEPA v1.")
    parser.add_argument("--config", type=str, default="configs/v1.yaml", help="Path to config.")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Optional key=value config overrides, e.g. training.lr=1e-4",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Optional model checkpoint to initialize weights from before training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    distributed, rank, local_rank, world_size = init_distributed()
    main_process = is_main_process(rank)

    experiment_cfg = cfg.get("experiment", {})
    output_dir = Path(args.output_dir or experiment_cfg.get("output_dir", "outputs/genosm_jepa_v1"))
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, is_main_process=main_process)
    logger = logging.getLogger("train")

    config_dump_path = output_dir / "resolved_config.json"
    if main_process:
        with config_dump_path.open("w", encoding="utf-8") as handle:
            json.dump(cfg, handle, indent=2)

    seed = int(experiment_cfg.get("seed", 42))
    set_seed(seed + rank)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    else:
        device = torch.device("cpu")
    logger.info(
        "Using device: %s | distributed=%s | rank=%d | world_size=%d",
        device,
        distributed,
        rank,
        world_size,
    )

    wandb_cfg = cfg.get("wandb", {})
    wandb_logger = None
    if main_process and bool(wandb_cfg.get("enabled", False)):
        default_run_name = output_dir.name or experiment_cfg.get("name", "genosm_readout_jepa")
        wandb_logger = WandbLogger(
            enabled=True,
            project=wandb_cfg.get("project"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("name", default_run_name),
            tags=list(wandb_cfg.get("tags", [])),
            mode=wandb_cfg.get("mode"),
            run_dir=wandb_cfg.get("dir", output_dir),
            config=cfg,
        )

    data_cfg = cfg["data"]
    data_root = data_cfg.get("data_roots", data_cfg["data_root"])
    data_root_names = data_cfg.get("data_root_names")
    dataset = HiddenStateDataset(
        data_root=data_root,
        hidden_key=data_cfg.get("hidden_key", "hidden_states"),
        attention_mask_key=data_cfg.get("attention_mask_key", "attention_mask"),
        label_key=data_cfg.get("label_key", "label"),
        hidden_dtype=resolve_torch_dtype(data_cfg.get("hidden_dtype", "float32")),
        max_length=data_cfg.get("max_length"),
        random_crop=bool(data_cfg.get("random_crop", False)),
        data_root_names=data_root_names,
    )
    sampler = None
    batch_sampler = None
    num_workers = int(data_cfg.get("num_workers", 0))
    batch_size = int(cfg["training"].get("batch_size", 2))
    bucket_weights = data_cfg.get("bucket_weights")
    steps_per_epoch = data_cfg.get("steps_per_epoch")
    if bucket_weights is not None:
        if steps_per_epoch is None:
            raise ValueError("data.steps_per_epoch is required when using data.bucket_weights")
        bucket_batch_sizes_cfg = data_cfg.get("bucket_batch_sizes")
        batch_sampler = WeightedBucketBatchSampler(
            group_to_indices=dataset.group_to_indices,
            batch_size=batch_size,
            steps_per_epoch=int(steps_per_epoch),
            bucket_weights={str(key): float(value) for key, value in bucket_weights.items()},
            bucket_batch_sizes=(
                {str(key): int(value) for key, value in bucket_batch_sizes_cfg.items()}
                if bucket_batch_sizes_cfg is not None
                else None
            ),
            seed=seed,
            rank=rank,
            world_size=world_size,
        )
    else:
        sampler = DistributedSampler(dataset, shuffle=True) if distributed else None

    dataloader_kwargs = {
        "dataset": dataset,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "prefetch_factor": int(data_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None,
        "collate_fn": hidden_state_collate_fn,
    }
    if batch_sampler is not None:
        dataloader = DataLoader(
            batch_sampler=batch_sampler,
            **dataloader_kwargs,
        )
    else:
        dataloader = DataLoader(
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=False,
            **dataloader_kwargs,
        )
    logger.info(
        "Loaded %d samples from %s | batch_size_per_rank=%d | num_workers=%d",
        len(dataset),
        data_root,
        batch_size,
        num_workers,
    )

    readout = build_readout(cfg["model"])
    span_cfg = cfg["jepa"]["span"]
    span_sampler = ContiguousSpanSampler(
        min_tokens=int(span_cfg.get("min_tokens", 64)),
        max_tokens=int(span_cfg.get("max_tokens", 2048)),
        span_ratio=float(span_cfg.get("span_ratio", 0.1)),
    )

    model = TeacherStudentJEPA(
        student_readout=readout,
        d_in=int(cfg["model"].get("d_in", 1024)),
        d_out=int(cfg["model"].get("d_out", 512)),
        predictor_hidden_dim=int(cfg["jepa"].get("predictor_hidden_dim", 1024)),
        use_span_position=bool(cfg["jepa"].get("use_span_position", True)),
        ema_decay=float(cfg["jepa"].get("ema_decay", 0.99)),
        span_sampler=span_sampler,
    )
    model.to(device)

    init_checkpoint = args.init_checkpoint or cfg["training"].get("init_checkpoint")
    if init_checkpoint:
        payload = load_checkpoint(
            checkpoint_path=init_checkpoint,
            model=model,
            optimizer=None,
            scaler=None,
            map_location="cpu",
        )
        logger.info(
            "Initialized model from checkpoint %s (epoch=%s, global_step=%s)",
            init_checkpoint,
            payload.get("epoch"),
            payload.get("global_step"),
        )
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=bool(cfg["training"].get("ddp_broadcast_buffers", False)),
            find_unused_parameters=bool(
                cfg["training"].get("ddp_find_unused_parameters", True)
            ),
        )

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(cfg["training"].get("lr", 1e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
    )
    loss_fn = JEPALoss(
        variance_weight=float(cfg["loss"].get("variance_weight", 0.0)),
        covariance_weight=float(cfg["loss"].get("covariance_weight", 0.0)),
        variance_floor=float(cfg["loss"].get("variance_floor", 1.0)),
    )

    trainer = JEPATrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        output_dir=output_dir,
        ema_decay=float(cfg["jepa"].get("ema_decay", 0.99)),
        mask_ratio=float(cfg["corruption"].get("mask_ratio", 0.15)),
        noise_std=float(cfg["corruption"].get("noise_std", 0.0)),
        zero_replace_prob=float(cfg["corruption"].get("zero_replace_prob", 1.0)),
        force_mask_target_span=bool(cfg["corruption"].get("mask_target_span", True)),
        grad_clip_norm=cfg["training"].get("grad_clip_norm"),
        amp_enabled=bool(cfg["training"].get("amp", True)),
        log_interval=int(cfg["training"].get("log_interval", 10)),
        train_sampler=batch_sampler if batch_sampler is not None else sampler,
        is_main_process=main_process,
        world_size=world_size,
        wandb_logger=wandb_logger,
    )
    try:
        trainer.train(
            train_loader=dataloader,
            num_epochs=int(cfg["training"].get("epochs", 10)),
        )
        if main_process:
            logger.info("Training completed. Checkpoints saved in %s", output_dir)
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()
        cleanup_distributed()


if __name__ == "__main__":
    main()
