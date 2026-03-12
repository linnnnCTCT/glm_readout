"""JEPA trainer with EMA teacher updates and mixed precision."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.masking import build_student_view
from data.span_sampler import ContiguousSpanSampler
from losses.jepa_losses import JEPALoss
from models.jepa import TeacherStudentJEPA
from utils.checkpointing import save_checkpoint


class JEPATrainer:
    """Training loop wrapper for TeacherStudentJEPA."""

    def __init__(
        self,
        model: TeacherStudentJEPA,
        optimizer: torch.optim.Optimizer,
        loss_fn: JEPALoss,
        device: torch.device,
        output_dir: str | Path,
        ema_decay: float = 0.99,
        mask_ratio: float = 0.15,
        noise_std: float = 0.0,
        zero_replace_prob: float = 1.0,
        force_mask_target_span: bool = True,
        grad_clip_norm: float | None = None,
        amp_enabled: bool = True,
        log_interval: int = 50,
        scheduler: Any | None = None,
        train_sampler: Any | None = None,
        is_main_process: bool = True,
        world_size: int = 1,
        wandb_logger: Any | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ema_decay = ema_decay
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std
        self.zero_replace_prob = zero_replace_prob
        self.force_mask_target_span = force_mask_target_span
        self.grad_clip_norm = grad_clip_norm
        self.amp_enabled = amp_enabled and device.type == "cuda"
        self.log_interval = max(log_interval, 1)
        self.scheduler = scheduler
        self.train_sampler = train_sampler
        self.is_main_process = is_main_process
        self.world_size = max(world_size, 1)
        self.wandb_logger = wandb_logger
        self.logger = logging.getLogger(__name__)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.global_step = 0

    def _unwrap_model(self) -> TeacherStudentJEPA:
        return self.model.module if hasattr(self.model, "module") else self.model

    def train(self, train_loader: DataLoader, num_epochs: int) -> None:
        """Runs end-to-end training for `num_epochs`."""
        self.model.to(self.device)
        self.model.train()

        for epoch in range(1, num_epochs + 1):
            if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
                self.train_sampler.set_epoch(epoch)
            metrics = self.train_one_epoch(train_loader=train_loader, epoch=epoch, num_epochs=num_epochs)
            if self.scheduler is not None:
                self.scheduler.step()

            if self.is_main_process:
                checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    epoch=epoch,
                    global_step=self.global_step,
                    extra_state={"metrics": metrics},
                )
                self.logger.info(
                    "Epoch %d/%d finished | loss=%.4f | jepa=%.4f | var=%.4f | cov=%.4f",
                    epoch,
                    num_epochs,
                    metrics["loss"],
                    metrics["loss_jepa"],
                    metrics["loss_var"],
                    metrics["loss_cov"],
                )
                if self.wandb_logger is not None:
                    self.wandb_logger.log(
                        {
                            "epoch": epoch,
                            "epoch/loss": metrics["loss"],
                            "epoch/loss_jepa": metrics["loss_jepa"],
                            "epoch/loss_var": metrics["loss_var"],
                            "epoch/loss_cov": metrics["loss_cov"],
                            "epoch/lr": float(self.optimizer.param_groups[0]["lr"]),
                        },
                        step=self.global_step,
                    )
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

    def train_one_epoch(
        self, train_loader: DataLoader, epoch: int, num_epochs: int
    ) -> dict[str, float]:
        running = {"loss": 0.0, "loss_jepa": 0.0, "loss_var": 0.0, "loss_cov": 0.0}

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs}",
            leave=False,
            disable=not self.is_main_process,
        )
        steps = 0
        for step, batch in enumerate(progress, start=1):
            hidden_states = batch["hidden_states"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            spans = self._unwrap_model().span_sampler.sample(attention_mask=attention_mask)
            forced_mask: torch.Tensor | None = None
            if self.force_mask_target_span:
                forced_mask = ContiguousSpanSampler.spans_to_mask(
                    spans=spans,
                    seq_length=hidden_states.shape[1],
                ) & attention_mask.bool()
            student_view, _ = build_student_view(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                mask_ratio=self.mask_ratio,
                noise_std=self.noise_std,
                zero_replace_prob=self.zero_replace_prob,
                forced_mask=forced_mask,
            )

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                outputs = self.model(
                    full_hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    student_hidden_states=student_view,
                    spans=spans,
                )
                loss_dict = self.loss_fn(
                    pred=outputs["pred"],
                    target=outputs["target"],
                    z_seq=outputs["z_seq_student"],
                )

            self.scaler.scale(loss_dict["loss"]).backward()
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._unwrap_model().update_teacher(decay=self.ema_decay)

            for key in running:
                running[key] += float(loss_dict[key].detach().item())

            self.global_step += 1
            steps = step
            if step % self.log_interval == 0:
                denom = float(step)
                avg_loss = running["loss"] / denom
                avg_jepa = running["loss_jepa"] / denom
                avg_var = running["loss_var"] / denom
                avg_cov = running["loss_cov"] / denom
                progress.set_postfix(
                    loss=avg_loss,
                    jepa=avg_jepa,
                )
                if self.is_main_process and self.wandb_logger is not None:
                    self.wandb_logger.log(
                        {
                            "train/epoch": epoch,
                            "train/step_in_epoch": step,
                            "train/loss": avg_loss,
                            "train/loss_jepa": avg_jepa,
                            "train/loss_var": avg_var,
                            "train/loss_cov": avg_cov,
                            "train/lr": float(self.optimizer.param_groups[0]["lr"]),
                        },
                        step=self.global_step,
                    )

        metrics_tensor = torch.tensor(
            [
                running["loss"],
                running["loss_jepa"],
                running["loss_var"],
                running["loss_cov"],
                float(max(steps, 1)),
            ],
            device=self.device,
            dtype=torch.float64,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        denom = max(float(metrics_tensor[-1].item()), 1.0)
        return {
            "loss": metrics_tensor[0].item() / denom,
            "loss_jepa": metrics_tensor[1].item() / denom,
            "loss_var": metrics_tensor[2].item() / denom,
            "loss_cov": metrics_tensor[3].item() / denom,
        }
