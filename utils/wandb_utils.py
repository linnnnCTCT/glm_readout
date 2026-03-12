"""Optional Weights & Biases logging helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class WandbLogger:
    """Thin wrapper around wandb to keep training code optional-dependency safe."""

    def __init__(
        self,
        enabled: bool,
        project: str | None = None,
        entity: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        mode: str | None = None,
        run_dir: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self._wandb = None
        self._run = None
        if not self.enabled:
            return

        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "wandb logging is enabled but the 'wandb' package is not installed. "
                "Install it with `pip install wandb` or disable wandb.enabled."
            ) from exc

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=tags,
            mode=mode,
            dir=str(run_dir) if run_dir is not None else None,
            config=config,
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.finish()
