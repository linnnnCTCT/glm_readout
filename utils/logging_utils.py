"""Logging utilities."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(
    log_dir: str | Path,
    filename: str = "train.log",
    is_main_process: bool = True,
) -> None:
    """Configures console + file logging."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    level = logging.INFO
    if is_main_process:
        handlers.append(logging.FileHandler(log_path / filename))
    else:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
