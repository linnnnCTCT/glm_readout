"""Config loading and lightweight CLI overrides."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    elif config_path.suffix.lower() == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    else:
        raise ValueError("Config must be YAML or JSON")

    overrides = overrides or []
    for item in overrides:
        key, value = _parse_override(item)
        _set_nested(config, key.split("."), value)
    return config


def _parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"Invalid override '{text}'. Expected key=value.")
    key, raw_value = text.split("=", 1)
    try:
        value = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        value = raw_value
    return key.strip(), value


def _set_nested(payload: dict[str, Any], keys: list[str], value: Any) -> None:
    current = payload
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
