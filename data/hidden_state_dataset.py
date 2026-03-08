"""Dataset for precomputed Genos-m hidden states."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class HiddenStateDataset(Dataset):
    """Loads precomputed hidden states from disk.

    Supported file formats:
    - `.pt` / `.pth`: tensor or dict containing `hidden_states`
    - `.npy`: array of shape [L, D]
    - `.npz`: array under `hidden_key`
    """

    def __init__(
        self,
        data_root: str | Path,
        hidden_key: str = "hidden_states",
        attention_mask_key: str = "attention_mask",
        label_key: str = "label",
        hidden_dtype: torch.dtype | None = torch.float32,
        max_length: int | None = None,
        random_crop: bool = False,
        extensions: tuple[str, ...] = (".pt", ".pth", ".npy", ".npz"),
    ) -> None:
        self.data_root = Path(data_root)
        self.hidden_key = hidden_key
        self.attention_mask_key = attention_mask_key
        self.label_key = label_key
        self.hidden_dtype = hidden_dtype
        self.max_length = max_length
        self.random_crop = random_crop
        self.extensions = extensions
        self.files = self._discover_files()

        if not self.files:
            raise FileNotFoundError(
                f"No hidden-state files found under {self.data_root} with {self.extensions}"
            )

    def _discover_files(self) -> list[Path]:
        files: list[Path] = []
        for extension in self.extensions:
            files.extend(sorted(self.data_root.rglob(f"*{extension}")))
        return sorted(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        file_path = self.files[index]
        payload = self._load_file(file_path)

        hidden_states = self._to_tensor(payload[self.hidden_key], dtype=self.hidden_dtype)
        if hidden_states.ndim != 2:
            raise ValueError(
                f"{file_path} expected hidden_states [L, D], got {tuple(hidden_states.shape)}"
            )

        seq_len = hidden_states.shape[0]
        attention_mask = payload.get(
            self.attention_mask_key, torch.ones(seq_len, dtype=torch.bool)
        )
        attention_mask = self._to_tensor(attention_mask, dtype=torch.bool)
        if attention_mask.shape != (seq_len,):
            raise ValueError(
                f"{file_path} expected attention_mask [L], got {tuple(attention_mask.shape)}"
            )

        hidden_states, attention_mask = self._crop_if_needed(hidden_states, attention_mask)

        sample: dict[str, Any] = {
            "id": file_path.stem,
            "path": str(file_path),
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
        }
        if self.label_key in payload:
            label = payload[self.label_key]
            sample["label"] = self._to_tensor(label)
        return sample

    def _crop_if_needed(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_length is None or hidden_states.shape[0] <= self.max_length:
            return hidden_states, attention_mask

        seq_len = hidden_states.shape[0]
        if self.random_crop:
            start = torch.randint(0, seq_len - self.max_length + 1, (1,)).item()
        else:
            start = 0
        end = start + self.max_length
        return hidden_states[start:end], attention_mask[start:end]

    def _load_file(self, file_path: Path) -> dict[str, Any]:
        suffix = file_path.suffix.lower()
        if suffix in {".pt", ".pth"}:
            payload = torch.load(file_path, map_location="cpu")
            if isinstance(payload, torch.Tensor):
                return {self.hidden_key: payload}
            if isinstance(payload, dict):
                return payload
            raise TypeError(f"Unsupported payload type in {file_path}: {type(payload)}")

        if suffix == ".npy":
            array = np.load(file_path)
            return {self.hidden_key: array}

        if suffix == ".npz":
            archive = np.load(file_path, allow_pickle=False)
            if self.hidden_key not in archive:
                raise KeyError(f"{file_path} does not contain key '{self.hidden_key}'")
            payload: dict[str, Any] = {key: archive[key] for key in archive.files}
            return payload

        raise ValueError(f"Unsupported file extension: {file_path.suffix}")

    @staticmethod
    def _to_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        else:
            tensor = torch.as_tensor(value)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor
