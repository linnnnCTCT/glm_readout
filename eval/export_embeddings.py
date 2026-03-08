"""Export sequence embeddings using a trained readout head."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any
import sys

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import HiddenStateDataset, hidden_state_collate_fn
from models import build_readout
from utils import load_config, resolve_torch_dtype, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export sequence embeddings.")
    parser.add_argument("--config", type=str, default="configs/v1.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--strict", action="store_true", help="Strict state_dict loading.")
    return parser.parse_args()


def load_student_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "student_readout_state_dict" in payload:
        return payload["student_readout_state_dict"]
    if "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        prefix = "student_readout."
        student_state = {
            key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)
        }
        if student_state:
            return student_state
        return state_dict
    return payload


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(output_path.parent, filename="export.log")
    logger = logging.getLogger("export")

    data_cfg = cfg["data"]
    data_root = args.data_root or data_cfg["data_root"]
    dataset = HiddenStateDataset(
        data_root=data_root,
        hidden_key=data_cfg.get("hidden_key", "hidden_states"),
        attention_mask_key=data_cfg.get("attention_mask_key", "attention_mask"),
        label_key=data_cfg.get("label_key", "label"),
        hidden_dtype=resolve_torch_dtype(data_cfg.get("hidden_dtype", "float32")),
        max_length=data_cfg.get("max_length"),
        random_crop=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=hidden_state_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    readout = build_readout(cfg["model"]).to(device)
    readout.eval()

    payload = torch.load(args.checkpoint, map_location="cpu")
    student_state = load_student_state(payload)
    load_result = readout.load_state_dict(student_state, strict=args.strict)
    if not args.strict:
        if load_result.missing_keys:
            logger.warning("Missing keys during load: %s", load_result.missing_keys)
        if load_result.unexpected_keys:
            logger.warning("Unexpected keys during load: %s", load_result.unexpected_keys)
    logger.info("Loaded readout from %s", args.checkpoint)

    all_embeddings: list[torch.Tensor] = []
    all_ids: list[str] = []
    all_labels: list[torch.Tensor] = []
    has_labels = False

    with torch.no_grad():
        progress = tqdm(dataloader, desc="Export", leave=False)
        for batch in progress:
            hidden_states = batch["hidden_states"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = readout(hidden_states=hidden_states, attention_mask=attention_mask)
            embeddings = outputs["z_seq"].cpu()
            all_embeddings.append(embeddings)
            all_ids.extend(batch["ids"])

            if "labels" in batch:
                has_labels = True
                all_labels.append(batch["labels"].cpu())

    export_payload: dict[str, Any] = {
        "embeddings": torch.cat(all_embeddings, dim=0),
        "ids": all_ids,
    }
    if has_labels:
        export_payload["labels"] = torch.cat(all_labels, dim=0)

    torch.save(export_payload, output_path)
    logger.info(
        "Saved embeddings to %s | num_samples=%d | dim=%d",
        output_path,
        export_payload["embeddings"].shape[0],
        export_payload["embeddings"].shape[1],
    )


if __name__ == "__main__":
    main()
