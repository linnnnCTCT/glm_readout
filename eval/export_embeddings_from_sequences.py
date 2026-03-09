"""Export sequence embeddings directly from raw sequence manifests."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import build_readout
from utils import load_config, resolve_torch_dtype, setup_logging


class SequenceManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        sequence_key: str = "sequence",
        id_key: str = "sample_id",
        label_key: str = "label",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.sequence_key = sequence_key
        self.id_key = id_key
        self.label_key = label_key
        self.rows = self._load_rows()

    def _load_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if self.sequence_key not in payload:
                    raise KeyError(
                        f"{self.manifest_path}:{line_number} missing '{self.sequence_key}'"
                    )
                rows.append(payload)
        if not rows:
            raise ValueError(f"No rows loaded from {self.manifest_path}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        payload = self.rows[index]
        sample: dict[str, Any] = {
            "id": str(payload.get(self.id_key, f"sample_{index:08d}")),
            "sequence": str(payload[self.sequence_key]),
        }
        if self.label_key in payload:
            sample["label"] = payload[self.label_key]
        return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export readout embeddings from raw sequences.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--sequence-key", type=str, default="sequence")
    parser.add_argument("--id-key", type=str, default="sample_id")
    parser.add_argument("--label-key", type=str, default="label")
    parser.add_argument("--model-dtype", type=str, default="bfloat16")
    parser.add_argument("--strict", action="store_true")
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
    if args.num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(output_path.parent, filename=f"{output_path.stem}.log")
    logger = logging.getLogger("export_seq")

    dataset = SequenceManifestDataset(
        manifest_path=args.manifest,
        sequence_key=args.sequence_key,
        id_key=args.id_key,
        label_key=args.label_key,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model_dtype = resolve_torch_dtype(args.model_dtype)
    if model_dtype is None:
        raise ValueError(f"Unsupported --model-dtype {args.model_dtype}")

    backbone = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)
    backbone.eval()

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

    max_length = args.max_length or int(cfg["data"].get("max_length", 4096))

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        sequences = [item["sequence"] for item in batch]
        encoded = tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        output: dict[str, Any] = {
            "ids": [item["id"] for item in batch],
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"].bool(),
        }
        labels = [item["label"] for item in batch if "label" in item]
        if labels and len(labels) == len(batch):
            output["labels"] = torch.as_tensor(labels)
        return output

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    all_embeddings: list[torch.Tensor] = []
    all_ids: list[str] = []
    all_labels: list[torch.Tensor] = []
    has_labels = False

    with torch.no_grad():
        progress = tqdm(dataloader, desc="Export", leave=False)
        for batch in progress:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                last_hidden = outputs.hidden_states[-1]
                readout_outputs = readout(hidden_states=last_hidden, attention_mask=attention_mask)

            all_embeddings.append(readout_outputs["z_seq"].detach().cpu())
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
