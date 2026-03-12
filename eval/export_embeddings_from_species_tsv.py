"""Export qformer-readout embeddings from a GTDB-style two-column TSV."""

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


class SpeciesTsvDataset(Dataset):
    """Reads lines with `SEQUENCE<TAB>GENOME_ID` or whitespace-delimited pairs."""

    def __init__(
        self,
        input_path: str | Path,
        label_to_id: dict[str, int] | None = None,
        id_prefix: str | None = None,
    ) -> None:
        self.input_path = Path(input_path)
        self.rows = self._load_rows()
        self.id_prefix = id_prefix or self.input_path.stem
        self.label_to_id = label_to_id or self._build_label_to_id(self.rows)
        self.id_to_label = [label for label, _ in sorted(self.label_to_id.items(), key=lambda item: item[1])]

    @staticmethod
    def _is_header(fields: list[str]) -> bool:
        if len(fields) < 2:
            return False
        left = fields[0].strip().lower()
        right = fields[1].strip().lower()
        return left in {"sequence", "seq"} or right in {
            "genome",
            "genome_id",
            "species",
            "species_id",
            "label",
            "accession",
        }

    def _parse_line(self, line: str, line_number: int) -> tuple[str, str]:
        fields = line.strip().split()
        if len(fields) < 2:
            raise ValueError(
                f"{self.input_path}:{line_number} expected at least 2 whitespace-delimited columns."
            )
        sequence = fields[0].strip()
        genome_id = fields[1].strip()
        if not sequence or not genome_id:
            raise ValueError(f"{self.input_path}:{line_number} contains an empty sequence or genome id.")
        return sequence, genome_id

    def _load_rows(self) -> list[tuple[str, str, int]]:
        rows: list[tuple[str, str, int]] = []
        with self.input_path.open("r", encoding="utf-8") as handle:
            first_data_line_seen = False
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                fields = line.split()
                if not first_data_line_seen and self._is_header(fields):
                    first_data_line_seen = True
                    continue
                first_data_line_seen = True
                sequence, genome_id = self._parse_line(line, line_number)
                rows.append((sequence, genome_id, line_number))
        if not rows:
            raise ValueError(f"No valid rows loaded from {self.input_path}")
        return rows

    @staticmethod
    def _build_label_to_id(rows: list[tuple[str, str, int]]) -> dict[str, int]:
        labels = sorted({genome_id for _, genome_id, _ in rows})
        return {label: index for index, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence, genome_id, line_number = self.rows[index]
        if genome_id not in self.label_to_id:
            raise KeyError(
                f"Genome id '{genome_id}' from {self.input_path}:{line_number} missing from label mapping."
            )
        return {
            "id": f"{self.id_prefix}:{line_number}",
            "sequence": sequence,
            "genome_id": genome_id,
            "label": self.label_to_id[genome_id],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export qformer-readout embeddings from GTDB-style TSV.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-tsv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--label-map-in", type=str, default=None)
    parser.add_argument("--label-map-out", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--model-dtype", type=str, default="bfloat16")
    parser.add_argument("--save-dtype", type=str, default="float16")
    parser.add_argument("--id-prefix", type=str, default=None)
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


def load_label_mapping(path: str | Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    mapping_path = Path(path)
    with mapping_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "label_to_id" in payload:
        payload = payload["label_to_id"]
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported label map format: {mapping_path}")
    return {str(label): int(index) for label, index in payload.items()}


def save_label_mapping(path: str | Path, label_to_id: dict[str, int]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label_to_id": label_to_id,
        "id_to_label": [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def cast_embeddings(embeddings: torch.Tensor, save_dtype: str) -> torch.Tensor:
    dtype = resolve_torch_dtype(save_dtype)
    if dtype is None:
        return embeddings
    return embeddings.to(dtype=dtype)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    if args.num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(output_path.parent, filename=f"{output_path.stem}.log")
    logger = logging.getLogger("export_species_tsv")

    label_to_id = load_label_mapping(args.label_map_in)
    dataset = SpeciesTsvDataset(
        input_path=args.input_tsv,
        label_to_id=label_to_id,
        id_prefix=args.id_prefix,
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
        return {
            "ids": [item["id"] for item in batch],
            "genome_ids": [item["genome_id"] for item in batch],
            "labels": torch.as_tensor([item["label"] for item in batch], dtype=torch.long),
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"].bool(),
        }

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    all_embeddings: list[torch.Tensor] = []
    all_ids: list[str] = []
    all_genome_ids: list[str] = []
    all_labels: list[torch.Tensor] = []

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

            all_embeddings.append(cast_embeddings(readout_outputs["z_seq"].detach().cpu(), args.save_dtype))
            all_ids.extend(batch["ids"])
            all_genome_ids.extend(batch["genome_ids"])
            all_labels.append(batch["labels"].cpu())

    export_payload: dict[str, Any] = {
        "embeddings": torch.cat(all_embeddings, dim=0),
        "ids": all_ids,
        "genome_ids": all_genome_ids,
        "labels": torch.cat(all_labels, dim=0),
        "label_to_id": dataset.label_to_id,
        "id_to_label": dataset.id_to_label,
        "input_tsv": str(Path(args.input_tsv).resolve()),
        "num_samples": len(dataset),
    }

    torch.save(export_payload, output_path)
    logger.info(
        "Saved embeddings to %s | num_samples=%d | dim=%d | num_classes=%d",
        output_path,
        export_payload["embeddings"].shape[0],
        export_payload["embeddings"].shape[1],
        len(dataset.label_to_id),
    )

    if args.label_map_out:
        save_label_mapping(args.label_map_out, dataset.label_to_id)
        logger.info("Saved label mapping to %s", args.label_map_out)


if __name__ == "__main__":
    main()
