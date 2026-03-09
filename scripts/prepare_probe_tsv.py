"""Prepare split-aware JSONL manifests from a labeled TSV/CSV sequence dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare split manifests from a labeled TSV/CSV.")
    parser.add_argument("--input", type=str, required=True, help="Input TSV/CSV with sequence/label/split.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output root for manifests.")
    parser.add_argument("--task", type=str, default="auto", choices=["auto", "classification", "regression"])
    parser.add_argument("--sequence-column", type=str, default="sequence")
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--split-column", type=str, default="split")
    parser.add_argument("--id-column", type=str, default="protein_id")
    return parser.parse_args()


def normalize_split(raw_value: str) -> str:
    key = raw_value.strip().lower()
    if key not in SPLIT_ALIASES:
        raise ValueError(f"Unsupported split '{raw_value}'. Expected train/validation/test aliases.")
    return SPLIT_ALIASES[key]


def infer_delimiter(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        return ","
    return "\t"


def configure_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def infer_task(label_values: list[str]) -> str:
    parsed_floats: list[float] = []
    for value in label_values:
        try:
            parsed_floats.append(float(value))
        except ValueError:
            return "classification"

    if all(float(item).is_integer() for item in parsed_floats):
        return "classification"
    return "regression"


def build_label_encoder(train_rows: list[dict[str, str]], label_column: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for row in train_rows:
        label = row[label_column]
        if label not in mapping:
            mapping[label] = len(mapping)
    return mapping


def make_sample_id(row: dict[str, str], id_column: str, row_index: int) -> str:
    if id_column in row and row[id_column].strip():
        return row[id_column].strip()
    return f"sample_{row_index:08d}"


def main() -> None:
    args = parse_args()
    configure_csv_field_limit()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    delimiter = infer_delimiter(input_path)
    with input_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError(f"No rows loaded from {input_path}")

    required_columns = {args.sequence_column, args.label_column, args.split_column}
    missing = sorted(required_columns.difference(rows[0].keys()))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for row in rows:
        row[args.split_column] = normalize_split(row[args.split_column])

    task = args.task if args.task != "auto" else infer_task([row[args.label_column] for row in rows])
    train_rows = [row for row in rows if row[args.split_column] == "train"]
    if not train_rows:
        raise ValueError("Training split is empty.")

    label_mapping: dict[str, int] | None = None
    if task == "classification":
        label_mapping = build_label_encoder(train_rows, args.label_column)

    handles = {
        split: (manifest_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "validation", "test")
    }

    split_counts = Counter()
    label_counts = Counter()
    try:
        for row_index, row in enumerate(rows, start=1):
            split = row[args.split_column]
            sequence = row[args.sequence_column].strip()
            if not sequence:
                continue

            raw_label = row[args.label_column].strip()
            if task == "classification":
                if raw_label not in label_mapping:
                    raise ValueError(
                        f"Label '{raw_label}' first appears outside train split; add it to train or predefine mapping."
                    )
                label_value: int | float = label_mapping[raw_label]
            else:
                label_value = float(raw_label)

            sample_id = make_sample_id(row, args.id_column, row_index)
            payload: dict[str, Any] = {
                "sample_id": sample_id,
                "sequence": sequence,
                "label": label_value,
                "meta": {
                    "raw_label": raw_label,
                    "split": split,
                },
            }
            for key, value in row.items():
                if key in {args.sequence_column, args.label_column, args.split_column}:
                    continue
                payload["meta"][key] = value

            handles[split].write(json.dumps(payload, ensure_ascii=False) + "\n")
            split_counts[split] += 1
            label_counts[raw_label] += 1
    finally:
        for handle in handles.values():
            handle.close()

    summary = {
        "input": str(input_path),
        "task": task,
        "split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "label_mapping": label_mapping,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
