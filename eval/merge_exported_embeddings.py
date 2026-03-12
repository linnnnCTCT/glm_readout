"""Merge multiple exported embedding shards into one payload."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge exported embedding shards.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def load_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if "embeddings" not in payload or "labels" not in payload:
        raise ValueError(f"Invalid embedding payload: {path}")
    return payload


def ensure_equal(name: str, left: Any, right: Any) -> None:
    if left != right:
        raise ValueError(f"Mismatched {name}: {left!r} != {right!r}")


def shard_sort_key(payload: dict[str, Any], fallback_index: int) -> tuple[int, int]:
    shard_index = payload.get("shard_index")
    num_shards = payload.get("num_shards")
    if shard_index is None:
        shard_index = fallback_index
    if num_shards is None:
        num_shards = 1
    return int(num_shards), int(shard_index)


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.inputs]
    payloads = [load_payload(path) for path in input_paths]

    indexed = sorted(
        list(enumerate(zip(input_paths, payloads))),
        key=lambda item: shard_sort_key(item[1][1], item[0]),
    )

    ordered_paths = [path for _, (path, _) in indexed]
    ordered_payloads = [payload for _, (_, payload) in indexed]
    reference = ordered_payloads[0]

    embeddings = []
    labels = []
    ids: list[str] = []
    genome_ids: list[str] = []
    num_samples = 0

    for path, payload in zip(ordered_paths, ordered_payloads):
        ensure_equal("embedding_dim", int(reference["embeddings"].shape[1]), int(payload["embeddings"].shape[1]))
        ensure_equal("label_to_id", reference.get("label_to_id"), payload.get("label_to_id"))
        ensure_equal("id_to_label", reference.get("id_to_label"), payload.get("id_to_label"))

        embeddings.append(payload["embeddings"])
        labels.append(payload["labels"].view(-1))
        ids.extend(list(payload.get("ids", [])))
        genome_ids.extend(list(payload.get("genome_ids", [])))
        num_samples += int(payload["embeddings"].shape[0])

    merged_payload = {
        "embeddings": torch.cat(embeddings, dim=0),
        "labels": torch.cat(labels, dim=0),
        "ids": ids,
        "genome_ids": genome_ids,
        "label_to_id": reference.get("label_to_id"),
        "id_to_label": reference.get("id_to_label"),
        "input_tsv": reference.get("input_tsv"),
        "num_samples": num_samples,
        "merged_from": [str(path.resolve()) for path in ordered_paths],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_payload, output_path)
    print(f"Merged {len(ordered_payloads)} shard(s) into {output_path}")


if __name__ == "__main__":
    main()
