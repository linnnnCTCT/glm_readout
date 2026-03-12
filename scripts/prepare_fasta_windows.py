"""Build train/val/test JSONL manifests from a FASTA file list."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "val", "test")
NON_ACGTN_RE = re.compile(r"[^ACGTN]")


@dataclass(frozen=True)
class BucketSpec:
    name: str
    length: int
    train_count: int
    val_count: int
    test_count: int

    def count_for_split(self, split: str) -> int:
        return {
            "train": self.train_count,
            "val": self.val_count,
            "test": self.test_count,
        }[split]


@dataclass(frozen=True)
class WindowCountSpec:
    name: str
    train_count: int
    val_count: int
    test_count: int

    def count_for_split(self, split: str) -> int:
        return {
            "train": self.train_count,
            "val": self.val_count,
            "test": self.test_count,
        }[split]


def default_worker_count() -> int:
    return max(1, min(16, os.cpu_count() or 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare JSONL manifests from a FASTA list.")
    parser.add_argument("--fasta-list", type=str, required=True, help="Text file with one FASTA path per line.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for manifests.")
    parser.add_argument(
        "--buckets",
        nargs="+",
        required=True,
        help="Bucket specs: name:length:train_count:val_count:test_count",
    )
    parser.add_argument(
        "--split-ratios",
        nargs=3,
        type=float,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Genome-level split ratios.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concat-contigs", action="store_true", help="Concatenate contigs with spacer gaps.")
    parser.add_argument(
        "--contig-spacer-length",
        type=int,
        default=1,
        help="Number of spacer characters inserted between contigs when concatenating.",
    )
    parser.add_argument(
        "--contig-spacer-char",
        type=str,
        default="#",
        help="Single character inserted between contigs when concatenating.",
    )
    parser.add_argument(
        "--labels-tsv",
        type=str,
        default=None,
        help="Optional TSV or CSV with genome_id,label columns.",
    )
    parser.add_argument(
        "--windows-per-genome",
        nargs="*",
        default=[],
        help="Per-bucket window counts: name:train:val:test. Defaults to 1:1:1 for unspecified buckets.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Number of worker processes for FASTA scanning and window extraction.",
    )
    parser.add_argument(
        "--summary-chunksize",
        type=int,
        default=32,
        help="Chunk size for the genome-summary multiprocessing phase.",
    )
    parser.add_argument(
        "--extract-chunksize",
        type=int,
        default=8,
        help="Chunk size for the window-extraction multiprocessing phase.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress every N completed genomes per phase.",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="independent",
        choices=["independent", "nested"],
        help="Genome selection mode across buckets. 'nested' makes shorter buckets include longer-bucket genomes.",
    )
    parser.add_argument(
        "--reuse-metadata",
        action="store_true",
        help="Reuse existing metadata/genome_metadata.json when compatible.",
    )
    return parser.parse_args()


def parse_bucket_spec(text: str) -> BucketSpec:
    parts = text.split(":")
    if len(parts) != 5:
        raise ValueError(f"Invalid bucket spec '{text}'. Expected name:length:train:val:test.")
    name, length, train_count, val_count, test_count = parts
    return BucketSpec(
        name=name,
        length=int(length),
        train_count=int(train_count),
        val_count=int(val_count),
        test_count=int(test_count),
    )


def parse_window_count_spec(text: str) -> WindowCountSpec:
    parts = text.split(":")
    if len(parts) != 4:
        raise ValueError(f"Invalid window count spec '{text}'. Expected name:train:val:test.")
    name, train_count, val_count, test_count = parts
    return WindowCountSpec(
        name=name,
        train_count=int(train_count),
        val_count=int(val_count),
        test_count=int(test_count),
    )


def resolve_window_count_specs(
    bucket_specs: list[BucketSpec],
    raw_specs: list[str],
) -> dict[str, WindowCountSpec]:
    resolved = {
        bucket.name: WindowCountSpec(name=bucket.name, train_count=1, val_count=1, test_count=1)
        for bucket in bucket_specs
    }
    for text in raw_specs:
        spec = parse_window_count_spec(text)
        if spec.name not in resolved:
            raise ValueError(f"Unknown bucket name '{spec.name}' in --windows-per-genome")
        resolved[spec.name] = spec
    return resolved


def read_fasta_paths(fasta_list_path: Path) -> list[Path]:
    with fasta_list_path.open("r", encoding="utf-8") as handle:
        return [Path(line.strip()) for line in handle if line.strip()]


def normalize_sequence(text: str) -> str:
    return NON_ACGTN_RE.sub("N", text.upper())


def build_genome_jobs(fasta_paths: list[Path], labels: dict[str, str]) -> list[dict[str, Any]]:
    stem_counts = Counter(path.stem for path in fasta_paths)
    jobs: list[dict[str, Any]] = []
    for fasta_path in fasta_paths:
        stem = fasta_path.stem
        if stem_counts[stem] == 1:
            genome_id = stem
        else:
            digest = hashlib.sha1(str(fasta_path).encode("utf-8")).hexdigest()[:8]
            genome_id = f"{stem}_{digest}"

        job: dict[str, Any] = {
            "genome_id": genome_id,
            "fasta_path": str(fasta_path),
        }
        if genome_id in labels:
            job["label"] = labels[genome_id]
        jobs.append(job)
    return jobs


def read_labels(labels_path: Path | None) -> dict[str, str]:
    if labels_path is None:
        return {}

    with labels_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        if not header:
            return {}
        delimiter = "\t" if "\t" in header else ","
        columns = header.split(delimiter)
        try:
            genome_index = columns.index("genome_id")
            label_index = columns.index("label")
        except ValueError as exc:
            raise ValueError("Label file must contain 'genome_id' and 'label' columns.") from exc

        labels: dict[str, str] = {}
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(delimiter)
            labels[parts[genome_index]] = parts[label_index]
        return labels


def summarize_genome(path: str, concat_contigs: bool, contig_spacer_length: int) -> dict[str, int]:
    num_contigs = 0
    max_contig_length = 0
    total_sequence_length = 0
    current_contig_length = 0
    saw_header = False

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if saw_header:
                    total_sequence_length += current_contig_length
                    if current_contig_length > max_contig_length:
                        max_contig_length = current_contig_length
                num_contigs += 1
                current_contig_length = 0
                saw_header = True
            else:
                current_contig_length += len(line)

    if saw_header:
        total_sequence_length += current_contig_length
        if current_contig_length > max_contig_length:
            max_contig_length = current_contig_length

    total_length = total_sequence_length
    if concat_contigs and num_contigs > 1:
        total_length += contig_spacer_length * (num_contigs - 1)
    if not concat_contigs:
        total_length = max_contig_length

    return {
        "num_contigs": num_contigs,
        "max_contig_length": max_contig_length,
        "total_length": total_length,
    }


def summarize_genome_job(job: dict[str, Any]) -> dict[str, Any]:
    summary = summarize_genome(
        path=str(job["fasta_path"]),
        concat_contigs=bool(job["concat_contigs"]),
        contig_spacer_length=int(job["contig_spacer_length"]),
    )
    payload: dict[str, Any] = {
        "genome_id": job["genome_id"],
        "fasta_path": job["fasta_path"],
        **summary,
    }
    if "label" in job:
        payload["label"] = job["label"]
    return payload


def assign_splits(genome_ids: list[str], seed: int, split_ratios: tuple[float, float, float]) -> dict[str, str]:
    train_ratio, val_ratio, test_ratio = split_ratios
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    shuffled = list(genome_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    split_map: dict[str, str] = {}
    for index, genome_id in enumerate(shuffled):
        if index < train_end:
            split_map[genome_id] = "train"
        elif index < val_end:
            split_map[genome_id] = "val"
        else:
            split_map[genome_id] = "test"
    return split_map


def select_genomes_for_bucket(
    bucket: BucketSpec,
    split: str,
    eligible_genome_ids: list[str],
    seed: int,
) -> list[str]:
    target_count = bucket.count_for_split(split)
    if not eligible_genome_ids:
        return []

    rng = random.Random(f"{seed}:{bucket.name}:{split}")
    candidates = list(eligible_genome_ids)
    rng.shuffle(candidates)
    if len(candidates) < target_count:
        print(
            f"[WARN] {bucket.name}/{split} requested {target_count} genomes, "
            f"but only {len(candidates)} are eligible. Using all available genomes.",
            flush=True,
        )
        return candidates
    return candidates[:target_count]


def select_genomes_nested(
    bucket_specs: list[BucketSpec],
    eligible_by_bucket: dict[str, dict[str, list[str]]],
    seed: int,
) -> dict[str, dict[str, list[str]]]:
    selected_by_bucket: dict[str, dict[str, list[str]]] = {bucket.name: {} for bucket in bucket_specs}
    buckets_desc = sorted(bucket_specs, key=lambda bucket: bucket.length, reverse=True)

    for split in SPLITS:
        required_genomes: list[str] = []
        for bucket in buckets_desc:
            eligible = eligible_by_bucket[bucket.name][split]
            eligible_set = set(eligible)
            missing_required = [genome_id for genome_id in required_genomes if genome_id not in eligible_set]
            if missing_required:
                raise ValueError(
                    f"Nested selection failed for bucket={bucket.name} split={split}: "
                    f"{len(missing_required)} longer-bucket genomes are not eligible."
                )

            requested = bucket.count_for_split(split)
            selected = list(required_genomes)
            if requested < len(selected):
                print(
                    f"[WARN] nested selection for {bucket.name}/{split} requested {requested}, "
                    f"but {len(selected)} genomes are already required from longer buckets. "
                    "Keeping the required genomes.",
                    flush=True,
                )
            else:
                remaining = [genome_id for genome_id in eligible if genome_id not in set(selected)]
                rng = random.Random(f"{seed}:{bucket.name}:{split}:nested")
                rng.shuffle(remaining)
                selected.extend(remaining[: max(requested - len(selected), 0)])

            selected_by_bucket[bucket.name][split] = selected
            required_genomes = selected

    return selected_by_bucket


def print_progress(phase: str, completed: int, total: int, started_at: float) -> None:
    elapsed = max(time.time() - started_at, 1e-6)
    rate = completed / elapsed
    print(
        f"[{phase}] {completed}/{total} completed | {rate:.2f} genomes/s | elapsed={elapsed/60:.1f} min",
        flush=True,
    )


def run_parallel_map(
    jobs: list[dict[str, Any]],
    worker_fn,
    workers: int,
    chunksize: int,
    progress_every: int,
    phase: str,
) -> list[dict[str, Any]]:
    if not jobs:
        return []

    total = len(jobs)
    started_at = time.time()
    if workers <= 1:
        results = []
        for index, job in enumerate(jobs, start=1):
            results.append(worker_fn(job))
            if index % progress_every == 0 or index == total:
                print_progress(phase, index, total, started_at)
        return results

    with ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(worker_fn, jobs, chunksize=max(chunksize, 1))
        results = []
        for index, result in enumerate(iterator, start=1):
            results.append(result)
            if index % progress_every == 0 or index == total:
                print_progress(phase, index, total, started_at)
    return results


def load_cached_metadata(metadata_path: Path) -> dict[str, dict[str, Any]]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected cached metadata dict in {metadata_path}")
    return {str(key): dict(value) for key, value in payload.items()}


def build_prepare_settings(
    concat_contigs: bool, contig_spacer_length: int, contig_spacer_char: str
) -> dict[str, Any]:
    return {
        "concat_contigs": bool(concat_contigs),
        "contig_spacer_length": int(contig_spacer_length),
        "contig_spacer_char": str(contig_spacer_char),
    }


def load_prepare_settings(settings_path: Path) -> dict[str, Any]:
    with settings_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {settings_path}")
    return payload


def merge_metadata(
    genome_jobs: list[dict[str, Any]],
    cached_metadata: dict[str, dict[str, Any]],
    concat_contigs: bool,
    contig_spacer_length: int,
    workers: int,
    summary_chunksize: int,
    progress_every: int,
) -> dict[str, dict[str, Any]]:
    jobs_to_compute: list[dict[str, Any]] = []
    genome_metadata: dict[str, dict[str, Any]] = {}

    for job in genome_jobs:
        genome_id = str(job["genome_id"])
        cached = cached_metadata.get(genome_id)
        if cached is not None and cached.get("fasta_path") == job["fasta_path"]:
            cached_copy = dict(cached)
            if "label" in job:
                cached_copy["label"] = job["label"]
            else:
                cached_copy.pop("label", None)
            genome_metadata[genome_id] = cached_copy
            continue

        compute_job = dict(job)
        compute_job["concat_contigs"] = concat_contigs
        compute_job["contig_spacer_length"] = contig_spacer_length
        jobs_to_compute.append(compute_job)

    if jobs_to_compute:
        print(
            f"[summary] computing metadata for {len(jobs_to_compute)} / {len(genome_jobs)} genomes "
            f"with workers={workers}",
            flush=True,
        )
        for result in run_parallel_map(
            jobs=jobs_to_compute,
            worker_fn=summarize_genome_job,
            workers=workers,
            chunksize=summary_chunksize,
            progress_every=progress_every,
            phase="summary",
        ):
            genome_metadata[str(result["genome_id"])] = result
    else:
        print("[summary] reuse_metadata hit for all genomes; no FASTA rescan needed.", flush=True)

    ordered_metadata: dict[str, dict[str, Any]] = {}
    for job in genome_jobs:
        genome_id = str(job["genome_id"])
        ordered_metadata[genome_id] = genome_metadata[genome_id]
    return ordered_metadata


def consume_segment_for_requests(
    segment: str,
    segment_start: int,
    requests: list[dict[str, Any]],
    buffers: list[list[str]],
    active_index: int,
) -> tuple[int, int]:
    if not segment or active_index >= len(requests):
        return active_index, segment_start

    segment_end = segment_start + len(segment)
    while active_index < len(requests) and int(requests[active_index]["window_end"]) <= segment_start:
        active_index += 1

    request_index = active_index
    while request_index < len(requests):
        request = requests[request_index]
        window_start = int(request["window_start"])
        window_end = int(request["window_end"])
        if window_start >= segment_end:
            break

        overlap_start = max(segment_start, window_start)
        overlap_end = min(segment_end, window_end)
        if overlap_start < overlap_end:
            left = overlap_start - segment_start
            right = overlap_end - segment_start
            buffers[request_index].append(segment[left:right])
        request_index += 1

    while active_index < len(requests) and int(requests[active_index]["window_end"]) <= segment_end:
        active_index += 1
    return active_index, segment_end


def extract_concat_window_records(
    path: str,
    requests: list[dict[str, Any]],
    contig_spacer_length: int,
    contig_spacer_char: str,
) -> list[dict[str, Any]]:
    buffers = [[] for _ in requests]
    active_index = 0
    position = 0
    saw_sequence = False
    pending_spacer = False
    spacer = str(contig_spacer_char) * max(contig_spacer_length, 0)

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if saw_sequence and contig_spacer_length > 0:
                    pending_spacer = True
                else:
                    saw_sequence = True
                continue

            if pending_spacer:
                active_index, position = consume_segment_for_requests(
                    segment=spacer,
                    segment_start=position,
                    requests=requests,
                    buffers=buffers,
                    active_index=active_index,
                )
                pending_spacer = False
                if active_index >= len(requests):
                    break

            active_index, position = consume_segment_for_requests(
                segment=normalize_sequence(line),
                segment_start=position,
                requests=requests,
                buffers=buffers,
                active_index=active_index,
            )
            if active_index >= len(requests):
                break

    records: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        sequence = "".join(buffers[index])
        expected_length = int(request["sequence_length"])
        if len(sequence) != expected_length:
            raise ValueError(
                f"Window extraction failed for {request['sample_id']}: "
                f"expected {expected_length}, got {len(sequence)}"
            )
        record = dict(request)
        record["sequence"] = sequence
        records.append(record)
    return records


def extract_longest_contig_records(path: str, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    longest_sequence = ""
    current_chunks: list[str] = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_chunks:
                    candidate = "".join(current_chunks)
                    if len(candidate) > len(longest_sequence):
                        longest_sequence = candidate
                    current_chunks = []
                continue
            current_chunks.append(normalize_sequence(line))

    if current_chunks:
        candidate = "".join(current_chunks)
        if len(candidate) > len(longest_sequence):
            longest_sequence = candidate

    records: list[dict[str, Any]] = []
    for request in requests:
        start = int(request["window_start"])
        end = int(request["window_end"])
        sequence = longest_sequence[start:end]
        expected_length = int(request["sequence_length"])
        if len(sequence) != expected_length:
            raise ValueError(
                f"Window extraction failed for {request['sample_id']}: "
                f"expected {expected_length}, got {len(sequence)}"
            )
        record = dict(request)
        record["sequence"] = sequence
        records.append(record)
    return records


def extract_windows_job(job: dict[str, Any]) -> dict[str, Any]:
    requests = sorted(job["requests"], key=lambda item: (int(item["window_start"]), str(item["sample_id"])))
    if bool(job["concat_contigs"]):
        records = extract_concat_window_records(
            path=str(job["fasta_path"]),
            requests=requests,
            contig_spacer_length=int(job["contig_spacer_length"]),
            contig_spacer_char=str(job.get("contig_spacer_char", "#")),
        )
    else:
        records = extract_longest_contig_records(path=str(job["fasta_path"]), requests=requests)

    return {
        "genome_id": job["genome_id"],
        "records": records,
    }


def sample_window_starts(max_start: int, num_windows: int, rng: random.Random) -> list[int]:
    if max_start < 0:
        return []
    if num_windows <= 0:
        return []

    num_positions = max_start + 1
    if num_positions <= num_windows:
        starts = list(range(num_positions))
        rng.shuffle(starts)
        return sorted(starts)

    starts = rng.sample(range(num_positions), k=num_windows)
    starts.sort()
    return starts


def build_extraction_jobs(
    genome_jobs: list[dict[str, Any]],
    genome_metadata: dict[str, dict[str, Any]],
    selected_by_bucket: dict[str, dict[str, list[str]]],
    bucket_lookup: dict[str, BucketSpec],
    window_count_lookup: dict[str, WindowCountSpec],
    seed: int,
    concat_contigs: bool,
    contig_spacer_length: int,
    contig_spacer_char: str,
) -> list[dict[str, Any]]:
    selected_index: dict[str, list[tuple[str, str]]] = {}
    for bucket_name, split_map_for_bucket in selected_by_bucket.items():
        for split in SPLITS:
            for genome_id in split_map_for_bucket[split]:
                selected_index.setdefault(genome_id, []).append((bucket_name, split))

    extraction_jobs: list[dict[str, Any]] = []
    for job in genome_jobs:
        genome_id = str(job["genome_id"])
        targets = selected_index.get(genome_id)
        if not targets:
            continue

        meta = genome_metadata[genome_id]
        total_length = int(meta["total_length"])
        requests: list[dict[str, Any]] = []
        for bucket_name, split in targets:
            bucket = bucket_lookup[bucket_name]
            window_count = window_count_lookup[bucket_name].count_for_split(split)
            max_start = total_length - bucket.length
            if max_start < 0 or window_count <= 0:
                continue

            rng = random.Random(f"{seed}:{genome_id}:{bucket_name}:{split}")
            for start in sample_window_starts(max_start=max_start, num_windows=window_count, rng=rng):
                end = start + bucket.length
                request: dict[str, Any] = {
                    "sample_id": f"{genome_id}__{bucket_name}__{start}_{end}",
                    "genome_id": genome_id,
                    "fasta_path": meta["fasta_path"],
                    "split": split,
                    "bucket": bucket_name,
                    "window_start": start,
                    "window_end": end,
                    "sequence_length": bucket.length,
                }
                if "label" in meta:
                    request["label"] = meta["label"]
                requests.append(request)

        if not requests:
            continue

        extraction_jobs.append(
            {
                "genome_id": genome_id,
                "fasta_path": meta["fasta_path"],
                "concat_contigs": concat_contigs,
                "contig_spacer_length": contig_spacer_length,
                "contig_spacer_char": contig_spacer_char,
                "requests": requests,
            }
        )
    return extraction_jobs


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir / "manifests"
    metadata_dir = output_dir / "metadata"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    labels = read_labels(Path(args.labels_tsv) if args.labels_tsv else None)
    bucket_specs = [parse_bucket_spec(text) for text in args.buckets]
    bucket_lookup = {bucket.name: bucket for bucket in bucket_specs}
    window_count_lookup = resolve_window_count_specs(
        bucket_specs=bucket_specs,
        raw_specs=list(args.windows_per_genome),
    )
    fasta_paths = read_fasta_paths(Path(args.fasta_list))
    genome_jobs = build_genome_jobs(fasta_paths=fasta_paths, labels=labels)

    metadata_path = metadata_dir / "genome_metadata.json"
    settings_path = metadata_dir / "prepare_settings.json"
    prepare_settings = build_prepare_settings(
        concat_contigs=bool(args.concat_contigs),
        contig_spacer_length=int(args.contig_spacer_length),
        contig_spacer_char=str(args.contig_spacer_char),
    )
    cached_metadata: dict[str, dict[str, Any]] = {}
    if args.reuse_metadata and metadata_path.exists() and settings_path.exists():
        cached_settings = load_prepare_settings(settings_path)
        if cached_settings == prepare_settings:
            print(f"[summary] loading cached metadata from {metadata_path}", flush=True)
            cached_metadata = load_cached_metadata(metadata_path)
        else:
            print(
                "[summary] cached metadata settings mismatch; ignoring cache and rescanning FASTA files.",
                flush=True,
            )

    genome_metadata = merge_metadata(
        genome_jobs=genome_jobs,
        cached_metadata=cached_metadata,
        concat_contigs=bool(args.concat_contigs),
        contig_spacer_length=int(args.contig_spacer_length),
        workers=args.workers,
        summary_chunksize=args.summary_chunksize,
        progress_every=args.progress_every,
    )

    split_map = assign_splits(
        genome_ids=list(genome_metadata),
        seed=args.seed,
        split_ratios=tuple(args.split_ratios),
    )
    for genome_id, split in split_map.items():
        genome_metadata[genome_id]["split"] = split

    eligible_by_bucket: dict[str, dict[str, list[str]]] = {}
    for bucket in bucket_specs:
        eligible_by_bucket[bucket.name] = {}
        for split in SPLITS:
            eligible_by_bucket[bucket.name][split] = [
                genome_id
                for genome_id, meta in genome_metadata.items()
                if meta["split"] == split and int(meta["total_length"]) >= bucket.length
            ]

    if args.selection_mode == "nested":
        selected_by_bucket = select_genomes_nested(
            bucket_specs=bucket_specs,
            eligible_by_bucket=eligible_by_bucket,
            seed=args.seed,
        )
    else:
        selected_by_bucket: dict[str, dict[str, list[str]]] = {}
        for bucket in bucket_specs:
            selected_by_bucket[bucket.name] = {}
            for split in SPLITS:
                selected_by_bucket[bucket.name][split] = select_genomes_for_bucket(
                    bucket=bucket,
                    split=split,
                    eligible_genome_ids=eligible_by_bucket[bucket.name][split],
                    seed=args.seed,
                )

    manifest_handles: dict[tuple[str, str], Any] = {}
    for bucket in bucket_specs:
        split_dir = manifest_dir / bucket.name
        split_dir.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            manifest_handles[(bucket.name, split)] = (split_dir / f"{split}.jsonl").open(
                "w", encoding="utf-8"
            )

    try:
        extraction_jobs = build_extraction_jobs(
            genome_jobs=genome_jobs,
            genome_metadata=genome_metadata,
            selected_by_bucket=selected_by_bucket,
            bucket_lookup=bucket_lookup,
            window_count_lookup=window_count_lookup,
            seed=args.seed,
            concat_contigs=bool(args.concat_contigs),
            contig_spacer_length=int(args.contig_spacer_length),
            contig_spacer_char=str(args.contig_spacer_char),
        )

        print(
            f"[extract] preparing {len(extraction_jobs)} selected genomes with workers={args.workers}",
            flush=True,
        )
        for payload in run_parallel_map(
            jobs=extraction_jobs,
            worker_fn=extract_windows_job,
            workers=args.workers,
            chunksize=args.extract_chunksize,
            progress_every=args.progress_every,
            phase="extract",
        ):
            for record in payload["records"]:
                handle = manifest_handles[(str(record["bucket"]), str(record["split"]))]
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    finally:
        for handle in manifest_handles.values():
            handle.close()

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(genome_metadata, handle, indent=2)
    with settings_path.open("w", encoding="utf-8") as handle:
        json.dump(prepare_settings, handle, indent=2)

    summary: dict[str, dict[str, dict[str, int]]] = {}
    for bucket in bucket_specs:
        summary[bucket.name] = {}
        for split in SPLITS:
            windows_per_genome = window_count_lookup[bucket.name].count_for_split(split)
            selected_genomes = len(selected_by_bucket[bucket.name][split])
            summary[bucket.name][split] = {
                "requested": bucket.count_for_split(split),
                "eligible_genomes": len(eligible_by_bucket[bucket.name][split]),
                "selected_genomes": selected_genomes,
                "windows_per_genome": windows_per_genome,
                "requested_windows": selected_genomes * windows_per_genome,
                "length": bucket.length,
            }
    with (metadata_dir / "bucket_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
