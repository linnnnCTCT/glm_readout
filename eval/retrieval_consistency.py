"""Unsupervised retrieval / consistency evaluation on exported embeddings."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EmbeddingView:
    name: str
    path: Path
    embeddings: torch.Tensor
    ids: list[str]
    genome_ids: list[str]
    buckets: list[str]
    windows: list[tuple[int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unsupervised retrieval / consistency evaluation.")
    parser.add_argument(
        "--embeddings",
        nargs="+",
        required=True,
        help="Embedding sources. Use either /path/to/embeddings.pt or name=/path/to/embeddings.pt",
    )
    parser.add_argument(
        "--topk",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="Top-k retrieval metrics to report.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=2000,
        help="Max positive pairs per source-pair to keep in the JSON dump.",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def infer_view_name(path: Path) -> str:
    parent_name = path.parent.name
    if parent_name and parent_name != ".":
        return parent_name
    return path.stem


def parse_embedding_arg(text: str) -> tuple[str, Path]:
    if "=" in text:
        name, raw_path = text.split("=", 1)
        path = Path(raw_path)
        return name.strip(), path
    path = Path(text)
    return infer_view_name(path), path


def parse_sample_id(sample_id: str) -> tuple[str, str, tuple[int, int]]:
    try:
        genome_id, bucket, span = sample_id.rsplit("__", 2)
        start_text, end_text = span.split("_", 1)
        return genome_id, bucket, (int(start_text), int(end_text))
    except ValueError as exc:
        raise ValueError(
            f"Failed to parse sample id '{sample_id}'. Expected genome__bucket__start_end."
        ) from exc


def load_view(name: str, path: Path) -> EmbeddingView:
    payload = torch.load(path, map_location="cpu")
    embeddings = payload["embeddings"].float()
    ids = [str(item) for item in payload["ids"]]

    genome_ids: list[str] = []
    buckets: list[str] = []
    windows: list[tuple[int, int]] = []
    for sample_id in ids:
        genome_id, bucket, window = parse_sample_id(sample_id)
        genome_ids.append(genome_id)
        buckets.append(bucket)
        windows.append(window)

    embeddings = F.normalize(embeddings, p=2, dim=-1)
    return EmbeddingView(
        name=name,
        path=path,
        embeddings=embeddings,
        ids=ids,
        genome_ids=genome_ids,
        buckets=buckets,
        windows=windows,
    )


def build_genome_index(genome_ids: list[str]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for row, genome_id in enumerate(genome_ids):
        index.setdefault(genome_id, []).append(row)
    return index


def tensor_mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def evaluate_pair(
    query_view: EmbeddingView,
    target_view: EmbeddingView,
    topk_values: list[int],
    max_pairs: int,
) -> dict[str, Any]:
    target_index = build_genome_index(target_view.genome_ids)

    query_rows: list[int] = []
    positive_targets: list[list[int]] = []
    shared_genome_ids: list[str] = []
    for row, genome_id in enumerate(query_view.genome_ids):
        if genome_id in target_index:
            query_rows.append(row)
            positive_targets.append(target_index[genome_id])
            shared_genome_ids.append(genome_id)

    num_queries = len(query_rows)
    if num_queries == 0:
        return {
            "query_view": query_view.name,
            "target_view": target_view.name,
            "num_queries": 0,
            "topk": {},
            "mrr": None,
            "positive_cosine_mean": None,
            "negative_cosine_mean": None,
            "cosine_gap_mean": None,
            "consistency_examples": [],
        }

    query_emb = query_view.embeddings[query_rows]
    target_emb = target_view.embeddings
    sim = query_emb @ target_emb.T
    positive_scores = torch.empty(num_queries, dtype=sim.dtype)
    best_positive_cols: list[int] = []
    positive_col_sets = [set(indices) for indices in positive_targets]

    for row, cols in enumerate(positive_targets):
        col_tensor = torch.tensor(cols, dtype=torch.long)
        positive_values = sim[row, col_tensor]
        best_offset = int(torch.argmax(positive_values).item())
        best_col = cols[best_offset]
        best_positive_cols.append(best_col)
        positive_scores[row] = positive_values[best_offset]

    ranking = sim.argsort(dim=1, descending=True)
    topk_metrics: dict[str, float] = {}
    for k in sorted(set(topk_values)):
        top_hits = ranking[:, : min(k, ranking.shape[1])]
        success_flags: list[float] = []
        for row in range(num_queries):
            hit = any(int(candidate) in positive_col_sets[row] for candidate in top_hits[row].tolist())
            success_flags.append(1.0 if hit else 0.0)
        success = float(sum(success_flags) / max(len(success_flags), 1))
        topk_metrics[f"recall@{k}"] = float(success)

    rank_values: list[float] = []
    for row in range(num_queries):
        min_rank = None
        for rank, candidate in enumerate(ranking[row].tolist(), start=1):
            if int(candidate) in positive_col_sets[row]:
                min_rank = rank
                break
        if min_rank is None:
            raise RuntimeError("Failed to find a positive target in the retrieval ranking.")
        rank_values.append(float(min_rank))
    reciprocal_ranks = 1.0 / torch.tensor(rank_values, dtype=torch.float32)
    mrr = float(reciprocal_ranks.mean().item())

    negative_means: list[float] = []
    cosine_gaps: list[float] = []
    examples: list[dict[str, Any]] = []
    for row in range(num_queries):
        positives = float(positive_scores[row].item())
        row_scores = sim[row]
        mask = torch.ones_like(row_scores, dtype=torch.bool)
        for positive_col in positive_targets[row]:
            mask[positive_col] = False
        negatives = row_scores[mask]
        negative_mean = float(negatives.mean().item()) if negatives.numel() > 0 else 0.0
        hardest_negative = float(negatives.max().item()) if negatives.numel() > 0 else 0.0
        cosine_gap = positives - hardest_negative
        negative_means.append(negative_mean)
        cosine_gaps.append(cosine_gap)

        if len(examples) < max_pairs:
            q_start, q_end = query_view.windows[query_rows[row]]
            best_positive_col = best_positive_cols[row]
            t_start, t_end = target_view.windows[best_positive_col]
            examples.append(
                {
                    "genome_id": shared_genome_ids[row],
                    "query_id": query_view.ids[query_rows[row]],
                    "target_id": target_view.ids[best_positive_col],
                    "query_bucket": query_view.buckets[query_rows[row]],
                    "target_bucket": target_view.buckets[best_positive_col],
                    "query_window": [q_start, q_end],
                    "target_window": [t_start, t_end],
                    "num_positive_targets": len(positive_targets[row]),
                    "positive_cosine": positives,
                    "negative_mean": negative_mean,
                    "hardest_negative_gap": cosine_gap,
                }
            )

    return {
        "query_view": query_view.name,
        "target_view": target_view.name,
        "num_queries": num_queries,
        "topk": topk_metrics,
        "mrr": mrr,
        "positive_cosine_mean": float(positive_scores.mean().item()),
        "negative_cosine_mean": tensor_mean(negative_means),
        "cosine_gap_mean": tensor_mean(cosine_gaps),
        "consistency_examples": examples,
    }


def evaluate_global_consistency(views: list[EmbeddingView]) -> dict[str, Any]:
    genome_to_views: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for view in views:
        for genome_id, embedding in zip(view.genome_ids, view.embeddings):
            genome_to_views.setdefault(genome_id, []).append((view.name, embedding))

    pairwise_cosines: list[float] = []
    genomes_with_multiple_views = 0
    for entries in genome_to_views.values():
        if len(entries) < 2:
            continue
        genomes_with_multiple_views += 1
        for left in range(len(entries)):
            for right in range(left + 1, len(entries)):
                cosine = float((entries[left][1] * entries[right][1]).sum().item())
                pairwise_cosines.append(cosine)

    if not pairwise_cosines:
        return {
            "num_genomes_with_multiple_views": 0,
            "pairwise_cosine_mean": None,
            "pairwise_cosine_std": None,
            "pairwise_cosine_min": None,
            "pairwise_cosine_max": None,
        }

    values = torch.tensor(pairwise_cosines, dtype=torch.float32)
    return {
        "num_genomes_with_multiple_views": genomes_with_multiple_views,
        "pairwise_cosine_mean": float(values.mean().item()),
        "pairwise_cosine_std": float(values.std(unbiased=False).item()),
        "pairwise_cosine_min": float(values.min().item()),
        "pairwise_cosine_max": float(values.max().item()),
    }


def print_pair_summary(payload: dict[str, Any]) -> None:
    print(
        f"{payload['query_view']} -> {payload['target_view']} | "
        f"queries={payload['num_queries']} | "
        f"R@1={payload['topk'].get('recall@1')} | "
        f"R@5={payload['topk'].get('recall@5')} | "
        f"MRR={payload['mrr']} | "
        f"pos_cos={payload['positive_cosine_mean']} | "
        f"gap={payload['cosine_gap_mean']}"
    )


def main() -> None:
    args = parse_args()
    views = [load_view(name=name, path=path) for name, path in (parse_embedding_arg(item) for item in args.embeddings)]

    pair_results: list[dict[str, Any]] = []
    for query_view in views:
        for target_view in views:
            if query_view.name == target_view.name:
                continue
            result = evaluate_pair(
                query_view=query_view,
                target_view=target_view,
                topk_values=args.topk,
                max_pairs=args.max_pairs,
            )
            pair_results.append(result)
            print_pair_summary(result)

    global_consistency = evaluate_global_consistency(views)
    print(
        "global_consistency | "
        f"genomes={global_consistency['num_genomes_with_multiple_views']} | "
        f"mean_cos={global_consistency['pairwise_cosine_mean']} | "
        f"std={global_consistency['pairwise_cosine_std']}"
    )

    output_payload = {
        "views": [
            {
                "name": view.name,
                "path": str(view.path),
                "num_samples": len(view.ids),
                "bucket_examples": sorted(set(view.buckets)),
            }
            for view in views
        ],
        "pair_results": pair_results,
        "global_consistency": global_consistency,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(output_payload, handle, indent=2)
    else:
        print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
