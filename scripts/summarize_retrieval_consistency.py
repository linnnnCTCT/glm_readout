#!/usr/bin/env python3
"""Summarize retrieval_consistency.json files into compact text/TSV outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize retrieval consistency JSON files.")
    parser.add_argument("inputs", nargs="+", help="One or more retrieval_consistency.json files.")
    parser.add_argument("--tsv-output", type=str, default=None, help="Optional TSV summary output.")
    return parser.parse_args()


def mean_or_none(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def summarize_file(path: Path) -> dict[str, float | str | None]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    pair_results = payload.get("pair_results", [])
    global_consistency = payload.get("global_consistency", {})

    summary = {
        "file": str(path),
        "mean_r1": mean_or_none([item.get("topk", {}).get("recall@1") for item in pair_results]),
        "mean_r5": mean_or_none([item.get("topk", {}).get("recall@5") for item in pair_results]),
        "mean_r10": mean_or_none([item.get("topk", {}).get("recall@10") for item in pair_results]),
        "mean_mrr": mean_or_none([item.get("mrr") for item in pair_results]),
        "mean_pos": mean_or_none([item.get("positive_cosine_mean") for item in pair_results]),
        "mean_neg": mean_or_none([item.get("negative_cosine_mean") for item in pair_results]),
        "mean_gap": mean_or_none([item.get("cosine_gap_mean") for item in pair_results]),
        "global_mean_cos": global_consistency.get("pairwise_cosine_mean"),
        "global_std_cos": global_consistency.get("pairwise_cosine_std"),
    }
    return summary


def main() -> None:
    args = parse_args()
    summaries = [summarize_file(Path(item)) for item in args.inputs]

    print(
        "experiment\tmean_r1\tmean_r5\tmean_r10\tmean_mrr\tmean_pos\tmean_neg\tmean_gap\tglobal_mean_cos\tglobal_std_cos"
    )
    for summary in summaries:
        print(
            "\t".join(
                [
                    summary["file"],
                    fmt(summary["mean_r1"]),
                    fmt(summary["mean_r5"]),
                    fmt(summary["mean_r10"]),
                    fmt(summary["mean_mrr"]),
                    fmt(summary["mean_pos"]),
                    fmt(summary["mean_neg"]),
                    fmt(summary["mean_gap"]),
                    fmt(summary["global_mean_cos"]),
                    fmt(summary["global_std_cos"]),
                ]
            )
        )

    if args.tsv_output:
        output_path = Path(args.tsv_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write(
                "experiment\tmean_r1\tmean_r5\tmean_r10\tmean_mrr\tmean_pos\tmean_neg\tmean_gap\tglobal_mean_cos\tglobal_std_cos\n"
            )
            for summary in summaries:
                handle.write(
                    "\t".join(
                        [
                            str(summary["file"]),
                            fmt(summary["mean_r1"]),
                            fmt(summary["mean_r5"]),
                            fmt(summary["mean_r10"]),
                            fmt(summary["mean_mrr"]),
                            fmt(summary["mean_pos"]),
                            fmt(summary["mean_neg"]),
                            fmt(summary["mean_gap"]),
                            fmt(summary["global_mean_cos"]),
                            fmt(summary["global_std_cos"]),
                        ]
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
