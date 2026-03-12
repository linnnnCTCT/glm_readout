"""Extract hidden states from a local checkpoint using txt or JSONL input."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
import sys
from typing import Any

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import resolve_torch_dtype


logger = logging.getLogger("extract_hidden_states")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract hidden states from a local HF checkpoint.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to local checkpoint folder.")
    parser.add_argument("--input", type=str, required=True, help="Input .txt or .jsonl manifest.")
    parser.add_argument("--input-format", type=str, default="auto", choices=["auto", "txt", "jsonl"])
    parser.add_argument("--sequence-key", type=str, default="sequence", help="Sequence key for JSONL.")
    parser.add_argument("--id-key", type=str, default="sample_id", help="Sample id key for JSONL.")
    parser.add_argument("--label-key", type=str, default="label", help="Optional label key for JSONL.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for per-sample .pt.")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model-dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_available() else "float32",
        help="Model inference dtype: float32 / float16 / bfloat16.",
    )
    parser.add_argument(
        "--save-dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_available() else "float32",
        help="Saved hidden state dtype: float32 / float16 / bfloat16.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose output .pt file already exists.",
    )
    parser.add_argument(
        "--force-causal-lm",
        action="store_true",
        help="Force loading with AutoModelForCausalLM instead of preferring AutoModel.",
    )
    return parser.parse_args()


def infer_input_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    return "txt"


def sanitize_sample_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return text or "sample"


def load_examples(
    input_path: Path,
    input_format: str,
    sequence_key: str,
    id_key: str,
    label_key: str,
) -> list[dict[str, Any]]:
    if input_format == "txt":
        with input_path.open("r", encoding="utf-8") as handle:
            return [
                {"sample_id": f"sample_{index:06d}", "sequence": line.strip()}
                for index, line in enumerate(handle)
                if line.strip()
            ]

    examples: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if sequence_key not in payload:
                raise KeyError(f"Line {line_number} missing sequence key '{sequence_key}'")
            sample_id = payload.get(id_key, f"sample_{line_number:06d}")
            example = {
                "sample_id": str(sample_id),
                "sequence": str(payload[sequence_key]),
            }
            if label_key in payload:
                example["label"] = payload[label_key]

            meta = {key: value for key, value in payload.items() if key not in {sequence_key, id_key}}
            if meta:
                example["meta"] = meta
            examples.append(example)
    return examples


def chunked_examples(examples: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [examples[start : start + batch_size] for start in range(0, len(examples), batch_size)]


def load_backbone(
    model_path: str,
    model_dtype: torch.dtype,
    force_causal_lm: bool,
) -> tuple[torch.nn.Module, str]:
    if not force_causal_lm:
        try:
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=model_dtype,
            )
            return model, "AutoModel"
        except Exception as exc:
            logger.warning(
                "AutoModel load failed for %s; falling back to AutoModelForCausalLM. Error: %s",
                model_path,
                exc,
            )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    return model, "AutoModelForCausalLM"


def extract_last_hidden(outputs: Any) -> torch.Tensor:
    last_hidden = getattr(outputs, "last_hidden_state", None)
    if last_hidden is not None:
        return last_hidden

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states[-1]

    if isinstance(outputs, tuple) and outputs:
        if isinstance(outputs[0], torch.Tensor):
            return outputs[0]

    raise ValueError("Could not extract last hidden state from model outputs.")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    device = torch.device(args.device)

    input_path = Path(args.input)
    input_format = infer_input_format(input_path, args.input_format)
    examples = load_examples(
        input_path=input_path,
        input_format=input_format,
        sequence_key=args.sequence_key,
        id_key=args.id_key,
        label_key=args.label_key,
    )

    if args.skip_existing:
        pending_examples: list[dict[str, Any]] = []
        for example in examples:
            sample_id = sanitize_sample_id(example["sample_id"])
            if not (output_dir / f"{sample_id}.pt").exists():
                pending_examples.append(example)
        examples = pending_examples

    if not examples:
        print("No samples to process.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_dtype = resolve_torch_dtype(args.model_dtype)
    if model_dtype is None:
        raise ValueError(f"Unsupported --model-dtype {args.model_dtype}")

    model, model_loader_name = load_backbone(
        model_path=args.model_path,
        model_dtype=model_dtype,
        force_causal_lm=bool(args.force_causal_lm),
    )
    model.to(device)
    model.eval()
    logger.info("Loaded backbone with %s", model_loader_name)

    save_dtype = resolve_torch_dtype(args.save_dtype)
    if save_dtype is None:
        raise ValueError("--save-dtype must resolve to a concrete torch dtype.")

    for batch_examples in chunked_examples(examples, args.batch_size):
        batch_sequences = [example["sequence"] for example in batch_examples]
        encoded = tokenizer(
            batch_sequences,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(**encoded, use_cache=False, return_dict=True)
            last_hidden = extract_last_hidden(outputs).detach().to(dtype=save_dtype).cpu()
            attention_mask = encoded["attention_mask"].bool().cpu()

        for row, example in enumerate(batch_examples):
            sample_id = sanitize_sample_id(example["sample_id"])
            payload: dict[str, Any] = {
                "hidden_states": last_hidden[row],
                "attention_mask": attention_mask[row],
            }
            if "label" in example:
                payload["label"] = torch.as_tensor(example["label"])
            if "meta" in example:
                payload["meta"] = example["meta"]
            torch.save(payload, output_dir / f"{sample_id}.pt")


if __name__ == "__main__":
    main()
