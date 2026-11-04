#!/usr/bin/env python3
"""
Generate manipulated food descriptions from an input CSV using a Qwen3.5-27B text model.

Input CSV columns required:
    image_path, dataset_name, class_id, is_human, description, manipulation_instruction, gen_model

Output CSV columns:
    all input columns + manipulated_description

Example:
    python generate_manipulated_food_descriptions.py \
        --model Qwen/Qwen3.5-27B-Instruct \
        --input input.csv \
        --output output.csv \
        --batch_size 4 \
        --save_every 50 \
        --max_new_tokens 1200 \
        --seed 42

Notes:
    - This is text-only; it does not load or inspect images.
    - The model rewrites the existing food/image description as if the manipulation is visible.
    - Existing output rows are reused when --output already exists and contains non-empty manipulated_description.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer


# ── Model registry ────────────────────────────────────────────────────────────
# The attached Qwen3.5-VL script used Qwen3_5ForConditionalGeneration.
# For text-only models, this script tries that first by default, with an optional
# AutoModelForCausalLM fallback for checkpoints that expose only causal LM APIs.
MODEL_FAMILIES: dict[str, dict[str, str]] = {
    "qwen3_5": {
        "class_name": "Qwen3_5ForConditionalGeneration",
        "import_path": "transformers",
    },
    "auto_causal_lm": {
        "class_name": "AutoModelForCausalLM",
        "import_path": "transformers",
    },
}


# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("manipulated_description_generator")


def setup_logging(log_file: str) -> None:
    log.handlers.clear()
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    log.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    log.addHandler(file_handler)


# ── Constants ─────────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "image_path",
    "dataset_name",
    "class_id",
    # "is_human",
    "description",
    "manipulation_instruction",
    "gen_model",
]
DEFAULT_OUTPUT_COLUMN = "manipulated_description"

SYSTEM_PROMPT = """You are a careful food-image description editor for a food quality-control deepfake detection dataset.
Your job is to rewrite an existing detailed food image description so that it describes the SAME image after exactly one specified visual manipulation has been applied.

Rules:
- Preserve the original food identity, serving vessel, background, camera angle, lighting, composition, and general writing style.
- Incorporate the manipulation as a subtle but visible detail in the most relevant section of the description.
- Do not add unrelated defects, extra objects, new ingredients, people, text, watermarks, or camera changes.
- Do not mention that this is an instruction, manipulation prompt, fake image, AI generation, or dataset item.
- Do not explain your reasoning.
- Return only the final rewritten description, with no markdown fence and no preamble.
"""

USER_PROMPT_TEMPLATE = """Rewrite the original description so it describes the image AFTER the requested manipulation is visible.

ORIGINAL DESCRIPTION:
{description}

REQUESTED VISUAL MANIPULATION:
{manipulation_instruction}

FINAL REWRITTEN DESCRIPTION:"""


# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class RunStats:
    total_rows: int = 0
    already_done: int = 0
    processed: int = 0
    errors: int = 0
    empty_inputs: int = 0


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Model helpers ─────────────────────────────────────────────────────────────
def load_model_class(model_family: str):
    if model_family not in MODEL_FAMILIES:
        raise ValueError(
            f"Unknown --model_family {model_family!r}. "
            f"Choose from: {list(MODEL_FAMILIES)}"
        )
    spec = MODEL_FAMILIES[model_family]
    module = importlib.import_module(spec["import_path"])
    return getattr(module, spec["class_name"])


def load_model_and_tokenizer(args):
    log.info(f"Loading model: {args.model}")
    log.info(f"Model family: {args.model_family}")

    ModelClass = load_model_class(args.model_family)
    model = ModelClass.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.generation_config, "pad_token_id", None) is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    log.info("Model and tokenizer loaded successfully.")
    return model, tokenizer


# ── Prompt / output helpers ───────────────────────────────────────────────────
def build_messages(description: str, manipulation_instruction: str) -> list[dict[str, str]]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        description=description.strip(),
        manipulation_instruction=manipulation_instruction.strip(),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def render_chat(tokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    # Fallback for tokenizers without a chat template.
    return (
        f"System:\n{messages[0]['content']}\n\n"
        f"User:\n{messages[1]['content']}\n\n"
        "Assistant:\n"
    )


def clean_generated_text(text: str) -> str:
    """Remove common chat/thinking artifacts while preserving the description text."""
    text = text.split("</think>")[-1]
    replacements = [
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
        "<|assistant|>",
        "FINAL REWRITTEN DESCRIPTION:",
        "Final rewritten description:",
    ]
    for item in replacements:
        text = text.replace(item, "")
    return text.strip()


def row_key(row: dict[str, Any]) -> str:
    """Stable key for resume support, robust to duplicate image_path values."""
    parts = [str(row.get(col, "")) for col in REQUIRED_COLUMNS]
    joined = "\u241f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def short_preview(text: str, n: int = 100) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:n] + ("..." if len(text) > n else "")


# ── CSV helpers ───────────────────────────────────────────────────────────────
def load_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {path}")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Input CSV is missing required column(s): {missing}. "
                f"Found columns: {reader.fieldnames}"
            )
        rows = [dict(row) for row in reader]
    log.info(f"Loaded {len(rows)} row(s) from {path}")
    return rows


def load_existing_completed_rows(path: str, output_column: str) -> dict[str, dict[str, str]]:
    completed: dict[str, dict[str, str]] = {}
    if not Path(path).exists():
        return completed

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or output_column not in reader.fieldnames:
            return completed
        for row in reader:
            if str(row.get(output_column, "")).strip():
                completed[row_key(row)] = dict(row)

    log.info(f"Resume: found {len(completed)} completed row(s) in {path}")
    return completed


def write_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ── Generation ────────────────────────────────────────────────────────────────
def generate_batch(
    rows: list[dict[str, str]],
    model,
    tokenizer,
    args,
) -> list[str]:
    prompts = []
    for row in rows:
        messages = build_messages(
            description=row.get("description", ""),
            manipulation_instruction=row.get("manipulation_instruction", ""),
        )
        prompts.append(render_chat(tokenizer, messages))

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_tokens,
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    # Greedy decoding is deterministic and best for reproducible dataset creation.
    torch.manual_seed(args.seed)
    with torch.no_grad():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Decode only newly generated tokens.
    input_len = encoded["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_len:]
    raw_outputs = tokenizer.batch_decode(
        new_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return [clean_generated_text(x) for x in raw_outputs]


# ── Run log ───────────────────────────────────────────────────────────────────
def build_run_log(args, stats: RunStats, run_start: float, run_end: float) -> dict[str, Any]:
    elapsed = run_end - run_start
    model_class = MODEL_FAMILIES[args.model_family]["class_name"]
    return {
        "script": Path(__file__).name,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "model_id": args.model,
            "model_family": args.model_family,
            "model_class": model_class,
            "device_map": args.device_map,
            "trust_remote_code": args.trust_remote_code,
        },
        "prompts": {
            "system": SYSTEM_PROMPT,
            "user_template": USER_PROMPT_TEMPLATE,
        },
        "generation": {
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "seed": args.seed,
        },
        "io": {
            "input_csv": str(Path(args.input).resolve()),
            "output_csv": str(Path(args.output).resolve()),
            "output_column": args.output_column,
            "batch_size": args.batch_size,
            "save_every": args.save_every,
        },
        "summary": asdict(stats),
        "timing": {
            "run_start_utc": datetime.fromtimestamp(run_start, tz=timezone.utc).isoformat(),
            "run_end_utc": datetime.fromtimestamp(run_end, tz=timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "elapsed_human": f"{int(elapsed // 3600):02d}h {int((elapsed % 3600) // 60):02d}m {elapsed % 60:.1f}s",
            "avg_seconds_per_processed_row": round(elapsed / stats.processed, 2) if stats.processed else None,
        },
        "environment": {
            "python_torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_names": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
        },
    }


def write_run_log(path: str, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info(f"Run log saved to: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate manipulated food descriptions using Qwen3.5-27B."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-27B-Instruct",
        help="Hugging Face model ID or local path.",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        choices=list(MODEL_FAMILIES),
        default="qwen3_5",
        help="Use qwen3_5 by default; use auto_causal_lm if your checkpoint requires AutoModelForCausalLM.",
    )
    parser.add_argument("--input", type=str, required=True, help="Input CSV path.")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path.")
    parser.add_argument(
        "--output_column",
        type=str,
        default=DEFAULT_OUTPUT_COLUMN,
        help="Name of the added output column.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Rows per generation batch. Increase if VRAM allows.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Checkpoint every N newly processed rows.",
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=4096,
        help="Maximum input tokens after chat formatting. Increase for very long descriptions if the model supports it.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1200,
        help="Maximum generated tokens for each manipulated description.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Generation uses greedy decoding for reproducibility.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map passed to from_pretrained, usually 'auto'.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face loaders if your local model requires it.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="manipulated_description_run.log",
        help="Plain-text log file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
    set_seed(args.seed)

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.save_every < 1:
        raise ValueError("--save_every must be >= 1")

    output_fieldnames = REQUIRED_COLUMNS + [args.output_column]
    run_start = time.time()

    rows = load_csv_rows(args.input)
    completed = load_existing_completed_rows(args.output, args.output_column)

    stats = RunStats(total_rows=len(rows))
    output_rows: list[dict[str, Any] | None] = [None] * len(rows)
    todo: list[tuple[int, dict[str, str]]] = []

    for idx, row in enumerate(rows):
        key = row_key(row)
        if key in completed:
            output_rows[idx] = completed[key]
            stats.already_done += 1
            continue

        description = str(row.get("description", "")).strip()
        instruction = str(row.get("manipulation_instruction", "")).strip()
        if not description or not instruction:
            out = dict(row)
            out[args.output_column] = "ERROR: missing description or manipulation_instruction"
            output_rows[idx] = out
            stats.empty_inputs += 1
            stats.errors += 1
            continue

        todo.append((idx, row))

    log.info(
        f"Total rows: {stats.total_rows} | To process: {len(todo)} | "
        f"Already done: {stats.already_done} | Empty/error inputs: {stats.empty_inputs}"
    )

    if todo:
        model, tokenizer = load_model_and_tokenizer(args)
    else:
        model = tokenizer = None
        log.info("No rows need model inference.")

    for batch_start in range(0, len(todo), args.batch_size):
        batch_items = todo[batch_start : batch_start + args.batch_size]
        batch_rows = [row for _, row in batch_items]
        batch_num = batch_start // args.batch_size + 1
        log.info(
            f"Batch {batch_num} | rows {batch_start + 1}-"
            f"{min(batch_start + args.batch_size, len(todo))} of {len(todo)}"
        )

        try:
            outputs = generate_batch(batch_rows, model, tokenizer, args)
            for (orig_idx, row), generated in zip(batch_items, outputs):
                out = dict(row)
                out[args.output_column] = generated
                output_rows[orig_idx] = out
                stats.processed += 1
                log.info(f"  ✓ {row.get('image_path', '')} → {short_preview(generated)}")
        except Exception as exc:
            log.exception(f"  ✗ Batch failed: {exc}")
            for orig_idx, row in batch_items:
                out = dict(row)
                out[args.output_column] = f"ERROR: {type(exc).__name__}: {exc}"
                output_rows[orig_idx] = out
                stats.errors += 1

        if stats.processed > 0 and stats.processed % args.save_every == 0:
            valid_rows = [r for r in output_rows if r is not None]
            write_csv(args.output, valid_rows, output_fieldnames)
            log.info(f"  💾 Checkpoint saved → {args.output}")

    # Final save.
    valid_rows = [r for r in output_rows if r is not None]
    write_csv(args.output, valid_rows, output_fieldnames)

    run_end = time.time()
    log_path = str(Path(args.output).with_suffix(".json"))
    write_run_log(log_path, build_run_log(args, stats, run_start, run_end))

    elapsed = run_end - run_start
    log.info(
        "Done. "
        f"processed={stats.processed} | already_done={stats.already_done} | "
        f"errors={stats.errors} | empty_inputs={stats.empty_inputs} | "
        f"elapsed={int(elapsed // 3600):02d}h {int((elapsed % 3600) // 60):02d}m {elapsed % 60:.1f}s"
    )
    log.info(f"Output CSV: {args.output}")
    log.info(f"Run log: {log_path}")


if __name__ == "__main__":
    main()