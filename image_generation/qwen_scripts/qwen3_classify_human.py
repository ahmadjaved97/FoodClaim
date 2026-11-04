"""
Classify images for human presence.
Supports both Qwen3-VL and Qwen3.5-VL model families.

Sample commands:

    # Qwen3-VL
    python qwen3vl_classify.py \
        --model Qwen/Qwen3-VL-7B-Thinking \
        --model_family qwen3vl \
        --input dataset.csv \
        --output classified.csv \
        --image_root /path/to/images \
        --save_every 50 \
        --seed 42

    # Qwen3.5-VL
    python qwen3vl_classify.py \
        --model Qwen/Qwen3.5-VL-7B-Instruct \
        --model_family qwen3_5vl \
        --input dataset.csv \
        --output classified.csv \
        --image_root /path/to/images \
        --save_every 50 \
        --seed 42

Input CSV must have columns: image_path, dataset_name, class_id
Output CSV adds columns:     is_human  (values: YES / NO / UNKNOWN / ERROR)

A JSON run log (same name as --output, .json extension) is written at the end
with everything needed to reproduce the run.
"""

import re
import json
import torch
import csv
import argparse
import logging
import importlib
import random
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

# ── Model family registry ─────────────────────────────────────────────────────
# To add a new family: insert one entry here — no other code changes needed.
MODEL_FAMILIES = {
    "qwen3vl": {
        "class_name": "Qwen3VLForConditionalGeneration",
        "import_path": "transformers",
    },
    "qwen3_5vl": {
        "class_name": "Qwen3_5ForConditionalGeneration",
        "import_path": "transformers",
    },
}


def load_model_class(family: str):
    """Dynamically import the correct model class for the given family key."""
    if family not in MODEL_FAMILIES:
        raise ValueError(
            f"Unknown --model_family '{family}'. "
            f"Choose from: {list(MODEL_FAMILIES.keys())}"
        )
    spec = MODEL_FAMILIES[family]
    module = importlib.import_module(spec["import_path"])
    return getattr(module, spec["class_name"])


# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("classify_run.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
INPUT_COLUMNS  = ["image_path", "dataset_name", "class_id"]
OUTPUT_COLUMNS = ["image_path", "dataset_name", "class_id", "is_human"]
VALID_ANSWERS  = {"YES", "NO"}

# ── Prompts ───────────────────────────────────────────────────────────────────
FOOD_SYSTEM_PROMPT = "You are a strict image classification assistant. Answer only with a single word: YES or NO."

FOOD_USER_PROMPT = """Look at this image and answer: Is this a food image?

A food image shows edible items, dishes, meals, beverages, ingredients, or food packaging as the primary subject.

Answer with a single word only:
- YES → the image is primarily about food
- NO  → the image is NOT primarily about food (e.g. landscape, people, objects, animals, etc.)

Your answer (one word only):"""

HUMAN_SYSTEM_PROMPT = "You are a strict image classification assistant. Answer only with a single word: YES or NO."

HUMAN_USER_PROMPT = """Look at this image and answer: Is a human body visible in this image beyond just hands?

Rules:
- If you see a person's face, torso, legs, arms, or full body → answer YES
- If you only see hands or part of a hand (e.g. someone holding food) → answer NO
- If no person is visible at all → answer NO

Answer with a single word only:

Your answer (one word only):"""


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Output cleaner ────────────────────────────────────────────────────────────
def clean_output(text: str) -> str:
    """Strip think blocks and special tokens, return the raw model answer."""
    text = text.split("</think>")[-1]
    text = (
        text.replace("<|im_end|>", "")
            .replace("<|im_start|>", "")
            .replace("<|endoftext|>", "")
    )
    return text.strip()


def parse_yes_no(raw: str) -> str:
    """
    Extract YES or NO from model output.
    Returns 'UNKNOWN' if the answer cannot be determined.
    """
    text = clean_output(raw).upper()

    # Try the first word first (model should give exactly one word)
    first_word = text.split()[0] if text.split() else ""
    if first_word in VALID_ANSWERS:
        return first_word

    # Fallback: scan the first line for YES / NO anywhere
    first_line = text.split("\n")[0]
    for token in VALID_ANSWERS:
        if token in first_line:
            return token

    log.warning(f"  Could not parse YES/NO from: {repr(raw[:120])}")
    return "UNKNOWN"


# ── Message builders ──────────────────────────────────────────────────────────
def build_human_messages(image_path: str) -> list:
    return [
        {"role": "system", "content": HUMAN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text",  "text": HUMAN_USER_PROMPT},
            ],
        },
    ]


# ── Core generate helper ──────────────────────────────────────────────────────
def _generate(model, processor, inputs, max_new_tokens: int, seed: int) -> list:
    """Shared generate call — greedy decoding for maximum determinism."""
    torch.manual_seed(seed)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # greedy — deterministic given fixed seed
        )
    return [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]


# ── Single-image inference ────────────────────────────────────────────────────
def classify_image(
    image_path: str,
    model,
    processor,
    max_new_tokens: int,
    seed: int,
) -> str:
    """
    Run human classification for a single image.
    Returns is_human — YES / NO / UNKNOWN.
    """
    messages = build_human_messages(image_path)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    trimmed = _generate(model, processor, inputs, max_new_tokens, seed)
    raw = processor.batch_decode(
        trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    return parse_yes_no(raw)

# ── Batch inference ───────────────────────────────────────────────────────────
def classify_batch(
    image_paths: list[str],
    model,
    processor,
    max_new_tokens: int,
    seed: int,
) -> list[str]:
    """
    Run human classification for a batch of images.
    Returns list of is_human answers in the same order as image_paths.
    """
    messages_list = [build_human_messages(p) for p in image_paths]
    all_texts, all_image_inputs, all_video_inputs = [], [], []
    for messages in messages_list:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        all_texts.append(text)
        all_image_inputs.extend(image_inputs)
        all_video_inputs.extend(video_inputs if video_inputs else [])

    inputs = processor(
        text=all_texts,
        images=all_image_inputs,
        videos=all_video_inputs if all_video_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    trimmed = _generate(model, processor, inputs, max_new_tokens, seed)
    raws = processor.batch_decode(
        trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return [parse_yes_no(r) for r in raws]

# ── Image path resolver ───────────────────────────────────────────────────────
def resolve_image_path(image_path: str, image_root: str | None) -> str:
    p = Path(image_path)
    if p.is_absolute() and p.exists():
        return str(p)
    if image_root:
        candidate = Path(image_root) / image_path
        if candidate.exists():
            return str(candidate.resolve())
    if p.exists():
        return str(p.resolve())
    raise FileNotFoundError(f"Image not found: {image_path!r} (image_root={image_root!r})")


# ── CSV helpers ───────────────────────────────────────────────────────────────
def load_input_csv(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    log.info(f"Loaded {len(rows)} row(s) from {csv_path}")
    return rows


def load_existing_output(csv_path: str) -> dict[str, dict]:
    """Load already-processed rows keyed by image_path (for resume support)."""
    done = {}
    if not Path(csv_path).exists():
        return done
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done[row["image_path"]] = row
    log.info(f"Resume: {len(done)} row(s) already in {csv_path} — will skip these.")
    return done


def write_csv(csv_path: str, rows: list[dict]):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ── Run log (JSON) ────────────────────────────────────────────────────────────
def build_run_log(
    args,
    run_start: float,
    run_end: float,
    n_total: int,
    n_processed: int,
    n_skipped: int,
    n_errors: int,
) -> dict:
    elapsed = run_end - run_start
    return {
        # ── identity ───────────────────────────────────────────────────────
        "script":            Path(__file__).name,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),

        # ── model ──────────────────────────────────────────────────────────
        "model_id":     args.model,
        "model_family": args.model_family,
        "model_class":  MODEL_FAMILIES[args.model_family]["class_name"],

        # ── prompts ────────────────────────────────────────────────────────
        "prompts": {
            "human": {
                "system": HUMAN_SYSTEM_PROMPT,
                "user":   HUMAN_USER_PROMPT,
            },
        },

        # ── generation settings ────────────────────────────────────────────
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample":      False,
            "seed":           args.seed,
        },

        # ── I/O ────────────────────────────────────────────────────────────
        "input_csv":   str(Path(args.input).resolve()),
        "output_csv":  str(Path(args.output).resolve()),
        "image_root":  str(Path(args.image_root).resolve()) if args.image_root else None,
        "batch_size":  args.batch_size,
        "save_every":  args.save_every,

        # ── run summary ────────────────────────────────────────────────────
        "summary": {
            "total_rows": n_total,
            "processed":  n_processed,
            "skipped":    n_skipped,
            "errors":     n_errors,
        },

        # ── timing ────────────────────────────────────────────────────────
        "timing": {
            "run_start_utc":         datetime.fromtimestamp(run_start, tz=timezone.utc).isoformat(),
            "run_end_utc":           datetime.fromtimestamp(run_end,   tz=timezone.utc).isoformat(),
            "elapsed_seconds":       round(elapsed, 2),
            "elapsed_human":         f"{int(elapsed // 3600):02d}h {int((elapsed % 3600) // 60):02d}m {elapsed % 60:.1f}s",
            "avg_seconds_per_image": round(elapsed / n_processed, 2) if n_processed else None,
        },

        # ── environment ────────────────────────────────────────────────────
        "environment": {
            "pytorch_version": torch.__version__,
            "cuda_available":  torch.cuda.is_available(),
            "gpu_name":        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
    }


def write_run_log(path: str, log_data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    log.info(f"Run log saved to: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Classify images for the is_human flag "
            "using Qwen3-VL or Qwen3.5-VL."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-7B-Thinking",
        help="HuggingFace model ID, e.g. Qwen/Qwen3.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        choices=list(MODEL_FAMILIES.keys()),
        default="qwen3vl",
        help=(
            f"Model class to load. Choices: {list(MODEL_FAMILIES.keys())}. "
            "Use 'qwen3vl' for Qwen3-VL-* models, 'qwen3_5vl' for Qwen3.5-VL-*."
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV with columns: image_path, dataset_name, class_id",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="classified.csv",
        help="Output CSV path. A matching .json run log is written alongside it.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Root directory prepended to relative image paths in the CSV.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Images per forward pass. Default 1 (safest). Increase if VRAM allows.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Checkpoint the output CSV every N newly processed rows (default: 50).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help=(
            "Max tokens to generate per question (default: 128). "
            "Thinking models may emit a <think> block before YES/NO — "
            "increase to 512 if you see truncated outputs."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed to torch before every generate() call (default: 42).",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # ── Load model ────────────────────────────────────────────────────────────
    log.info(f"Loading model '{args.model}' (family: {args.model_family})...")
    ModelClass = load_model_class(args.model_family)
    model = ModelClass.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    log.info("Model loaded successfully.\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    input_rows   = load_input_csv(args.input)
    already_done = load_existing_output(args.output)

    # Preserve original row order; pre-fill done rows, mark todo rows
    output_rows: list[dict | None] = []
    todo: list[tuple[int, dict]] = []   # (original_index, row)

    for idx, row in enumerate(input_rows):
        if row["image_path"] in already_done:
            output_rows.append(already_done[row["image_path"]])
        else:
            output_rows.append(None)    # placeholder — filled in below
            todo.append((idx, row))

    n_total     = len(input_rows)
    n_skipped   = n_total - len(todo)
    n_processed = 0
    n_errors    = 0

    log.info(
        f"Total rows: {n_total} | To process: {len(todo)} | "
        f"Already done (skipped): {n_skipped}\n"
    )

    if not todo:
        log.info("Nothing to process — all rows already present in output CSV.")
        write_csv(args.output, [r for r in output_rows if r is not None])
        return

    run_start = time.time()

    # ── Process ───────────────────────────────────────────────────────────────
    for batch_start in range(0, len(todo), args.batch_size):
        batch_items = todo[batch_start : batch_start + args.batch_size]

        # Resolve paths — skip images that can't be found
        resolved: list[tuple[int, dict, str]] = []
        for orig_idx, row in batch_items:
            try:
                rp = resolve_image_path(row["image_path"], args.image_root)
                resolved.append((orig_idx, row, rp))
            except FileNotFoundError as e:
                log.error(f"  ✗ {e}")
                output_rows[orig_idx] = {
                    "image_path":   row["image_path"],
                    "dataset_name": row.get("dataset_name", ""),
                    "class_id":     row.get("class_id", ""),
                    "is_human":     "ERROR",
                }
                n_errors += 1

        if not resolved:
            continue

        log.info(
            f"Batch {batch_start // args.batch_size + 1} | "
            f"rows {batch_start + 1}–{min(batch_start + args.batch_size, len(todo))} "
            f"of {len(todo)}"
        )

        if args.batch_size == 1:
            orig_idx, row, rp = resolved[0]
            try:
                is_human = classify_image(
                    rp, model, processor, args.max_new_tokens, args.seed
                )
                log.info(
                    f"  ✓ {row['image_path']} → is_human={is_human}"
                )
                output_rows[orig_idx] = {
                    "image_path":   row["image_path"],
                    "dataset_name": row.get("dataset_name", ""),
                    "class_id":     row.get("class_id", ""),
                    "is_human":     is_human,
                }
                n_processed += 1
            except Exception as e:
                log.error(f"  ✗ Failed on {row['image_path']}: {e}")
                output_rows[orig_idx] = {
                    "image_path":   row["image_path"],
                    "dataset_name": row.get("dataset_name", ""),
                    "class_id":     row.get("class_id", ""),
                    "is_human":     "ERROR",
                }
                n_errors += 1

        else:
            try:
                image_paths_batch = [rp for _, _, rp in resolved]
                answers = classify_batch(
                    image_paths_batch, model, processor, args.max_new_tokens, args.seed
                )
                for (orig_idx, row, rp), is_human in zip(resolved, answers):
                    log.info(
                        f"  ✓ {row['image_path']} → is_human={is_human}"
                    )
                    output_rows[orig_idx] = {
                        "image_path":   row["image_path"],
                        "dataset_name": row.get("dataset_name", ""),
                        "class_id":     row.get("class_id", ""),
                        "is_human":     is_human,
                    }
                    n_processed += 1
            except Exception as e:
                log.error(f"  ✗ Batch failed: {e}")
                for orig_idx, row, rp in resolved:
                    output_rows[orig_idx] = {
                        "image_path":   row["image_path"],
                        "dataset_name": row.get("dataset_name", ""),
                        "class_id":     row.get("class_id", ""),
                        "is_human":     "ERROR",
                    }
                    n_errors += 1

        # ── Periodic checkpoint ───────────────────────────────────────────
        if n_processed > 0 and n_processed % args.save_every == 0:
            valid = [r for r in output_rows if r is not None]
            write_csv(args.output, valid)
            log.info(
                f"  💾 Checkpoint saved ({n_processed} newly processed rows) → {args.output}"
            )

    # ── Final save ────────────────────────────────────────────────────────────
    run_end    = time.time()
    valid_rows = [r for r in output_rows if r is not None]
    write_csv(args.output, valid_rows)

    # ── Write run log ─────────────────────────────────────────────────────────
    log_path = str(Path(args.output).with_suffix(".json"))
    log_data = build_run_log(
        args,
        run_start=run_start,
        run_end=run_end,
        n_total=n_total,
        n_processed=n_processed,
        n_skipped=n_skipped,
        n_errors=n_errors,
    )
    write_run_log(log_path, log_data)

    elapsed = run_end - run_start
    log.info(
        f"\nDone. {n_processed} processed | {n_skipped} skipped | {n_errors} errors | "
        f"Total time: {int(elapsed // 3600):02d}h {int((elapsed % 3600) // 60):02d}m {elapsed % 60:.1f}s"
    )
    log.info(f"Output CSV : {args.output}")
    log.info(f"Run log    : {log_path}")


if __name__ == "__main__":
    main()