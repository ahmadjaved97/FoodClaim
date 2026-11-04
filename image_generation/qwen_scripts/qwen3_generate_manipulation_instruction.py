"""
Generate Flux manipulation instructions using a Qwen3.5-VL model.

This file reads the target manipulation from the input CSV column
`manipulation_prompt` and generates a Flux inpainting/manipulation instruction
using both the image and the text context. It does not sample from a
manipulative-practices list and does not run a separate candidate-selection
step.

Expected input CSV columns:
    image_path  dataset_name  class_id  is_human  description  gen_model
    manipulation_type  manipulation_prompt

Only these columns are used by the script:
    image_path, dataset_name, description, manipulation_prompt

Output CSV columns:
    image_path, dataset_name, manipulation_prompt, manipulation_instruction

Sample command:
    python manipulation_generation_from_csv2.py \
        --model_id Qwen/Qwen3.5-VL-7B-Instruct \
        --model_family qwen3_5 \
        --prompt_file prompts/select_best_issue.json \
        --input food_descriptions.csv \
        --output food_manipulations.csv \
        --batch_size 4 \
        --save_every 50
"""

import argparse
import csv
import importlib
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("manipulation_run.log"),
    ],
)
log = logging.getLogger(__name__)


# ── Model family registry ─────────────────────────────────────────────────────
MODEL_FAMILY_REGISTRY = {
    "qwen3_5": {
        "class_name": "Qwen3_5ForConditionalGeneration",
        "import_path": "transformers",
    },
}


def load_model_class(family: str):
    """Dynamically import and return the model class for the requested family."""
    if family not in MODEL_FAMILY_REGISTRY:
        log.error(
            f"Unknown --model_family '{family}'. "
            f"Valid options: {list(MODEL_FAMILY_REGISTRY.keys())}"
        )
        sys.exit(1)

    spec = MODEL_FAMILY_REGISTRY[family]
    try:
        module = importlib.import_module(spec["import_path"])
        cls = getattr(module, spec["class_name"])
        log.info(f"Using model class: {spec['import_path']}.{spec['class_name']}")
        return cls
    except AttributeError:
        log.error(
            f"transformers does not expose '{spec['class_name']}'. "
            "Make sure you have a recent enough version installed "
            "(pip install -U transformers)."
        )
        sys.exit(1)


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reset_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Generation hyperparameters ────────────────────────────────────────────────
GENERATION_KWARGS = dict(
    max_new_tokens=3072,
    temperature=0.4,
    top_k=20,
    do_sample=True,
)


# ── Prompt file loading ───────────────────────────────────────────────────────
REQUIRED_PROMPT_KEYS = {"name", "description", "system_prompt", "step2_prompt"}


def load_prompt_file(prompt_path: str) -> dict:
    """
    Loads and validates a prompt JSON file.

    Required JSON keys:
        name, description, system_prompt, step2_prompt

    Runtime placeholders for step2_prompt:
        {caption}, {selected_issue}

    The placeholder name {selected_issue} is kept so existing prompt files can
    be reused. It receives the CSV value from manipulation_prompt.
    """
    path = Path(prompt_path)
    if not path.exists():
        log.error(f"Prompt file not found: {prompt_path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    missing = REQUIRED_PROMPT_KEYS - data.keys()
    if missing:
        log.error(
            f"Prompt file '{prompt_path}' is missing required key(s): {missing}\n"
            f"Required keys: {REQUIRED_PROMPT_KEYS}"
        )
        sys.exit(1)

    log.info(f"Loaded prompt config: '{data['name']}' — {data['description']}")
    return data


# ── Prompt rendering ──────────────────────────────────────────────────────────
def render_generation_prompt(template: str, caption: str, manipulation_prompt: str) -> str:
    try:
        return template.format(
            caption=caption,
            selected_issue=manipulation_prompt,
            manipulation_prompt=manipulation_prompt,
        )
    except KeyError as e:
        log.error(
            f"Generation prompt template is missing placeholder: {e}. "
            "Supported placeholders are {caption}, {selected_issue}, and "
            "{manipulation_prompt}."
        )
        sys.exit(1)


# ── Output cleaner ────────────────────────────────────────────────────────────
_RAMBLE_PHRASES = re.compile(
    r"(got it|let me|let'?s|wait[,.]|wait—|think about|thinking|"
    r"first,|okay,|alright,|sure[,.]|certainly|of course|here is|"
    r"step by step|i need to|i should|i'll|i will|i think|"
    r"looking at|based on|considering|so,|hmm|actually)",
    re.IGNORECASE,
)


def clean_output(text: str) -> str:
    text = text.split("</think>")[-1]
    text = (
        text.replace("<|im_end|>", "")
            .replace("<|im_start|>", "")
            .replace("<|endoftext|>", "")
            .strip()
    )

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    clean_lines = []
    for line in lines:
        if _RAMBLE_PHRASES.search(line):
            log.debug(f"  Dropped ramble line: {line!r}")
            continue
        if line.endswith("?"):
            continue
        clean_lines.append(line)

    answer = clean_lines[-1] if clean_lines else (lines[-1] if lines else text)
    return answer.strip('"').strip("'").strip()


# ── Image path resolver ───────────────────────────────────────────────────────
def resolve_image_path(image_path: str, image_root: str | None) -> str:
    """Validate that an image exists before model inference."""
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


# ── Message / generation helpers ──────────────────────────────────────────────
def build_messages(prompt_text: str, system_prompt: str, image_path: str) -> list:
    """Build a Qwen-VL chat message containing both image and text input."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


def generate_batch(
    model,
    processor,
    messages_list: list,
    gen_kwargs: dict,
    seed: int,
    thinking: bool = False,
) -> list[str]:
    """Run one batched Qwen-VL generation pass using text and image inputs."""
    all_texts = []
    for messages in messages_list:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        all_texts.append(text)

    image_inputs, video_inputs = process_vision_info(messages_list)

    inputs = processor(
        text=all_texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    _reset_seed(seed)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


# ── Core per-batch generation ─────────────────────────────────────────────────
def generate_instructions_for_batch(
    batch_items: list[dict],
    model,
    processor,
    prompt_config: dict,
    seed: int,
) -> list[dict]:
    """
    Generate one manipulation instruction per row using that row's
    manipulation_prompt, description, and resolved image.
    """
    system_prompt = prompt_config["system_prompt"]
    template = prompt_config["step2_prompt"]

    messages_list = []
    for item in batch_items:
        manipulation_prompt = item["manipulation_prompt"].strip()
        log.info(f"  [{item['image_path']}] Using manipulation_prompt: {manipulation_prompt}")
        prompt_text = render_generation_prompt(
            template=template,
            caption=item["description"],
            manipulation_prompt=manipulation_prompt,
        )
        messages_list.append(
            build_messages(
                prompt_text=prompt_text,
                system_prompt=system_prompt,
                image_path=item["resolved_path"],
            )
        )

    raw_outputs = generate_batch(
        model=model,
        processor=processor,
        messages_list=messages_list,
        gen_kwargs=GENERATION_KWARGS,
        seed=seed,
        thinking=False,
    )

    results = []
    for item, raw_output in zip(batch_items, raw_outputs):
        instruction = clean_output(raw_output)
        log.info(f"  [{item['image_path']}] → instruction: {instruction[:120]}...")
        results.append({
            "image_path": item["image_path"],
            "dataset_name": item["dataset_name"],
            "manipulation_prompt": item["manipulation_prompt"],
            "manipulation_instruction": instruction,
        })

    return results


# ── CSV helpers ───────────────────────────────────────────────────────────────
OUTPUT_COLUMNS = [
    "image_path",
    "dataset_name",
    "manipulation_prompt",
    "manipulation_instruction",
]


def load_input_csv(csv_path: str) -> list[dict]:
    """
    Load the input CSV and extract only the columns needed for generation.
    Extra columns are ignored.
    """
    aliases = {
        "image_path": ["image_path", "image", "img_path", "filename", "file_path"],
        "dataset_name": ["dataset_name", "dataset", "source"],
        "description": ["description", "caption", "desc", "text"],
        "manipulation_prompt": ["manipulation_prompt"],
    }

    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        actual_columns = [col.strip() for col in (reader.fieldnames or [])]

        col_map: dict[str, str] = {}
        for internal_key, candidates in aliases.items():
            for candidate in candidates:
                if candidate in actual_columns:
                    col_map[internal_key] = candidate
                    break

        missing_keys = [key for key in aliases if key not in col_map]
        if missing_keys:
            log.error(
                f"Input CSV '{csv_path}' is missing required column(s): {missing_keys}\n"
                f"  Detected columns : {actual_columns}\n"
                f"  Accepted aliases : {aliases}"
            )
            sys.exit(1)

        log.info(
            "CSV column mapping: "
            + ", ".join(f"{key} ← '{value}'" for key, value in col_map.items())
        )

        for row in reader:
            normalised = {
                internal_key: row[actual_col].strip()
                for internal_key, actual_col in col_map.items()
            }
            if not normalised["description"] or normalised["description"] == "ERROR":
                log.warning(
                    f"Skipping {normalised.get('image_path')} — missing or errored description."
                )
                continue
            if not normalised["manipulation_prompt"] or normalised["manipulation_prompt"] == "ERROR":
                log.warning(
                    f"Skipping {normalised.get('image_path')} — missing or errored manipulation_prompt."
                )
                continue
            rows.append(normalised)

    log.info(f"Loaded {len(rows)} valid row(s) from {csv_path}")
    return rows


def resume_key(row: dict) -> tuple[str, str]:
    """Use both image path and manipulation prompt so repeated images are safe."""
    return (row.get("image_path", ""), row.get("manipulation_prompt", ""))


def load_existing_output(csv_path: str) -> dict[tuple[str, str], dict]:
    """Return rows already written, keyed by (image_path, manipulation_prompt)."""
    done = {}
    if not Path(csv_path).exists():
        return done

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [col for col in OUTPUT_COLUMNS if col not in (reader.fieldnames or [])]
        if missing:
            log.warning(
                f"Existing output '{csv_path}' is missing columns {missing}; "
                "resume data will be ignored."
            )
            return done

        for row in reader:
            done[resume_key(row)] = {col: row.get(col, "") for col in OUTPUT_COLUMNS}

    log.info(f"Resume: {len(done)} row(s) already in {csv_path} — will skip these.")
    return done


def write_csv(csv_path: str, rows: list[dict]):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ── Metadata JSON ─────────────────────────────────────────────────────────────
def write_metadata_json(
    args,
    prompt_config: dict,
    run_start: float,
    run_end: float,
    total_rows: int,
    n_processed: int,
    n_skipped: int,
    n_errors: int,
):
    output_path = Path(args.output)
    meta_path = output_path.with_suffix(".meta.json")
    elapsed = run_end - run_start

    metadata = {
        "script": Path(__file__).name,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "model_family": args.model_family,
        "model_class": MODEL_FAMILY_REGISTRY[args.model_family]["class_name"],
        "generation": GENERATION_KWARGS,
        "input_csv": str(Path(args.input).resolve()),
        "output_csv": str(output_path.resolve()),
        "image_root": str(Path(args.image_root).resolve()) if args.image_root else None,
        "manipulation_source": "input_csv.manipulation_prompt",
        "uses_image_input": True,
        "random_seed": args.seed,
        "batch_size": args.batch_size,
        "save_every": args.save_every,
        "summary": {
            "total_rows": total_rows,
            "processed": n_processed,
            "skipped": n_skipped,
            "errors": n_errors,
        },
        "timing": {
            "run_start_utc": datetime.fromtimestamp(run_start, tz=timezone.utc).isoformat(),
            "run_end_utc": datetime.fromtimestamp(run_end, tz=timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "elapsed_human": (
                f"{int(elapsed // 3600):02d}h "
                f"{int((elapsed % 3600) // 60):02d}m "
                f"{elapsed % 60:.1f}s"
            ),
            "avg_seconds_per_image": round(elapsed / n_processed, 2) if n_processed else None,
        },
        "environment": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
        "prompt_file": str(Path(args.prompt_file).resolve()),
        "prompt_name": prompt_config["name"],
        "prompt_description": prompt_config["description"],
        "system_prompt": prompt_config["system_prompt"],
        "step2_prompt_template": prompt_config["step2_prompt"],
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.info(f"Metadata written to: {meta_path}")
    return meta_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Flux manipulation instructions from CSV manipulation_prompt "
            "values using a Qwen3.5-VL model."
        )
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace model ID, e.g. Qwen/Qwen3.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        required=True,
        choices=list(MODEL_FAMILY_REGISTRY.keys()),
        help="Model family to use. This controls which transformers class is imported.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help=(
            "Path to a prompt JSON file. Must contain keys: "
            "name, description, system_prompt, step2_prompt."
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Path to the input CSV file with columns: image_path, dataset_name, "
            "class_id, is_human, description, gen_model, manipulation_type, "
            "manipulation_prompt."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="manipulation_instructions.csv",
        help="Path to the output CSV file. A .meta.json file is also written alongside it.",
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
        help="Rows per forward pass. Default: 1. Increase if VRAM allows.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Checkpoint the output CSV every N newly processed rows. Default: 50.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    log.info(f"Random seed set to {args.seed}")

    prompt_config = load_prompt_file(args.prompt_file)

    ModelClass = load_model_class(args.model_family)
    log.info(f"Loading model '{args.model_id}' (family: {args.model_family})...")
    model = ModelClass.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "left"
    log.info("Model loaded successfully.\n")

    log.info(f"Loading input CSV: {args.input}")
    input_rows = load_input_csv(args.input)

    already_done = load_existing_output(args.output)

    output_rows: list[dict | None] = []
    todo: list[tuple[int, dict]] = []

    for idx, row in enumerate(input_rows):
        key = resume_key(row)
        if key in already_done:
            output_rows.append(already_done[key])
        else:
            output_rows.append(None)
            todo.append((idx, row))

    n_total = len(input_rows)
    n_skipped = n_total - len(todo)
    n_processed = 0
    n_errors = 0

    log.info(
        f"Total rows: {n_total} | To process: {len(todo)} | "
        f"Already done skipped: {n_skipped}\n"
    )

    if not todo:
        log.info("Nothing to process — all rows already present in output CSV.")
        write_csv(args.output, [row for row in output_rows if row is not None])
        return

    run_start = time.time()

    for batch_start in range(0, len(todo), args.batch_size):
        batch_slice = todo[batch_start: batch_start + args.batch_size]
        log.info(
            f"Batch {batch_start // args.batch_size + 1} | "
            f"rows {batch_start + 1}–{min(batch_start + args.batch_size, len(todo))} "
            f"of {len(todo)}"
        )

        resolved_items: list[tuple[int, dict]] = []
        for orig_idx, row in batch_slice:
            try:
                resolved_path = resolve_image_path(row["image_path"], args.image_root)
                resolved_items.append((orig_idx, {**row, "resolved_path": resolved_path}))
            except FileNotFoundError as e:
                log.error(f"  ✗ {e}")
                output_rows[orig_idx] = {
                    "image_path": row["image_path"],
                    "dataset_name": row.get("dataset_name", ""),
                    "manipulation_prompt": row.get("manipulation_prompt", ""),
                    "manipulation_instruction": "ERROR",
                }
                n_errors += 1

        if not resolved_items:
            continue

        try:
            batch_dicts = [item for _, item in resolved_items]
            results = generate_instructions_for_batch(
                batch_items=batch_dicts,
                model=model,
                processor=processor,
                prompt_config=prompt_config,
                seed=args.seed,
            )
            for (orig_idx, _), result in zip(resolved_items, results):
                output_rows[orig_idx] = result
                log.info(f"  ✓ {result['image_path']}")
            n_processed += len(results)

        except torch.cuda.OutOfMemoryError:
            log.error(
                f"  ✗ CUDA OOM on batch starting at index {batch_start}. "
                "Try reducing --batch_size."
            )
            for orig_idx, item in resolved_items:
                output_rows[orig_idx] = {
                    "image_path": item["image_path"],
                    "dataset_name": item.get("dataset_name", ""),
                    "manipulation_prompt": item.get("manipulation_prompt", ""),
                    "manipulation_instruction": "ERROR",
                }
                n_errors += 1

        except Exception as e:
            log.error(f"  ✗ Batch failed: {e}")
            for orig_idx, item in resolved_items:
                output_rows[orig_idx] = {
                    "image_path": item["image_path"],
                    "dataset_name": item.get("dataset_name", ""),
                    "manipulation_prompt": item.get("manipulation_prompt", ""),
                    "manipulation_instruction": "ERROR",
                }
                n_errors += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if n_processed > 0 and n_processed % args.save_every == 0:
            valid_rows = [row for row in output_rows if row is not None]
            write_csv(args.output, valid_rows)
            log.info(
                f"  💾 Checkpoint saved ({n_processed} newly processed rows) → {args.output}"
            )

    run_end = time.time()
    valid_rows = [row for row in output_rows if row is not None]
    write_csv(args.output, valid_rows)

    write_metadata_json(
        args=args,
        prompt_config=prompt_config,
        run_start=run_start,
        run_end=run_end,
        total_rows=n_total,
        n_processed=n_processed,
        n_skipped=n_skipped,
        n_errors=n_errors,
    )

    elapsed = run_end - run_start
    log.info(
        f"\nDone. {n_processed} processed | {n_skipped} skipped | {n_errors} errors | "
        f"Total time: {int(elapsed // 3600):02d}h "
        f"{int((elapsed % 3600) // 60):02d}m {elapsed % 60:.1f}s"
    )
    log.info(f"Output CSV  : {args.output}")
    log.info(f"Metadata    : {Path(args.output).with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()