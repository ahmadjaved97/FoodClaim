"""
Supports both Qwen3-VL and Qwen3.5-VL model families.

Sample commands:

    # Qwen3-VL (original folder mode)
    python qwen3_description_generator.py \
        --model Qwen/Qwen3-VL-7B-Thinking \
        --model_family qwen3vl \
        --prompt_file prompts/detailed.json \
        --folder /ssd1/ajaved/thesis/food_detection/sample_data \
        --output food_descriptions.csv

    # CSV mode: only rows where is_human is NOT YES are processed.
    # image_path values in the CSV are resolved relative to --image_root.
    python qwen3_description_generator.py \
        --model Qwen/Qwen3-VL-7B-Thinking \
        --model_family qwen3vl \
        --prompt_file prompts/detailed.json \
        --input_csv input.csv \
        --image_root /ssd1/ajaved/thesis/food_detection \
        --output food_descriptions.csv \
        --save_every 100

    # Qwen3.5-VL
    python qwen3_description_generator.py \
        --model Qwen/Qwen3.5-VL-7B-Instruct \
        --model_family qwen3_5 \
        --prompt_file prompts/minimal.json \
        --folder /ssd1/ajaved/thesis/food_detection/sample_data \
        --output food_descriptions.csv

    # Increase max_new_tokens if you still see truncation:
    python qwen3_description_generator.py ... --max_new_tokens 4096

Prompt file format (.json) — required keys: system_prompt, user_prompt
    {
        "name": "detailed",
        "description": "Full structured description with image quality section.",
        "system_prompt": "You are a food photography analyst ...",
        "user_prompt": "Describe the food image in detail ..."
    }

    user_prompt should not require any class placeholder. The model infers the food/class
    directly from the image.
"""

import re
import json
import torch
import os
import csv
import argparse
import logging
import importlib
from pathlib import Path
from datetime import datetime, timezone

# ── Model family registry ─────────────────────────────────────────────────────
# To add a new family: insert one entry here, no other code changes needed.
MODEL_FAMILIES = {
    "qwen3vl": {
        "class_name": "Qwen3VLForConditionalGeneration",
        "import_path": "transformers",
    },
    "qwen3_5": {
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
        logging.FileHandler("run.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Supported image extensions ────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# ── Preamble-stripping regex ──────────────────────────────────────────────────
# Strips any text appearing before the first "FOOD:" section header.
# If no such header is found the full text is returned unchanged.
_PREAMBLE_RE = re.compile(
    r"^.*?"
    r"(?=FOOD\s*:)",
    re.DOTALL | re.IGNORECASE,
)


# ── Prompt file loader ────────────────────────────────────────────────────────
REQUIRED_PROMPT_KEYS = {"system_prompt", "user_prompt"}


def load_prompt_file(path: str) -> dict:
    """
    Load and validate a prompt JSON file.

    Required keys:  system_prompt, user_prompt
    Optional keys:  name, description  (recorded in the run log only)

    The user_prompt does not need any class placeholder. The model infers the
    food/class directly from the image.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    if p.suffix.lower() != ".json":
        raise ValueError(f"Prompt file must be a .json file, got: {path}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    missing = REQUIRED_PROMPT_KEYS - data.keys()
    if missing:
        raise ValueError(
            f"Prompt file '{path}' is missing required key(s): {missing}\n"
            f"Required: {REQUIRED_PROMPT_KEYS}"
        )
    log.info(
        f"Loaded prompt '{data.get('name', '<unnamed>')}' from {path}"
        + (f": {data['description']}" if "description" in data else "")
    )
    return data


def build_prompt(user_prompt_template: str) -> str:
    """Return the user prompt unchanged; the model infers the image class itself."""
    return user_prompt_template


# ── Output cleaner ────────────────────────────────────────────────────────────
def clean_output(text: str) -> str:
    text = text.split("</think>")[-1]
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("<|endoftext|>", "")
    if re.search(r"FOOD\s*:", text, re.IGNORECASE):
        text = _PREAMBLE_RE.sub("", text)
    #  Strip markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


# ── Message builder ───────────────────────────────────────────────────────────
def build_messages(image_path: str, prompt: dict) -> list:
    """
    Build the chat message list for a single image.
    prompt: the dict returned by load_prompt_file().
    """
    return [
        {
            "role": "system",
            "content": prompt["system_prompt"],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text",  "text": build_prompt(prompt["user_prompt"])},
            ],
        },
    ]


# ── Generation helper ─────────────────────────────────────────────────────────
def _generate(model, processor, inputs, max_new_tokens: int, seed: int) -> list:
    """Shared generate call with deterministic seed."""
    torch.manual_seed(seed)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_k=20,
        )
    return [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]


# ── Single image description ──────────────────────────────────────────────────
def describe_image(
    image_path: str,
    model,
    processor,
    prompt: dict,
    max_new_tokens: int = 3072,
    seed: int = 42,
) -> str:
    from qwen_vl_utils import process_vision_info

    messages = build_messages(image_path, prompt)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    trimmed = _generate(model, processor, inputs, max_new_tokens, seed)
    raw = processor.batch_decode(
        trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    return clean_output(raw)


# ── Batch description ─────────────────────────────────────────────────────────
def describe_batch(
    batch: list[tuple],
    model,
    processor,
    prompt: dict,
    max_new_tokens: int = 3072,
    seed: int = 42,
) -> list[tuple]:
    from qwen_vl_utils import process_vision_info

    all_texts, all_image_inputs, all_video_inputs = [], [], []

    for image_path, class_label in batch:
        messages = build_messages(image_path, prompt)
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
    raw_outputs = processor.batch_decode(
        trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    return [
        (image_path, class_label, clean_output(raw))
        for (image_path, class_label), raw in zip(batch, raw_outputs)
    ]


# ── Folder scanner ────────────────────────────────────────────────────────────
def collect_images(root_folder: str) -> list[tuple]:
    """
    Walks root_folder. Immediate sub-folder name -> class label.
    Images directly inside root get class label 'unknown'.
    Returns: list of (absolute_image_path, class_label)
    """
    root = Path(root_folder)
    items = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(root)
        parts = relative.parts
        class_label = "unknown" if len(parts) == 1 else parts[0].replace("_", " ")
        items.append((str(path.resolve()), class_label))
    return items


# ── CSV helpers ───────────────────────────────────────────────────────────────
def detect_csv_dialect(csv_path: str):
    """Detect CSV delimiter, falling back to comma if sniffing fails."""
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        return csv.excel


def get_class_label_from_csv_row(row: dict) -> str:
    """
    Choose the class label used in the prompt for CSV mode.
    Priority: class_label column -> class_id column -> parent folder from image_path -> unknown.
    """
    if row.get("class_label"):
        return str(row["class_label"]).replace("_", " ")
    if row.get("class_id"):
        return str(row["class_id"])
    image_path = row.get("image_path", "")
    parent = Path(image_path).parent.name
    return parent.replace("_", " ") if parent else "unknown"


def collect_images_from_csv(input_csv: str, image_root: str) -> tuple[list[dict], list[str]]:
    """
    Reads input_csv and keeps the original columns.
    Only rows where is_human is not YES are returned.
    Adds internal-only fields: _absolute_image_path and _class_label.
    Returns: (rows, original_fieldnames)
    """
    dialect = detect_csv_dialect(input_csv)
    image_root_path = Path(image_root)
    items = []

    with open(input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header: {input_csv}")
        if "image_path" not in reader.fieldnames:
            raise ValueError("Input CSV must contain an 'image_path' column.")

        original_fieldnames = list(reader.fieldnames)

        for row in reader:
            is_human_value = str(row.get("is_human", "")).strip().upper()
            if is_human_value == "YES":
                continue

            relative_image_path = row.get("image_path", "")
            if not relative_image_path:
                log.warning("Skipping CSV row with empty image_path.")
                continue

            absolute_image_path = image_root_path / relative_image_path
            row["_absolute_image_path"] = str(absolute_image_path.resolve())
            row["_class_label"] = get_class_label_from_csv_row(row)
            items.append(row)

    return items, original_fieldnames


def output_columns_for_csv_mode(original_fieldnames: list[str]) -> list[str]:
    """Preserve all original CSV columns and append description."""
    columns = [c for c in original_fieldnames if c not in {"description", "_absolute_image_path", "_class_label"}]
    columns.append("description")
    return columns


def load_existing_output(output_csv: str) -> tuple[list[dict], set[str]]:
    """
    Load prior output for resume support.
    Uses image_path as the resume key.
    """
    existing_rows = []
    done_image_paths = set()
    output_path = Path(output_csv)

    if not output_path.exists():
        return existing_rows, done_image_paths

    with open(output_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_rows.append(row)
            image_path = row.get("image_path")
            if image_path:
                done_image_paths.add(image_path)

    log.info(f"Resume enabled: found {len(done_image_paths)} already processed image(s) in {output_csv}.")
    return existing_rows, done_image_paths


def write_csv(csv_path: str, rows: list[dict], fieldnames: list[str]):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ── Run-log (JSON) ────────────────────────────────────────────────────────────
def build_run_log(args, prompt: dict, seed: int, n_images: int, n_errors: int,
                   run_start: float, run_end: float) -> dict:
    """
    Assembles a full reproducibility record for this run.
    Everything needed to re-run and get identical results is captured here.
    """
    output_path = Path(args.output).resolve()
    elapsed = run_end - run_start
    dataset_info = {
        "image_extensions": sorted(IMAGE_EXTENSIONS),
        "total_images_found": n_images,
    }
    if args.input_csv:
        dataset_info.update({
            "input_csv": str(Path(args.input_csv).resolve()),
            "image_root": str(Path(args.image_root).resolve()),
            "filter": "is_human != YES",
        })
    else:
        dataset_info.update({
            "root_folder": str(Path(args.folder).resolve()),
        })

    return {
        # ── identity ───────────────────────────────────────────────────────
        "script":              Path(__file__).name,
        "run_timestamp_utc":   datetime.now(timezone.utc).isoformat(),

        # ── model ──────────────────────────────────────────────────────────
        "model_id":            args.model,
        "model_family":        args.model_family,
        "model_class":         MODEL_FAMILIES[args.model_family]["class_name"],

        # ── prompt ─────────────────────────────────────────────────────────
        # Full prompt file contents are recorded so the run is self-contained.
        "prompt_file":         str(Path(args.prompt_file).resolve()),
        "prompt_name":         prompt.get("name", "<unnamed>"),
        "prompt_description":  prompt.get("description", ""),
        "system_prompt":       prompt["system_prompt"],
        "user_prompt_template": prompt["user_prompt"],

        # ── generation settings ────────────────────────────────────────────
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature":    0.7,
            "do_sample":      True,
            "top_k":          20,
            "seed":           seed,
        },

        # ── dataset / I/O ──────────────────────────────────────────────────
        "dataset": dataset_info,
        "batch_size":  args.batch_size,
        "save_every":  args.save_every,
        "output_csv":  str(output_path),

        # ── run summary ────────────────────────────────────────────────────
        "summary": {
            "total_to_process_this_run": n_images,
            "errors":          n_errors,
            "success":         n_images - n_errors,
        },
        # ── timing ────────────────────────────────────────────────────────
        "timing": {
            "run_start_utc":        datetime.fromtimestamp(run_start, tz=timezone.utc).isoformat(),
            "run_end_utc":          datetime.fromtimestamp(run_end,   tz=timezone.utc).isoformat(),
            "elapsed_seconds":      round(elapsed, 2),
            "elapsed_human":        f"{int(elapsed // 3600):02d}h {int((elapsed % 3600) // 60):02d}m {elapsed % 60:.1f}s",
            "avg_seconds_per_image": round(elapsed / n_images, 2) if n_images else None,
        },
    }


def write_run_log(args, prompt: dict, seed: int, n_images: int, n_errors: int,
                   run_start: float, run_end: float):
    log_data = build_run_log(args, prompt, seed, n_images, n_errors, run_start, run_end)
    json_path = Path(args.output).with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    log.info(f"Run log saved to: {json_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate food descriptions using a Qwen3-VL or Qwen3.5-VL model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-2B-Thinking",
        help="HuggingFace model ID, e.g. Qwen/Qwen3-VL-7B-Thinking",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        choices=list(MODEL_FAMILIES.keys()),
        default="qwen3vl",
        help=(
            f"Model class to load. Choices: {list(MODEL_FAMILIES.keys())}. "
            "Use 'qwen3vl' for Qwen3-VL-* models, 'qwen3_5' for Qwen3.5-VL-*."
        ),
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,   # always required — no silent fallback to hardcoded defaults
        help=(
            "Path to a JSON prompt file with keys: system_prompt, user_prompt. "
            "user_prompt does not need any class placeholder; the model infers the class from the image. "
            "Optional keys: name, description (logged but not used at runtime). "
            "See prompts/detailed.json for a ready-to-use example."
        ),
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Root folder containing sub-folders (class names) with images inside. Used when --input_csv is not provided.",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Input CSV containing image_path and is_human columns. Rows with is_human == YES are skipped.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Root directory used to resolve relative image_path values from --input_csv.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="descriptions.csv",
        help="Path to the output CSV. A matching .json run log is written alongside it.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Images to process at once (default: 1).",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save output CSV after every N newly processed images (default: 1).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=3072,
        help=(
            "Max tokens to generate per image (default: 3072). "
            "Thinking models use 200-600 tokens on <think> before real output, "
            "so keep this well above 1024."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed to torch before every generate() call (default: 42).",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.save_every < 1:
        raise ValueError("--save_every must be >= 1")
    if args.input_csv and not args.image_root:
        raise ValueError("--image_root is required when --input_csv is provided.")
    if not args.input_csv and not args.folder:
        raise ValueError("Provide either --input_csv with --image_root, or --folder.")

    # ── Load prompt file ──────────────────────────────────────────────────────
    # Done before loading the model so a bad prompt file fails fast and cheaply.
    prompt = load_prompt_file(args.prompt_file)

    # ── Load model ────────────────────────────────────────────────────────────
    log.info(f"Loading model '{args.model}' (family: {args.model_family})...")
    ModelClass = load_model_class(args.model_family)
    model = ModelClass.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    log.info("Model loaded successfully.\n")

    # ── Collect images ────────────────────────────────────────────────────────
    existing_rows, done_image_paths = load_existing_output(args.output)
    rows = existing_rows[:]

    if args.input_csv:
        log.info(f"Reading CSV: {args.input_csv}")
        csv_items, original_fieldnames = collect_images_from_csv(args.input_csv, args.image_root)
        output_fieldnames = output_columns_for_csv_mode(original_fieldnames)

        pending_items = [
            item for item in csv_items
            if item.get("image_path") not in done_image_paths
        ]
        total_candidates = len(csv_items)
        all_images = [
            (item["_absolute_image_path"], item["_class_label"], item)
            for item in pending_items
        ]
        log.info(
            f"CSV rows selected after is_human filter: {total_candidates}. "
            f"Pending after resume: {len(all_images)}.\n"
        )
    else:
        log.info(f"Scanning folder: {args.folder}")
        folder_items = collect_images(args.folder)
        output_fieldnames = ["image_path", "class_label", "description"]

        pending_items = [
            item for item in folder_items
            if item[0] not in done_image_paths
        ]
        all_images = [
            (image_path, class_label, None)
            for image_path, class_label in pending_items
        ]
        log.info(
            f"Found {len(folder_items)} image(s). "
            f"Pending after resume: {len(all_images)}.\n"
        )

    if not all_images:
        log.warning("No new images to process. Exiting.")
        return

    import time
    run_start = time.time()

    n_errors = 0
    total = len(all_images)
    newly_processed_since_save = 0

    for start in range(0, total, args.batch_size):
        batch_items = all_images[start : start + args.batch_size]
        generation_batch = [(image_path, class_label) for image_path, class_label, _ in batch_items]
        log.info(
            f"Processing images {start + 1}–"
            f"{min(start + args.batch_size, total)} of {total}..."
        )

        if args.batch_size == 1:
            image_path, class_label, source_row = batch_items[0]
            try:
                description = describe_image(
                    image_path, model, processor,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                )
                log.info(f"  ✓ {image_path}")
            except Exception as e:
                log.error(f"  ✗ Failed on {image_path}: {e}")
                description = "ERROR"
                n_errors += 1

            if source_row is not None:
                output_row = {k: source_row.get(k, "") for k in output_fieldnames if k != "description"}
                output_row["description"] = description
            else:
                output_row = {
                    "image_path": image_path,
                    "class_label": class_label,
                    "description": description,
                }
            rows.append(output_row)
            newly_processed_since_save += 1
        else:
            try:
                results = describe_batch(
                    generation_batch, model, processor,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                )
                descriptions_by_path = {
                    image_path: description
                    for image_path, _class_label, description in results
                }
                for image_path, class_label, source_row in batch_items:
                    description = descriptions_by_path.get(image_path, "ERROR")
                    if description == "ERROR":
                        n_errors += 1
                    if source_row is not None:
                        output_row = {k: source_row.get(k, "") for k in output_fieldnames if k != "description"}
                        output_row["description"] = description
                    else:
                        output_row = {
                            "image_path": image_path,
                            "class_label": class_label,
                            "description": description,
                        }
                    rows.append(output_row)
                    newly_processed_since_save += 1
                    log.info(f"  ✓ {image_path}")
            except Exception as e:
                log.error(
                    f"  ✗ Batch failed "
                    f"({generation_batch[0][0]} ... {generation_batch[-1][0]}): {e}"
                )
                for image_path, class_label, source_row in batch_items:
                    if source_row is not None:
                        output_row = {k: source_row.get(k, "") for k in output_fieldnames if k != "description"}
                        output_row["description"] = "ERROR"
                    else:
                        output_row = {
                            "image_path": image_path,
                            "class_label": class_label,
                            "description": "ERROR",
                        }
                    rows.append(output_row)
                    n_errors += 1
                    newly_processed_since_save += 1

        if newly_processed_since_save >= args.save_every:
            write_csv(args.output, rows, output_fieldnames)
            log.info(f"Progress saved to: {args.output}")
            newly_processed_since_save = 0

    # Save at the end, even if the last save interval was not reached.
    write_csv(args.output, rows, output_fieldnames)

    # ── Write run log ─────────────────────────────────────────────────────────
    run_end = time.time()
    write_run_log(args, prompt=prompt, seed=args.seed, n_images=total, n_errors=n_errors,
                  run_start=run_start, run_end=run_end)

    log.info(f"\nDone. Results saved to: {args.output}")


if __name__ == "__main__":
    main()