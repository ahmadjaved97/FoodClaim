#!/usr/bin/env python3
"""
food_image_batch_batch_generation.py
=======================
Generate food images from a CSV using the OpenAI Batch API (gpt-image-2).

Mirrors the structure of the flux2_klein and gemini_batch generation scripts:
  - Three explicit modes: submit | poll | download
  - Structured logging to both stdout and a shared log file
  - generation_metadata.json with reproducibility, progress, and state-history blocks
  - Output CSV adds only two columns to the original:  generated_image_path, status
  - Run output folder is auto-named  <output_folder>/<model>_<dataset>[_suffix]

Input CSV columns (same schema as the flux/gemini scripts):
    image_path, dataset_name, class_id, is_human, description, [gen_model, ...]

Prompt column behaviour:
    - Uses `human_description` if present, otherwise falls back to `description`.
    - Override with --prompt_column <col>.

Resume / reproducibility:
    - Metadata JSON lives inside the run output folder.
    - Re-running submit with the same config reuses the existing JSONL/batch.
    - Re-running download skips images that are already saved on disk.
    - --skip_existing skips rows whose image_path already appears in the output CSV.

Examples
--------
1) Submit a batch for the whole CSV:
    python food_image_batch_batch_generation.py --mode submit \\
        --input input.csv --output_folder generated_outputs \\
        --model gpt-image-2

2) Submit only the first 100 rows:
    python food_image_batch_batch_generation.py --mode submit \\
        --input input.csv --limit 100

3) Poll until complete:
    python food_image_batch_batch_generation.py --mode poll \\
        --metadata_file generated_outputs/gpt-image-2_ethz_food101/generation_metadata.json \\
        --poll_until_done

4) Download images and write output CSV:
    python food_image_batch_batch_generation.py --mode download \\
        --metadata_file generated_outputs/gpt-image-2_ethz_food101/generation_metadata.json

5) Submit, then immediately poll + download in one command:
    python food_image_batch_batch_generation.py --mode submit \\
        --input input.csv --poll_until_done --download_after_poll

6) Incremental: add rows 101-200 without re-submitting 1-100:
    python food_image_batch_batch_generation.py --mode submit \\
        --input input.csv --limit 200 --skip_existing

Before running:
    pip install openai
    export OPENAI_API_KEY="your_api_key_here"
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import logging
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_VERSION = "2026-06-24.v3-modal-batch"
ENDPOINT = "/v1/images/generations"
COMPLETION_WINDOW = "24h"

SYSTEM_PROMPT = ""
BASE_PROMPT_TEMPLATE = (
    "Generate a high-quality, realistic food photograph based only on this food description. "
    "Match the food appearance, ingredients, serving vessel, scene composition, lighting, "
    "camera angle, and image quality details as closely as possible.\n\n"
    "{prompt_text}"
)

SUPPORTED_OUTPUT_FORMATS = {"png", "jpeg", "webp"}
EXTENSION_TO_FORMAT = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp"}
FORMAT_TO_EXTENSION = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}

IMAGE_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

ADDED_OUTPUT_COLUMNS = ["generated_image_path", "status"]
SUCCESS_STATUSES = {"generated", "reused"}

TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_name(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "model"


def infer_dataset_name(rows: List[Dict[str, str]], input_csv: Path) -> str:
    names = sorted(
        {(r.get("dataset_name") or "").strip() for r in rows
         if (r.get("dataset_name") or "").strip()}
    )
    if len(names) == 1:
        return sanitize_name(names[0])
    if len(names) > 1:
        preview = "_".join(sanitize_name(n) for n in names[:3])
        return f"{preview}_mixed_{len(names)}_datasets" if len(names) > 3 else preview
    return sanitize_name(input_csv.stem)


def build_run_output_folder(
    base: Path, model_name: str, dataset_name: str, quality: Optional[str], suffix: str
) -> Path:
    parts = [sanitize_name(model_name), sanitize_name(dataset_name)]
    if quality and quality.strip():
        parts.append(sanitize_name(quality))
    s = sanitize_name(suffix) if suffix.strip() else ""
    if s:
        parts.append(s)
    return base / "_".join(parts)


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("food_batch_openai")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "script": Path(__file__).name,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "script_version": SCRIPT_VERSION,
    }
    try:
        import openai
        info["openai_version"] = getattr(openai, "__version__", "unknown")
    except ImportError:
        pass
    return info


def response_to_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content.decode("utf-8")
    read = getattr(response, "read", None)
    if callable(read):
        data = read()
        if isinstance(data, bytes):
            return data.decode("utf-8")
        if isinstance(data, str):
            return data
    if isinstance(response, bytes):
        return response.decode("utf-8")
    if isinstance(response, str):
        return response
    raise RuntimeError("Could not read file content returned by OpenAI SDK.")


def require_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set. Set it with:\n"
            "  export OPENAI_API_KEY='your_api_key_here'",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# CSV / row helpers  (robust multiline-repair logic from v2)
# ---------------------------------------------------------------------------

def resolve_delimiter(delimiter_arg: Optional[str], csv_path: Path) -> csv.Dialect:
    if delimiter_arg:
        delimiter = "\t" if delimiter_arg == r"\t" else delimiter_arg

        class CustomDialect(csv.excel):
            pass

        CustomDialect.delimiter = delimiter
        return CustomDialect

    sample = csv_path.read_text(encoding="utf-8", errors="replace")[:65536]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        if "\t" in first_line:
            class TabDialect(csv.excel_tab):
                pass
            return TabDialect

        class CommaDialect(csv.excel):
            pass
        return CommaDialect


def delimiter_name(dialect: csv.Dialect) -> str:
    d = getattr(dialect, "delimiter", ",")
    return "\\t" if d == "\t" else d


def looks_like_image_path(value: str) -> bool:
    value = (value or "").strip().strip('"').strip("'")
    if not value:
        return False
    if value.endswith(":") and "/" not in value and "\\" not in value:
        return False
    suffix = PurePosixPath(value.replace("\\", "/")).suffix.lower()
    return suffix in IMAGE_FILE_EXTENSIONS


def best_multiline_append_column(headers: List[str]) -> str:
    preferred = ["human_description", "description", "prompt", "caption", "manipulation_prompt"]
    for name in preferred:
        if name in headers:
            return name
    for name in headers:
        if any(tok in name.lower() for tok in ("description", "prompt", "caption")):
            return name
    return headers[-1]


def clean_dict_row(raw_row: Dict[Any, Any], headers: List[str]) -> Dict[str, str]:
    return {k: (raw_row.get(k) or "") for k in headers}


def continuation_text_from_row(raw_row: Dict[Any, Any], headers: List[str], dialect: csv.Dialect) -> str:
    parts: List[str] = []
    for header in headers:
        value = raw_row.get(header)
        if value is not None and str(value).strip():
            parts.append(str(value).strip())
    extras = raw_row.get(None) or raw_row.get("__extra_fields__") or []
    if isinstance(extras, list):
        parts.extend(str(item).strip() for item in extras if str(item).strip())
    elif extras:
        parts.append(str(extras).strip())
    delimiter = getattr(dialect, "delimiter", ",")
    return delimiter.join(parts).strip()


def append_continuation_text(row: Dict[str, str], column: str, text_to_append: str) -> None:
    if not text_to_append:
        return
    existing = row.get(column, "").rstrip()
    row[column] = f"{existing}\n{text_to_append}" if existing else text_to_append


def load_rows(
    csv_path: Path,
    limit: Optional[int],
    delimiter: Optional[str],
) -> Tuple[List[Dict[str, str]], List[str], str]:
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if limit is not None and limit <= 0:
        print("ERROR: --limit must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    dialect = resolve_delimiter(delimiter, csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect, restkey="__extra_fields__")
        if not reader.fieldnames:
            print("ERROR: CSV appears to have no header row.", file=sys.stderr)
            sys.exit(1)

        headers = list(reader.fieldnames)
        append_column = best_multiline_append_column(headers)
        rows: List[Dict[str, str]] = []
        repaired = 0
        skipped_blank = 0

        for physical_line, raw_row in enumerate(reader, start=2):
            row = clean_dict_row(raw_row, headers)
            if not any(str(v).strip() for v in row.values()):
                skipped_blank += 1
                continue

            if "image_path" in headers:
                image_path = row.get("image_path", "").strip()
                if looks_like_image_path(image_path):
                    rows.append(row)
                    if limit is not None and len(rows) >= limit:
                        break
                    continue
                continuation_text = continuation_text_from_row(raw_row, headers, dialect)
                if rows and continuation_text:
                    append_continuation_text(rows[-1], append_column, continuation_text)
                    repaired += 1
                    continue
                print(
                    f"ERROR: Invalid image_path at physical line {physical_line}: {image_path!r}.",
                    file=sys.stderr,
                )
                sys.exit(1)
            else:
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break

    if repaired:
        print(f"Repaired {repaired} unquoted multiline CSV/TSV continuation lines.")
    if skipped_blank:
        print(f"Skipped {skipped_blank} blank physical lines.")

    return rows, headers, delimiter_name(dialect)


def choose_prompt_column(headers: List[str], requested: Optional[str]) -> str:
    if requested:
        if requested not in headers:
            print(
                f"ERROR: Requested prompt column '{requested}' not found. Available: {headers}",
                file=sys.stderr,
            )
            sys.exit(1)
        return requested
    if "human_description" in headers:
        return "human_description"
    if "description" in headers:
        return "description"
    print(
        "ERROR: No usable prompt column found. Expected 'human_description' or 'description'.",
        file=sys.stderr,
    )
    sys.exit(1)


def safe_relative_path(raw_image_path: str) -> Path:
    normalized = raw_image_path.strip().replace("\\", "/")
    if not normalized:
        print("ERROR: Encountered an empty image_path value.", file=sys.stderr)
        sys.exit(1)
    pure = PurePosixPath(normalized)
    safe_parts = [
        part for part in pure.parts
        if part not in ("", ".", "/", "\\") and part != ".." and not part.endswith(":")
    ]
    if not safe_parts:
        print(
            f"ERROR: image_path could not be converted to a safe relative path: {raw_image_path!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    return Path(*safe_parts)


def output_path_and_format(image_path: str, output_format_arg: str) -> Tuple[Path, str]:
    rel = safe_relative_path(image_path)
    suffix = rel.suffix.lower()
    if output_format_arg == "match":
        fmt = EXTENSION_TO_FORMAT.get(suffix, "png")
        if suffix in EXTENSION_TO_FORMAT:
            return rel, fmt
        return rel.with_suffix(".png"), "png"
    fmt = output_format_arg
    if fmt not in SUPPORTED_OUTPUT_FORMATS:
        print(f"ERROR: Unsupported output format: {fmt}", file=sys.stderr)
        sys.exit(1)
    return rel.with_suffix(FORMAT_TO_EXTENSION[fmt]), fmt


def build_prompt(description: str) -> str:
    description = description.strip()
    if not description:
        print("ERROR: Encountered an empty description value.", file=sys.stderr)
        sys.exit(1)
    prompt = BASE_PROMPT_TEMPLATE.format(prompt_text=description)
    if SYSTEM_PROMPT.strip():
        prompt = f"{SYSTEM_PROMPT.strip()}\n\n{prompt}"
    return prompt


def custom_id_for_row(row_number: int, image_path: str) -> str:
    digest = hashlib.sha1(image_path.encode("utf-8")).hexdigest()[:10]
    return f"row-{row_number:06d}-{digest}"


def build_output_image_path(
    image_path: str,
    output_folder: Path,
    output_format_arg: str,
) -> Tuple[Path, str]:
    """
    Returns (absolute_output_path, api_format_string).
    Mirrors the logic of output_path_and_format() but resolves under output_folder.
    """
    rel, fmt = output_path_and_format(image_path, output_format_arg)
    return output_folder / rel, fmt


# ---------------------------------------------------------------------------
# CSV output helpers  (flux/gemini style: two added columns)
# ---------------------------------------------------------------------------

def output_fieldnames(input_fieldnames: List[str]) -> List[str]:
    fields = list(input_fieldnames)
    for col in ADDED_OUTPUT_COLUMNS:
        if col not in fields:
            fields.append(col)
    return fields


def read_completed_image_paths(output_csv: Path) -> Set[str]:
    """Return set of image_path values already successfully generated."""
    completed: Set[str] = set()
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return completed
    with output_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ip = (row.get("image_path") or "").strip()
            status = (row.get("status") or "").strip().lower()
            gip = (row.get("generated_image_path") or "").strip()
            if ip and status in SUCCESS_STATUSES and gip:
                completed.add(ip)
    return completed


def append_output_row(output_csv: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists() and output_csv.stat().st_size > 0
    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


# ---------------------------------------------------------------------------
# JSONL + manifest builder
# ---------------------------------------------------------------------------

def generation_signature(config: Dict[str, Any]) -> str:
    sig_fields = {
        "input_csv_sha256": config["input_csv_sha256"],
        "limit": config["limit"],
        "detected_delimiter": config["detected_delimiter"],
        "prompt_column": config["prompt_column"],
        "model": config["model"],
        "size": config["size"],
        "quality": config["quality"],
        "output_format": config["output_format"],
        "n": config["n"],
        "system_prompt": SYSTEM_PROMPT,
        "base_prompt_template": BASE_PROMPT_TEMPLATE,
        "endpoint": ENDPOINT,
    }
    encoded = json.dumps(sig_fields, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_batch_jsonl(
    *,
    rows: List[Dict[str, str]],
    headers: List[str],
    prompt_column: str,
    model: str,
    size: str,
    quality: Optional[str],
    output_format: str,
    n: int,
    output_folder: Path,
    jsonl_path: Path,
    manifest_path: Path,
    skip_image_paths: Set[str],
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build JSONL and manifest for pending rows.

    Returns:
        pending_items: list of dicts with keys: custom_id, row, output_path, fmt
        manifest:      dict mapping custom_id -> metadata
    """
    pending_items: List[Dict[str, Any]] = []
    skipped = 0

    for row_idx, row in enumerate(rows, start=1):
        image_path = row.get("image_path", "").strip()
        prompt_text = row.get(prompt_column, "").strip()

        if not image_path:
            logger.warning("Row %d: empty image_path — skipping.", row_idx)
            skipped += 1
            continue
        if not prompt_text or prompt_text.upper() == "ERROR":
            logger.warning("Row %d: empty/error prompt (%s) — skipping.", row_idx, prompt_column)
            skipped += 1
            continue

        if image_path in skip_image_paths:
            skipped += 1
            continue

        abs_output_path, row_fmt = build_output_image_path(image_path, output_folder, output_format)
        custom_id = custom_id_for_row(row_idx, image_path)

        pending_items.append({
            "custom_id": custom_id,
            "row": row,
            "row_idx": row_idx,
            "output_path": abs_output_path,
            "fmt": row_fmt,
        })

    if skipped:
        logger.info("Skipped %d rows (already completed or invalid).", skipped)

    manifest: Dict[str, Any] = {
        "_meta": {
            "created_at": utc_now(),
            "model": model,
            "size": size,
            "quality": quality,
            "output_format": output_format,
            "output_folder": str(output_folder),
            "rows_pending": len(pending_items),
            "rows_skipped": skipped,
        }
    }

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in pending_items:
            prompt = build_prompt(item["row"].get(prompt_column, ""))
            body: Dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "output_format": item["fmt"],
            }
            if quality:
                body["quality"] = quality
            if n != 1:
                body["n"] = n

            request = {
                "custom_id": item["custom_id"],
                "method": "POST",
                "url": ENDPOINT,
                "body": body,
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

            row = item["row"]
            manifest[item["custom_id"]] = {
                "row_number": item["row_idx"],
                "image_path": row.get("image_path", "").strip(),
                "output_path": str(item["output_path"]),
                "output_format": item["fmt"],
                "dataset_name": row.get("dataset_name", ""),
                "class_id": row.get("class_id", ""),
                "is_human": row.get("is_human", ""),
            }

    atomic_write_json(manifest_path, manifest)
    logger.info("Wrote JSONL  : %s  (%d requests)", jsonl_path, len(pending_items))
    logger.info("Wrote manifest: %s", manifest_path)

    return pending_items, manifest


# ---------------------------------------------------------------------------
# MODE: submit
# ---------------------------------------------------------------------------

def mode_submit(args: argparse.Namespace) -> None:
    require_api_key()

    input_csv = Path(args.input)
    base_output_folder = Path(args.output_folder)

    rows, input_fieldnames, detected_delimiter = load_rows(
        input_csv, args.limit, args.delimiter
    )
    if not rows:
        print("ERROR: No rows found in the CSV after applying the limit.", file=sys.stderr)
        sys.exit(1)

    dataset_name = infer_dataset_name(rows, input_csv)
    model_name = args.model

    output_folder = (
        Path(args.run_output_folder)
        if args.run_output_folder
        else build_run_output_folder(base_output_folder, model_name, dataset_name, args.quality or "", args.output_folder_suffix)
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    output_csv = Path(args.output_csv) if args.output_csv else output_folder / "generated_images.csv"
    json_log = Path(args.metadata_file) if args.metadata_file else output_folder / "generation_metadata.json"
    text_log = Path(args.text_log) if args.text_log else base_output_folder / "generation.log"
    jsonl_path = output_folder / "batch_input.jsonl"
    manifest_path = output_folder / "batch_manifest.json"

    logger = setup_logger(text_log)

    prompt_column = choose_prompt_column(input_fieldnames, args.prompt_column)
    quality: Optional[str] = args.quality if args.quality else None

    # Rows already done in a previous run
    skip_image_paths: Set[str] = set()
    if args.skip_existing:
        skip_image_paths = read_completed_image_paths(output_csv)

    config: Dict[str, Any] = {
        "input_csv_sha256": sha256_file(input_csv),
        "limit": args.limit,
        "detected_delimiter": detected_delimiter,
        "prompt_column": prompt_column,
        "model": model_name,
        "size": args.size,
        "quality": quality,
        "output_format": args.output_format,
        "n": args.n,
    }
    sig = generation_signature(config)

    logger.info("=" * 80)
    logger.info("OpenAI Batch Image Generation — SUBMIT")
    logger.info("Input CSV        : %s", input_csv)
    logger.info("Model            : %s", model_name)
    logger.info("Dataset          : %s", dataset_name)
    logger.info("Run output       : %s", output_folder)
    logger.info("Output CSV       : %s", output_csv)
    logger.info("Metadata JSON    : %s", json_log)
    logger.info("Prompt column    : %s", prompt_column)
    logger.info("Size / quality   : %s / %s", args.size, quality or "<omitted>")
    logger.info("Output format    : %s", args.output_format)
    logger.info("Total rows       : %d", len(rows))
    logger.info("Already done     : %d", len(skip_image_paths))
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Resume: reuse existing JSONL/batch if signature matches
    # ------------------------------------------------------------------
    existing_meta: Dict[str, Any] = {}
    if json_log.exists():
        try:
            existing_meta = json.loads(json_log.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    can_reuse_jsonl = (
        not args.force_prepare
        and existing_meta.get("generation_signature") == sig
        and jsonl_path.exists()
        and manifest_path.exists()
    )

    if can_reuse_jsonl and not args.skip_existing:
        logger.info("Resume: generation signature matches — reusing existing JSONL.")
    else:
        # Build fresh JSONL
        pending_items, manifest = build_batch_jsonl(
            rows=rows,
            headers=input_fieldnames,
            prompt_column=prompt_column,
            model=model_name,
            size=args.size,
            quality=quality,
            output_format=args.output_format,
            n=args.n,
            output_folder=output_folder,
            jsonl_path=jsonl_path,
            manifest_path=manifest_path,
            skip_image_paths=skip_image_paths,
            logger=logger,
        )
        if not pending_items:
            logger.info("No pending rows to submit. Nothing to do.")
            return

    # ------------------------------------------------------------------
    # Resume: reuse existing OpenAI batch ID if it's already submitted
    # ------------------------------------------------------------------
    openai_info = existing_meta.get("openai", {})
    existing_batch_id = openai_info.get("batch_id")
    existing_batch_status = openai_info.get("batch_status", "")

    client = OpenAI()

    if (
        not args.force_prepare
        and existing_batch_id
        and existing_batch_status not in TERMINAL_BATCH_STATUSES
    ):
        logger.info("Resume: reusing existing batch ID %s (status=%s).", existing_batch_id, existing_batch_status)
        batch = client.batches.retrieve(existing_batch_id)
        batch_id = existing_batch_id
    else:
        # Upload JSONL
        logger.info("Uploading JSONL to OpenAI Files API…")
        with jsonl_path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        input_file_id = uploaded.id
        logger.info("Uploaded file ID : %s", input_file_id)

        # Submit batch
        logger.info("Submitting batch job…")
        batch = client.batches.create(
            input_file_id=input_file_id,
            endpoint=ENDPOINT,
            completion_window=COMPLETION_WINDOW,
        )
        batch_id = batch.id
        logger.info("Batch ID created : %s", batch_id)
        logger.info("Initial status   : %s", batch.status)

    # ------------------------------------------------------------------
    # Save / update metadata
    # ------------------------------------------------------------------
    # Reload manifest for row_index_map
    manifest_data: Dict[str, Any] = {}
    if manifest_path.exists():
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    row_index_map = {
        k: v for k, v in manifest_data.items() if k != "_meta"
    }

    run_data: Dict[str, Any] = {
        "submitted_at": utc_now(),
        "script_version": SCRIPT_VERSION,
        "generation_signature": sig,
        "model": model_name,
        "input_csv": str(input_csv.resolve()),
        "jsonl_path": str(jsonl_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "output_folder": str(output_folder.resolve()),
        "output_csv": str(output_csv.resolve()),
        "json_log": str(json_log.resolve()),
        "text_log": str(text_log.resolve()),
        "dataset_name": dataset_name,
        "output_folder_suffix": args.output_folder_suffix,
        "openai": {
            "batch_id": batch_id,
            "batch_status": getattr(batch, "status", None),
            "input_file_id": getattr(batch, "input_file_id", None),
            "output_file_id": getattr(batch, "output_file_id", None),
            "error_file_id": getattr(batch, "error_file_id", None),
            "last_checked_at": utc_now(),
        },
        "settings": {
            "limit": args.limit,
            "delimiter": args.delimiter,
            "prompt_column": prompt_column,
            "size": args.size,
            "quality": quality,
            "output_format": args.output_format,
            "n": args.n,
            "skip_existing": args.skip_existing,
        },
        "reproducibility": {
            "system_prompt": SYSTEM_PROMPT,
            "base_prompt_template": BASE_PROMPT_TEMPLATE,
            "prompt_construction": "BASE_PROMPT_TEMPLATE.format(prompt_text=<value from prompt_column>)",
            "model": model_name,
            "prompt_column": prompt_column,
            "dataset_name": dataset_name,
            "output_folder_suffix": args.output_folder_suffix,
            "generation_signature": sig,
            "input_csv_sha256": config["input_csv_sha256"],
        },
        "environment": environment_info(),
        "progress": {
            "total_input_rows": len(rows),
            "skipped_already_completed": len(skip_image_paths),
            "submitted_this_run": len(row_index_map),
        },
        "batch_state_history": [
            {"timestamp": utc_now(), "status": getattr(batch, "status", "unknown")}
        ],
        "row_index_map": row_index_map,
        "input_fieldnames": input_fieldnames,
        "download_completed": False,
    }
    atomic_write_json(json_log, run_data)

    logger.info("=" * 80)
    logger.info("Submission complete.")
    logger.info("Batch ID      : %s", batch_id)
    logger.info("Metadata saved: %s", json_log)
    logger.info("")
    logger.info("Next steps:")
    logger.info(
        "  Poll:     python %s --mode poll     --metadata_file %s",
        Path(sys.argv[0]).name, json_log,
    )
    logger.info(
        "  Download: python %s --mode download --metadata_file %s",
        Path(sys.argv[0]).name, json_log,
    )
    logger.info("=" * 80)

    # Optionally continue to poll + download in the same invocation
    if args.poll_until_done or args.download_after_poll:
        _args_for_poll = argparse.Namespace(
            metadata_file=str(json_log),
            poll_until_done=True,
            poll_interval=args.poll_interval,
        )
        mode_poll(_args_for_poll, _existing_logger=logger)

    if args.download_after_poll:
        _args_for_dl = argparse.Namespace(
            metadata_file=str(json_log),
            poll_until_done=False,
            poll_interval=args.poll_interval,
            overwrite_images=getattr(args, "overwrite_images", False),
        )
        mode_download(_args_for_dl, _existing_logger=logger)


# ---------------------------------------------------------------------------
# MODE: poll
# ---------------------------------------------------------------------------

def mode_poll(
    args: argparse.Namespace,
    _existing_logger: Optional[logging.Logger] = None,
) -> None:
    require_api_key()

    meta_path = Path(args.metadata_file)
    if not meta_path.exists():
        print(f"ERROR: Metadata file not found: {meta_path}", file=sys.stderr)
        sys.exit(1)

    run_data = json.loads(meta_path.read_text(encoding="utf-8"))
    text_log = Path(run_data.get("text_log", str(meta_path.parent / "generation.log")))
    logger = _existing_logger or setup_logger(text_log)

    batch_id = run_data.get("openai", {}).get("batch_id")
    if not batch_id:
        logger.error("No batch_id found in metadata. Cannot poll.")
        sys.exit(1)

    poll_interval = getattr(args, "poll_interval", 60)

    logger.info("=" * 80)
    logger.info("OpenAI Batch Image Generation — POLL")
    logger.info("Batch ID  : %s", batch_id)
    logger.info("Model     : %s", run_data.get("model", "?"))
    logger.info("Submitted : %s", run_data.get("submitted_at", "?"))
    logger.info("=" * 80)

    client = OpenAI()

    while True:
        batch = client.batches.retrieve(batch_id)
        status = getattr(batch, "status", "unknown")
        logger.info("Current status: %s", status)

        run_data.setdefault("openai", {}).update({
            "batch_status": status,
            "output_file_id": getattr(batch, "output_file_id", None),
            "error_file_id": getattr(batch, "error_file_id", None),
            "last_checked_at": utc_now(),
        })
        run_data.setdefault("batch_state_history", []).append(
            {"timestamp": utc_now(), "status": status}
        )
        atomic_write_json(meta_path, run_data)

        if status in TERMINAL_BATCH_STATUSES:
            if status == "completed":
                logger.info(
                    "Batch completed. Output file ID: %s",
                    getattr(batch, "output_file_id", "?"),
                )
                logger.info(
                    "Run with --mode download to save images."
                )
            else:
                logger.warning("Batch ended with status: %s", status)
                if getattr(batch, "error_file_id", None):
                    logger.warning("Error file ID: %s", batch.error_file_id)
            break

        if not getattr(args, "poll_until_done", False):
            logger.info(
                "Not done yet (status=%s). Re-run with --mode poll --poll_until_done to wait.", status
            )
            break

        logger.info("Waiting %d seconds…", poll_interval)
        time.sleep(poll_interval)

    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# MODE: download
# ---------------------------------------------------------------------------

def extract_image_base64(data_item: Dict[str, Any]) -> Optional[str]:
    for key in ("b64_json", "image_base64", "base64"):
        value = data_item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def indexed_output_path(path: Path, index: int) -> Path:
    if index == 0:
        return path
    return path.with_name(f"{path.stem}_{index + 1}{path.suffix}")


def mode_download(
    args: argparse.Namespace,
    _existing_logger: Optional[logging.Logger] = None,
) -> None:
    require_api_key()

    meta_path = Path(args.metadata_file)
    if not meta_path.exists():
        print(f"ERROR: Metadata file not found: {meta_path}", file=sys.stderr)
        sys.exit(1)

    run_data = json.loads(meta_path.read_text(encoding="utf-8"))
    text_log = Path(run_data.get("text_log", str(meta_path.parent / "generation.log")))
    logger = _existing_logger or setup_logger(text_log)

    output_csv = Path(run_data["output_csv"])
    model_name = run_data["model"]
    output_folder = Path(run_data["output_folder"])
    row_index_map: Dict[str, Dict[str, Any]] = run_data.get("row_index_map", {})
    input_fieldnames: List[str] = run_data.get("input_fieldnames", [])
    fields = output_fieldnames(input_fieldnames)
    overwrite_images = getattr(args, "overwrite_images", False)

    logger.info("=" * 80)
    logger.info("OpenAI Batch Image Generation — DOWNLOAD")
    logger.info("Batch ID      : %s", run_data.get("openai", {}).get("batch_id", "?"))
    logger.info("Model         : %s", model_name)
    logger.info("Output folder : %s", output_folder)
    logger.info("Output CSV    : %s", output_csv)

    # Optionally poll until done first
    if getattr(args, "poll_until_done", False):
        poll_interval = getattr(args, "poll_interval", 60)
        logger.info("--poll_until_done set; waiting for batch to complete…")
        client = OpenAI()
        batch_id = run_data.get("openai", {}).get("batch_id")
        while True:
            batch = client.batches.retrieve(batch_id)
            status = getattr(batch, "status", "unknown")
            run_data.setdefault("openai", {}).update({
                "batch_status": status,
                "output_file_id": getattr(batch, "output_file_id", None),
                "last_checked_at": utc_now(),
            })
            run_data.setdefault("batch_state_history", []).append(
                {"timestamp": utc_now(), "status": status}
            )
            atomic_write_json(meta_path, run_data)
            if status in TERMINAL_BATCH_STATUSES:
                break
            logger.info("Status: %s. Waiting %d seconds…", status, poll_interval)
            time.sleep(poll_interval)

    # Verify batch completed
    batch_status = run_data.get("openai", {}).get("batch_status", "")
    if batch_status != "completed":
        # Try a fresh status check before giving up
        try:
            client = OpenAI()
            batch_id = run_data.get("openai", {}).get("batch_id")
            batch = client.batches.retrieve(batch_id)
            batch_status = getattr(batch, "status", "unknown")
            run_data.setdefault("openai", {}).update({
                "batch_status": batch_status,
                "output_file_id": getattr(batch, "output_file_id", None),
                "last_checked_at": utc_now(),
            })
            atomic_write_json(meta_path, run_data)
        except Exception:
            pass

    if batch_status != "completed":
        logger.error(
            "Batch is not completed (status=%s). "
            "Add --poll_until_done to wait, or run --mode poll first.",
            batch_status,
        )
        sys.exit(1)

    output_file_id = run_data.get("openai", {}).get("output_file_id")
    if not output_file_id:
        logger.error("No output_file_id in metadata. Cannot download.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Download output JSONL from OpenAI Files API
    # ------------------------------------------------------------------
    output_jsonl_path = output_folder / "batch_output.jsonl"

    # Resume: skip re-download if we already have it for the same file ID
    already_downloaded = (
        run_data.get("outputs", {}).get("downloaded_output_file_id") == output_file_id
        and output_jsonl_path.exists()
        and output_jsonl_path.stat().st_size > 0
    )

    if already_downloaded:
        logger.info("Resume: reusing existing output JSONL: %s", output_jsonl_path)
    else:
        logger.info("Downloading output JSONL (file_id=%s)…", output_file_id)
        client = OpenAI()
        response = client.files.content(output_file_id)
        text = response_to_text(response)
        atomic_write_text(output_jsonl_path, text)
        run_data.setdefault("outputs", {})["downloaded_output_file_id"] = output_file_id
        atomic_write_json(meta_path, run_data)
        logger.info("Downloaded: %s", output_jsonl_path)

    # ------------------------------------------------------------------
    # Parse output JSONL, decode images, write output CSV
    # ------------------------------------------------------------------
    output_folder.mkdir(parents=True, exist_ok=True)

    generated = 0
    reused = 0
    errors = 0
    skipped = 0
    download_start = time.perf_counter()

    with output_jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Line %d: JSON parse error: %s", line_num, exc)
                errors += 1
                continue

            custom_id = record.get("custom_id", "")
            manifest_entry = row_index_map.get(custom_id)

            if manifest_entry is None:
                logger.warning("Line %d: unknown custom_id '%s' — skipping.", line_num, custom_id)
                skipped += 1
                continue

            # Reconstruct the original CSV row from row_index_map
            image_path = manifest_entry.get("image_path", "").strip()
            abs_output_path = Path(manifest_entry["output_path"])

            # Pull original row from run_data if available (stored by submit mode)
            orig_row: Dict[str, str] = {}
            # row_index_map stores manifest entries, not raw rows;
            # reconstruct what we can from manifest + input_fieldnames
            for col in input_fieldnames:
                orig_row[col] = manifest_entry.get(col, "")
            orig_row["image_path"] = image_path

            if record.get("error"):
                logger.warning(
                    "[%s] Request-level error: %s", custom_id, record["error"]
                )
                out_row = dict(orig_row)
                out_row.update({"generated_image_path": "", "status": "error"})
                append_output_row(output_csv, fields, out_row)
                errors += 1
                continue

            response = record.get("response") or {}
            status_code = response.get("status_code")
            if status_code is not None and int(status_code) >= 400:
                logger.warning(
                    "[%s] HTTP %s error.", custom_id, status_code
                )
                out_row = dict(orig_row)
                out_row.update({"generated_image_path": "", "status": "error"})
                append_output_row(output_csv, fields, out_row)
                errors += 1
                continue

            body = response.get("body") or {}
            data = body.get("data") or []

            if not data:
                logger.warning("[%s] No image data in response.", custom_id)
                out_row = dict(orig_row)
                out_row.update({"generated_image_path": "", "status": "error"})
                append_output_row(output_csv, fields, out_row)
                errors += 1
                continue

            # Save image(s) — use first image for the output CSV path
            first_saved_path: Optional[Path] = None

            for img_idx, item in enumerate(data):
                final_path = indexed_output_path(abs_output_path, img_idx)

                if not overwrite_images and final_path.exists() and final_path.stat().st_size > 0:
                    logger.info("[%d] Reused: %s", generated + reused + 1, final_path)
                    if first_saved_path is None:
                        first_saved_path = final_path
                    reused += 1
                    continue

                image_b64 = extract_image_base64(item)
                if not image_b64:
                    logger.warning(
                        "[%s] Image %d: no base64 data field found.", custom_id, img_idx + 1
                    )
                    continue

                image_bytes = base64.b64decode(image_b64)
                atomic_write_bytes(final_path, image_bytes)
                logger.info("[%d] Saved: %s", generated + reused + 1, final_path)
                if first_saved_path is None:
                    first_saved_path = final_path
                generated += 1

            if first_saved_path is not None:
                out_row = dict(orig_row)
                out_row.update({
                    "generated_image_path": str(first_saved_path),
                    "status": "reused" if (generated == 0 and reused > 0) else "generated",
                })
                append_output_row(output_csv, fields, out_row)
            else:
                out_row = dict(orig_row)
                out_row.update({"generated_image_path": "", "status": "error"})
                append_output_row(output_csv, fields, out_row)
                errors += 1

    elapsed = round(time.perf_counter() - download_start, 2)

    run_data["download_completed"] = True
    run_data["download_completed_at"] = utc_now()
    run_data["download_stats"] = {
        "generated": generated,
        "reused": reused,
        "errors": errors,
        "skipped": skipped,
        "elapsed_seconds": elapsed,
    }
    run_data.setdefault("batch_state_history", []).append(
        {"timestamp": utc_now(), "status": "DOWNLOAD_COMPLETE"}
    )
    atomic_write_json(meta_path, run_data)

    logger.info("=" * 80)
    logger.info("Download complete in %.1fs", elapsed)
    logger.info("Generated : %d", generated)
    logger.info("Reused    : %d", reused)
    logger.info("Errors    : %d", errors)
    logger.info("Output CSV: %s", output_csv.resolve())
    logger.info("Metadata  : %s", meta_path.resolve())
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate food images via the OpenAI Batch API from a CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- mode ----
    parser.add_argument(
        "--mode",
        required=True,
        choices=["submit", "poll", "download"],
        help=(
            "submit   : Convert CSV → JSONL, upload, start batch job, save metadata.\n"
            "poll     : Check or wait for a previously submitted batch job.\n"
            "download : Download completed results, decode and save images, write output CSV."
        ),
    )

    # ---- submit-only args ----
    parser.add_argument("--input", default=None, help="[submit] Input CSV/TSV file.")
    parser.add_argument("--model", default="gpt-image-2", help="[submit] OpenAI image model.")
    parser.add_argument("--size", default="1024x1024", help="[submit] Image size.")
    parser.add_argument(
        "--quality", default="low",
        help="[submit] Image quality (low/medium/high). Pass empty string '' to omit."
    )
    parser.add_argument(
        "--output_format",
        default="match",
        choices=["match", "png", "jpeg", "webp"],
        help="[submit] 'match' preserves the input file extension; otherwise forces the given format.",
    )
    parser.add_argument("--n", type=int, default=1, help="[submit] Images per request.")
    parser.add_argument("--limit", type=int, default=None, help="[submit] Max CSV rows to include.")
    parser.add_argument("--delimiter", default=None, help=r"[submit] CSV delimiter override, e.g. '\t'.")
    parser.add_argument(
        "--prompt_column", default=None,
        help="[submit] CSV column for prompt text. Defaults to human_description or description."
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="[submit] Skip rows whose image_path already appears as a success in the output CSV.",
    )
    parser.add_argument("--output_folder", default="generated_outputs", help="[submit] Base output folder.")
    parser.add_argument(
        "--output_folder_suffix", default="",
        help="[submit] Optional tag appended to the auto-created run folder name."
    )
    parser.add_argument(
        "--run_output_folder", default=None,
        help="[submit] Override the auto-generated run output folder path entirely."
    )
    parser.add_argument("--output_csv", default=None, help="[submit] Override output CSV path.")
    parser.add_argument(
        "--force_prepare",
        action="store_true",
        help="[submit] Regenerate JSONL/manifest even if the generation signature matches.",
    )

    # ---- poll/download args ----
    parser.add_argument(
        "--metadata_file", default=None,
        help="[poll/download] Path to generation_metadata.json from a previous --mode submit."
    )
    parser.add_argument(
        "--poll_until_done",
        action="store_true",
        help="[poll/download] Keep polling until the batch reaches a terminal state.",
    )
    parser.add_argument(
        "--poll_interval", type=int, default=60,
        help="[poll/download] Seconds between status polls when --poll_until_done is set."
    )
    parser.add_argument(
        "--download_after_poll",
        action="store_true",
        help="[submit] After polling to completion, automatically run download too.",
    )
    parser.add_argument(
        "--overwrite_images",
        action="store_true",
        help="[download] Overwrite existing image files instead of reusing them.",
    )

    # ---- logging ----
    parser.add_argument(
        "--text_log", default=None,
        help="Override the log file path (default: <output_folder>/generation.log)."
    )
    # ---- metadata ----
    parser.add_argument(
        "--metadata_file_submit", default=None,
        help="[submit] Override the metadata JSON path (default: <run_output_folder>/generation_metadata.json)."
    )

    args = parser.parse_args()

    # Normalize: --metadata_file_submit -> --metadata_file for submit mode
    if args.mode == "submit" and hasattr(args, "metadata_file_submit") and args.metadata_file_submit:
        args.metadata_file = args.metadata_file_submit

    if args.mode == "submit":
        if not args.input:
            parser.error("--mode submit requires --input")
        mode_submit(args)

    elif args.mode == "poll":
        if not args.metadata_file:
            parser.error("--mode poll requires --metadata_file")
        mode_poll(args)

    elif args.mode == "download":
        if not args.metadata_file:
            parser.error("--mode download requires --metadata_file")
        mode_download(args)


if __name__ == "__main__":
    main()