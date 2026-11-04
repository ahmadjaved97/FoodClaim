#!/usr/bin/env python3
"""
food_image_edit_parallel.py
===========================
Modify images from a CSV using the normal OpenAI Images Edit API, not Batch API.
Batch Image editing was not present with gpt-2 at the time of using this script.
This script was used for image editing for gpt-2

This script is intended as a direct/non-batch version of the image manipulation
batch runner:
  - Reads an input CSV/TSV.
  - Resolves relative image paths from --input_root.
  - Uses one input image + one prompt per row.
  - Calls client.images.edit(...) with model gpt-image-2 by default.
  - Runs direct API calls concurrently with --workers.
  - Writes edited images to an output folder using the CSV image_path structure.
  - Writes an output CSV with the original columns plus: generated_image_path, status.
  - Supports resume-style behavior with --skip_existing and --overwrite_images.

Expected CSV columns
--------------------
Default input image column search order:
    image_path, input_image_path, source_image_path, original_image_path, path

Default prompt column search order:
    manipulation_prompt, edit_prompt, prompt, human_description, description

Examples
--------
1) Run direct image edits with 4 workers:
    python food_image_edit_parallel.py \
        --input input.csv \
        --input_root /path/to/images \
        --output_folder edited_outputs \
        --workers 4

2) Explicit image/prompt columns:
    python food_image_edit_parallel.py \
        --input input.csv \
        --input_root /path/to/images \
        --input_image_column image_path \
        --prompt_column manipulation_prompt

3) True sequential processing:
    python food_image_edit_parallel.py \
        --input input.csv \
        --input_root /path/to/images \
        --workers 1

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
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from openai import OpenAI
except ImportError:  # Keeps --help / py_compile usable before installing openai.
    OpenAI = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_VERSION = "2026-06-27.v1-direct-image-edit-parallel"

SYSTEM_PROMPT = ""
BASE_PROMPT_TEMPLATE = (
    "Modify the provided input image according to the following instruction. "
    "Preserve all parts of the image that the instruction does not ask to change. "
    "Keep the result realistic, coherent, and high quality.\n\n"
    "{prompt_text}"
)

SUPPORTED_INPUT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_OUTPUT_FORMATS = {"png", "jpeg", "webp"}
EXTENSION_TO_FORMAT = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp"}
FORMAT_TO_EXTENSION = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}
IMAGE_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
MAX_GPT_IMAGE_INPUT_BYTES = 50 * 1024 * 1024

DEFAULT_IMAGE_COLUMNS = ["image_path", "input_image_path", "source_image_path", "original_image_path", "path"]
DEFAULT_PROMPT_COLUMNS = ["manipulation_prompt", "edit_prompt", "prompt", "human_description", "description"]
ADDED_OUTPUT_COLUMNS = ["generated_image_path", "status"]
SUCCESS_STATUSES = {"generated", "reused"}

_thread_local = threading.local()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EditTask:
    row_idx: int
    row: Dict[str, str]
    input_image_path_raw: str
    input_image_path_abs: Path
    output_path_abs: Path
    output_format: str
    prompt: str
    n: int


@dataclass
class EditResult:
    row_idx: int
    row: Dict[str, str]
    input_image_path_raw: str
    first_output_path: str
    status: str
    generated_count: int = 0
    reused_count: int = 0
    error: str = ""


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
    base: Path,
    model_name: str,
    dataset_name: str,
    quality: Optional[str],
    suffix: str,
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
    logger = logging.getLogger("image_edit_direct_openai")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def require_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set. Set it with:\n"
            "  export OPENAI_API_KEY='your_api_key_here'",
            file=sys.stderr,
        )
        sys.exit(1)


def require_openai_sdk() -> None:
    if OpenAI is None:
        print("ERROR: The openai package is not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "script_version": SCRIPT_VERSION,
    }
    try:
        import openai
        info["openai_version"] = getattr(openai, "__version__", "unknown")
    except ImportError:
        info["openai_version"] = "not-installed"
    return info


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def get_client() -> Any:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenAI()
        _thread_local.client = client
    return client


# ---------------------------------------------------------------------------
# CSV / row helpers
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
    for name in ["manipulation_prompt", "edit_prompt", "prompt", "human_description", "description", "caption"]:
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


def load_rows(csv_path: Path, limit: Optional[int], delimiter: Optional[str]) -> Tuple[List[Dict[str, str]], List[str], str]:
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
        image_column_for_repair = next((c for c in DEFAULT_IMAGE_COLUMNS if c in headers), None)
        rows: List[Dict[str, str]] = []
        repaired = 0
        skipped_blank = 0

        for physical_line, raw_row in enumerate(reader, start=2):
            row = clean_dict_row(raw_row, headers)
            if not any(str(v).strip() for v in row.values()):
                skipped_blank += 1
                continue

            if image_column_for_repair:
                image_path = row.get(image_column_for_repair, "").strip()
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
                    f"ERROR: Invalid image path in column '{image_column_for_repair}' "
                    f"at physical line {physical_line}: {image_path!r}.",
                    file=sys.stderr,
                )
                sys.exit(1)

            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break

    if repaired:
        print(f"Repaired {repaired} unquoted multiline CSV/TSV continuation lines.")
    if skipped_blank:
        print(f"Skipped {skipped_blank} blank physical lines.")

    return rows, headers, delimiter_name(dialect)


def choose_column(headers: List[str], requested: Optional[str], defaults: List[str], label: str) -> str:
    if requested:
        if requested not in headers:
            print(f"ERROR: Requested {label} column '{requested}' not found. Available: {headers}", file=sys.stderr)
            sys.exit(1)
        return requested
    for name in defaults:
        if name in headers:
            return name
    print(
        f"ERROR: No usable {label} column found. Tried: {defaults}. Available: {headers}",
        file=sys.stderr,
    )
    sys.exit(1)


def safe_relative_path(raw_image_path: str) -> Path:
    normalized = raw_image_path.strip().replace("\\", "/")
    if not normalized:
        raise ValueError("empty image path")
    pure = PurePosixPath(normalized)
    safe_parts = [
        part for part in pure.parts
        if part not in ("", ".", "/", "\\") and part != ".." and not part.endswith(":")
    ]
    if not safe_parts:
        raise ValueError(f"image path could not be converted to a safe relative path: {raw_image_path!r}")
    return Path(*safe_parts)


def resolve_input_image_path(raw_image_path: str, input_root: Path) -> Path:
    raw = raw_image_path.strip().strip('"').strip("'")
    if not raw:
        raise ValueError("empty input image path")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return input_root / path


def output_path_and_format(image_path: str, output_format_arg: str) -> Tuple[Path, str]:
    rel = safe_relative_path(image_path)
    suffix = rel.suffix.lower()
    if output_format_arg == "match":
        fmt = EXTENSION_TO_FORMAT.get(suffix, "png")
        if suffix in EXTENSION_TO_FORMAT:
            return rel.with_suffix(FORMAT_TO_EXTENSION[fmt]), fmt
        return rel.with_suffix(".png"), "png"
    fmt = output_format_arg
    if fmt not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {fmt}")
    return rel.with_suffix(FORMAT_TO_EXTENSION[fmt]), fmt


def indexed_output_path(path: Path, index: int) -> Path:
    if index == 0:
        return path
    return path.with_name(f"{path.stem}_{index + 1}{path.suffix}")


def output_fieldnames(input_fieldnames: List[str]) -> List[str]:
    fields = list(input_fieldnames)
    for col in ADDED_OUTPUT_COLUMNS:
        if col not in fields:
            fields.append(col)
    return fields


def read_completed_image_paths(output_csv: Path, image_column: str) -> Set[str]:
    completed: Set[str] = set()
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return completed
    with output_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ip = (row.get(image_column) or row.get("image_path") or "").strip()
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


def build_prompt(prompt_text: str) -> str:
    prompt_text = prompt_text.strip()
    if not prompt_text:
        raise ValueError("empty prompt")
    prompt = BASE_PROMPT_TEMPLATE.format(prompt_text=prompt_text)
    if SYSTEM_PROMPT.strip():
        prompt = f"{SYSTEM_PROMPT.strip()}\n\n{prompt}"
    return prompt


def decode_first_available_base64(data_item: Any) -> Optional[str]:
    # OpenAI SDK objects usually expose .b64_json; dict handling makes tests/mocks easier.
    value = getattr(data_item, "b64_json", None)
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(data_item, dict):
        for key in ("b64_json", "image_base64", "base64"):
            value = data_item.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


# ---------------------------------------------------------------------------
# Task preparation and API execution
# ---------------------------------------------------------------------------


def validate_input_image(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"input image not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_INPUT_EXTENSIONS:
        raise ValueError(
            f"unsupported input image extension {suffix!r}; supported for GPT image models: "
            f"{sorted(SUPPORTED_INPUT_EXTENSIONS)}"
        )
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"input image is empty: {path}")
    if size >= MAX_GPT_IMAGE_INPUT_BYTES:
        raise ValueError(f"input image is >= 50MB and may be rejected: {path}")


def prepare_tasks(
    *,
    rows: List[Dict[str, str]],
    input_image_column: str,
    prompt_column: str,
    input_root: Path,
    output_folder: Path,
    output_format: str,
    n: int,
    skip_image_paths: Set[str],
    overwrite_images: bool,
) -> Tuple[List[EditTask], List[EditResult], int]:
    tasks: List[EditTask] = []
    immediate_results: List[EditResult] = []
    skipped_existing_csv = 0

    for row_idx, row in enumerate(rows, start=1):
        raw_image_path = (row.get(input_image_column) or "").strip()
        raw_prompt = (row.get(prompt_column) or "").strip()

        if raw_image_path in skip_image_paths:
            skipped_existing_csv += 1
            continue

        try:
            if not raw_image_path:
                raise ValueError(f"empty {input_image_column}")
            if not raw_prompt or raw_prompt.upper() == "ERROR":
                raise ValueError(f"empty/error prompt in {prompt_column}")

            input_path_abs = resolve_input_image_path(raw_image_path, input_root).resolve()
            validate_input_image(input_path_abs)

            rel_output, row_fmt = output_path_and_format(raw_image_path, output_format)
            output_path_abs = output_folder / rel_output
            prompt = build_prompt(raw_prompt)

            if not overwrite_images and all(indexed_output_path(output_path_abs, i).exists() for i in range(n)):
                first_output = str(indexed_output_path(output_path_abs, 0))
                immediate_results.append(EditResult(
                    row_idx=row_idx,
                    row=row,
                    input_image_path_raw=raw_image_path,
                    first_output_path=first_output,
                    status="reused",
                    generated_count=0,
                    reused_count=n,
                ))
                continue

            tasks.append(EditTask(
                row_idx=row_idx,
                row=row,
                input_image_path_raw=raw_image_path,
                input_image_path_abs=input_path_abs,
                output_path_abs=output_path_abs,
                output_format=row_fmt,
                prompt=prompt,
                n=n,
            ))
        except Exception as exc:
            immediate_results.append(EditResult(
                row_idx=row_idx,
                row=row,
                input_image_path_raw=raw_image_path,
                first_output_path="",
                status="error",
                error=str(exc),
            ))

    return tasks, immediate_results, skipped_existing_csv


def edit_one_task(task: EditTask, args: argparse.Namespace) -> EditResult:
    backoff = max(0.1, float(args.retry_initial_sleep))
    max_sleep = max(backoff, float(args.retry_max_sleep))
    last_error = ""

    for attempt in range(args.max_retries + 1):
        try:
            client = get_client()
            request_kwargs: Dict[str, Any] = {
                "model": args.model,
                "prompt": task.prompt,
                "size": args.size,
                "output_format": task.output_format,
                "n": task.n,
            }
            if args.quality:
                request_kwargs["quality"] = args.quality
            if args.background:
                request_kwargs["background"] = args.background
            # Do not send input_fidelity by default. gpt-image-2 currently rejects
            # this parameter on Images Edit with invalid_input_fidelity_model.
            # Keep the CLI option for compatibility with models/endpoints that may
            # support it, but silently omit it for gpt-image-2.
            if args.input_fidelity and args.model != "gpt-image-2":
                request_kwargs["input_fidelity"] = args.input_fidelity
            if args.user:
                request_kwargs["user"] = args.user
            if args.output_compression is not None:
                request_kwargs["output_compression"] = args.output_compression

            with task.input_image_path_abs.open("rb") as image_file:
                result = client.images.edit(
                    image=image_file,
                    **request_kwargs,
                )

            data = getattr(result, "data", None)
            if data is None and isinstance(result, dict):
                data = result.get("data")
            if not data:
                raise RuntimeError("Images Edit response had no data")

            first_output: Optional[Path] = None
            generated_count = 0
            reused_count = 0

            for img_idx, data_item in enumerate(data):
                final_path = indexed_output_path(task.output_path_abs, img_idx)

                if not args.overwrite_images and final_path.exists() and final_path.stat().st_size > 0:
                    if first_output is None:
                        first_output = final_path
                    reused_count += 1
                    continue

                image_b64 = decode_first_available_base64(data_item)
                if not image_b64:
                    raise RuntimeError(f"response image {img_idx + 1} had no base64 data")

                image_bytes = base64.b64decode(image_b64)
                if not image_bytes:
                    raise RuntimeError(f"response image {img_idx + 1} decoded to empty bytes")

                atomic_write_bytes(final_path, image_bytes)
                if first_output is None:
                    first_output = final_path
                generated_count += 1

            if first_output is None:
                raise RuntimeError("no images were saved")

            return EditResult(
                row_idx=task.row_idx,
                row=task.row,
                input_image_path_raw=task.input_image_path_raw,
                first_output_path=str(first_output),
                status="generated" if generated_count > 0 else "reused",
                generated_count=generated_count,
                reused_count=reused_count,
            )

        except Exception as exc:
            last_error = str(exc)
            if attempt >= args.max_retries:
                break
            sleep_s = min(max_sleep, backoff * (2 ** attempt)) + random.uniform(0, 0.25)
            time.sleep(sleep_s)

    return EditResult(
        row_idx=task.row_idx,
        row=task.row,
        input_image_path_raw=task.input_image_path_raw,
        first_output_path="",
        status="error",
        error=last_error,
    )


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    require_api_key()
    require_openai_sdk()

    input_csv = Path(args.input).expanduser().resolve()
    rows, input_fieldnames, detected_delimiter = load_rows(input_csv, args.limit, args.delimiter)
    if not rows:
        print("ERROR: No rows found in the CSV after applying the limit.", file=sys.stderr)
        sys.exit(1)

    input_image_column = choose_column(input_fieldnames, args.input_image_column, DEFAULT_IMAGE_COLUMNS, "input image")
    prompt_column = choose_column(input_fieldnames, args.prompt_column, DEFAULT_PROMPT_COLUMNS, "prompt")

    input_root = Path(args.input_root).expanduser().resolve() if args.input_root else input_csv.parent
    base_output_folder = Path(args.output_folder).expanduser()
    dataset_name = infer_dataset_name(rows, input_csv)
    quality = args.quality if args.quality else None

    output_folder = (
        Path(args.run_output_folder).expanduser()
        if args.run_output_folder
        else build_run_output_folder(base_output_folder, args.model, dataset_name, quality, args.output_folder_suffix)
    ).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else output_folder / "edited_images.csv"
    metadata_file = Path(args.metadata_file).expanduser().resolve() if args.metadata_file else output_folder / "generation_metadata.json"
    text_log = Path(args.text_log).expanduser().resolve() if args.text_log else base_output_folder / "generation.log"
    fields = output_fieldnames(input_fieldnames)

    logger = setup_logger(text_log)

    if args.workers < 1:
        logger.error("--workers must be >= 1")
        sys.exit(1)
    if args.n < 1 or args.n > 10:
        logger.error("--n must be between 1 and 10")
        sys.exit(1)
    if args.output_format not in ({"match"} | SUPPORTED_OUTPUT_FORMATS):
        logger.error("Unsupported --output_format: %s", args.output_format)
        sys.exit(1)
    if args.model.startswith("gpt-image-2") and args.background == "transparent":
        logger.error("gpt-image-2 does not support --background transparent. Use auto or opaque.")
        sys.exit(1)

    skip_image_paths: Set[str] = set()
    if args.skip_existing:
        skip_image_paths = read_completed_image_paths(output_csv, input_image_column)

    logger.info("=" * 80)
    logger.info("OpenAI Direct Image Manipulation — RUN")
    logger.info("Input CSV          : %s", input_csv)
    logger.info("Input root         : %s", input_root)
    logger.info("Input image column : %s", input_image_column)
    logger.info("Prompt column      : %s", prompt_column)
    logger.info("Model              : %s", args.model)
    logger.info("Size / quality     : %s / %s", args.size, quality or "<omitted>")
    logger.info("Output format      : %s", args.output_format)
    logger.info("Workers            : %d", args.workers)
    logger.info("Output folder      : %s", output_folder)
    logger.info("Output CSV         : %s", output_csv)
    logger.info("Metadata JSON      : %s", metadata_file)
    logger.info("Total rows         : %d", len(rows))
    logger.info("Already done       : %d", len(skip_image_paths))
    logger.info("=" * 80)

    tasks, immediate_results, skipped_existing_csv = prepare_tasks(
        rows=rows,
        input_image_column=input_image_column,
        prompt_column=prompt_column,
        input_root=input_root,
        output_folder=output_folder,
        output_format=args.output_format,
        n=args.n,
        skip_image_paths=skip_image_paths,
        overwrite_images=args.overwrite_images,
    )

    run_data: Dict[str, Any] = {
        "started_at": utc_now(),
        "script_version": SCRIPT_VERSION,
        "model": args.model,
        "input_csv": str(input_csv),
        "input_csv_sha256": sha256_file(input_csv),
        "input_root": str(input_root),
        "output_folder": str(output_folder),
        "output_csv": str(output_csv),
        "metadata_file": str(metadata_file),
        "text_log": str(text_log),
        "dataset_name": dataset_name,
        "settings": {
            "limit": args.limit,
            "delimiter": args.delimiter,
            "detected_delimiter": detected_delimiter,
            "input_image_column": input_image_column,
            "prompt_column": prompt_column,
            "size": args.size,
            "quality": quality,
            "background": args.background,
            "input_fidelity": args.input_fidelity,
            "output_format": args.output_format,
            "output_compression": args.output_compression,
            "n": args.n,
            "workers": args.workers,
            "skip_existing": args.skip_existing,
            "overwrite_images": args.overwrite_images,
            "max_retries": args.max_retries,
        },
        "reproducibility": {
            "system_prompt": SYSTEM_PROMPT,
            "base_prompt_template": BASE_PROMPT_TEMPLATE,
            "prompt_construction": "BASE_PROMPT_TEMPLATE.format(prompt_text=<value from prompt_column>)",
            "api_method": "client.images.edit",
        },
        "environment": environment_info(),
        "progress": {
            "total_input_rows": len(rows),
            "skipped_existing_csv": skipped_existing_csv,
            "immediate_results": len(immediate_results),
            "api_tasks": len(tasks),
            "generated": 0,
            "reused": 0,
            "errors": 0,
        },
        "completed": False,
    }
    atomic_write_json(metadata_file, run_data)

    start_time = time.perf_counter()
    generated = 0
    reused = 0
    errors = 0

    # Write immediate results first: validation errors or locally reused files.
    for result in immediate_results:
        out_row = dict(result.row)
        out_row.update({
            "generated_image_path": result.first_output_path,
            "status": result.status,
        })
        append_output_row(output_csv, fields, out_row)
        if result.status == "reused":
            reused += 1
            logger.info("[%d] Reused: %s", result.row_idx, result.first_output_path)
        elif result.status == "error":
            errors += 1
            logger.warning("[%d] Error before API call: %s", result.row_idx, result.error)

    if not tasks:
        logger.info("No API tasks to run after skipping/reusing/validation.")
    else:
        logger.info("Prepared %d API tasks. Starting direct calls with %d worker(s).", len(tasks), args.workers)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(edit_one_task, task, args): task for task in tasks}
            completed = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed += 1
                try:
                    result = future.result()
                except Exception as exc:  # Defensive; edit_one_task should return errors instead.
                    result = EditResult(
                        row_idx=task.row_idx,
                        row=task.row,
                        input_image_path_raw=task.input_image_path_raw,
                        first_output_path="",
                        status="error",
                        error=str(exc),
                    )

                out_row = dict(result.row)
                out_row.update({
                    "generated_image_path": result.first_output_path,
                    "status": result.status,
                })
                append_output_row(output_csv, fields, out_row)

                if result.status == "generated":
                    generated += 1
                    logger.info(
                        "[%d/%d] Row %d generated: %s",
                        completed,
                        len(tasks),
                        result.row_idx,
                        result.first_output_path,
                    )
                elif result.status == "reused":
                    reused += 1
                    logger.info(
                        "[%d/%d] Row %d reused: %s",
                        completed,
                        len(tasks),
                        result.row_idx,
                        result.first_output_path,
                    )
                else:
                    errors += 1
                    logger.warning(
                        "[%d/%d] Row %d error: %s",
                        completed,
                        len(tasks),
                        result.row_idx,
                        result.error,
                    )

                if completed % max(1, args.metadata_update_every) == 0 or completed == len(tasks):
                    run_data["progress"].update({
                        "generated": generated,
                        "reused": reused,
                        "errors": errors,
                        "api_tasks_completed": completed,
                        "api_tasks_total": len(tasks),
                        "last_updated_at": utc_now(),
                    })
                    atomic_write_json(metadata_file, run_data)

    elapsed = round(time.perf_counter() - start_time, 2)
    run_data["completed"] = True
    run_data["completed_at"] = utc_now()
    run_data["progress"].update({
        "generated": generated,
        "reused": reused,
        "errors": errors,
        "elapsed_seconds": elapsed,
    })
    atomic_write_json(metadata_file, run_data)

    logger.info("=" * 80)
    logger.info("Run complete in %.1fs", elapsed)
    logger.info("Generated : %d", generated)
    logger.info("Reused    : %d", reused)
    logger.info("Errors    : %d", errors)
    logger.info("Output CSV: %s", output_csv.resolve())
    logger.info("Metadata  : %s", metadata_file.resolve())
    logger.info("=" * 80)

    if errors:
        logger.warning("Some rows failed. Check the log above and rerun with --skip_existing after fixing inputs.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Modify images via the normal OpenAI Images Edit API from a CSV, with optional parallel workers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input", required=True, help="Input CSV/TSV file.")
    parser.add_argument(
        "--input_root",
        default=None,
        help="Root directory used to resolve relative image paths. Defaults to the input CSV parent directory.",
    )
    parser.add_argument("--input_image_column", default=None, help="CSV column containing input image paths.")
    parser.add_argument("--prompt_column", default=None, help="CSV column containing edit/manipulation prompts.")
    parser.add_argument("--delimiter", default=None, help=r"CSV delimiter override, e.g. '\t'.")
    parser.add_argument("--limit", type=int, default=None, help="Max CSV rows to process.")

    parser.add_argument("--model", default="gpt-image-2", help="OpenAI image edit model.")
    parser.add_argument("--size", default="1024x1024", help="Edited image size, e.g. 1024x1024 or 1536x1024.")
    parser.add_argument(
        "--quality",
        default="auto",
        help="Image quality. Use empty string '' to omit. Common values: auto, low, medium, high.",
    )
    parser.add_argument(
        "--background",
        default="auto",
        choices=["auto", "opaque", "transparent", ""],
        help="Background setting. For gpt-image-2, use auto or opaque; transparent is not supported.",
    )
    parser.add_argument(
        "--input_fidelity",
        default=None,
        choices=["low", "high", ""],
        help=(
            "Optional input fidelity parameter. Omitted by default because "
            "gpt-image-2 rejects input_fidelity on Images Edit. If set with "
            "--model gpt-image-2, the script will ignore it."
        ),
    )
    parser.add_argument(
        "--output_format",
        default="match",
        choices=["match", "png", "jpeg", "webp"],
        help="'match' preserves jpg/png/webp extension; otherwise forces the given format.",
    )
    parser.add_argument(
        "--output_compression",
        type=int,
        default=None,
        help="Compression 0-100 for jpeg/webp outputs. Omit for API default.",
    )
    parser.add_argument("--n", type=int, default=1, help="Images per row/request, 1-10.")
    parser.add_argument("--user", default=None, help="Optional end-user identifier passed to OpenAI.")

    parser.add_argument("--workers", type=int, default=4, help="Parallel direct API calls. Set to 1 for true sequential.")
    parser.add_argument("--max_retries", type=int, default=3, help="Retries per row after API/file errors.")
    parser.add_argument("--retry_initial_sleep", type=float, default=2.0, help="Initial retry backoff seconds.")
    parser.add_argument("--retry_max_sleep", type=float, default=30.0, help="Maximum retry backoff seconds.")
    parser.add_argument("--metadata_update_every", type=int, default=10, help="Write metadata after this many completed API tasks.")

    parser.add_argument("--output_folder", default="edited_outputs", help="Base output folder.")
    parser.add_argument("--output_folder_suffix", default="", help="Optional tag appended to the auto-created run folder name.")
    parser.add_argument("--run_output_folder", default=None, help="Override the auto-generated run output folder path entirely.")
    parser.add_argument("--output_csv", default=None, help="Override output CSV path.")
    parser.add_argument("--metadata_file", default=None, help="Override metadata JSON path.")
    parser.add_argument("--text_log", default=None, help="Override text log path.")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip rows whose image path already appears as generated/reused in the output CSV.",
    )
    parser.add_argument(
        "--overwrite_images",
        action="store_true",
        help="Overwrite existing output image files instead of reusing them.",
    )

    args = parser.parse_args()

    # Normalize empty-string options to None where appropriate.
    if args.quality == "":
        args.quality = None
    if args.background == "":
        args.background = None
    if args.input_fidelity == "":
        args.input_fidelity = None

    if args.model == "gpt-image-2" and args.input_fidelity:
        print(
            "WARNING: --input_fidelity is not supported by gpt-image-2 for Images Edit; omitting it.",
            file=sys.stderr,
        )

    run(args)


if __name__ == "__main__":
    main()
