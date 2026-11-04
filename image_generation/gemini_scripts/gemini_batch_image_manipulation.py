"""
gemini_batch_image_manipulation.py
=====================================
Edit/manipulate food images from a CSV using the Gemini Batch API
(Google AI Studio — genai.Client with api_key, NOT Vertex AI).

How it works vs the one-by-one script
--------------------------------------
Instead of calling generate_content() for each row individually, this script:
  1. Reads the CSV and pre-validates every source image.
  2. Embeds each source image as base64 inline_data directly inside the JSONL
     request (no GCS or File API needed for the *input* images).
  3. Uploads the JSONL file via the Gemini File API and submits ONE batch job.
  4. Polls / downloads results later via --mode poll / --mode download.

Why base64 in JSONL?
  The Gemini Batch API accepts full GenerateContentRequest objects per line,
  including inline_data image parts — exactly the same multimodal format used
  by the real-time API. This is the only way to include per-row source images
  in a batch without needing GCS (which is Vertex AI only).

  The tradeoff: the JSONL file can be large (each image ~1–4 MB base64).
  The File API accepts up to 2 GB, so this works for thousands of images.

Supported models:
  - gemini-3.1-flash-image  (fast, efficient — default)
  - gemini-2.5-flash-image  (high-volume, low-latency)

Required CSV columns:
  image_path, dataset_name, manipulation_instruction

Optional columns preserved in output:
  class_id, is_human, description, gen_model, candidate_issues, selected_issue

Three modes via --mode:
  submit   : Validate images → build JSONL (with embedded base64 images) →
             upload via File API → submit batch job → save metadata.
  poll     : Check job status. Add --poll_until_done to wait in a loop.
  download : Download result JSONL → decode output images → write output CSV.

Authentication:
  Pass --api_key or set the GEMINI_API_KEY environment variable.

Install:
  pip install google-genai Pillow

Example — submit:
  python gemini_batch_image_manipulation.py \\
    --mode submit \\
    --input edits.csv \\
    --input_root /data/original_images \\
    --output_folder /data/edited_runs \\
    --model gemini-3.1-flash-image \\
    --api_key $GEMINI_API_KEY

Example — poll (non-blocking status check):
  python gemini_batch_image_manipulation.py \\
    --mode poll \\
    --metadata_file /data/edited_runs/.../generation_metadata.json \\
    --api_key $GEMINI_API_KEY

Example — poll until done (blocking):
  python gemini_batch_image_manipulation.py \\
    --mode poll --poll_until_done --poll_interval 60 \\
    --metadata_file /data/edited_runs/.../generation_metadata.json \\
    --api_key $GEMINI_API_KEY

Example — download results:
  python gemini_batch_image_manipulation.py \\
    --mode download \\
    --metadata_file /data/edited_runs/.../generation_metadata.json \\
    --api_key $GEMINI_API_KEY

Example — download and wait automatically:
  python gemini_batch_image_manipulation.py \\
    --mode download --poll_until_done \\
    --metadata_file /data/edited_runs/.../generation_metadata.json \\
    --api_key $GEMINI_API_KEY
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
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    from google import genai
    from google.genai import types
except ImportError as exc:
    raise ImportError(
        "Could not import google-genai. Install it with:\n  pip install google-genai"
    ) from exc

try:
    from PIL import Image
except ImportError as exc:
    raise ImportError(
        "Could not import Pillow. Install it with:\n  pip install Pillow"
    ) from exc


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_IDS: Dict[str, str] = {
    "gemini-3.1-flash-image": "gemini-3.1-flash-image",
    "gemini-2.5-flash-image": "gemini-2.5-flash-image",
}

# ---------------------------------------------------------------------------
# CSV schema  (same as flux manipulation script)
# ---------------------------------------------------------------------------

BASE_REQUIRED_COLUMNS = [
    "image_path",
    "dataset_name",
    "manipulation_instruction",
]
ADDED_OUTPUT_COLUMNS = ["generated_image_path", "status"]
SUCCESS_STATUSES = {"generated", "reused"}

# ---------------------------------------------------------------------------
# Prompt template  (mirrors flux BASE_EDIT_PROMPT_TEMPLATE)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = ""
BASE_EDIT_PROMPT_TEMPLATE = "{prompt_text}{edit_suffix}"
DEFAULT_EDIT_SUFFIX = (
    ", nearly indistinguishable from the surrounding food texture, not catching strong "
    "specular highlights, lying naturally in the scene without obvious artificial edges, "
    "visually subordinate to the dish"
)

# Batch job terminal states
COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_name(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "item"


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
    base: Path, model_name: str, dataset_name: str, suffix: str
) -> Path:
    parts = [sanitize_name(model_name), sanitize_name(dataset_name)]
    s = sanitize_name(suffix) if suffix.strip() else ""
    if s:
        parts.append(s)
    return base / "_".join(parts)


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gemini_batch_manipulation")
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


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def environment_info() -> Dict[str, Any]:
    return {
        "script": Path(__file__).name,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "google_genai_version": getattr(genai, "__version__", "unknown"),
        "pillow_version": getattr(Image, "__version__", "unknown"),
    }


def get_job_state(batch_job: Any) -> str:
    state = getattr(batch_job, "state", None)
    if state is None:
        return "UNKNOWN"
    return getattr(state, "name", str(state))


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_input_rows(
    input_csv: Path, prompt_column: str
) -> Tuple[List[Dict[str, str]], List[str]]:
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        required = list(BASE_REQUIRED_COLUMNS)
        if prompt_column not in required:
            required.append(prompt_column)
        missing = [c for c in required if c not in fieldnames]
        if missing:
            raise ValueError(
                f"Input CSV missing required columns: {missing}. Found: {fieldnames}"
            )
        return [dict(r) for r in reader], fieldnames


def output_fieldnames(input_fieldnames: Iterable[str]) -> List[str]:
    fields = list(input_fieldnames)
    for col in ADDED_OUTPUT_COLUMNS:
        if col not in fields:
            fields.append(col)
    return fields


def completed_key(row: Dict[str, str], model_name: str) -> Tuple[str, str, str, str]:
    """Unique resume key — mirrors the one-by-one script exactly."""
    return (
        (row.get("image_path") or "").strip(),
        (row.get("selected_issue") or "").strip(),
        (row.get("manipulation_instruction") or "").strip(),
        model_name,
    )


def read_completed_keys(
    output_csv: Path, model_name: str
) -> Set[Tuple[str, str, str, str]]:
    completed: Set[Tuple[str, str, str, str]] = set()
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return completed
    with output_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = (row.get("status") or "").strip().lower()
            gip = (row.get("generated_image_path") or "").strip()
            if status in SUCCESS_STATUSES and gip:
                completed.add(completed_key(row, model_name))
    return completed


def append_output_row(
    output_csv: Path, fieldnames: List[str], row: Dict[str, Any]
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists() and output_csv.stat().st_size > 0
    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


# ---------------------------------------------------------------------------
# Image / prompt helpers
# ---------------------------------------------------------------------------

def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def build_edit_prompt(prompt_text: str, edit_suffix: str) -> str:
    return BASE_EDIT_PROMPT_TEMPLATE.format(
        prompt_text=prompt_text.strip(),
        edit_suffix=edit_suffix or "",
    )


def detect_mime_type(path: Path) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(path.suffix.lower(), "image/png")


def image_to_base64(image: Image.Image, mime_type: str = "image/png") -> str:
    """Encode a PIL Image to a base64 string suitable for inline_data."""
    fmt = "JPEG" if "jpeg" in mime_type else "PNG"
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_output_image_path(
    row_index: int,
    row: Dict[str, str],
    output_folder: Path,
    model_name: str,
    image_extension: str,
) -> Path:
    """Mirrors flux build_output_image_path: stem + model + issue + sha1 digest."""
    image_path = (row.get("image_path") or "").strip()
    rel = Path(image_path)
    suffix = image_extension if image_extension.startswith(".") else f".{image_extension}"
    issue = sanitize_name(row.get("selected_issue", "edit"))
    instruction = (row.get("manipulation_instruction") or "").strip()
    digest = hashlib.sha1(
        f"{row_index}-{image_path}-{issue}-{instruction}-{model_name}".encode()
    ).hexdigest()[:8]
    filename = f"{rel.stem}_{sanitize_name(model_name)}_{issue}_{digest}{suffix}"
    return output_folder / rel.parent / filename


def row_to_batch_key(row_index: int, row: Dict[str, str]) -> str:
    """Stable, filesystem-safe key used to cross-reference batch results to CSV rows."""
    image_path = (row.get("image_path") or "").strip()
    issue = (row.get("selected_issue") or "edit").strip()
    digest = hashlib.sha1(
        f"{row_index}-{image_path}-{issue}".encode()
    ).hexdigest()[:8]
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", image_path)[:60]
    return f"req-{row_index:06d}-{safe}-{digest}"


# ---------------------------------------------------------------------------
# JSONL builder  — the key difference from text-only batch generation
# ---------------------------------------------------------------------------

def build_jsonl(
    pending: List[Tuple[int, Dict[str, str]]],
    input_root: Path,
    prompt_column: str,
    edit_suffix: str,
    model_name: str,
    image_extension: str,
    output_folder: Path,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[Tuple[int, Dict[str, str]]]]:
    """
    Build batch JSONL requests with base64-embedded source images.

    Each JSONL line contains a full GenerateContentRequest:
      - Part 1: text  (the manipulation instruction)
      - Part 2: inline_data  (the source image as base64)

    Returns:
      requests       : list of dicts ready to be serialised as JSONL lines
      row_index_map  : {batch_key -> {row, row_index, output_image_path}}
                       saved in metadata so download mode can stitch results back
      skipped        : rows that failed pre-validation (missing file, bad prompt, etc.)
    """
    requests: List[Dict[str, Any]] = []
    row_index_map: Dict[str, Dict[str, Any]] = {}
    skipped: List[Tuple[int, Dict[str, str]]] = []

    for row_index, row in pending:
        image_path = (row.get("image_path") or "").strip()
        prompt_text = (row.get(prompt_column) or "").strip()

        # Pre-validate
        if not image_path:
            logger.warning("Row %d: empty image_path — skipping.", row_index)
            skipped.append((row_index, row))
            continue
        if not prompt_text or prompt_text.upper() == "ERROR":
            logger.warning(
                "Row %d (%s): empty/error prompt — skipping.", row_index, image_path
            )
            skipped.append((row_index, row))
            continue

        source_abs_path = input_root / image_path
        if not source_abs_path.exists():
            logger.warning(
                "Row %d: source image not found: %s — skipping.",
                row_index, source_abs_path,
            )
            skipped.append((row_index, row))
            continue

        # Load image and encode as base64
        try:
            img = load_rgb(source_abs_path)
            mime_type = detect_mime_type(source_abs_path)
            b64_data = image_to_base64(img, mime_type)
        except Exception as exc:
            logger.warning(
                "Row %d (%s): could not load/encode image: %s — skipping.",
                row_index, image_path, exc,
            )
            skipped.append((row_index, row))
            continue

        prompt = build_edit_prompt(prompt_text, edit_suffix)
        key = row_to_batch_key(row_index, row)
        output_image_path = build_output_image_path(
            row_index, row, output_folder, model_name, image_extension
        )

        requests.append({
            "key": key,
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            # Part 1: text instruction
                            {"text": prompt},
                            # Part 2: source image embedded as base64
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": b64_data,
                                }
                            },
                        ],
                    }
                ],
                "generation_config": {
                    "responseModalities": ["TEXT", "IMAGE"],
                },
            },
        })

        row_index_map[key] = {
            "row": row,
            "row_index": row_index,
            "output_image_path": str(output_image_path),
            "image_path": image_path,
            "selected_issue": row.get("selected_issue", ""),
            "prompt_text": prompt_text,
            "full_prompt": prompt,
        }

    return requests, row_index_map, skipped


# ---------------------------------------------------------------------------
# MODE: submit
# ---------------------------------------------------------------------------

def mode_submit(args: argparse.Namespace, logger: logging.Logger) -> None:
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.error("Provide --api_key or set GEMINI_API_KEY.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    model_id = MODEL_IDS[args.model]
    input_csv = Path(args.input)
    input_root = Path(args.input_root)
    base_output_folder = Path(args.output_folder)

    rows, input_fieldnames = load_input_rows(input_csv, args.prompt_column)
    dataset_name = infer_dataset_name(rows, input_csv)
    output_folder = build_run_output_folder(
        base_output_folder, args.model, dataset_name, args.output_folder_suffix
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    output_csv = (
        Path(args.output_csv) if args.output_csv
        else output_folder / "generated_images.csv"
    )
    json_log = (
        Path(args.metadata_file) if args.metadata_file
        else output_folder / "generation_metadata.json"
    )
    text_log = (
        Path(args.text_log) if args.text_log
        else base_output_folder / "generation.log"
    )
    logger = setup_logger(text_log)

    completed_keys = set() if args.overwrite else read_completed_keys(output_csv, args.model)
    pending = [
        (idx, row) for idx, row in enumerate(rows)
        if completed_key(row, args.model) not in completed_keys
    ]

    logger.info("=" * 80)
    logger.info("Gemini Batch Image Manipulation (AI Studio) — SUBMIT")
    logger.info("Input CSV      : %s", input_csv)
    logger.info("Input root     : %s", input_root)
    logger.info("Model          : %s  (%s)", args.model, model_id)
    logger.info("Dataset        : %s", dataset_name)
    logger.info("Run output     : %s", output_folder)
    logger.info("Prompt column  : %s", args.prompt_column)
    logger.info("Edit suffix    : %s", repr(args.edit_suffix))
    logger.info("Total rows     : %d", len(rows))
    logger.info("Completed      : %d", len(completed_keys))
    logger.info("Pending        : %d", len(pending))
    logger.info("=" * 80)

    if not pending:
        logger.info("No pending rows. Nothing to submit.")
        return

    # Build JSONL — images are base64-embedded inside each request
    logger.info("Loading and encoding %d source images into JSONL…", len(pending))
    requests, row_index_map, skipped = build_jsonl(
        pending=pending,
        input_root=input_root,
        prompt_column=args.prompt_column,
        edit_suffix=args.edit_suffix,
        model_name=args.model,
        image_extension=args.image_extension,
        output_folder=output_folder,
        logger=logger,
    )

    # Write skipped rows immediately to output CSV as errors
    fields = output_fieldnames(input_fieldnames)
    for row_index, row in skipped:
        out_row = dict(row)
        out_row.update({"generated_image_path": "", "status": "error_skipped_at_submit"})
        append_output_row(output_csv, fields, out_row)

    if not requests:
        logger.error("No valid requests after pre-validation. Nothing to submit.")
        sys.exit(1)

    logger.info(
        "JSONL built: %d valid requests, %d skipped.", len(requests), len(skipped)
    )

    # Write JSONL to disk (useful for debugging)
    jsonl_path = output_folder / "batch_input.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    jsonl_size_mb = jsonl_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Wrote JSONL to %s (%.1f MB)", jsonl_path, jsonl_size_mb
    )

    # Upload JSONL via File API
    logger.info("Uploading JSONL to Gemini File API…")
    uploaded_file = client.files.upload(
        file=str(jsonl_path),
        config=types.UploadFileConfig(
            display_name=f"{sanitize_name(args.model)}_{dataset_name}_manip_input",
            mime_type="jsonl",
        ),
    )
    logger.info("Uploaded file: %s", uploaded_file.name)

    # Submit batch job
    display_name = (
        f"{sanitize_name(args.model)}_{dataset_name}_{utc_now_iso()[:10]}"
    )
    logger.info("Submitting batch job (display_name=%s)…", display_name)
    batch_job = client.batches.create(
        model=model_id,
        src=uploaded_file.name,
        config={"display_name": display_name},
    )
    state = get_job_state(batch_job)
    logger.info("Batch job created : %s", batch_job.name)
    logger.info("Initial state     : %s", state)

    # Save metadata
    run_data: Dict[str, Any] = {
        "submitted_at": utc_now_iso(),
        "batch_job_name": batch_job.name,
        "batch_job_display_name": display_name,
        "model": args.model,
        "model_id": model_id,
        # paths
        "input_csv": str(input_csv.resolve()),
        "input_root": str(input_root.resolve()),
        "jsonl_path": str(jsonl_path.resolve()),
        "uploaded_file_name": uploaded_file.name,
        "output_folder": str(output_folder.resolve()),
        "output_csv": str(output_csv.resolve()),
        "json_log": str(json_log.resolve()),
        "text_log": str(text_log.resolve()),
        "dataset_name": dataset_name,
        "output_folder_suffix": args.output_folder_suffix,
        "settings": {
            "prompt_column": args.prompt_column,
            "edit_suffix": args.edit_suffix,
            "edit_suffix_enabled": bool(args.edit_suffix),
            "image_extension": args.image_extension,
            "overwrite": args.overwrite,
        },
        "reproducibility": {
            "system_prompt": SYSTEM_PROMPT,
            "base_edit_prompt_template": BASE_EDIT_PROMPT_TEMPLATE,
            "edit_suffix": args.edit_suffix,
            "prompt_construction": (
                "BASE_EDIT_PROMPT_TEMPLATE.format("
                "prompt_text=<prompt_column value>, edit_suffix=<edit_suffix>)"
            ),
            "model": args.model,
            "model_id": model_id,
            "prompt_column": args.prompt_column,
            "dataset_name": dataset_name,
            "note": (
                "Gemini image generation is non-deterministic. "
                "Re-running with the same inputs produces different images."
            ),
        },
        "environment": environment_info(),
        "progress": {
            "total_input_rows": len(rows),
            "skipped_already_completed": len(completed_keys),
            "submitted_this_run": len(requests),
            "skipped_invalid_at_submit": len(skipped),
        },
        "batch_state_history": [
            {"timestamp": utc_now_iso(), "state": state}
        ],
        # row_index_map lets download mode reconnect results to CSV rows
        # NOTE: base64 image data is NOT stored here — only metadata
        "row_index_map": {
            k: {kk: vv for kk, vv in v.items() if kk != "b64_data"}
            for k, v in row_index_map.items()
        },
        "input_fieldnames": input_fieldnames,
        "download_completed": False,
    }
    atomic_write_json(json_log, run_data)

    logger.info("=" * 80)
    logger.info("Submission complete.")
    logger.info("Batch job name   : %s", batch_job.name)
    logger.info("Metadata saved   : %s", json_log)
    logger.info("JSONL size       : %.1f MB", jsonl_size_mb)
    logger.info("")
    logger.info("Next steps:")
    logger.info(
        "  Poll:     python %s --mode poll     --metadata_file %s --api_key <KEY>",
        Path(sys.argv[0]).name, json_log,
    )
    logger.info(
        "  Download: python %s --mode download --metadata_file %s --api_key <KEY>",
        Path(sys.argv[0]).name, json_log,
    )
    logger.info(
        "  Console:  https://aistudio.google.com (Batch jobs section)"
    )
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# MODE: poll
# ---------------------------------------------------------------------------

def mode_poll(args: argparse.Namespace, logger: logging.Logger) -> None:
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.error("Provide --api_key or set GEMINI_API_KEY.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    meta_path = Path(args.metadata_file)
    if not meta_path.exists():
        logger.error("Metadata file not found: %s", meta_path)
        sys.exit(1)

    run_data = json.loads(meta_path.read_text(encoding="utf-8"))
    job_name = run_data["batch_job_name"]

    text_log = Path(run_data.get("text_log", str(meta_path.parent / "generation.log")))
    logger = setup_logger(text_log)

    logger.info("=" * 80)
    logger.info("Gemini Batch Image Manipulation (AI Studio) — POLL")
    logger.info("Batch job  : %s", job_name)
    logger.info("Model      : %s  (%s)", run_data["model"], run_data["model_id"])
    logger.info("Submitted  : %s", run_data["submitted_at"])

    poll_interval = getattr(args, "poll_interval", 60)

    while True:
        batch_job = client.batches.get(name=job_name)
        state = get_job_state(batch_job)
        logger.info("Current state: %s", state)

        run_data.setdefault("batch_state_history", []).append(
            {"timestamp": utc_now_iso(), "state": state}
        )
        atomic_write_json(meta_path, run_data)

        if state in COMPLETED_STATES:
            logger.info("Job reached terminal state: %s", state)
            if state == "JOB_STATE_FAILED":
                logger.error(
                    "Job FAILED. Error: %s", getattr(batch_job, "error", "unknown")
                )
            elif state == "JOB_STATE_SUCCEEDED":
                logger.info(
                    "Job SUCCEEDED. Run with --mode download to save edited images."
                )
            break

        if not getattr(args, "poll_until_done", False):
            logger.info(
                "Job not done (state=%s). Re-run with --mode poll to check again, "
                "or add --poll_until_done to wait here.",
                state,
            )
            break

        logger.info("Waiting %d seconds before next poll…", poll_interval)
        time.sleep(poll_interval)

    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# MODE: download
# ---------------------------------------------------------------------------

def mode_download(args: argparse.Namespace, logger: logging.Logger) -> None:
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.error("Provide --api_key or set GEMINI_API_KEY.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    meta_path = Path(args.metadata_file)
    if not meta_path.exists():
        logger.error("Metadata file not found: %s", meta_path)
        sys.exit(1)

    run_data = json.loads(meta_path.read_text(encoding="utf-8"))
    job_name = run_data["batch_job_name"]
    output_folder = Path(run_data["output_folder"])
    output_csv = Path(run_data["output_csv"])
    model_name = run_data["model"]
    image_extension = run_data["settings"]["image_extension"]
    row_index_map: Dict[str, Dict[str, Any]] = run_data.get("row_index_map", {})
    input_fieldnames: List[str] = run_data.get("input_fieldnames", [])
    fields = output_fieldnames(input_fieldnames)

    text_log = Path(run_data.get("text_log", str(meta_path.parent / "generation.log")))
    logger = setup_logger(text_log)

    logger.info("=" * 80)
    logger.info("Gemini Batch Image Manipulation (AI Studio) — DOWNLOAD")
    logger.info("Batch job     : %s", job_name)
    logger.info("Model         : %s  (%s)", model_name, run_data["model_id"])
    logger.info("Output folder : %s", output_folder)

    # Optionally wait for the job to finish before downloading
    if getattr(args, "poll_until_done", False):
        poll_interval = getattr(args, "poll_interval", 60)
        logger.info("--poll_until_done set; waiting for job to finish…")
        while True:
            batch_job = client.batches.get(name=job_name)
            state = get_job_state(batch_job)
            run_data.setdefault("batch_state_history", []).append(
                {"timestamp": utc_now_iso(), "state": state}
            )
            atomic_write_json(meta_path, run_data)
            if state in COMPLETED_STATES:
                break
            logger.info("State: %s. Waiting %d seconds…", state, poll_interval)
            time.sleep(poll_interval)
    else:
        batch_job = client.batches.get(name=job_name)

    state = get_job_state(batch_job)
    logger.info("Job state: %s", state)

    if state != "JOB_STATE_SUCCEEDED":
        logger.error(
            "Job is not JOB_STATE_SUCCEEDED (current: %s). "
            "Add --poll_until_done to wait, or check AI Studio.",
            state,
        )
        run_data.setdefault("batch_state_history", []).append(
            {"timestamp": utc_now_iso(), "state": state}
        )
        atomic_write_json(meta_path, run_data)
        sys.exit(1)

    # Download result JSONL from the Gemini File API
    result_file_name = batch_job.dest.file_name
    logger.info("Downloading result file: %s", result_file_name)
    file_content_bytes: bytes = client.files.download(file=result_file_name)
    file_content = file_content_bytes.decode("utf-8")
    logger.info(
        "Downloaded result file (%.1f MB)", len(file_content_bytes) / (1024 * 1024)
    )

    # Parse result lines and save images
    generated = 0
    errors = 0
    download_start = time.perf_counter()

    for line in file_content.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse result line: %s", exc)
            errors += 1
            continue

        key = parsed.get("key", "")
        entry = row_index_map.get(key)

        if entry is None:
            logger.warning(
                "No matching CSV row for batch key '%s'. Skipping.", key
            )
            errors += 1
            continue

        row: Dict[str, str] = entry["row"]
        image_path: str = entry["image_path"]
        output_image_path = Path(entry["output_image_path"])

        # Per-request API error
        if "error" in parsed and parsed["error"]:
            logger.warning(
                "API error for key '%s' (image_path=%s): %s",
                key, image_path, parsed["error"],
            )
            out_row = dict(row)
            out_row.update({"generated_image_path": "", "status": "error"})
            append_output_row(output_csv, fields, out_row)
            errors += 1
            continue

        # Extract the first image part from the response
        response = parsed.get("response", {})
        candidates = response.get("candidates", [])
        image_saved = False

        for candidate in candidates:
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            for part in parts:
                inline_data = part.get("inlineData") or part.get("inline_data")
                if inline_data:
                    mime_type = inline_data.get("mimeType", "image/png")
                    data_b64 = inline_data.get("data", "")
                    image_bytes = base64.b64decode(data_b64)

                    output_image_path.parent.mkdir(parents=True, exist_ok=True)

                    # Re-encode to the requested extension if needed
                    if output_image_path.suffix.lower() in {".jpg", ".jpeg"}:
                        img = Image.open(BytesIO(image_bytes)).convert("RGB")
                        img.save(output_image_path, quality=95)
                    else:
                        output_image_path.write_bytes(image_bytes)

                    logger.info(
                        "[%d] saved: %s  (mime=%s, %d bytes)",
                        generated + 1,
                        output_image_path,
                        mime_type,
                        len(image_bytes),
                    )
                    image_saved = True
                    break
            if image_saved:
                break

        if image_saved:
            out_row = dict(row)
            out_row.update(
                {
                    "generated_image_path": str(output_image_path),
                    "status": "generated",
                }
            )
            append_output_row(output_csv, fields, out_row)
            generated += 1
        else:
            text_parts = [
                p.get("text", "")
                for c in candidates
                for p in c.get("content", {}).get("parts", [])
                if p.get("text")
            ]
            logger.warning(
                "No image in response for key '%s' (image_path=%s). "
                "Model text: %s",
                key, image_path, " | ".join(text_parts)[:200],
            )
            out_row = dict(row)
            out_row.update({"generated_image_path": "", "status": "error"})
            append_output_row(output_csv, fields, out_row)
            errors += 1

    elapsed = round(time.perf_counter() - download_start, 2)

    run_data["download_completed"] = True
    run_data["download_completed_at"] = utc_now_iso()
    run_data["download_stats"] = {
        "generated": generated,
        "errors": errors,
        "elapsed_seconds": elapsed,
    }
    run_data.setdefault("batch_state_history", []).append(
        {"timestamp": utc_now_iso(), "state": "DOWNLOAD_COMPLETE"}
    )
    atomic_write_json(meta_path, run_data)

    logger.info("=" * 80)
    logger.info("Download complete in %.1fs", elapsed)
    logger.info("Edited images saved : %d", generated)
    logger.info("Errors              : %d", errors)
    logger.info("Output CSV          : %s", output_csv.resolve())
    logger.info("Metadata JSON       : %s", meta_path.resolve())
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Edit/manipulate food images in batch using the Gemini Batch API "
            "(Google AI Studio)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- mode ----
    parser.add_argument(
        "--mode",
        required=True,
        choices=["submit", "poll", "download"],
        help=(
            "submit   : Load CSV, encode source images, build JSONL, upload, "
            "start batch job, save metadata.\n"
            "poll     : Check job status. Add --poll_until_done to wait in a loop.\n"
            "download : Download completed results, decode images, write output CSV."
        ),
    )

    # ---- auth ----
    parser.add_argument(
        "--api_key", default=None,
        help="Google AI Studio API key. Falls back to GEMINI_API_KEY env var.",
    )

    # ---- submit-only args ----
    parser.add_argument(
        "--input", "--csv", dest="input", default=None,
        help="[submit] Input CSV path.",
    )
    parser.add_argument(
        "--input_root", "--image_root", dest="input_root", default=None,
        help="[submit] Root folder for relative image_path values in the CSV.",
    )
    parser.add_argument(
        "--model", default="gemini-3.1-flash-image",
        choices=sorted(MODEL_IDS),
        help="[submit] Gemini model to use for editing.",
    )
    parser.add_argument(
        "--output_folder", "--out_dir", dest="output_folder",
        default="edited_outputs",
        help=(
            "[submit] Base folder for run output. "
            "Actual run folder: <output_folder>/<model>_<dataset>[_suffix]."
        ),
    )
    parser.add_argument(
        "--output_folder_suffix", default="",
        help="[submit] Optional tag appended to the auto-created run folder name.",
    )
    parser.add_argument(
        "--prompt_column", default="manipulation_instruction",
        help="[submit] CSV column to use as the edit instruction.",
    )
    parser.add_argument(
        "--edit_suffix", default=DEFAULT_EDIT_SUFFIX,
        help="[submit] Suffix appended to every instruction. Pass '' to disable.",
    )
    parser.add_argument(
        "--image_extension", default=".png",
        help="[submit/download] File extension for saved output images (.png or .jpg).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="[submit] Re-submit rows that already appear as successful in the output CSV.",
    )
    parser.add_argument(
        "--output_csv", default=None,
        help="[submit] Override the default output CSV path.",
    )

    # ---- poll/download args ----
    parser.add_argument(
        "--metadata_file", default=None,
        help=(
            "[poll/download] Path to generation_metadata.json saved by --mode submit.\n"
            "[submit] Override the default metadata file path."
        ),
    )
    parser.add_argument(
        "--poll_until_done", action="store_true",
        help=(
            "[poll/download] Keep polling in a loop until the job reaches a "
            "terminal state (SUCCEEDED / FAILED / CANCELLED / EXPIRED)."
        ),
    )
    parser.add_argument(
        "--poll_interval", type=int, default=60,
        help="[poll/download] Seconds between status checks when --poll_until_done is set.",
    )

    # ---- logging ----
    parser.add_argument(
        "--text_log", default=None,
        help="Override the log file path (default: <output_folder>/generation.log).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _tmp_logger = logging.getLogger("gemini_batch_manip_pre")
    _tmp_logger.setLevel(logging.INFO)
    if not _tmp_logger.handlers:
        _tmp_logger.addHandler(logging.StreamHandler(sys.stdout))

    if args.mode == "submit":
        if not args.input:
            print("ERROR: --mode submit requires --input", file=sys.stderr)
            sys.exit(1)
        if not args.input_root:
            print("ERROR: --mode submit requires --input_root", file=sys.stderr)
            sys.exit(1)
        mode_submit(args, _tmp_logger)

    elif args.mode == "poll":
        if not args.metadata_file:
            print("ERROR: --mode poll requires --metadata_file", file=sys.stderr)
            sys.exit(1)
        mode_poll(args, _tmp_logger)

    elif args.mode == "download":
        if not args.metadata_file:
            print("ERROR: --mode download requires --metadata_file", file=sys.stderr)
            sys.exit(1)
        mode_download(args, _tmp_logger)


if __name__ == "__main__":
    main()