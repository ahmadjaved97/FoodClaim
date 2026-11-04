"""
gemini_batch_image_generation.py
=================================
Generate food images from a CSV using the Gemini Batch API.

Supported model names:
  - gemini-3.1-flash-image  -> gemini-3.1-flash-image
  - gemini-2.5-flash-image  -> gemini-2.5-flash-preview-05-20

Input CSV columns required (same schema as flux2_klein script):
  image_path, dataset_name, class_id, is_human, description, gen_model

Key behaviour:
  - Converts CSV rows to a JSONL file, uploads it, and submits a Gemini Batch job.
  - Saves a JSON metadata file with everything needed to reproduce or resume:
      batch job name, model, prompt template, all settings, timestamps.
  - Three modes via --mode:
      submit   : Build JSONL, upload, submit batch job, save metadata. Exit.
      poll     : Load a previous metadata file, check job status, wait if needed.
      download : Load a previous metadata file, download results, decode images,
                 write output CSV (mirrors the flux script output format).

Example — submit:
  python gemini_batch_image_generation.py \\
    --mode submit \\
    --input input.csv \\
    --output_folder generated_outputs \\
    --model gemini-3.1-flash-image \\
    --api_key $GEMINI_API_KEY

Example — poll (check if job is done):
  python gemini_batch_image_generation.py \\
    --mode poll \\
    --metadata_file generated_outputs/gemini-3.1-flash-image_mydata/generation_metadata.json \\
    --api_key $GEMINI_API_KEY

Example — download (save images + output CSV):
  python gemini_batch_image_generation.py \\
    --mode download \\
    --metadata_file generated_outputs/gemini-3.1-flash-image_mydata/generation_metadata.json \\
    --api_key $GEMINI_API_KEY \\
    --poll_until_done   # optionally wait for completion before downloading
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from google import genai
    from google.genai import types
except ImportError as exc:
    raise ImportError(
        "Could not import google-genai. Install it with:\n"
        "  pip install google-genai"
    ) from exc


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REPOS: Dict[str, str] = {
    "gemini-3.1-flash-image": "gemini-3.1-flash-image",
    "gemini-2.5-flash-image": "gemini-2.5-flash-image",
}

# ---------------------------------------------------------------------------
# CSV schema (matches flux2_klein script)
# ---------------------------------------------------------------------------

BASE_REQUIRED_COLUMNS = [
    "image_path", "dataset_name", "class_id", "gen_model"
]
ADDED_OUTPUT_COLUMNS = ["generated_image_path", "status"]
SUCCESS_STATUSES = {"generated", "reused"}

# ---------------------------------------------------------------------------
# Prompt template (mirrors BASE_PROMPT_TEMPLATE in the flux script)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = ""
BASE_PROMPT_TEMPLATE = (
    "Generate a high-quality, realistic food photograph based only on this food description. "
    "Match the food appearance, ingredients, serving vessel, scene composition, lighting, "
    "camera angle, and image quality details as closely as possible.\n\n"
    "{prompt_text}"
)

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
    base: Path, model_name: str, dataset_name: str, suffix: str
) -> Path:
    parts = [sanitize_name(model_name), sanitize_name(dataset_name)]
    s = sanitize_name(suffix) if suffix.strip() else ""
    if s:
        parts.append(s)
    return base / "_".join(parts)


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gemini_batch_image_gen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")]:
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
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "google_genai_version": getattr(genai, "__version__", "unknown"),
    }


def build_prompt(prompt_text: str) -> str:
    return BASE_PROMPT_TEMPLATE.format(prompt_text=prompt_text.strip())


def build_output_image_path(
    image_path: str, output_folder: Path, model_name: str, image_extension: str
) -> Path:
    rel = Path(image_path)
    suffix = image_extension if image_extension.startswith(".") else f".{image_extension}"
    return output_folder / rel.parent / f"{rel.stem}_{sanitize_name(model_name)}{suffix}"


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


def output_fieldnames(input_fieldnames: List[str]) -> List[str]:
    fields = list(input_fieldnames)
    for col in ADDED_OUTPUT_COLUMNS:
        if col not in fields:
            fields.append(col)
    return fields


def read_completed_keys(output_csv: Path, model_name: str) -> Set[str]:
    """Return set of image_path values that were already successfully generated."""
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
# JSONL builder
# ---------------------------------------------------------------------------

def build_jsonl(
    rows: List[Dict[str, str]],
    prompt_column: str,
    model_name: str,
    image_extension: str,
    output_folder: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """
    Build a list of JSONL request dicts.

    Returns:
        requests   : list of dicts ready to be written as JSONL lines
        row_index  : parallel list mapping each request back to its CSV row
                     (stored separately so the JSONL stays clean)
    """
    requests: List[Dict[str, Any]] = []
    row_index: List[Dict[str, str]] = []

    for row in rows:
        image_path = (row.get("image_path") or "").strip()
        prompt_text = (row.get(prompt_column) or "").strip()

        if not image_path or not prompt_text or prompt_text.upper() == "ERROR":
            continue  # skip invalid rows; they'll be logged as errors at download time

        key = f"request-{image_path.replace('/', '_').replace(' ', '_')}"
        requests.append({
            "key": key,
            "request": {
                "contents": [{"parts": [{"text": build_prompt(prompt_text)}]}],
                "generation_config": {
                    "responseModalities": ["TEXT", "IMAGE"],
                },
            },
        })
        row_index.append(row)

    return requests, row_index


# ---------------------------------------------------------------------------
# MODE: submit
# ---------------------------------------------------------------------------

def mode_submit(args: argparse.Namespace, logger: logging.Logger) -> None:
    api_key = args.api_key or _require_env("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    input_csv = Path(args.input)
    base_output_folder = Path(args.output_folder)
    repo_id = MODEL_REPOS[args.model]

    rows, input_fieldnames = load_input_rows(input_csv, args.prompt_column)
    dataset_name = infer_dataset_name(rows, input_csv)
    output_folder = build_run_output_folder(
        base_output_folder, args.model, dataset_name, args.output_folder_suffix
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    output_csv = Path(args.output_csv) if args.output_csv else output_folder / "generated_images.csv"
    json_log = Path(args.metadata_file) if args.metadata_file else output_folder / "generation_metadata.json"
    text_log = Path(args.text_log) if args.text_log else base_output_folder / "generation.log"
    logger = setup_logger(text_log)

    # Skip already completed rows
    completed_keys = set() if args.overwrite else read_completed_keys(output_csv, args.model)
    pending_rows = [r for r in rows if (r.get("image_path") or "").strip() not in completed_keys]

    logger.info("=" * 80)
    logger.info("Gemini Batch Image Generation — SUBMIT")
    logger.info("Input CSV     : %s", input_csv)
    logger.info("Model         : %s  (%s)", args.model, repo_id)
    logger.info("Dataset       : %s", dataset_name)
    logger.info("Run output    : %s", output_folder)
    logger.info("Prompt column : %s", args.prompt_column)
    logger.info("Total rows    : %d", len(rows))
    logger.info("Completed     : %d", len(completed_keys))
    logger.info("Pending       : %d", len(pending_rows))
    logger.info("=" * 80)

    if not pending_rows:
        logger.info("No pending rows. Nothing to submit.")
        return

    # Build JSONL
    requests, row_index = build_jsonl(
        pending_rows, args.prompt_column, args.model,
        args.image_extension, output_folder
    )

    if not requests:
        logger.error("All pending rows had empty/invalid prompts. Nothing to submit.")
        sys.exit(1)

    jsonl_path = output_folder / "batch_input.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    logger.info("Wrote %d requests to %s", len(requests), jsonl_path)

    # Upload JSONL via File API
    logger.info("Uploading JSONL to Gemini File API…")
    uploaded_file = client.files.upload(
        file=str(jsonl_path),
        config=types.UploadFileConfig(
            display_name=f"{sanitize_name(args.model)}_{dataset_name}_batch_input",
            mime_type="jsonl",
        ),
    )
    logger.info("Uploaded file: %s", uploaded_file.name)

    # Submit batch job
    display_name = f"{sanitize_name(args.model)}_{dataset_name}_{utc_now_iso()[:10]}"
    logger.info("Submitting batch job (display_name=%s)…", display_name)
    batch_job = client.batches.create(
        model=repo_id,
        src=uploaded_file.name,
        config={"display_name": display_name},
    )
    logger.info("Batch job created: %s", batch_job.name)
    logger.info(
        "Initial state: %s",
        getattr(batch_job.state, "name", str(batch_job.state)),
    )

    # Build row_index_map: key -> csv row  (so download can stitch results back)
    row_index_map = {
        f"request-{(r.get('image_path') or '').strip().replace('/', '_').replace(' ', '_')}": r
        for r in row_index
    }

    # Save metadata JSON
    run_data: Dict[str, Any] = {
        "submitted_at": utc_now_iso(),
        "batch_job_name": batch_job.name,
        "batch_job_display_name": display_name,
        "model": args.model,
        "repo_id": repo_id,
        "input_csv": str(input_csv.resolve()),
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
            "image_extension": args.image_extension,
            "overwrite": args.overwrite,
        },
        "reproducibility": {
            "system_prompt": SYSTEM_PROMPT,
            "base_prompt_template": BASE_PROMPT_TEMPLATE,
            "prompt_construction": "BASE_PROMPT_TEMPLATE.format(prompt_text=<value from prompt_column>)",
            "model": args.model,
            "repo_id": repo_id,
            "prompt_column": args.prompt_column,
            "dataset_name": dataset_name,
            "output_folder_suffix": args.output_folder_suffix,
        },
        "environment": environment_info(),
        "progress": {
            "total_input_rows": len(rows),
            "skipped_already_completed": len(completed_keys),
            "submitted_this_run": len(requests),
        },
        "batch_state_history": [
            {
                "timestamp": utc_now_iso(),
                "state": getattr(batch_job.state, "name", str(batch_job.state)),
            }
        ],
        # row_index_map lets download mode know which CSV row each batch key belongs to
        "row_index_map": row_index_map,
        "input_fieldnames": input_fieldnames,
        "download_completed": False,
    }
    atomic_write_json(json_log, run_data)

    logger.info("=" * 80)
    logger.info("Submission complete.")
    logger.info("Batch job name : %s", batch_job.name)
    logger.info("Metadata saved : %s", json_log)
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
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# MODE: poll
# ---------------------------------------------------------------------------

def mode_poll(args: argparse.Namespace, logger: logging.Logger) -> None:
    api_key = args.api_key or _require_env("GEMINI_API_KEY")
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
    logger.info("Gemini Batch Image Generation — POLL")
    logger.info("Batch job : %s", job_name)
    logger.info("Model     : %s  (%s)", run_data["model"], run_data["repo_id"])
    logger.info("Submitted : %s", run_data["submitted_at"])

    poll_interval = getattr(args, "poll_interval", 60)

    while True:
        batch_job = client.batches.get(name=job_name)
        state = getattr(batch_job.state, "name", str(batch_job.state))
        logger.info("Current state: %s", state)

        # Record state history
        run_data.setdefault("batch_state_history", []).append(
            {"timestamp": utc_now_iso(), "state": state}
        )
        atomic_write_json(meta_path, run_data)

        if state in COMPLETED_STATES:
            logger.info("Job finished with state: %s", state)
            if state == "JOB_STATE_FAILED":
                logger.error("Batch job failed. Error: %s", getattr(batch_job, "error", "unknown"))
            elif state in {"JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
                logger.warning("Batch job ended with state: %s", state)
            else:
                logger.info(
                    "Job succeeded. Run with --mode download to save images."
                )
            break

        if not getattr(args, "poll_until_done", False):
            logger.info(
                "Job not done yet (state=%s). Re-run with --mode poll to check again, "
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
    api_key = args.api_key or _require_env("GEMINI_API_KEY")
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
    row_index_map: Dict[str, Dict[str, str]] = run_data.get("row_index_map", {})
    input_fieldnames: List[str] = run_data.get("input_fieldnames", [])
    fields = output_fieldnames(input_fieldnames)

    text_log = Path(run_data.get("text_log", str(meta_path.parent / "generation.log")))
    logger = setup_logger(text_log)

    logger.info("=" * 80)
    logger.info("Gemini Batch Image Generation — DOWNLOAD")
    logger.info("Batch job     : %s", job_name)
    logger.info("Model         : %s  (%s)", model_name, run_data["repo_id"])
    logger.info("Output folder : %s", output_folder)

    # Optionally wait for completion
    if getattr(args, "poll_until_done", False):
        poll_interval = getattr(args, "poll_interval", 60)
        logger.info("--poll_until_done set; waiting for job to finish…")
        while True:
            batch_job = client.batches.get(name=job_name)
            state = getattr(batch_job.state, "name", str(batch_job.state))
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

    state = getattr(batch_job.state, "name", str(batch_job.state))
    logger.info("Job state: %s", state)

    if state != "JOB_STATE_SUCCEEDED":
        logger.error(
            "Job is not in JOB_STATE_SUCCEEDED (current: %s). "
            "Cannot download results. Add --poll_until_done to wait.",
            state,
        )
        run_data.setdefault("batch_state_history", []).append(
            {"timestamp": utc_now_iso(), "state": state}
        )
        atomic_write_json(meta_path, run_data)
        sys.exit(1)

    # Download result JSONL from Gemini File API
    result_file_name = batch_job.dest.file_name
    logger.info("Downloading result file: %s", result_file_name)
    file_content_bytes: bytes = client.files.download(file=result_file_name)
    file_content = file_content_bytes.decode("utf-8")

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
        except json.JSONDecodeError as e:
            logger.warning("Could not parse result line: %s", e)
            errors += 1
            continue

        key = parsed.get("key", "")
        row = row_index_map.get(key)

        if row is None:
            logger.warning("No matching CSV row for key '%s'. Skipping.", key)
            errors += 1
            continue

        image_path = (row.get("image_path") or "").strip()
        output_image_path = build_output_image_path(
            image_path, output_folder, model_name, image_extension
        )

        # Check for per-request error
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

        # Extract image from response
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
                    output_image_path.write_bytes(image_bytes)
                    logger.info(
                        "[%d] saved: %s  (mime=%s, %d bytes)",
                        generated + 1, output_image_path, mime_type, len(image_bytes),
                    )
                    image_saved = True
                    break
            if image_saved:
                break

        if image_saved:
            out_row = dict(row)
            out_row.update(
                {"generated_image_path": str(output_image_path), "status": "generated"}
            )
            append_output_row(output_csv, fields, out_row)
            generated += 1
        else:
            logger.warning(
                "No image data found in response for key '%s' (image_path=%s). "
                "Text parts: %s",
                key, image_path,
                [p.get("text", "") for c in candidates
                 for p in c.get("content", {}).get("parts", []) if p.get("text")],
            )
            out_row = dict(row)
            out_row.update({"generated_image_path": "", "status": "error"})
            append_output_row(output_csv, fields, out_row)
            errors += 1

    elapsed = round(time.perf_counter() - download_start, 2)

    # Update metadata with download results
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
    logger.info("Generated images : %d", generated)
    logger.info("Errors           : %d", errors)
    logger.info("Output CSV       : %s", output_csv.resolve())
    logger.info("Metadata JSON    : %s", meta_path.resolve())
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_env(var: str) -> str:
    import os
    val = os.environ.get(var, "")
    if not val:
        print(f"ERROR: --api_key not provided and environment variable {var} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate images via the Gemini Batch API from a CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- mode ----
    parser.add_argument(
        "--mode",
        required=True,
        choices=["submit", "poll", "download"],
        help=(
            "submit   : Convert CSV → JSONL, upload, start batch job, save metadata.\n"
            "poll     : Check the status of a previously submitted batch job.\n"
            "download : Download completed batch results and decode images to disk."
        ),
    )

    # ---- API key ----
    parser.add_argument(
        "--api_key",
        default=None,
        help="Google AI Studio API key. Falls back to GEMINI_API_KEY env var.",
    )

    # ---- submit-only args ----
    parser.add_argument("--input", default=None, help="[submit] Input CSV path.")
    parser.add_argument(
        "--model",
        default=None,
        choices=sorted(MODEL_REPOS),
        help="[submit] Model to use.",
    )
    parser.add_argument(
        "--output_folder",
        default="generated_outputs",
        help="[submit] Base folder for run output. Actual run folder: <output_folder>/<model>_<dataset>.",
    )
    parser.add_argument(
        "--output_folder_suffix",
        default="",
        help="[submit] Optional tag appended to the run folder name.",
    )
    parser.add_argument(
        "--prompt_column",
        default="description",
        help="[submit] CSV column used as the prompt text.",
    )
    parser.add_argument(
        "--image_extension",
        default=".png",
        help="[submit/download] File extension for saved images (.png or .jpg).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="[submit] Resubmit even rows that already have a generated image in the output CSV.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="[submit] Override the default output CSV path.",
    )

    # ---- poll/download args ----
    parser.add_argument(
        "--metadata_file",
        default=None,
        help=(
            "[poll/download] Path to the generation_metadata.json saved by --mode submit.\n"
            "[submit] Override default metadata file path."
        ),
    )
    parser.add_argument(
        "--poll_until_done",
        action="store_true",
        help="[poll/download] Keep polling until the job reaches a terminal state.",
    )
    parser.add_argument(
        "--poll_interval",
        type=int,
        default=60,
        help="[poll/download] Seconds between status checks when --poll_until_done is set.",
    )

    # ---- logging ----
    parser.add_argument(
        "--text_log",
        default=None,
        help="Override the log file path (default: <output_folder>/generation.log).",
    )

    args = parser.parse_args()

    # Minimal logger before we know the output folder
    _tmp_logger = logging.getLogger("gemini_batch_pre")
    _tmp_logger.setLevel(logging.INFO)
    if not _tmp_logger.handlers:
        _tmp_logger.addHandler(logging.StreamHandler(sys.stdout))

    if args.mode == "submit":
        if not args.input:
            parser.error("--mode submit requires --input")
        if not args.model:
            parser.error("--mode submit requires --model")
        mode_submit(args, _tmp_logger)

    elif args.mode == "poll":
        if not args.metadata_file:
            parser.error("--mode poll requires --metadata_file")
        mode_poll(args, _tmp_logger)

    elif args.mode == "download":
        if not args.metadata_file:
            parser.error("--mode download requires --metadata_file")
        mode_download(args, _tmp_logger)


if __name__ == "__main__":
    main()