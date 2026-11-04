#!/usr/bin/env python3
"""
openrouter_image_batch.py
==========================
Generate or edit food images from a CSV using the OpenRouter Image API.

Supports three modes:
  generate   – text-to-image from a prompt column
  edit       – image-to-image using a source image + instruction column
  auto       – pick "edit" if input_root + prompt_column look like editing rows,
               otherwise fall back to "generate"

Supported models (via OpenRouter):
  seedream-4.5   -> bytedance-seed/seedream-4.5
  flux-pro       -> black-forest-labs/flux.2-pro
  flux-max       -> black-forest-labs/flux.2-max

Output folder naming follows the same convention as food_image_batch_v3.py:
  <output_folder>/<model>_<dataset>[_suffix]

Output CSV adds two columns to the original: generated_image_path, status

Resume / reproducibility:
  - Rows already marked successful in the output CSV are skipped (--skip_existing)
  - Metadata JSON is written after every row and on completion
  - generation_signature in metadata makes runs reproducible

Before running:
  pip install requests pillow
  export OPENROUTER_API_KEY="sk-or-..."

Examples
--------
1) Text-to-image with seedream-4.5:
    python openrouter_image_batch.py --mode generate \\
        --input input.csv --model seedream-4.5

2) Image editing with flux-pro:
    python openrouter_image_batch.py --mode edit \\
        --input input.csv --input_root /data/images \\
        --model flux-pro --prompt_column manipulation_instruction

3) Auto-mode (detect from CSV):
    python openrouter_image_batch.py --mode auto \\
        --input input.csv --input_root /data/images --model flux-max

4) Skip already completed rows:
    python openrouter_image_batch.py --mode generate \\
        --input input.csv --model seedream-4.5 --skip_existing
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
import threading
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is required. Install it with: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_VERSION = "2026-06-24.v1-openrouter"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
IMAGE_ENDPOINT = f"{OPENROUTER_BASE_URL}/images"
CHAT_ENDPOINT   = f"{OPENROUTER_BASE_URL}/chat/completions"

MODEL_IDS: Dict[str, str] = {
    "seedream-4.5": "bytedance-seed/seedream-4.5",
    "flux-pro":     "black-forest-labs/flux.2-pro",
    "flux-max":     "black-forest-labs/flux.2-max",
}

# seedream-4.5 uses chat/completions with modalities=["image"]
# flux-pro / flux-max use /api/v1/images with input_references for editing
CHAT_MODELS: Set[str] = {"seedream-4.5"}
IMAGE_API_MODELS: Set[str] = {"flux-pro", "flux-max"}

IMAGE_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
ADDED_OUTPUT_COLUMNS = ["generated_image_path", "status"]
SUCCESS_STATUSES = {"generated", "reused"}

BASE_GENERATE_PROMPT_TEMPLATE = (
    "Generate a high-quality, realistic food photograph based only on this food description. "
    "Match the food appearance, ingredients, serving vessel, scene composition, lighting, "
    "camera angle, and image quality details as closely as possible.\n\n{prompt_text}"
)

SYSTEM_PROMPT = (
    "You are a photorealistic image editor. Apply only the described addition to the image. "
    "Preserve the original dish, background, lighting, and composition exactly. "
    "Do not redraw, reframe, or alter anything not described."
)
PRESERVATION_ANCHOR_TEMPLATE = (
    "Preserve the exact scene, dish, background, lighting, and composition unchanged. "
    "Only add: "
)
BASE_EDIT_PROMPT_TEMPLATE = "{preservation_anchor} {prompt_text}{edit_suffix}"
DEFAULT_EDIT_SUFFIX = (
    ", nearly indistinguishable from the surrounding food texture, not catching strong specular highlights, "
    "lying naturally in the scene without obvious artificial edges, visually subordinate to the dish"
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_name(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "item"


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("openrouter_image_batch")
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


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "script": Path(__file__).name,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "script_version": SCRIPT_VERSION,
        "pillow_available": HAS_PIL,
    }
    try:
        import requests as req
        info["requests_version"] = getattr(req, "__version__", "unknown")
    except ImportError:
        pass
    return info


def require_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        print(
            "ERROR: OPENROUTER_API_KEY is not set.\n"
            "  export OPENROUTER_API_KEY='sk-or-...'",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_rows(csv_path: Path, limit: Optional[int]) -> Tuple[List[Dict[str, str]], List[str]]:
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("ERROR: CSV has no header row.", file=sys.stderr)
            sys.exit(1)
        fieldnames = list(reader.fieldnames)
        rows = []
        for row in reader:
            rows.append(dict(row))
            if limit is not None and len(rows) >= limit:
                break
    return rows, fieldnames


def infer_dataset_name(rows: List[Dict[str, str]], input_csv: Path) -> str:
    names = sorted(
        {(r.get("dataset_name") or "").strip() for r in rows
         if (r.get("dataset_name") or "").strip()}
    )
    if len(names) == 1:
        return sanitize_name(names[0])
    if len(names) > 1:
        preview = "_".join(sanitize_name(n) for n in names[:3])
        return f"{preview}_mixed" if len(names) > 3 else preview
    return sanitize_name(input_csv.stem)


def build_run_output_folder(
    base: Path, model_key: str, dataset_name: str, suffix: str
) -> Path:
    parts = [sanitize_name(model_key), sanitize_name(dataset_name)]
    s = sanitize_name(suffix) if suffix.strip() else ""
    if s:
        parts.append(s)
    return base / "_".join(parts)


def choose_prompt_column(fieldnames: List[str], requested: Optional[str], mode: str) -> str:
    if requested:
        if requested not in fieldnames:
            print(
                f"ERROR: Prompt column '{requested}' not in CSV. Available: {fieldnames}",
                file=sys.stderr,
            )
            sys.exit(1)
        return requested
    # Auto-detect: prefer manipulation_instruction for edit, description for generate
    if mode == "edit":
        for col in ["manipulation_instruction", "instruction", "edit_instruction", "human_description", "description"]:
            if col in fieldnames:
                return col
    else:
        for col in ["human_description", "description", "prompt", "caption"]:
            if col in fieldnames:
                return col
    print(
        f"ERROR: Could not auto-detect a prompt column for mode '{mode}'. "
        f"Available: {fieldnames}. Use --prompt_column.",
        file=sys.stderr,
    )
    sys.exit(1)


def output_fieldnames(input_fieldnames: List[str]) -> List[str]:
    fields = list(input_fieldnames)
    for col in ADDED_OUTPUT_COLUMNS:
        if col not in fields:
            fields.append(col)
    return fields


def read_completed_image_paths(output_csv: Path) -> Set[str]:
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
# Path / image helpers
# ---------------------------------------------------------------------------

def safe_relative_path(raw: str) -> Path:
    normalized = raw.strip().replace("\\", "/")
    pure = PurePosixPath(normalized)
    parts = [
        p for p in pure.parts
        if p not in ("", ".", "/", "\\") and p != ".." and not p.endswith(":")
    ]
    if not parts:
        raise ValueError(f"Cannot build safe relative path from: {raw!r}")
    return Path(*parts)


def build_output_path(
    row: Dict[str, str], output_folder: Path, model_key: str,
    mode: str, output_extension: str
) -> Path:
    image_path = (row.get("image_path") or "").strip()
    rel = safe_relative_path(image_path)
    ext = output_extension if output_extension.startswith(".") else f".{output_extension}"
    issue = sanitize_name(row.get("selected_issue") or row.get("class_id") or "gen")
    instruction = (row.get("manipulation_instruction") or row.get("description") or "").strip()
    digest = hashlib.sha1(
        f"{image_path}-{issue}-{instruction[:80]}-{model_key}-{mode}".encode()
    ).hexdigest()[:8]
    filename = f"{rel.stem}_{sanitize_name(model_key)}_{issue}_{digest}{ext}"
    return output_folder / rel.parent / filename


def image_to_base64_data_url(path: Path) -> str:
    data = path.read_bytes()
    suffix = path.suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".webp": "image/webp"}.get(suffix, "image/png")
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def extract_b64_from_data_url(data_url: str) -> bytes:
    if "," in data_url:
        return base64.b64decode(data_url.split(",", 1)[1])
    return base64.b64decode(data_url)


def generation_signature(config: Dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_generate_prompt(description: str) -> str:
    return BASE_GENERATE_PROMPT_TEMPLATE.format(prompt_text=description.strip())


def build_edit_prompt(instruction: str, edit_suffix: str = "") -> str:
    """
    Builds the user-facing edit prompt using:
      BASE_EDIT_PROMPT_TEMPLATE = "{preservation_anchor} {prompt_text}{edit_suffix}"
    SYSTEM_PROMPT is kept separate and injected into the API call itself.
    """
    return BASE_EDIT_PROMPT_TEMPLATE.format(
        preservation_anchor=PRESERVATION_ANCHOR_TEMPLATE.rstrip(),
        prompt_text=instruction.strip(),
        edit_suffix=edit_suffix if edit_suffix.strip() else "",
    )


# ---------------------------------------------------------------------------
# OpenRouter API calls
# ---------------------------------------------------------------------------

def _headers(api_key: str, app_name: str = "openrouter_image_batch") -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": app_name,
    }


def call_image_api(
    *,
    api_key: str,
    model_id: str,
    prompt: str,
    aspect_ratio: str,
    quality: Optional[str],
    n: int,
    input_image_path: Optional[Path],
    timeout: int,
    logger: logging.Logger,
) -> List[bytes]:
    """
    Calls /api/v1/images — used by flux-pro and flux-max.
    Supports both text-to-image and image-to-image (via input_references).
    Returns a list of raw image bytes, one per generated image.
    """
    # /api/v1/images has no separate system role; prepend SYSTEM_PROMPT to the prompt
    # when editing so the model receives the preservation instruction inline.
    effective_prompt = (
        f"{SYSTEM_PROMPT}\n\n{prompt}"
        if input_image_path is not None and SYSTEM_PROMPT.strip()
        else prompt
    )
    payload: Dict[str, Any] = {
        "model": model_id,
        "prompt": effective_prompt,
        "n": n,
    }
    if aspect_ratio:
        payload["aspect_ratio"] = aspect_ratio
    if quality:
        payload["quality"] = quality

    if input_image_path is not None:
        data_url = image_to_base64_data_url(input_image_path)
        payload["input_references"] = [{"type": "image_url", "image_url": {"url": data_url}}]

    response = requests.post(
        IMAGE_ENDPOINT,
        headers=_headers(api_key),
        json=payload,
        timeout=timeout,
    )

    if not response.ok:
        raise RuntimeError(
            f"OpenRouter /images API error {response.status_code}: {response.text[:500]}"
        )

    data = response.json()
    images_out: List[bytes] = []

    for item in data.get("data") or []:
        url = item.get("url")
        b64 = item.get("b64_json")
        if b64:
            images_out.append(base64.b64decode(b64))
        elif url:
            if url.startswith("data:"):
                images_out.append(extract_b64_from_data_url(url))
            else:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                images_out.append(r.content)

    if not images_out:
        raise RuntimeError(f"No image data in response: {json.dumps(data)[:500]}")
    return images_out


def call_chat_api(
    *,
    api_key: str,
    model_id: str,
    prompt: str,
    aspect_ratio: str,
    input_image_path: Optional[Path],
    timeout: int,
    logger: logging.Logger,
) -> List[bytes]:
    """
    Calls /api/v1/chat/completions with modalities=["image"] — used by seedream-4.5.
    For editing, the source image is sent as an image_url message part.
    Returns a list of raw image bytes.
    """
    content: List[Dict[str, Any]] = []

    if input_image_path is not None:
        data_url = image_to_base64_data_url(input_image_path)
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    content.append({"type": "text", "text": prompt})

    messages: List[Dict[str, Any]] = []
    if input_image_path is not None and SYSTEM_PROMPT.strip():
        # For edit mode: inject system prompt so the model knows to preserve the scene
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": content})

    payload: Dict[str, Any] = {
        "model": model_id,
        "modalities": ["image"],
        "messages": messages,
    }
    if aspect_ratio:
        payload.setdefault("image_config", {})["aspect_ratio"] = aspect_ratio

    response = requests.post(
        CHAT_ENDPOINT,
        headers=_headers(api_key),
        json=payload,
        timeout=timeout,
    )

    if not response.ok:
        raise RuntimeError(
            f"OpenRouter chat/completions API error {response.status_code}: {response.text[:500]}"
        )

    data = response.json()
    images_out: List[bytes] = []

    # Response format: choices[0].message.images[] or choices[0].message.content[].image_url
    message = (data.get("choices") or [{}])[0].get("message") or {}

    # Variant A: message.images[]
    for img in message.get("images") or []:
        url = (img.get("image_url") or {}).get("url", "")
        if url.startswith("data:"):
            images_out.append(extract_b64_from_data_url(url))
        elif url:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            images_out.append(r.content)

    # Variant B: message.content[] with type="image_url"
    if not images_out:
        for part in message.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url.startswith("data:"):
                    images_out.append(extract_b64_from_data_url(url))
                elif url:
                    r = requests.get(url, timeout=60)
                    r.raise_for_status()
                    images_out.append(r.content)

    if not images_out:
        raise RuntimeError(
            f"No image data in chat completions response: {json.dumps(data)[:500]}"
        )
    return images_out


def generate_or_edit(
    *,
    api_key: str,
    model_key: str,
    model_id: str,
    prompt: str,
    aspect_ratio: str,
    quality: Optional[str],
    n: int,
    input_image_path: Optional[Path],
    timeout: int,
    logger: logging.Logger,
) -> List[bytes]:
    """Dispatch to the correct API based on model_key."""
    if model_key in IMAGE_API_MODELS:
        return call_image_api(
            api_key=api_key,
            model_id=model_id,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            quality=quality,
            n=n,
            input_image_path=input_image_path,
            timeout=timeout,
            logger=logger,
        )
    elif model_key in CHAT_MODELS:
        return call_chat_api(
            api_key=api_key,
            model_id=model_id,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            input_image_path=input_image_path,
            timeout=timeout,
            logger=logger,
        )
    else:
        raise ValueError(f"Unknown model_key: {model_key}")


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def call_with_retry(
    fn,
    *,
    max_retries: int,
    retry_delay: float,
    logger: logging.Logger,
    label: str = "",
) -> Any:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = retry_delay * (2 ** attempt)
                logger.warning(
                    "[%s] Attempt %d/%d failed: %s — retrying in %.1fs",
                    label, attempt + 1, max_retries + 1, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error("[%s] All %d attempts failed: %s", label, max_retries + 1, exc)
    raise last_exc  # type: ignore


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    api_key = require_api_key()

    input_csv = Path(args.input)
    base_output_folder = Path(args.output_folder)
    model_key: str = args.model
    model_id: str = MODEL_IDS[model_key]
    mode: str = args.mode

    rows, input_fieldnames = load_rows(input_csv, args.limit)
    if not rows:
        print("ERROR: No rows loaded from CSV.", file=sys.stderr)
        sys.exit(1)

    # Auto-detect mode
    if mode == "auto":
        has_instruction_col = any(
            c in input_fieldnames for c in
            ["manipulation_instruction", "instruction", "edit_instruction"]
        )
        has_input_root = bool(args.input_root)
        mode = "edit" if (has_instruction_col and has_input_root) else "generate"

    input_root = Path(args.input_root) if args.input_root else None
    if mode == "edit" and input_root is None:
        print("ERROR: --mode edit requires --input_root.", file=sys.stderr)
        sys.exit(1)

    dataset_name = infer_dataset_name(rows, input_csv)
    output_folder = (
        Path(args.run_output_folder) if args.run_output_folder
        else build_run_output_folder(base_output_folder, model_key, dataset_name, args.output_folder_suffix)
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    output_csv = Path(args.output_csv) if args.output_csv else output_folder / "generated_images.csv"
    json_log = Path(args.json_log) if args.json_log else output_folder / "generation_metadata.json"
    text_log = Path(args.text_log) if args.text_log else base_output_folder / "generation.log"

    logger = setup_logger(text_log)

    prompt_column = choose_prompt_column(input_fieldnames, args.prompt_column, mode)
    quality: Optional[str] = args.quality if args.quality else None

    # Resume
    skip_image_paths: Set[str] = set()
    if args.skip_existing:
        skip_image_paths = read_completed_image_paths(output_csv)

    sig_config = {
        "input_csv_sha256": sha256_file(input_csv),
        "model_key": model_key,
        "model_id": model_id,
        "mode": mode,
        "prompt_column": prompt_column,
        "aspect_ratio": args.aspect_ratio,
        "quality": quality,
        "n": args.n,
        "output_extension": args.output_extension,
        "generate_prompt_template": BASE_GENERATE_PROMPT_TEMPLATE,
        "edit_prompt_template": BASE_EDIT_PROMPT_TEMPLATE,
        "edit_suffix": args.edit_suffix if mode == "edit" else "",
    }
    sig = generation_signature(sig_config)

    fields = output_fieldnames(input_fieldnames)

    logger.info("=" * 80)
    logger.info("OpenRouter Batch Image Generation/Editing")
    logger.info("Mode           : %s", mode)
    logger.info("Model          : %s  (%s)", model_key, model_id)
    logger.info("Input CSV      : %s", input_csv)
    logger.info("Dataset        : %s", dataset_name)
    logger.info("Output folder  : %s", output_folder)
    logger.info("Output CSV     : %s", output_csv)
    logger.info("Metadata JSON  : %s", json_log)
    logger.info("Prompt column  : %s", prompt_column)
    if mode == "edit":
        logger.info("Input root     : %s", input_root)
    logger.info("Aspect ratio   : %s", args.aspect_ratio)
    logger.info("Quality        : %s", quality or "<omitted>")
    logger.info("N per row      : %d", args.n)
    logger.info("Total rows     : %d", len(rows))
    logger.info("Already done   : %d", len(skip_image_paths))
    logger.info("=" * 80)

    run_data: Dict[str, Any] = {
        "run_started_at": utc_now(),
        "run_finished_at": None,
        "total_time_seconds": None,
        "script_version": SCRIPT_VERSION,
        "generation_signature": sig,
        "task": f"openrouter_image_{mode}",
        "model_key": model_key,
        "model_id": model_id,
        "mode": mode,
        "input_csv": str(input_csv.resolve()),
        "input_root": str(input_root.resolve()) if input_root else None,
        "output_folder": str(output_folder.resolve()),
        "output_csv": str(output_csv.resolve()),
        "json_log": str(json_log.resolve()),
        "text_log": str(text_log.resolve()),
        "dataset_name": dataset_name,
        "output_folder_suffix": args.output_folder_suffix,
        "settings": {
            "prompt_column": prompt_column,
            "aspect_ratio": args.aspect_ratio,
            "quality": quality,
            "n": args.n,
            "output_extension": args.output_extension,
            "workers": args.workers,
            "max_retries": args.max_retries,
            "retry_delay": args.retry_delay,
            "timeout": args.timeout,
            "row_delay": args.row_delay,
            "overwrite": args.overwrite,
            "skip_existing": args.skip_existing,
            "limit": args.limit,
            "edit_suffix": args.edit_suffix if mode == "edit" else "",
        },
        "reproducibility": {
            "generation_signature": sig,
            "generate_prompt_template": BASE_GENERATE_PROMPT_TEMPLATE,
            "system_prompt": SYSTEM_PROMPT,
            "preservation_anchor_template": PRESERVATION_ANCHOR_TEMPLATE,
            "edit_prompt_template": BASE_EDIT_PROMPT_TEMPLATE,
            "default_edit_suffix": DEFAULT_EDIT_SUFFIX,
            "edit_suffix_used": args.edit_suffix if mode == "edit" else "",
            "model_key": model_key,
            "model_id": model_id,
            "prompt_column": prompt_column,
            "dataset_name": dataset_name,
        },
        "environment": environment_info(),
        "progress": {
            "total_input_rows": len(rows),
            "skipped_already_completed": len(skip_image_paths),
            "processed_this_run": 0,
            "generated_this_run": 0,
            "reused_this_run": 0,
            "error_this_run": 0,
        },
        "items": [],
    }
    atomic_write_json(json_log, run_data)

    run_start = time.perf_counter()
    n_rows = len(rows)

    # Shared mutable state — all writes go through this lock
    write_lock = threading.Lock()
    counters = {"generated": 0, "reused": 0, "errors": 0}

    # -----------------------------------------------------------------------
    # Per-row worker function (runs in a thread when workers > 1)
    # -----------------------------------------------------------------------
    def process_row(row_idx: int, row: Dict[str, str], worker_stagger: float) -> None:
        """
        Validates, optionally reuses, calls the API, saves the image, and
        writes one line to the output CSV + updates shared run_data.
        All shared-state mutations are serialised via write_lock.
        """
        if worker_stagger > 0:
            time.sleep(worker_stagger)

        image_path = (row.get("image_path") or "").strip()
        prompt_text = (row.get(prompt_column) or "").strip()
        label = f"{row_idx}/{n_rows}"

        def record_error(reason: str, item_log: Optional[Dict[str, Any]] = None) -> None:
            out_row = dict(row)
            out_row.update({"generated_image_path": "", "status": "error"})
            with write_lock:
                append_output_row(output_csv, fields, out_row)
                counters["errors"] += 1
                run_data["progress"]["error_this_run"] += 1
                run_data["progress"]["processed_this_run"] += 1
                if item_log:
                    run_data["items"].append(item_log)
                atomic_write_json(json_log, run_data)
            logger.warning("[%s] skipped/error: %s", label, reason)

        # --- validation ---
        if not image_path:
            record_error("empty image_path")
            return
        if not prompt_text or prompt_text.upper() == "ERROR":
            record_error(f"empty/error prompt column '{prompt_column}'")
            return
        if image_path in skip_image_paths:
            with write_lock:
                counters["reused"] += 1
            return

        # --- build output path ---
        try:
            abs_output_path = build_output_path(
                row, output_folder, model_key, mode, args.output_extension
            )
        except ValueError as exc:
            record_error(f"bad image_path: {exc}")
            return

        # --- reuse existing file on disk ---
        if abs_output_path.exists() and not args.overwrite:
            out_row = dict(row)
            out_row.update({"generated_image_path": str(abs_output_path), "status": "reused"})
            with write_lock:
                append_output_row(output_csv, fields, out_row)
                counters["reused"] += 1
                run_data["progress"]["reused_this_run"] += 1
                run_data["progress"]["processed_this_run"] += 1
            logger.info("[%s] reused: %s", label, abs_output_path)
            return

        # --- build prompt ---
        prompt = (
            build_edit_prompt(prompt_text, args.edit_suffix)
            if mode == "edit"
            else build_generate_prompt(prompt_text)
        )

        # --- locate source image for edit mode ---
        source_path: Optional[Path] = None
        if mode == "edit":
            try:
                rel = safe_relative_path(image_path)
            except ValueError as exc:
                record_error(f"bad image_path: {exc}")
                return
            assert input_root is not None
            source_path = input_root / rel
            if not source_path.exists():
                record_error(
                    f"source image not found: {source_path}",
                    item_log={
                        "row_index": row_idx, "image_path": image_path,
                        "status": "error", "error": f"source_not_found: {source_path}",
                        "started_at": utc_now(), "finished_at": utc_now(), "time_seconds": 0,
                    },
                )
                return

        # --- call API with retry ---
        item_start = time.perf_counter()
        item_start_at = utc_now()

        try:
            image_bytes_list = call_with_retry(
                lambda: generate_or_edit(
                    api_key=api_key,
                    model_key=model_key,
                    model_id=model_id,
                    prompt=prompt,
                    aspect_ratio=args.aspect_ratio,
                    quality=quality,
                    n=args.n,
                    input_image_path=source_path,
                    timeout=args.timeout,
                    logger=logger,
                ),
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
                logger=logger,
                label=label,
            )
        except Exception as exc:
            elapsed = round(time.perf_counter() - item_start, 3)
            record_error(
                f"API failed after retries: {exc}",
                item_log={
                    "row_index": row_idx, "image_path": image_path,
                    "status": "error", "error": str(exc),
                    "started_at": item_start_at, "finished_at": utc_now(),
                    "time_seconds": elapsed,
                },
            )
            return

        # --- save image(s) to disk ---
        saved_path: Optional[Path] = None
        for img_idx, img_bytes in enumerate(image_bytes_list):
            out_path = (
                abs_output_path if img_idx == 0
                else abs_output_path.with_name(
                    f"{abs_output_path.stem}_{img_idx + 1}{abs_output_path.suffix}"
                )
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(out_path, img_bytes)
            if img_idx == 0:
                saved_path = out_path

        elapsed = round(time.perf_counter() - item_start, 3)
        logger.info("[%s] generated → %s (%.1fs)", label, saved_path, elapsed)

        out_row = dict(row)
        out_row.update({"generated_image_path": str(saved_path), "status": "generated"})
        item_log = {
            "row_index": row_idx,
            "image_path": image_path,
            "generated_image_path": str(saved_path),
            "status": "generated",
            "mode": mode,
            "model_key": model_key,
            "model_id": model_id,
            "prompt_column": prompt_column,
            "started_at": item_start_at,
            "finished_at": utc_now(),
            "time_seconds": elapsed,
            "error": None,
        }
        with write_lock:
            append_output_row(output_csv, fields, out_row)
            counters["generated"] += 1
            run_data["progress"]["generated_this_run"] += 1
            run_data["progress"]["processed_this_run"] += 1
            run_data["items"].append(item_log)
            atomic_write_json(json_log, run_data)

    # -----------------------------------------------------------------------
    # Dispatch: serial (workers=1) or parallel (workers>1)
    # -----------------------------------------------------------------------
    workers = args.workers
    logger.info("Parallelism    : %d worker%s", workers, "s" if workers > 1 else "")

    if workers <= 1:
        for row_idx, row in enumerate(rows, start=1):
            process_row(row_idx, row, worker_stagger=0.0)
            if args.row_delay > 0 and row_idx < n_rows:
                time.sleep(args.row_delay)
    else:
        # Stagger thread starts by row_delay seconds each to spread the
        # initial burst of requests (e.g. --workers 4 --row_delay 1 staggers
        # threads 0s, 1s, 2s, 3s apart, then they run fully in parallel).
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    process_row,
                    row_idx,
                    row,
                    (row_idx - 1) * args.row_delay,  # per-worker stagger
                ): row_idx
                for row_idx, row in enumerate(rows, start=1)
            }
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    # Unhandled exception inside process_row (shouldn't happen,
                    # but log it just in case)
                    row_idx = futures[future]
                    logger.error("Unhandled exception in worker for row %d: %s", row_idx, exc)

    total_elapsed = round(time.perf_counter() - run_start, 2)
    run_data["run_finished_at"] = utc_now()
    run_data["total_time_seconds"] = total_elapsed
    run_data["progress"]["reused_this_run"] = counters["reused"]
    atomic_write_json(json_log, run_data)

    logger.info("=" * 80)
    logger.info("Run complete in %.1fs", total_elapsed)
    logger.info("Generated  : %d", counters["generated"])
    logger.info("Reused     : %d", counters["reused"])
    logger.info("Errors     : %d", counters["errors"])
    logger.info("Workers    : %d", workers)
    logger.info("Output CSV : %s", output_csv.resolve())
    logger.info("Metadata   : %s", json_log.resolve())
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch image generation/editing via OpenRouter (seedream-4.5, flux-pro, flux-max).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core
    parser.add_argument("--input", required=True, help="Input CSV file.")
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_IDS),
        help="Model alias: seedream-4.5 | flux-pro | flux-max",
    )
    parser.add_argument(
        "--mode",
        default="generate",
        choices=["generate", "edit", "auto"],
        help=(
            "generate : text-to-image from --prompt_column.\n"
            "edit     : image-to-image using --input_root + source image + --prompt_column.\n"
            "auto     : detect from CSV columns and --input_root presence."
        ),
    )

    # Edit-mode
    parser.add_argument(
        "--input_root", default=None,
        help="[edit] Root folder for relative image_path values in the CSV."
    )
    parser.add_argument(
        "--edit_suffix", default=DEFAULT_EDIT_SUFFIX,
        help=(
            "[edit] Text appended after the instruction prompt. "
            "Defaults to DEFAULT_EDIT_SUFFIX (natural food-scene blending cues). "
            "Pass an empty string '' to disable."
        ),
    )

    # Prompt
    parser.add_argument(
        "--prompt_column", default=None,
        help="CSV column to use as prompt. Auto-detected if omitted."
    )

    # Image settings
    parser.add_argument(
        "--aspect_ratio", default="1:1",
        help="Aspect ratio: 1:1 | 16:9 | 9:16 | 4:3 | 3:2 | etc."
    )
    parser.add_argument(
        "--quality", default="",
        help="Quality hint: low | medium | high. Leave empty to omit."
    )
    parser.add_argument("--n", type=int, default=1, help="Number of images per row.")
    parser.add_argument(
        "--output_extension", default=".png",
        help="File extension for saved images: .png | .jpg | .webp"
    )

    # Output
    parser.add_argument("--output_folder", default="openrouter_outputs", help="Base output folder.")
    parser.add_argument("--output_folder_suffix", default="", help="Optional suffix added to the run folder name.")
    parser.add_argument("--run_output_folder", default=None, help="Override the auto-generated run output folder path.")
    parser.add_argument("--output_csv", default=None, help="Override the output CSV path.")
    parser.add_argument("--json_log", default=None, help="Override metadata JSON path.")
    parser.add_argument("--text_log", default=None, help="Override text log path.")

    # Resume
    parser.add_argument("--skip_existing", action="store_true", help="Skip rows already successfully processed in the output CSV.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if the output image already exists on disk.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N rows.")

    # Reliability & parallelism
    parser.add_argument("--workers", type=int, default=1,
        help=(
            "Number of parallel API calls. 1 = serial (default). "
            "4-8 is a good range for most OpenRouter models. "
            "Note: output CSV rows are written in completion order, not input order, when workers > 1."
        ),
    )
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries on API error.")
    parser.add_argument("--retry_delay", type=float, default=5.0,
        help=(
            "Base retry delay in seconds, doubled each attempt (exponential backoff). "
            "When --workers > 1, also used as a per-worker stagger on startup "
            "to spread the initial burst (worker N waits N * row_delay seconds before its first call)."
        ),
    )
    parser.add_argument("--timeout", type=int, default=120, help="HTTP request timeout in seconds.")
    parser.add_argument("--row_delay", type=float, default=0.0,
        help=(
            "Serial mode: sleep this many seconds between rows. "
            "Parallel mode: stagger thread starts by this many seconds "
            "(e.g. --workers 4 --row_delay 1 starts threads at 0s, 1s, 2s, 3s)."
        ),
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()