"""
qwen_image_edit_manipulation.py
================================
Edit/manipulate food images from a CSV using Qwen-Image-Edit-2511.

It has the following the CSV/run-management features:
  - Qwen-Image-Edit-2511 model name and optional repo override,
  - append-only output CSV,
  - resumable runs based on existing successful rows,
  - JSON metadata log with detailed per-item errors/timing/seed info,
  - text logging,
  - auto-created run output folder,
  - dtype selection,
  - optional CPU offload,
  - multi-GPU multiprocessing where the main process is the only CSV/JSON writer.

Model: https://huggingface.co/Qwen/Qwen-Image-Edit-2511
Pipeline class: diffusers.QwenImageEditPlusPipeline

Key Qwen-specific parameters (versus FLUX.2-klein):
  - true_cfg_scale  : classifier-free guidance scale applied inside the pipeline
                      (separate from guidance_scale; default 4.0)
  - negative_prompt : negative text prompt (default " " as recommended by Qwen)
  - num_inference_steps: 40 recommended (vs 4 for FLUX.2-klein)
  - height / width  : NOT supported by QwenImageEditPlusPipeline — output
                      resolution is determined by the input image.

Required CSV columns:
  image_path,dataset_name,candidate_issues,selected_issue,manipulation_instruction

Recommended/optional columns are preserved if present:
  class_id,is_human,description,gen_model

Example:
python qwen_image_edit_manipulation.py \
  --input edits.csv \
  --input_root /data/original_images \
  --output_folder /data/edited_runs \
  --model Qwen-Image-Edit-2511 \
  --num_gpus 4 \
  --seed 42

Backward-compatible aliases from the old edit script are also supported:
  --csv -> --input
  --image_root -> --input_root
  --out_dir -> --output_folder
  --steps -> --num_inference_steps
  --guidance -> --guidance_scale
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import multiprocessing as mp
import platform
import queue as queue_mod
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from PIL import Image

try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError as exc:
    raise ImportError("Could not import QwenImageEditPlusPipeline. Install/upgrade diffusers first.") from exc


MODEL_REPOS: Dict[str, str] = {
    "Qwen-Image-Edit-2511": "Qwen/Qwen-Image-Edit-2511",
}

BASE_REQUIRED_COLUMNS = [
    "image_path",
    "dataset_name",
    # "description",
    "manipulation_instruction", 
    # "class_id", 
    # "is_human", 
    # "gen_model",
]
ADDED_OUTPUT_COLUMNS = ["generated_image_path", "status"]
SUCCESS_STATUSES = {"generated", "reused"}

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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_name(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "item"


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def infer_dataset_name(rows: List[Dict[str, str]], input_csv: Path) -> str:
    dataset_names = sorted({(row.get("dataset_name") or "").strip() for row in rows if (row.get("dataset_name") or "").strip()})
    if len(dataset_names) == 1:
        return sanitize_name(dataset_names[0])
    if len(dataset_names) > 1:
        preview = "_".join(sanitize_name(name) for name in dataset_names[:3])
        if len(dataset_names) > 3:
            return f"{preview}_mixed_{len(dataset_names)}_datasets"
        return preview
    return sanitize_name(input_csv.stem)


def build_run_output_folder(base_output_folder: Path, model_name: str, dataset_name: str, output_folder_suffix: str) -> Path:
    parts = [sanitize_name(model_name), sanitize_name(dataset_name)]
    suffix = sanitize_name(output_folder_suffix) if output_folder_suffix.strip() else ""
    if suffix:
        parts.append(suffix)
    return base_output_folder / "_".join(parts)


def resolve_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_str.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_str}'. Choose from: {sorted(mapping)}")
    return mapping[key]


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("qwen_image_edit_csv")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def worker_logger(worker_id: int) -> logging.Logger:
    logger = logging.getLogger(f"qwen_edit_worker_{worker_id}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(f"%(asctime)s [worker {worker_id}] [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def load_input_rows(input_csv: Path, prompt_column: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        required_columns = list(BASE_REQUIRED_COLUMNS)
        if prompt_column not in required_columns:
            required_columns.append(prompt_column)
        missing = [c for c in required_columns if c not in fieldnames]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {missing}. Found: {fieldnames}")
        return [dict(r) for r in reader], fieldnames


def output_fieldnames(input_fieldnames: Iterable[str]) -> List[str]:
    fields = list(input_fieldnames)
    for col in ADDED_OUTPUT_COLUMNS:
        if col not in fields:
            fields.append(col)
    return fields


def completed_key(row: Dict[str, str], model_name: str) -> Tuple[str, str, str, str]:
    return (
        (row.get("image_path") or "").strip(),
        (row.get("selected_issue") or "").strip(),
        (row.get("manipulation_instruction") or "").strip(),
        model_name,
    )


def read_completed_keys(output_csv: Path, model_name: str) -> Set[Tuple[str, str, str, str]]:
    completed: Set[Tuple[str, str, str, str]] = set()
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return completed
    with output_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = (row.get("status") or "").strip().lower()
            generated_image_path = (row.get("generated_image_path") or "").strip()
            old_generated_model = (row.get("generated_model") or "").strip()
            model_matches = (not old_generated_model) or (old_generated_model == model_name)
            if status in SUCCESS_STATUSES and generated_image_path and model_matches:
                completed.add(completed_key(row, model_name))
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


def build_edit_prompt(prompt_text: str, edit_suffix: str, preservation_anchor: str = PRESERVATION_ANCHOR_TEMPLATE) -> str:
    suffix = edit_suffix or ""
    return BASE_EDIT_PROMPT_TEMPLATE.format(
        preservation_anchor=preservation_anchor,
        prompt_text=prompt_text.strip(),
        edit_suffix=suffix,
    )


def build_output_image_path(row_index: int, row: Dict[str, str], output_folder: Path, model_name: str, image_extension: str) -> Path:
    image_path = (row.get("image_path") or "").strip()
    rel = Path(image_path)
    suffix = image_extension if image_extension.startswith(".") else f".{image_extension}"
    issue = sanitize_name(row.get("selected_issue", "issue"))
    instruction = (row.get("manipulation_instruction") or "").strip()
    digest = hashlib.sha1(f"{row_index}-{image_path}-{issue}-{instruction}-{model_name}".encode("utf-8")).hexdigest()[:8]
    filename = f"{rel.stem}_{sanitize_name(model_name)}_{issue}_{digest}{suffix}"
    return output_folder / rel.parent / filename


def load_pipeline(repo_id: str, dtype: torch.dtype, device: str, load_strategy: str, cpu_offload: bool, logger: logging.Logger) -> QwenImageEditPlusPipeline:
    """Load the Qwen-Image-Edit-2511 pipeline.

    The QwenImageEditPlusPipeline does not expose separate transformer /
    text_encoder constructors, so only a single from_pretrained path is used.
    The ``load_strategy`` argument is accepted for CLI compatibility but
    ``"components"`` silently falls back to ``"full"``.
    """
    logger.info("Loading Qwen-Image-Edit-2511 pipeline from %s on %s", repo_id, device)

    pipe = QwenImageEditPlusPipeline.from_pretrained(repo_id, torch_dtype=dtype)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpus"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append({"index": i, "name": torch.cuda.get_device_name(i), "total_memory_gb": round(props.total_memory / (1024**3), 2)})
    return info


def make_error_result(row_index: int, row: Dict[str, str], status_error: str, output_image_path: Optional[Path], args_dict: Dict[str, Any], item_started: float, item_started_at: str) -> Dict[str, Any]:
    image_path = (row.get("image_path") or "").strip()
    return {
        "row_index": row_index,
        "row": row,
        "status": "error",
        "generated_image_path": "",
        "generated_model": args_dict["model"],
        "hf_model_repo_id": args_dict["repo_id"],
        "item_log": {
            "row_index": row_index,
            "source_image_path": image_path,
            "source_image_abs_path": str(Path(args_dict["input_root"]) / image_path) if image_path else "",
            "input_gen_model_value": row.get("gen_model", ""),
            "selected_issue": row.get("selected_issue", ""),
            "generated_model": args_dict["model"],
            "hf_model_repo_id": args_dict["repo_id"],
            "generated_image_path": str(output_image_path) if output_image_path else "",
            "prompt_column": args_dict["prompt_column"],
            "seed": args_dict["seed"],
            "seed_mode": args_dict["seed_mode"],
            "started_at": item_started_at,
            "finished_at": utc_now_iso(),
            "time_seconds": round(time.perf_counter() - item_started, 4),
            "status": "error",
            "error": status_error,
        },
    }


def worker(worker_id: int, device: str, indexed_rows: List[Tuple[int, Dict[str, str]]], args_dict: Dict[str, Any], result_queue: "mp.Queue[Dict[str, Any]]") -> None:
    logger = worker_logger(worker_id)
    dtype = resolve_dtype(args_dict["dtype"])
    try:
        pipe = load_pipeline(args_dict["repo_id"], dtype, device, args_dict["load_strategy"], args_dict["cpu_offload"], logger)
    except Exception as exc:
        for row_index, row in indexed_rows:
            now_perf = time.perf_counter()
            now_iso = utc_now_iso()
            result_queue.put(make_error_result(row_index, row, f"pipeline_load_error: {exc}", None, args_dict, now_perf, now_iso))
        return

    input_root = Path(args_dict["input_root"])
    output_folder = Path(args_dict["output_folder"])

    for row_index, row in indexed_rows:
        item_started = time.perf_counter()
        item_started_at = utc_now_iso()
        image_path = (row.get("image_path") or "").strip()
        prompt_column = args_dict["prompt_column"]
        prompt_text = (row.get(prompt_column) or "").strip()
        output_image_path = build_output_image_path(row_index, row, output_folder, args_dict["model"], args_dict["image_extension"])

        if not image_path:
            result_queue.put(make_error_result(row_index, row, "empty_image_path", output_image_path, args_dict, item_started, item_started_at))
            continue
        if not prompt_text or prompt_text.upper() == "ERROR":
            result_queue.put(make_error_result(row_index, row, f"empty_or_error_prompt_column: {prompt_column}", output_image_path, args_dict, item_started, item_started_at))
            continue

        source_abs_path = input_root / image_path
        if not source_abs_path.exists():
            result_queue.put(make_error_result(row_index, row, f"source_image_not_found: {source_abs_path}", output_image_path, args_dict, item_started, item_started_at))
            continue

        output_image_path.parent.mkdir(parents=True, exist_ok=True)

        if output_image_path.exists() and not args_dict["overwrite"]:
            result_queue.put({
                "row_index": row_index,
                "row": row,
                "status": "reused",
                "generated_image_path": str(output_image_path),
                "generated_model": args_dict["model"],
                "hf_model_repo_id": args_dict["repo_id"],
                "item_log": {
                    "row_index": row_index,
                    "source_image_path": image_path,
                    "source_image_abs_path": str(source_abs_path),
                    "input_gen_model_value": row.get("gen_model", ""),
                    "selected_issue": row.get("selected_issue", ""),
                    "generated_model": args_dict["model"],
                    "hf_model_repo_id": args_dict["repo_id"],
                    "generated_image_path": str(output_image_path),
                    "prompt_column": prompt_column,
                    "seed": args_dict["seed"],
                    "seed_mode": args_dict["seed_mode"],
                    "started_at": item_started_at,
                    "finished_at": utc_now_iso(),
                    "time_seconds": round(time.perf_counter() - item_started, 4),
                    "status": "reused",
                    "error": None,
                },
            })
            continue

        try:
            input_image = load_rgb(source_abs_path)
            prompt = build_edit_prompt(prompt_text, args_dict["edit_suffix"], args_dict["preservation_anchor"])
            seed_value = args_dict["seed"] if args_dict["seed_mode"] == "fixed" else args_dict["seed"] + row_index
            generator = torch.Generator(device=device).manual_seed(seed_value)
            # QwenImageEditPlusPipeline does not accept height/width — output
            # resolution follows the input image. true_cfg_scale and
            # negative_prompt are Qwen-specific parameters.
            gen_kwargs: Dict[str, Any] = {
                "image": input_image,
                "prompt": prompt,
                "generator": generator,
                "num_inference_steps": args_dict["num_inference_steps"],
                "guidance_scale": args_dict["guidance_scale"],
                "true_cfg_scale": args_dict["true_cfg_scale"],
                "negative_prompt": args_dict["negative_prompt"],
                "num_images_per_prompt": 1,
            }

            with torch.inference_mode():
                result = pipe(**gen_kwargs)
            image = result.images[0]
            if output_image_path.suffix.lower() in {".jpg", ".jpeg"}:
                image = image.convert("RGB")
                image.save(output_image_path, quality=95)
            else:
                image.save(output_image_path)

            result_queue.put({
                "row_index": row_index,
                "row": row,
                "status": "generated",
                "generated_image_path": str(output_image_path),
                "generated_model": args_dict["model"],
                "hf_model_repo_id": args_dict["repo_id"],
                "item_log": {
                    "row_index": row_index,
                    "source_image_path": image_path,
                    "source_image_abs_path": str(source_abs_path),
                    "input_gen_model_value": row.get("gen_model", ""),
                    "selected_issue": row.get("selected_issue", ""),
                    "generated_model": args_dict["model"],
                    "hf_model_repo_id": args_dict["repo_id"],
                    "generated_image_path": str(output_image_path),
                    "prompt_column": prompt_column,
                    "seed": seed_value,
                    "seed_mode": args_dict["seed_mode"],
                    "started_at": item_started_at,
                    "finished_at": utc_now_iso(),
                    "time_seconds": round(time.perf_counter() - item_started, 4),
                    "status": "generated",
                    "error": None,
                },
            })
        except Exception as exc:
            result_queue.put(make_error_result(row_index, row, str(exc), output_image_path, args_dict, item_started, item_started_at))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edit/manipulate images from a CSV using Qwen-Image-Edit-2511.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "--csv", dest="input", required=True, help="Input CSV path.")
    parser.add_argument("--input_root", "--image_root", dest="input_root", required=True, help="Root folder for relative image_path values in the input CSV.")
    parser.add_argument("--output_folder", "--out_dir", dest="output_folder", default="edited_outputs", help="Base folder for edited run folders. Default run folder: <output_folder>/<model>_<dataset>[_output_folder_suffix].")
    parser.add_argument("--model", default="Qwen-Image-Edit-2511", choices=sorted(MODEL_REPOS), help="Model to use for every edited image.")
    parser.add_argument("--repo_id", default=None, help="Optional Hugging Face repo/local path override. If omitted, selected from --model.")
    parser.add_argument("--output_csv", default=None, help="Default: <run_output_folder>/generated_images.csv")
    parser.add_argument("--json_log", default=None, help="Default: <run_output_folder>/generation_metadata.json")
    parser.add_argument("--text_log", default=None, help="Default: <output_folder>/generation.log, shared/appended across runs")
    parser.add_argument("--output_folder_suffix", default="", help="Optional word/tag appended to the auto-created run output folder name.")
    parser.add_argument("--prompt_column", default="manipulation_instruction", help="CSV column to use as the edit instruction/prompt text.")
    parser.add_argument("--edit_suffix", default=DEFAULT_EDIT_SUFFIX, help="Suffix appended to the manipulation instruction. Use '' to disable.")
    # Qwen does not accept height/width — output resolution follows the input image.
    parser.add_argument("--num_inference_steps", "--steps", dest="num_inference_steps", type=int, default=40,
                        help="Number of denoising steps. Qwen-Image-Edit-2511 recommendation: 40.")
    parser.add_argument("--guidance_scale", "--guidance", dest="guidance_scale", type=float, default=1.0,
                        help="Guidance scale (flow matching CFG). Qwen recommendation: 1.0.")
    parser.add_argument("--true_cfg_scale", type=float, default=4.0,
                        help="True classifier-free guidance scale (Qwen-specific). Default: 4.0.")
    parser.add_argument("--negative_prompt", type=str, default=" ",
                        help="Negative text prompt. Qwen recommendation: a single space ' '. Set '' to disable.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_mode", choices=["fixed", "row_index"], default="fixed", help="fixed uses the same seed for every row; row_index uses seed + row_index.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"])
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs. 0 means all visible GPUs. Uses CPU if CUDA is unavailable.")
    parser.add_argument("--cpu_offload", action="store_true", help="Enable diffusers CPU offload (recommended for GPUs < 40GB VRAM).")
    parser.add_argument("--load_strategy", choices=["auto", "full", "components"], default="auto",
                        help="Load strategy. 'components' falls back to 'full' for Qwen (no separate component API).")
    parser.add_argument("--image_extension", default=".png")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if the output image already exists.")
    parser.add_argument("--save_json_every", type=int, default=10, help="Update JSON log every N collected results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_start = time.perf_counter()
    input_csv = Path(args.input)
    input_root = Path(args.input_root)
    base_output_folder = Path(args.output_folder)
    repo_id = args.repo_id or MODEL_REPOS[args.model]

    rows, input_fieldnames = load_input_rows(input_csv, args.prompt_column)
    dataset_name = infer_dataset_name(rows, input_csv)
    output_folder = build_run_output_folder(base_output_folder, args.model, dataset_name, args.output_folder_suffix)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_csv = Path(args.output_csv) if args.output_csv else output_folder / "generated_images.csv"
    json_log = Path(args.json_log) if args.json_log else output_folder / "generation_metadata.json"
    text_log = Path(args.text_log) if args.text_log else base_output_folder / "generation.log"
    logger = setup_logger(text_log)

    fields = output_fieldnames(input_fieldnames)
    completed_keys = set() if args.overwrite else read_completed_keys(output_csv, args.model)

    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if cuda_count == 0:
        devices = ["cpu"]
    else:
        use_count = cuda_count if args.num_gpus == 0 else min(args.num_gpus, cuda_count)
        devices = [f"cuda:{i}" for i in range(use_count)]

    pending = [(idx, row) for idx, row in enumerate(rows) if completed_key(row, args.model) not in completed_keys]

    logger.info("=" * 80)
    logger.info("Qwen-Image-Edit-2511 CSV image editing")
    logger.info("Input CSV     : %s", input_csv)
    logger.info("Input root    : %s", input_root)
    logger.info("Base output   : %s", base_output_folder)
    logger.info("Dataset name  : %s", dataset_name)
    logger.info("Run output    : %s", output_folder)
    logger.info("Folder suffix : %s", args.output_folder_suffix or "<empty>")
    logger.info("Output CSV    : %s", output_csv)
    logger.info("JSON log      : %s", json_log)
    logger.info("Text log      : %s", text_log)
    logger.info("Model         : %s", args.model)
    logger.info("Repo ID       : %s", repo_id)
    logger.info("Prompt column : %s", args.prompt_column)
    logger.info("Seed          : %s", args.seed)
    logger.info("Seed mode     : %s", args.seed_mode)
    logger.info("Devices       : %s", devices)
    logger.info("Input rows    : %d", len(rows))
    logger.info("Completed     : %d", len(completed_keys))
    logger.info("Pending       : %d", len(pending))
    logger.info("=" * 80)

    run_data: Dict[str, Any] = {
        "run_started_at": utc_now_iso(),
        "run_finished_at": None,
        "total_time_seconds": None,
        "task": "qwen_image_edit_2511_from_csv",
        "input_csv": str(input_csv),
        "input_root": str(input_root),
        "base_output_folder": str(base_output_folder),
        "dataset_name": dataset_name,
        "output_folder_suffix": args.output_folder_suffix,
        "output_folder": str(output_folder),
        "output_csv": str(output_csv),
        "json_log": str(json_log),
        "text_log": str(text_log),
        "model": args.model,
        "repo_id": repo_id,
        "python_version": platform.python_version(),
        "settings": {
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "true_cfg_scale": args.true_cfg_scale,
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
            "seed_mode": args.seed_mode,
            "dtype": args.dtype,
            "devices": devices,
            "cpu_offload": args.cpu_offload,
            "load_strategy": args.load_strategy,
            "image_extension": args.image_extension,
            "overwrite": args.overwrite,
            "prompt_column": args.prompt_column,
            "edit_suffix_enabled": bool(args.edit_suffix),
            "output_folder_suffix": args.output_folder_suffix,
        },
        "environment": environment_info(),
        "progress": {
            "total_input_rows": len(rows),
            "skipped_already_completed": len(completed_keys),
            "pending_this_run": len(pending),
            "processed_this_run": 0,
            "generated_this_run": 0,
            "reused_existing_image_this_run": 0,
            "error_this_run": 0,
        },
        "items": [],
        "reproducibility": {
            "prompt_column": args.prompt_column,
            "dataset_name": dataset_name,
            "output_folder_suffix": args.output_folder_suffix,
            "system_prompt": SYSTEM_PROMPT,
            "preservation_anchor_template": PRESERVATION_ANCHOR_TEMPLATE,
            "base_edit_prompt_template": BASE_EDIT_PROMPT_TEMPLATE,
            "edit_suffix": args.edit_suffix,
            "prompt_construction": "BASE_EDIT_PROMPT_TEMPLATE.format(preservation_anchor=PRESERVATION_ANCHOR_TEMPLATE, prompt_text=<value from prompt_column>, edit_suffix=<edit_suffix>)",
            "per_item_prompts_stored": False,
            "model": args.model,
            "repo_id": repo_id,
            "python_version": platform.python_version(),
            "settings": {
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "true_cfg_scale": args.true_cfg_scale,
                "negative_prompt": args.negative_prompt,
                "seed": args.seed,
                "seed_mode": args.seed_mode,
                "dtype": args.dtype,
                "cpu_offload": args.cpu_offload,
                "load_strategy": args.load_strategy,
                "image_extension": args.image_extension,
            },
        },
    }
    atomic_write_json(json_log, run_data)

    if not pending:
        logger.info("No pending rows. Nothing to do.")
        run_data["run_finished_at"] = utc_now_iso()
        run_data["total_time_seconds"] = round(time.perf_counter() - run_start, 4)
        atomic_write_json(json_log, run_data)
        return

    partitions: List[List[Tuple[int, Dict[str, str]]]] = [[] for _ in devices]
    for i, item in enumerate(pending):
        partitions[i % len(devices)].append(item)

    args_dict = {
        "input_root": str(input_root),
        "output_folder": str(output_folder),
        "dataset_name": dataset_name,
        "output_folder_suffix": args.output_folder_suffix,
        "model": args.model,
        "repo_id": repo_id,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "seed_mode": args.seed_mode,
        "dtype": args.dtype,
        "cpu_offload": args.cpu_offload,
        "load_strategy": args.load_strategy,
        "image_extension": args.image_extension,
        "overwrite": args.overwrite,
        "prompt_column": args.prompt_column,
        "edit_suffix": args.edit_suffix,
        "preservation_anchor": PRESERVATION_ANCHOR_TEMPLATE,
    }

    mp_ctx = mp.get_context("spawn")
    result_queue: "mp.Queue[Dict[str, Any]]" = mp_ctx.Queue()
    processes: List[mp.Process] = []

    if len(devices) == 1:
        worker(0, devices[0], partitions[0], args_dict, result_queue)
    else:
        for worker_id, device in enumerate(devices):
            p = mp_ctx.Process(target=worker, args=(worker_id, device, partitions[worker_id], args_dict, result_queue), name=f"qwen-image-edit-worker-{worker_id}")
            p.start()
            processes.append(p)
            logger.info("Started worker %d on %s with %d row(s).", worker_id, device, len(partitions[worker_id]))

    collected = 0
    since_json_save = 0
    while collected < len(pending):
        try:
            res = result_queue.get(timeout=600)
        except queue_mod.Empty:
            if processes and all(not p.is_alive() for p in processes):
                logger.error("All workers stopped before all rows were returned.")
                break
            continue

        row = dict(res["row"])
        row.update({
            "generated_image_path": res.get("generated_image_path", ""),
            "status": res.get("status", "error"),
        })
        append_output_row(output_csv, fields, row)

        item_log = res["item_log"]
        run_data["items"].append(item_log)
        run_data["progress"]["processed_this_run"] += 1
        if res["status"] == "generated":
            run_data["progress"]["generated_this_run"] += 1
            logger.info("[%d/%d] generated edit: %s", collected + 1, len(pending), row.get("image_path", ""))
        elif res["status"] == "reused":
            run_data["progress"]["reused_existing_image_this_run"] += 1
            logger.info("[%d/%d] reused edit: %s", collected + 1, len(pending), row.get("image_path", ""))
        else:
            run_data["progress"]["error_this_run"] += 1
            logger.warning("[%d/%d] error: %s | %s", collected + 1, len(pending), row.get("image_path", ""), item_log.get("error"))

        collected += 1
        since_json_save += 1
        if since_json_save >= args.save_json_every:
            atomic_write_json(json_log, run_data)
            since_json_save = 0

    for p in processes:
        p.join(timeout=60)

    run_data["run_finished_at"] = utc_now_iso()
    run_data["total_time_seconds"] = round(time.perf_counter() - run_start, 4)
    atomic_write_json(json_log, run_data)

    logger.info("=" * 80)
    logger.info("Run complete")
    logger.info("Generated edits this run: %d", run_data["progress"]["generated_this_run"])
    logger.info("Reused edits this run   : %d", run_data["progress"]["reused_existing_image_this_run"])
    logger.info("Errors this run         : %d", run_data["progress"]["error_this_run"])
    logger.info("Output CSV              : %s", output_csv.resolve())
    logger.info("JSON log                : %s", json_log.resolve())
    logger.info("Text log                : %s", text_log.resolve())
    logger.info("=" * 80)


if __name__ == "__main__":
    main()