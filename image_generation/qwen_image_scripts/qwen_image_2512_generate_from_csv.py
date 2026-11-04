"""
qwen_image_2512_generate_from_csv.py
====================================
Generate food images from a CSV using Qwen-Image-2512.

Model:
  - Qwen-Image-2512 -> Qwen/Qwen-Image-2512

Input CSV columns required:
  image_path,dataset_name,class_id,is_human,description,gen_model

Key behavior:
  - image_path values are relative to --input_root, matching the original FLUX script.
  - generation prompt uses only the description column.
  - only one output aspect ratio is used: 1:1, fixed at 1024x1024 by default.
  - output CSV keeps the original columns and adds only:
      generated_image_path, generated_model, hf_model_repo_id, status
  - errors are recorded in the output CSV only as status="error".
  - detailed error/seed/prompt/timing info is written to JSON, not the CSV.
  - output CSV is append-only and resume is based on existing successful rows.
  - multi-GPU parallelism is optional; the main process is the only CSV/JSON writer.

Install:
  pip install -U diffusers transformers accelerate torch pillow

Example:
python qwen_image_2512_generate_from_csv.py \
  --input input.csv \
  --input_root /data/original_images \
  --output_folder /data/generated_qwen \
  --model Qwen-Image-2512 \
  --num_gpus 4 \
  --seed 42
"""

from __future__ import annotations

import argparse
import csv
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

try:
    from diffusers import DiffusionPipeline
except ImportError as exc:
    raise ImportError(
        "Could not import DiffusionPipeline. Install/upgrade diffusers first: "
        "pip install -U diffusers transformers accelerate"
    ) from exc


MODEL_REPOS: Dict[str, str] = {
    "Qwen-Image-2512": "Qwen/Qwen-Image-2512",
}

REQUIRED_COLUMNS = ["image_path", "dataset_name", "class_id", "is_human", "description", "gen_model"]
ADDED_OUTPUT_COLUMNS = ["generated_image_path", "generated_model", "hf_model_repo_id", "status"]
SUCCESS_STATUSES = {"generated", "reused"}

DEFAULT_NEGATIVE_PROMPT = (
    "low resolution, low quality, blurry, distorted anatomy, deformed objects, "
    "oversaturated, waxy texture, artificial AI look, messy composition, unreadable text"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_name(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "model"


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
    logger = logging.getLogger("qwen_image_csv_gen")
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
    logger = logging.getLogger(f"qwen_image_worker_{worker_id}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(f"%(asctime)s [worker {worker_id}] [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def load_input_rows(input_csv: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {missing}. Found: {fieldnames}")
        return [dict(r) for r in reader], fieldnames


def output_fieldnames(input_fieldnames: Iterable[str]) -> List[str]:
    fields = list(input_fieldnames)
    for col in ADDED_OUTPUT_COLUMNS:
        if col not in fields:
            fields.append(col)
    return fields


def read_completed_keys(output_csv: Path, model_name: str) -> Set[Tuple[str, str]]:
    completed: Set[Tuple[str, str]] = set()
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return completed
    with output_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = (row.get("image_path") or "").strip()
            generated_model = (row.get("generated_model") or row.get("gen_model") or "").strip()
            status = (row.get("status") or "").strip().lower()
            generated_image_path = (row.get("generated_image_path") or "").strip()
            if image_path and generated_model == model_name and status in SUCCESS_STATUSES and generated_image_path:
                completed.add((image_path, generated_model))
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


def build_prompt(description: str) -> str:
    return (
        "Generate a high-quality, realistic 1024x1024 square food photograph based only on this food description. "
        "Match the food appearance, ingredients, serving vessel, scene composition, lighting, "
        "camera angle, and image quality details as closely as possible. Do not add text or watermarks.\n\n"
        f"{description.strip()}"
    )


def build_output_image_path(image_path: str, output_folder: Path, model_name: str, image_extension: str) -> Path:
    rel = Path(image_path)
    suffix = image_extension if image_extension.startswith(".") else f".{image_extension}"
    return output_folder / rel.parent / f"{rel.stem}_{sanitize_name(model_name)}{suffix}"


def load_pipeline(
    repo_id: str,
    dtype: torch.dtype,
    device: str,
    cpu_offload: bool,
    logger: logging.Logger,
) -> DiffusionPipeline:
    logger.info("Loading Qwen-Image pipeline from %s on %s", repo_id, device)
    pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=dtype)

    if cpu_offload and device.startswith("cuda"):
        # Works with recent diffusers/accelerate. For older versions, fall back to .to(device).
        try:
            gpu_id = int(device.split(":", 1)[1]) if ":" in device else 0
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
            return pipe
        except TypeError:
            logger.warning("enable_model_cpu_offload(gpu_id=...) not supported by this diffusers version; using .to(%s).", device)
        except Exception as exc:
            logger.warning("CPU offload failed (%s); using .to(%s).", exc, device)

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
            info["gpus"].append(
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )
    return info


def make_error_result(
    row_index: int,
    row: Dict[str, str],
    status_error: str,
    output_image_path: Optional[Path],
    args_dict: Dict[str, Any],
    item_started: float,
    item_started_at: str,
) -> Dict[str, Any]:
    image_path = (row.get("image_path") or "").strip()
    description = (row.get("description") or "").strip()
    prompt = build_prompt(description) if description else ""
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
            "generated_model": args_dict["model"],
            "hf_model_repo_id": args_dict["repo_id"],
            "generated_image_path": str(output_image_path) if output_image_path else "",
            "prompt": prompt,
            "negative_prompt": args_dict["negative_prompt"],
            "seed": args_dict["seed"],
            "width": args_dict["width"],
            "height": args_dict["height"],
            "started_at": item_started_at,
            "finished_at": utc_now_iso(),
            "time_seconds": round(time.perf_counter() - item_started, 4),
            "status": "error",
            "error": status_error,
        },
    }


def make_generator(device: str, seed: int) -> torch.Generator:
    if device.startswith("cuda"):
        return torch.Generator(device=device).manual_seed(seed)
    return torch.Generator(device="cpu").manual_seed(seed)


def worker(
    worker_id: int,
    device: str,
    indexed_rows: List[Tuple[int, Dict[str, str]]],
    args_dict: Dict[str, Any],
    result_queue: "mp.Queue[Dict[str, Any]]",
) -> None:
    logger = worker_logger(worker_id)
    dtype = resolve_dtype(args_dict["dtype"])
    try:
        pipe = load_pipeline(args_dict["repo_id"], dtype, device, args_dict["cpu_offload"], logger)
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
        description = (row.get("description") or "").strip()
        output_image_path = build_output_image_path(image_path, output_folder, args_dict["model"], args_dict["image_extension"])

        if not image_path:
            result_queue.put(make_error_result(row_index, row, "empty_image_path", output_image_path, args_dict, item_started, item_started_at))
            continue
        if not description or description.upper() == "ERROR":
            result_queue.put(make_error_result(row_index, row, "empty_or_error_description", output_image_path, args_dict, item_started, item_started_at))
            continue

        source_abs_path = input_root / image_path
        if args_dict["check_source_exists"] and not source_abs_path.exists():
            result_queue.put(make_error_result(row_index, row, f"source_image_not_found: {source_abs_path}", output_image_path, args_dict, item_started, item_started_at))
            continue

        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        prompt = build_prompt(description)

        if output_image_path.exists() and not args_dict["overwrite"]:
            result_queue.put(
                {
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
                        "generated_model": args_dict["model"],
                        "hf_model_repo_id": args_dict["repo_id"],
                        "generated_image_path": str(output_image_path),
                        "prompt": prompt,
                        "negative_prompt": args_dict["negative_prompt"],
                        "seed": args_dict["seed"],
                        "width": args_dict["width"],
                        "height": args_dict["height"],
                        "started_at": item_started_at,
                        "finished_at": utc_now_iso(),
                        "time_seconds": round(time.perf_counter() - item_started, 4),
                        "status": "reused",
                        "error": None,
                    },
                }
            )
            continue

        try:
            generator = make_generator(device, args_dict["seed"])
            gen_kwargs: Dict[str, Any] = {
                "prompt": prompt,
                "negative_prompt": args_dict["negative_prompt"],
                "width": args_dict["width"],
                "height": args_dict["height"],
                "num_inference_steps": args_dict["num_inference_steps"],
                "true_cfg_scale": args_dict["true_cfg_scale"],
                "generator": generator,
                "max_sequence_length": args_dict["max_sequence_length"],
            }
            if args_dict["guidance_scale"] is not None:
                gen_kwargs["guidance_scale"] = args_dict["guidance_scale"]

            result = pipe(**gen_kwargs)
            result.images[0].save(output_image_path)
            result_queue.put(
                {
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
                        "generated_model": args_dict["model"],
                        "hf_model_repo_id": args_dict["repo_id"],
                        "generated_image_path": str(output_image_path),
                        "prompt": prompt,
                        "negative_prompt": args_dict["negative_prompt"],
                        "seed": args_dict["seed"],
                        "width": args_dict["width"],
                        "height": args_dict["height"],
                        "started_at": item_started_at,
                        "finished_at": utc_now_iso(),
                        "time_seconds": round(time.perf_counter() - item_started, 4),
                        "status": "generated",
                        "error": None,
                    },
                }
            )
        except Exception as exc:
            result_queue.put(make_error_result(row_index, row, str(exc), output_image_path, args_dict, item_started, item_started_at))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 1024x1024 images from a CSV using Qwen-Image-2512.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--input_root", required=True, help="Root folder for the relative image_path values in the input CSV.")
    parser.add_argument("--output_folder", required=True, help="Folder for generated images and default logs.")
    parser.add_argument("--model", default="Qwen-Image-2512", choices=sorted(MODEL_REPOS), help="Model to use for every generated image.")
    parser.add_argument("--repo_id", default=None, help="Optional Hugging Face repo/local path override. If omitted, selected from --model.")
    parser.add_argument("--output_csv", default=None, help="Default: <output_folder>/generated_images.csv")
    parser.add_argument("--json_log", default=None, help="Default: <output_folder>/generation_metadata.json")
    parser.add_argument("--text_log", default=None, help="Default: <output_folder>/generation.log")
    parser.add_argument("--height", type=int, default=1024, help="Fixed output height. Keep 1024 for 1:1.")
    parser.add_argument("--width", type=int, default=1024, help="Fixed output width. Keep 1024 for 1:1.")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0, help="Qwen-Image classifier-free guidance scale.")
    parser.add_argument("--guidance_scale", type=float, default=None, help="Optional distilled guidance scale; normally leave unset for Qwen-Image-2512.")
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--max_sequence_length", type=int, default=1024, help="Maximum prompt sequence length passed to the Qwen-Image pipeline.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed used for every image.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"])
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs. 0 means all visible GPUs. Uses CPU if CUDA is unavailable.")
    parser.add_argument("--cpu_offload", action="store_true", help="Enable diffusers CPU offload when CUDA is used.")
    parser.add_argument("--image_extension", default=".png")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if the output image already exists.")
    parser.add_argument("--save_json_every", type=int, default=10, help="Update JSON log every N collected results.")
    parser.add_argument(
        "--no_source_check",
        action="store_true",
        help="Do not require input_root/image_path files to exist. Useful because Qwen-Image generation uses text only.",
    )
    args = parser.parse_args()

    if args.width != args.height:
        raise ValueError("This script is restricted to one aspect ratio: 1:1. Use equal --width and --height, e.g. 1024 and 1024.")

    run_start = time.perf_counter()
    input_csv = Path(args.input)
    input_root = Path(args.input_root)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv) if args.output_csv else output_folder / "generated_images.csv"
    json_log = Path(args.json_log) if args.json_log else output_folder / "generation_metadata.json"
    text_log = Path(args.text_log) if args.text_log else output_folder / "generation.log"
    logger = setup_logger(text_log)

    repo_id = args.repo_id or MODEL_REPOS[args.model]
    rows, input_fieldnames = load_input_rows(input_csv)
    fields = output_fieldnames(input_fieldnames)
    completed_keys = set() if args.overwrite else read_completed_keys(output_csv, args.model)

    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if cuda_count == 0:
        devices = ["cpu"]
    else:
        use_count = cuda_count if args.num_gpus == 0 else min(args.num_gpus, cuda_count)
        devices = [f"cuda:{i}" for i in range(use_count)]

    pending = [
        (idx, row)
        for idx, row in enumerate(rows)
        if (((row.get("image_path") or "").strip(), args.model) not in completed_keys)
    ]

    logger.info("=" * 80)
    logger.info("Qwen-Image-2512 CSV image generation")
    logger.info("Input CSV     : %s", input_csv)
    logger.info("Input root    : %s", input_root)
    logger.info("Output folder : %s", output_folder)
    logger.info("Output CSV    : %s", output_csv)
    logger.info("JSON log      : %s", json_log)
    logger.info("Text log      : %s", text_log)
    logger.info("Model         : %s", args.model)
    logger.info("Repo ID       : %s", repo_id)
    logger.info("Prompt fields : description only")
    logger.info("Aspect ratio  : 1:1")
    logger.info("Resolution    : %dx%d", args.width, args.height)
    logger.info("Seed          : %s", args.seed)
    logger.info("Devices       : %s", devices)
    logger.info("Input rows    : %d", len(rows))
    logger.info("Completed     : %d", len(completed_keys))
    logger.info("Pending       : %d", len(pending))
    logger.info("=" * 80)

    run_data: Dict[str, Any] = {
        "run_started_at": utc_now_iso(),
        "run_finished_at": None,
        "total_time_seconds": None,
        "input_csv": str(input_csv),
        "input_root": str(input_root),
        "output_folder": str(output_folder),
        "output_csv": str(output_csv),
        "json_log": str(json_log),
        "text_log": str(text_log),
        "model": args.model,
        "repo_id": repo_id,
        "settings": {
            "height": args.height,
            "width": args.width,
            "aspect_ratio": "1:1",
            "num_inference_steps": args.num_inference_steps,
            "true_cfg_scale": args.true_cfg_scale,
            "guidance_scale": args.guidance_scale,
            "negative_prompt": args.negative_prompt,
            "max_sequence_length": args.max_sequence_length,
            "seed": args.seed,
            "seed_mode": "global_fixed_same_seed_for_every_row",
            "dtype": args.dtype,
            "devices": devices,
            "cpu_offload": args.cpu_offload,
            "image_extension": args.image_extension,
            "overwrite": args.overwrite,
            "check_source_exists": not args.no_source_check,
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
        "model": args.model,
        "repo_id": repo_id,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "true_cfg_scale": args.true_cfg_scale,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": args.negative_prompt,
        "max_sequence_length": args.max_sequence_length,
        "seed": args.seed,
        "dtype": args.dtype,
        "cpu_offload": args.cpu_offload,
        "image_extension": args.image_extension,
        "overwrite": args.overwrite,
        "check_source_exists": not args.no_source_check,
    }

    mp_ctx = mp.get_context("spawn")
    result_queue: "mp.Queue[Dict[str, Any]]" = mp_ctx.Queue()
    processes: List[mp.Process] = []

    if len(devices) == 1:
        # In-process mode is easier to debug and avoids multiprocessing overhead.
        worker(0, devices[0], partitions[0], args_dict, result_queue)
    else:
        for worker_id, device in enumerate(devices):
            p = mp_ctx.Process(
                target=worker,
                args=(worker_id, device, partitions[worker_id], args_dict, result_queue),
                name=f"qwen-image-worker-{worker_id}",
            )
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
        row.update(
            {
                "generated_image_path": res.get("generated_image_path", ""),
                "generated_model": res.get("generated_model", args.model),
                "hf_model_repo_id": res.get("hf_model_repo_id", repo_id),
                "status": res.get("status", "error"),
            }
        )
        append_output_row(output_csv, fields, row)

        item_log = res["item_log"]
        run_data["items"].append(item_log)
        run_data["progress"]["processed_this_run"] += 1
        if res["status"] == "generated":
            run_data["progress"]["generated_this_run"] += 1
            logger.info("[%d/%d] generated: %s", collected + 1, len(pending), row.get("image_path", ""))
        elif res["status"] == "reused":
            run_data["progress"]["reused_existing_image_this_run"] += 1
            logger.info("[%d/%d] reused: %s", collected + 1, len(pending), row.get("image_path", ""))
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
    logger.info("Generated this run: %d", run_data["progress"]["generated_this_run"])
    logger.info("Reused this run   : %d", run_data["progress"]["reused_existing_image_this_run"])
    logger.info("Errors this run   : %d", run_data["progress"]["error_this_run"])
    logger.info("Output CSV        : %s", output_csv.resolve())
    logger.info("JSON log          : %s", json_log.resolve())
    logger.info("Text log          : %s", text_log.resolve())
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
