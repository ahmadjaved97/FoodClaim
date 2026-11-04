#!/usr/bin/env python3
"""
Batch summarize long CSV prompts/descriptions into FLUX-token-safe prompts using Qwen3.5-VL.

What this script does:
  1. Reads a CSV containing long prompts/descriptions.
  2. Loads ONLY the FLUX tokenizer, never the FLUX model.
  3. Loads Qwen3.5-VL for text summarization.
  4. Runs batched summarization in two possible passes:
       Pass 1: summarize normally, then check FLUX token count.
       Pass 2: only for rows still above the FLUX limit, summarize again with a stricter prompt.
       If pass 2 is still above the limit, the pass-2 prompt is kept as-is. No hard truncation.
  5. Saves checkpoint CSV every N completed rows.
  6. Resumes automatically from an existing output CSV.
  7. Writes a JSON metadata file for reproducibility.

Example:
  python3 summarize_flux_prompts_qwen3_5_vl_batched.py \
    --input /path/to/FLUX1-dev.csv \
    --output_folder /path/to/output_summaries \
    --qwen_model /path/to/Qwen3.5-VL \
    --flux_model black-forest-labs/FLUX.1-dev \
    --description_col description \
    --max_flux_tokens 512 \
    --target_flux_tokens 447 \
    --batch_size 4 \
    --save_every 50 \
    --reserved_flux_gpu 0 \
    --qwen_devices 1,2,3
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
try:
    from transformers import AutoProcessor, AutoTokenizer
    try:
        from transformers import Qwen3_5ForConditionalGeneration
    except ImportError:  # pragma: no cover - depends on installed transformers version
        Qwen3_5ForConditionalGeneration = None
    _TRANSFORMERS_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment dependent
    AutoProcessor = None
    AutoTokenizer = None
    Qwen3_5ForConditionalGeneration = None
    _TRANSFORMERS_IMPORT_ERROR = exc


# ----------------------------- Defaults -------------------------------------

DEFAULT_QWEN_MODEL = "./models/Qwen3.5-27B"
DEFAULT_FLUX_MODEL = "./models/FLUX.1-dev"

DEFAULT_DESCRIPTION_COL = "description"
DEFAULT_OUTPUT_COL = "summarized_description"

FINAL_STATUSES = {
    "ok_after_pass1",
    "ok_after_pass2",
    "too_long_after_pass2_kept",
    "empty_input",
}


# ----------------------------- Utilities ------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(value: str) -> List[int]:
    if value is None or str(value).strip() == "":
        return []
    return [int(x.strip()) for x in str(value).split(",") if x.strip() != ""]


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logging(output_folder: Path, log_level: str = "INFO") -> Path:
    safe_mkdir(output_folder)
    log_path = output_folder / "summarization.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    return log_path


def get_environment() -> Dict[str, Any]:
    gpus: List[Dict[str, Any]] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
            })

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_imported": _TRANSFORMERS_IMPORT_ERROR is None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda,
        "gpus": gpus,
    }


def clean_output(text: Any) -> str:
    if not isinstance(text, str):
        return ""

    # Qwen thinking/special-token cleanup.
    text = text.split("</think>")[-1]
    for token in [
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
        "<|end|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ]:
        text = text.replace(token, "")

    text = text.replace("\n", " ")
    text = " ".join(text.split()).strip()

    prefixes = [
        "Summarized prompt:",
        "Summary:",
        "Prompt:",
        "Final prompt:",
        "summarized_description:",
        "Output:",
        "Answer:",
    ]
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    return text.strip().strip('"').strip("'").strip()


# ----------------------------- Prompts --------------------------------------

SYSTEM_PROMPT_PASS1 = (
    "You are a strict prompt compression assistant for an image generation model. "
    "Rewrite long dataset-style image descriptions into concise visual prompts. "
    "Return only the final prompt. Do not use markdown, bullets, headings, quotes, or explanations."
)

USER_PROMPT_TEMPLATE_PASS1 = """Summarize the following image-generation prompt for FLUX.

Rules:
- Keep the main subject identity.
- Preserve visually important details: colors, ingredients or object parts, plate or vessel, composition, lighting, camera angle, and image quality if visually relevant.
- Preserve any food defect, manipulation, foreign object, contamination, missing ingredient, or abnormal visual detail if present.
- Remove dataset-style labels such as FOOD, SERVING VESSEL, BACKGROUND, IMAGE QUALITY.
- Remove irrelevant tiny background objects unless they affect the scene.
- Do not invent details.
- Write one concise natural-language prompt.
- Keep the answer under about {target_flux_tokens} FLUX tokens.

Original prompt:
{description}
"""

SYSTEM_PROMPT_PASS2 = (
    "You are an aggressive prompt shortener for an image generation model. "
    "Return only one compact prompt. No markdown, no headings, no bullets, no explanation."
)

USER_PROMPT_TEMPLATE_PASS2 = """The prompt below is still too long for FLUX.

Current FLUX token count: {current_flux_tokens}
Hard FLUX token limit: {max_flux_tokens}
Target FLUX token count: {target_flux_tokens}

Rewrite it more strictly.

Strict rules:
- Keep the food or main subject.
- Keep the defect, manipulation, foreign object, contamination, or abnormal visual detail if present.
- Keep only the most important visible colors, vessel, composition, lighting, and camera angle.
- Remove all repeated wording and non-visual details.
- Use one compact natural-language sentence or two very short sentences.
- Do not add details.
- Do not mention token counts or these rules.

Prompt to shorten:
{prompt}
"""


def build_messages_pass1(description: str, target_flux_tokens: int) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_PASS1},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE_PASS1.format(
                description=description,
                target_flux_tokens=target_flux_tokens,
            ),
        },
    ]


def build_messages_pass2(
    prompt: str,
    current_flux_tokens: int,
    max_flux_tokens: int,
    target_flux_tokens: int,
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_PASS2},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE_PASS2.format(
                prompt=prompt,
                current_flux_tokens=current_flux_tokens,
                max_flux_tokens=max_flux_tokens,
                target_flux_tokens=target_flux_tokens,
            ),
        },
    ]


# ----------------------------- Tokenizers -----------------------------------

def load_flux_tokenizer(flux_model_id: str, flux_tokenizer_subfolder: str):
    """
    Load only the FLUX tokenizer.

    Note: tokenizers are CPU objects in Hugging Face. There is no FLUX model loaded here.
    The script reserves the requested GPU by excluding it from Qwen's device_map.
    """
    kwargs = {"trust_remote_code": True}
    if flux_tokenizer_subfolder:
        kwargs["subfolder"] = flux_tokenizer_subfolder
    if AutoTokenizer is None:
        raise ImportError(f"transformers is required to load the FLUX tokenizer: {_TRANSFORMERS_IMPORT_ERROR}")
    return AutoTokenizer.from_pretrained(flux_model_id, **kwargs)


def flux_token_count(text: Any, flux_tokenizer) -> int:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    encoded = flux_tokenizer(text, truncation=False, add_special_tokens=True)
    return int(len(encoded["input_ids"]))


# ----------------------------- Model loading --------------------------------

def dtype_from_string(dtype_name: str):
    name = str(dtype_name).lower()
    if name in {"auto", ""}:
        return "auto"
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def build_max_memory(
    qwen_devices: List[int],
    reserved_flux_gpu: Optional[int],
    max_memory_per_gpu: str,
) -> Optional[Dict[Any, str]]:
    if not torch.cuda.is_available():
        return None

    visible_count = torch.cuda.device_count()
    allowed = set(qwen_devices)
    max_memory: Dict[Any, str] = {}

    for idx in range(visible_count):
        if reserved_flux_gpu is not None and idx == reserved_flux_gpu:
            max_memory[idx] = "0GiB"
        elif idx in allowed:
            if max_memory_per_gpu:
                max_memory[idx] = max_memory_per_gpu
            else:
                total_gib = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
                max_memory[idx] = f"{int(total_gib * 0.92)}GiB"
        else:
            max_memory[idx] = "0GiB"

    # Allow a little CPU room for dispatch bookkeeping, not for intentional offload.
    max_memory["cpu"] = "64GiB"
    return max_memory


def load_qwen3_5_vl_model(args) -> Tuple[Any, Any]:
    if _TRANSFORMERS_IMPORT_ERROR is not None:
        raise ImportError(f"transformers is required to load Qwen3.5-VL: {_TRANSFORMERS_IMPORT_ERROR}")

    if Qwen3_5ForConditionalGeneration is None:
        raise ImportError(
            "Qwen3_5ForConditionalGeneration is not available in your installed transformers. "
            "Install/upgrade a transformers version that supports Qwen3.5-VL."
        )

    logging.info("Loading Qwen3.5-VL model: %s", args.qwen_model)

    qwen_devices = parse_int_list(args.qwen_devices)
    max_memory = build_max_memory(
        qwen_devices=qwen_devices,
        reserved_flux_gpu=args.reserved_flux_gpu,
        max_memory_per_gpu=args.max_memory_per_gpu,
    )

    if torch.cuda.is_available():
        logging.info("Qwen allowed CUDA devices: %s", qwen_devices)
        logging.info("Reserved GPU excluded from Qwen device_map: cuda:%s", args.reserved_flux_gpu)
        logging.info("max_memory: %s", max_memory)

    model_kwargs = {
        "torch_dtype": dtype_from_string(args.dtype),
        "device_map": args.device_map,
        "max_memory": max_memory,
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.qwen_model,
        **model_kwargs,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.qwen_model, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.qwen_model, trust_remote_code=True)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info("Qwen3.5-VL model loaded successfully.")
    return model, tokenizer


def get_model_input_device(model) -> torch.device:
    """Find the correct first device for input tensors when using device_map='auto'."""
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict) and hf_device_map:
        preferred_keys = [
            "model.embed_tokens",
            "model.language_model.embed_tokens",
            "language_model.model.embed_tokens",
            "visual",
        ]
        for key in preferred_keys:
            if key in hf_device_map and str(hf_device_map[key]) not in {"cpu", "disk"}:
                return torch.device(hf_device_map[key])
        for _, device in hf_device_map.items():
            if str(device) not in {"cpu", "disk"}:
                return torch.device(device)

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ----------------------------- Generation -----------------------------------

@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    batch_messages: List[List[Dict[str, str]]],
    max_new_tokens: int,
    seed: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> List[str]:
    if not batch_messages:
        return []

    torch.manual_seed(seed)

    texts = []
    for messages in batch_messages:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Some tokenizer versions do not expose enable_thinking.
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        texts.append(text)

    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    input_device = get_model_input_device(model)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    generated_ids = model.generate(**inputs, **generation_kwargs)

    input_len = inputs["input_ids"].shape[1]
    output_ids = generated_ids[:, input_len:]

    raw_outputs = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return [clean_output(x) for x in raw_outputs]


# ----------------------------- Metadata -------------------------------------

def initial_metadata(args, output_csv: Path, json_log: Path, text_log: Path) -> Dict[str, Any]:
    return {
        "run_started_at": utc_now_iso(),
        "run_finished_at": None,
        "total_time_seconds": None,
        "input_csv": str(Path(args.input).resolve()),
        "output_folder": str(Path(args.output_folder).resolve()),
        "output_csv": str(output_csv.resolve()),
        "json_log": str(json_log.resolve()),
        "text_log": str(text_log.resolve()),
        "task": "csv_prompt_description_summarization",
        "models": {
            "summarization_model": args.qwen_model,
            "summarization_model_class": "Qwen3_5ForConditionalGeneration",
            "flux_model_for_tokenizer_only": args.flux_model,
            "flux_tokenizer_subfolder": args.flux_tokenizer_subfolder,
            "flux_model_loaded": False,
        },
        "settings": {
            "description_col": args.description_col,
            "output_col": args.output_col,
            "sep": args.sep,
            "batch_size": args.batch_size,
            "save_every": args.save_every,
            "max_flux_tokens": args.max_flux_tokens,
            "target_flux_tokens": args.target_flux_tokens,
            "pass1_max_new_tokens": args.pass1_max_new_tokens,
            "pass2_max_new_tokens": args.pass2_max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "dtype": args.dtype,
            "device_map": args.device_map,
            "reserved_flux_gpu": args.reserved_flux_gpu,
            "qwen_devices": parse_int_list(args.qwen_devices),
            "max_memory_per_gpu": args.max_memory_per_gpu,
            "attn_implementation": args.attn_implementation,
            "resume": not args.no_resume,
        },
        "environment": get_environment(),
        "progress": {
            "total_input_rows": 0,
            "skipped_already_completed": 0,
            "pending_this_run": 0,
            "processed_this_run": 0,
            "pass1_completed_this_run": 0,
            "pass2_completed_this_run": 0,
            "ok_after_pass1": 0,
            "ok_after_pass2": 0,
            "too_long_after_pass2_kept": 0,
            "empty_input": 0,
            "error_this_run": 0,
        },
        "reproducibility": {
            "system_prompt_pass1": SYSTEM_PROMPT_PASS1,
            "user_prompt_template_pass1": USER_PROMPT_TEMPLATE_PASS1,
            "system_prompt_pass2": SYSTEM_PROMPT_PASS2,
            "user_prompt_template_pass2": USER_PROMPT_TEMPLATE_PASS2,
            "prompt_column": args.description_col,
            "prompt_construction_pass1": "USER_PROMPT_TEMPLATE_PASS1.format(description=<value>, target_flux_tokens=<target>)",
            "prompt_construction_pass2": "USER_PROMPT_TEMPLATE_PASS2.format(prompt=<pass1_summary>, current_flux_tokens=<count>, max_flux_tokens=<limit>, target_flux_tokens=<target>)",
            "per_item_prompts_stored": bool(args.store_item_logs),
        },
        "items": [] if args.store_item_logs else None,
    }


def write_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


# ----------------------------- Data handling --------------------------------

def output_paths(input_csv: str, output_folder: str) -> Tuple[Path, Path]:
    input_path = Path(input_csv)
    out_dir = Path(output_folder)
    safe_mkdir(out_dir)
    output_csv = out_dir / f"{input_path.stem}_summarized.csv"
    json_log = out_dir / f"{input_path.stem}_summarization_metadata.json"
    return output_csv, json_log


def prepare_dataframe(args, output_csv: Path) -> pd.DataFrame:
    if args.sep == "\\t":
        args.sep = "\t"

    resume_enabled = not args.no_resume
    if resume_enabled and output_csv.exists():
        logging.info("Resuming from existing output CSV: %s", output_csv)
        df = pd.read_csv(output_csv, sep=args.sep, engine="python")
    else:
        logging.info("Loading input CSV: %s", args.input)
        df = pd.read_csv(args.input, sep=args.sep, engine="python")

    if args.description_col not in df.columns:
        raise ValueError(
            f"Description column '{args.description_col}' not found. Available columns: {list(df.columns)}"
        )

    output_col = args.output_col
    flux_col = f"{output_col}_flux_tokens"
    pass_col = f"{output_col}_pass"
    status_col = f"{output_col}_status"
    error_col = f"{output_col}_error"

    if output_col not in df.columns:
        df[output_col] = ""
    if flux_col not in df.columns:
        df[flux_col] = pd.Series([pd.NA] * len(df), dtype="Int64")
    else:
        df[flux_col] = pd.to_numeric(df[flux_col], errors="coerce").astype("Int64")
    if pass_col not in df.columns:
        df[pass_col] = pd.Series([pd.NA] * len(df), dtype="Int64")
    else:
        df[pass_col] = pd.to_numeric(df[pass_col], errors="coerce").astype("Int64")
    if status_col not in df.columns:
        df[status_col] = ""
    if error_col not in df.columns:
        df[error_col] = ""

    return df


def is_completed(row: pd.Series, output_col: str) -> bool:
    flux_col = f"{output_col}_flux_tokens"
    status_col = f"{output_col}_status"
    summary = row.get(output_col, "")
    status = row.get(status_col, "")
    token_count = row.get(flux_col, pd.NA)

    if str(status) in FINAL_STATUSES:
        return True

    # Backward-compatible resume for older output files without status.
    return isinstance(summary, str) and summary.strip() != "" and pd.notna(token_count)


def batched(iterable: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def save_checkpoint(df: pd.DataFrame, output_csv: Path, args, metadata: Dict[str, Any], json_log: Path) -> None:
    df.to_csv(output_csv, index=False, sep=args.sep)
    write_metadata(json_log, metadata)
    logging.info("Checkpoint saved: %s", output_csv)


# ----------------------------- Main processing ------------------------------

def process_batch(
    df: pd.DataFrame,
    indices: List[Any],
    model,
    tokenizer,
    flux_tokenizer,
    args,
    metadata: Dict[str, Any],
) -> int:
    output_col = args.output_col
    flux_col = f"{output_col}_flux_tokens"
    pass_col = f"{output_col}_pass"
    status_col = f"{output_col}_status"
    error_col = f"{output_col}_error"

    valid_indices: List[Any] = []
    pass1_messages: List[List[Dict[str, str]]] = []

    for idx in indices:
        description = df.at[idx, args.description_col]
        if not isinstance(description, str) or not description.strip():
            df.at[idx, output_col] = ""
            df.at[idx, flux_col] = 0
            df.at[idx, pass_col] = 0
            df.at[idx, status_col] = "empty_input"
            df.at[idx, error_col] = ""
            metadata["progress"]["empty_input"] += 1
            metadata["progress"]["processed_this_run"] += 1
            continue

        valid_indices.append(idx)
        pass1_messages.append(build_messages_pass1(description, args.target_flux_tokens))

    if not valid_indices:
        return len(indices)

    try:
        pass1_outputs = generate_batch(
            model=model,
            tokenizer=tokenizer,
            batch_messages=pass1_messages,
            max_new_tokens=args.pass1_max_new_tokens,
            seed=args.seed,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    except Exception as exc:
        logging.exception("Pass 1 batch failed for indices %s", valid_indices)
        for idx in valid_indices:
            df.at[idx, output_col] = ""
            df.at[idx, flux_col] = pd.NA
            df.at[idx, pass_col] = pd.NA
            df.at[idx, status_col] = "error"
            df.at[idx, error_col] = f"pass1_batch_error: {repr(exc)}"
            metadata["progress"]["error_this_run"] += 1
            metadata["progress"]["processed_this_run"] += 1
        return len(indices)

    pass2_indices: List[Any] = []
    pass2_messages: List[List[Dict[str, str]]] = []

    for idx, summary in zip(valid_indices, pass1_outputs):
        token_count = flux_token_count(summary, flux_tokenizer)
        metadata["progress"]["pass1_completed_this_run"] += 1

        if token_count <= args.max_flux_tokens:
            df.at[idx, output_col] = summary
            df.at[idx, flux_col] = token_count
            df.at[idx, pass_col] = 1
            df.at[idx, status_col] = "ok_after_pass1"
            df.at[idx, error_col] = ""
            metadata["progress"]["ok_after_pass1"] += 1
            metadata["progress"]["processed_this_run"] += 1

            if args.store_item_logs:
                metadata["items"].append({
                    "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                    "status": "ok_after_pass1",
                    "final_flux_tokens": token_count,
                    "final_pass": 1,
                })
        else:
            pass2_indices.append(idx)
            pass2_messages.append(
                build_messages_pass2(
                    prompt=summary,
                    current_flux_tokens=token_count,
                    max_flux_tokens=args.max_flux_tokens,
                    target_flux_tokens=args.target_flux_tokens,
                )
            )
            # Store pass-1 result temporarily in case pass 2 raises an exception.
            df.at[idx, output_col] = summary
            df.at[idx, flux_col] = token_count
            df.at[idx, pass_col] = 1
            df.at[idx, status_col] = "needs_pass2"
            df.at[idx, error_col] = ""

    if pass2_indices:
        try:
            pass2_outputs = generate_batch(
                model=model,
                tokenizer=tokenizer,
                batch_messages=pass2_messages,
                max_new_tokens=args.pass2_max_new_tokens,
                seed=args.seed,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as exc:
            logging.exception("Pass 2 batch failed for indices %s", pass2_indices)
            for idx in pass2_indices:
                df.at[idx, status_col] = "error"
                df.at[idx, error_col] = f"pass2_batch_error_kept_pass1: {repr(exc)}"
                metadata["progress"]["error_this_run"] += 1
                metadata["progress"]["processed_this_run"] += 1
            return len(indices)

        for idx, summary in zip(pass2_indices, pass2_outputs):
            token_count = flux_token_count(summary, flux_tokenizer)
            status = "ok_after_pass2" if token_count <= args.max_flux_tokens else "too_long_after_pass2_kept"

            df.at[idx, output_col] = summary
            df.at[idx, flux_col] = token_count
            df.at[idx, pass_col] = 2
            df.at[idx, status_col] = status
            df.at[idx, error_col] = ""

            metadata["progress"]["pass2_completed_this_run"] += 1
            metadata["progress"][status] += 1
            metadata["progress"]["processed_this_run"] += 1

            if args.store_item_logs:
                metadata["items"].append({
                    "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                    "status": status,
                    "final_flux_tokens": token_count,
                    "final_pass": 2,
                })

    return len(indices)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch summarize CSV prompts/descriptions with Qwen3.5-VL and validate using only the FLUX tokenizer."
    )

    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output_folder", required=True, help="Folder where <input_stem>_summarized.csv and metadata JSON will be saved.")
    parser.add_argument("--description_col", default=DEFAULT_DESCRIPTION_COL, help="Input column containing prompts/descriptions.")
    parser.add_argument("--output_col", default=DEFAULT_OUTPUT_COL, help="Output column for summarized prompts.")
    parser.add_argument("--sep", default=",", help="CSV separator. Use '\\t' for TSV.")

    parser.add_argument("--qwen_model", default=DEFAULT_QWEN_MODEL, help="Qwen3.5-VL model ID or local path.")
    parser.add_argument("--flux_model", default=DEFAULT_FLUX_MODEL, help="FLUX model ID or local path used only for tokenizer loading.")
    parser.add_argument("--flux_tokenizer_subfolder", default="tokenizer_2", help="FLUX tokenizer subfolder. Use '' if tokenizer is at repo root.")

    parser.add_argument("--max_flux_tokens", type=int, default=256, help="Hard FLUX token limit.")
    parser.add_argument("--target_flux_tokens", type=int, default=220, help="Soft target FLUX token count used inside prompts.")
    parser.add_argument("--pass1_max_new_tokens", type=int, default=256, help="Max new tokens for pass 1 summarization.")
    parser.add_argument("--pass2_max_new_tokens", type=int, default=160, help="Max new tokens for pass 2 strict shortening.")

    parser.add_argument("--batch_size", type=int, default=4, help="Rows to summarize per batch.")
    parser.add_argument("--save_every", type=int, default=50, help="Save CSV/metadata every N completed rows.")
    parser.add_argument("--no_resume", action="store_true", help="Disable automatic resume from existing output CSV.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling. Default is greedy decoding.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature if --do_sample is used.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p if --do_sample is used.")

    parser.add_argument("--dtype", default="bfloat16", help="Model dtype: auto, bfloat16, float16, float32.")
    parser.add_argument("--device_map", default="auto", help="HF device_map. Default: auto.")
    parser.add_argument("--reserved_flux_gpu", type=int, default=0, help="GPU to reserve/exclude from Qwen. Tokenizer itself is CPU-only.")
    parser.add_argument("--qwen_devices", default="1,2,3", help="Comma-separated CUDA device indices allowed for Qwen model shards.")
    parser.add_argument("--max_memory_per_gpu", default="", help="Optional max memory per allowed Qwen GPU, e.g. '60GiB'. Empty uses ~92%% of each GPU.")
    parser.add_argument("--attn_implementation", default="", help="Optional attention implementation, e.g. flash_attention_2, sdpa, eager.")

    parser.add_argument("--store_item_logs", action="store_true", help="Store compact per-row status entries in metadata JSON.")
    parser.add_argument("--log_level", default="INFO", help="Logging level.")

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.save_every <= 0:
        raise ValueError("--save_every must be > 0")
    if args.target_flux_tokens > args.max_flux_tokens:
        logging.warning("target_flux_tokens is greater than max_flux_tokens; this is probably not intended.")

    if args.flux_tokenizer_subfolder.strip() == "":
        args.flux_tokenizer_subfolder = ""

    output_csv, json_log = output_paths(args.input, args.output_folder)
    text_log = setup_logging(Path(args.output_folder), args.log_level)

    run_start = time.time()
    set_seed(args.seed)

    metadata = initial_metadata(args, output_csv, json_log, text_log)
    write_metadata(json_log, metadata)

    logging.info("Output CSV will be: %s", output_csv)
    logging.info("Metadata JSON will be: %s", json_log)

    df = prepare_dataframe(args, output_csv)

    total_rows = len(df)
    completed_mask = df.apply(lambda row: is_completed(row, args.output_col), axis=1)
    pending_indices = df.index[~completed_mask].tolist()

    metadata["progress"]["total_input_rows"] = int(total_rows)
    metadata["progress"]["skipped_already_completed"] = int(completed_mask.sum())
    metadata["progress"]["pending_this_run"] = int(len(pending_indices))
    write_metadata(json_log, metadata)

    if len(pending_indices) == 0:
        logging.info("No pending rows. Existing output appears complete.")
        metadata["run_finished_at"] = utc_now_iso()
        metadata["total_time_seconds"] = round(time.time() - run_start, 2)
        write_metadata(json_log, metadata)
        df.to_csv(output_csv, index=False, sep=args.sep)
        return

    logging.info("Loading FLUX tokenizer only: %s", args.flux_model)
    flux_tokenizer = load_flux_tokenizer(args.flux_model, args.flux_tokenizer_subfolder)
    logging.info("FLUX tokenizer loaded. FLUX model was not loaded.")

    model, tokenizer = load_qwen3_5_vl_model(args)

    completed_since_save = 0
    pbar = tqdm(total=len(pending_indices), desc="Summarizing pending rows")

    try:
        for batch_indices in batched(pending_indices, args.batch_size):
            before_processed = metadata["progress"]["processed_this_run"]

            process_batch(
                df=df,
                indices=batch_indices,
                model=model,
                tokenizer=tokenizer,
                flux_tokenizer=flux_tokenizer,
                args=args,
                metadata=metadata,
            )

            newly_processed = metadata["progress"]["processed_this_run"] - before_processed
            completed_since_save += newly_processed
            pbar.update(len(batch_indices))

            if completed_since_save >= args.save_every:
                save_checkpoint(df, output_csv, args, metadata, json_log)
                completed_since_save = 0
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        pbar.close()

    df.to_csv(output_csv, index=False, sep=args.sep)
    metadata["run_finished_at"] = utc_now_iso()
    metadata["total_time_seconds"] = round(time.time() - run_start, 2)
    write_metadata(json_log, metadata)

    logging.info("Done.")
    logging.info("Total rows: %s", total_rows)
    logging.info("Skipped already completed: %s", metadata["progress"]["skipped_already_completed"])
    logging.info("Processed this run: %s", metadata["progress"]["processed_this_run"])
    logging.info("Errors this run: %s", metadata["progress"]["error_this_run"])
    logging.info("Output CSV: %s", output_csv)
    logging.info("Metadata JSON: %s", json_log)


if __name__ == "__main__":
    main()