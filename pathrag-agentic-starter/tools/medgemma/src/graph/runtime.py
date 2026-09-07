from __future__ import annotations

import importlib
import inspect
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from transformers import pipeline


class MedGemmaConfigurationError(ValueError):
    """Raised before model loading when tool configuration is invalid."""


@dataclass(frozen=True)
class RuntimeSettings:
    model_id: str
    precision: str
    quantization: str
    max_new_tokens: int
    microbatch_size: int
    temperature: float


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _env(name: str, default: Any) -> Any:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _precision_dtype(precision: str) -> torch.dtype | str:
    if precision == "auto":
        return "auto"
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    raise MedGemmaConfigurationError("MEDGEMMA_PRECISION must be auto, bf16, or fp16")


def runtime_settings(config: dict[str, Any], model_override: str | None = None) -> RuntimeSettings:
    models = config.get("models", {})
    generation = config.get("generation", {})
    runtime = config.get("runtime", {})
    model_id = model_override or _env("MEDGEMMA_MODEL", models.get("medgemma_repo", "google/medgemma-1.5-4b-it"))
    precision = str(_env("MEDGEMMA_PRECISION", runtime.get("precision", "bf16"))).lower()
    quantization = str(_env("MEDGEMMA_QUANTIZATION", runtime.get("quantization", "4bit"))).lower()
    max_new_tokens = int(_env("MEDGEMMA_MAX_NEW_TOKENS", generation.get("max_new_tokens", 96)))
    microbatch_size = int(_env("MEDGEMMA_MICROBATCH_SIZE", runtime.get("microbatch_size", 1)))
    temperature = float(_env("MEDGEMMA_TEMPERATURE", generation.get("temperature", 0.0)))
    if not model_id:
        raise MedGemmaConfigurationError("MedGemma model ID must be non-empty")
    _precision_dtype(precision)
    if quantization not in {"none", "8bit", "4bit"}:
        raise MedGemmaConfigurationError("MEDGEMMA_QUANTIZATION must be none, 8bit, or 4bit")
    if max_new_tokens < 1:
        raise MedGemmaConfigurationError("MEDGEMMA_MAX_NEW_TOKENS must be at least 1")
    if microbatch_size != 1:
        raise MedGemmaConfigurationError(
            "MEDGEMMA_MICROBATCH_SIZE must be 1: only sequential single-item inference "
            "is currently supported; larger microbatches require a future measured optimization."
        )
    if temperature < 0:
        raise MedGemmaConfigurationError("MEDGEMMA_TEMPERATURE must be non-negative")
    return RuntimeSettings(model_id, precision, quantization, max_new_tokens, microbatch_size, temperature)


def pipeline_kwargs(settings: RuntimeSettings) -> dict[str, Any]:
    """Build load arguments without constructing a pipeline or loading weights."""
    precision_key = "dtype" if "dtype" in inspect.signature(pipeline).parameters else "torch_dtype"
    kwargs: dict[str, Any] = {"model": settings.model_id, "device_map": "auto", precision_key: _precision_dtype(settings.precision)}
    cache_dir = os.environ.get("HF_HOME")
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if settings.quantization == "none":
        return kwargs
    try:
        importlib.import_module("bitsandbytes")
    except ImportError as exc:
        raise MedGemmaConfigurationError(
            "MEDGEMMA_QUANTIZATION requires bitsandbytes in tools/medgemma. "
            "Install the tool requirements before requesting 4bit or 8bit mode."
        ) from exc
    from transformers import BitsAndBytesConfig

    if settings.quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        compute_dtype = torch.bfloat16 if settings.precision != "fp16" else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    return kwargs


@lru_cache(maxsize=4)
def load_pipe(model_id: str, precision: str = "bf16", quantization: str = "4bit"):
    settings = RuntimeSettings(model_id, precision, quantization, 1, 1, 0.0)
    return pipeline("image-text-to-text", **pipeline_kwargs(settings))


def _coerce_generated_text(output: Any) -> str:
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, list):
        for item in reversed(output):
            if isinstance(item, dict) and item.get("role") == "assistant":
                content = item.get("content", "")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    return " ".join(str(part.get("text", "")).strip() for part in content if isinstance(part, dict) and part.get("type") == "text" and part.get("text")).strip()
    if isinstance(output, dict):
        return _coerce_generated_text(output.get("generated_text", ""))
    return str(output).strip()


def infer_one(settings: RuntimeSettings, image_path: str | Path, prompt: str) -> str:
    pipe = load_pipe(settings.model_id, settings.precision, settings.quantization)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    outputs = pipe(text=messages, max_new_tokens=settings.max_new_tokens, return_full_text=False, do_sample=settings.temperature > 0, temperature=settings.temperature)
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
        text = _coerce_generated_text(outputs[0].get("generated_text", outputs[0]))
        if text:
            return text
    return _coerce_generated_text(outputs)
