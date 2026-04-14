from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from transformers import pipeline


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _bf16_or_auto():
    if torch.cuda.is_available():
        return torch.bfloat16
    return "auto"


@lru_cache(maxsize=2)
def load_pipe(model_id: str):
    cache_dir = os.environ.get("HF_HOME")
    kwargs: dict[str, Any] = {
        "model": model_id,
        "device_map": "auto",
        "torch_dtype": _bf16_or_auto(),
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return pipeline("image-text-to-text", **kwargs)


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
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append(str(part.get("text", "")))
                    return " ".join(p.strip() for p in parts if p.strip()).strip()
    if isinstance(output, dict):
        return _coerce_generated_text(output.get("generated_text", ""))
    return str(output).strip()


def infer_one(model_id: str, image_path: str | Path, prompt: str, max_new_tokens: int, temperature: float) -> str:
    pipe = load_pipe(model_id)
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    outputs = pipe(
        text=messages,
        max_new_tokens=max_new_tokens,
        return_full_text=False,
        do_sample=temperature > 0,
        temperature=temperature,
    )
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, dict):
            text = _coerce_generated_text(first.get("generated_text", first))
            if text:
                return text
    return _coerce_generated_text(outputs)
