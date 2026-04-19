from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _ensure_on_syspath(chief_root: Path) -> None:
    root = str(chief_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_models(chief_root: Path, model_dir: Path, device: str):
    _ensure_on_syspath(chief_root)
    from models.ctran import ConvStem
    from models.CHIEF import CHIEF
    from timm.models.swin_transformer import SwinTransformer

    with pushd(chief_root):
        backbone = SwinTransformer(
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
        backbone.patch_embed = ConvStem(
            img_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            norm_layer=nn.LayerNorm,
            flatten=True,
        )
        backbone.head = nn.Identity()
        td = torch.load(model_dir / "CHIEF_CTransPath.pth", map_location=device)
        backbone.load_state_dict(td["model"], strict=True)
        backbone.eval().to(device)

        chief = CHIEF(size_arg="small", dropout=True, n_classes=2)
        chief.load_state_dict(
            torch.load(model_dir / "CHIEF_pretraining.pth", map_location=device),
            strict=True,
        )
        chief.eval().to(device)
    return backbone, chief


def extract_patches(
    chief_root: Path,
    model_dir: Path,
    image_path: str,
    top_k: int,
    patch_size: int,
    slide_level: int,
    anatomical_label: int,
    valid_bounds: list[list[int]],
) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ensure_on_syspath(chief_root)
    from chief_heatmap import extract_top_k_patches

    backbone, chief = _load_models(chief_root, model_dir, device)
    with pushd(chief_root):
        top_k_patches, heatmap = extract_top_k_patches(
            svs_path=str(Path(image_path).resolve()),
            slide_level=int(slide_level),
            valid_bounds=[tuple(map(int, b)) for b in valid_bounds],
            k=int(top_k),
            backbone=backbone,
            chief=chief,
            anatomical_label=int(anatomical_label),
            patch_size=int(patch_size),
            device=device,
        )

    patch_coords = [
        {
            "x1": int(x),
            "y1": int(y),
            "x2": int(x + patch_size),
            "y2": int(y + patch_size),
        }
        for x, y in top_k_patches
    ]
    return {
        "top_k_patches": patch_coords,
        "heatmap": heatmap.tolist(),
        "slide_level": int(slide_level),
        "message": f"Successfully extracted {len(patch_coords)} patches at slide level {slide_level}",
        "device": device,
    }
