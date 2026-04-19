from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np
from PIL import Image
import yaml

SVS_EXTENSIONS = {".svs"}
FLAT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

_NUCLEI_DETECTOR = None
_FEATS_EXTRACTOR = None
_KNN_GRAPH_BUILDER = None


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_components():
    global _NUCLEI_DETECTOR, _FEATS_EXTRACTOR, _KNN_GRAPH_BUILDER
    if _NUCLEI_DETECTOR is None:
        from histocartography.preprocessing import (
            DeepFeatureExtractor,
            KNNGraphBuilder,
            NucleiExtractor,
        )

        _NUCLEI_DETECTOR = NucleiExtractor()
        _FEATS_EXTRACTOR = DeepFeatureExtractor(
            architecture="resnet34",
            patch_size=72,
            resize_size=224,
        )
        _KNN_GRAPH_BUILDER = KNNGraphBuilder(k=5, thresh=50, add_loc_feats=True)
    return _NUCLEI_DETECTOR, _FEATS_EXTRACTOR, _KNN_GRAPH_BUILDER


def _get_file_ext(path: str) -> str:
    return Path(path).suffix.lower()


def _is_svs(path: str) -> bool:
    return _get_file_ext(path) in SVS_EXTENSIONS


def _is_flat_image(path: str) -> bool:
    return _get_file_ext(path) in FLAT_IMAGE_EXTENSIONS


def svs_to_image(svs_path: str, level: int | None = None) -> Tuple[Image.Image, int]:
    import openslide

    slide = openslide.OpenSlide(svs_path)
    try:
        if level is None:
            level = slide.level_count - 1
        level_dims = slide.level_dimensions[level]
        img = slide.read_region((0, 0), level, level_dims).convert("RGB")
        return img, level
    finally:
        slide.close()


def load_input_image(image_path: str, svs_level: int | None = None) -> Tuple[Image.Image, int]:
    if _is_svs(image_path):
        return svs_to_image(image_path, svs_level)
    if _is_flat_image(image_path):
        return Image.open(image_path).convert("RGB"), 0
    raise ValueError(f"Unsupported image format: {image_path}")


def extract_patches(
    image: Image.Image,
    top_n: int,
    grid_size: int,
    nuclei_threshold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nuclei_detector, feats_extractor, knn_graph_builder = _get_components()

    image_array = np.array(image)
    nuclei_map, nuclei_centers = nuclei_detector.process(image_array)

    if nuclei_centers.shape[0] <= nuclei_threshold:
        return [], []

    features = feats_extractor.process(image_array, nuclei_map)
    knn_graph_builder.process(nuclei_map, features)

    width, height = image.size
    width_range = np.linspace(0, width, grid_size + 1, dtype=int)
    height_range = np.linspace(0, height, grid_size + 1, dtype=int)

    patch_nuclei_centers = []
    patch_coordinates = []

    for i in range(len(width_range) - 1):
        for j in range(len(height_range) - 1):
            left = int(width_range[i])
            upper = int(height_range[j])
            right = int(width_range[i + 1])
            lower = int(height_range[j + 1])

            center_list = []
            for center in nuclei_centers:
                if (
                    center[0] >= left
                    and center[0] <= right
                    and center[1] >= upper
                    and center[1] <= lower
                ):
                    center_list.append(center)

            patch_nuclei_centers.append(center_list)
            patch_coordinates.append((left, upper, right, lower))

    patch_center_length = [len(center) for center in patch_nuclei_centers]
    sorted_indices_desc = np.flip(np.argsort(patch_center_length))

    top_patch_info: list[dict[str, Any]] = []
    non_selected_patch_info: list[dict[str, Any]] = []
    actual_top_n = min(top_n, len(patch_coordinates))

    for patch_index in range(actual_top_n):
        idx = sorted_indices_desc[patch_index]
        coords = patch_coordinates[idx]
        nuclei_count = int(patch_center_length[idx])
        top_patch_info.append(
            {
                "patch_number": patch_index + 1,
                "x1": int(coords[0]),
                "y1": int(coords[1]),
                "x2": int(coords[2]),
                "y2": int(coords[3]),
                "nuclei_count": nuclei_count,
            }
        )

    for patch_index in range(actual_top_n, len(sorted_indices_desc)):
        idx = sorted_indices_desc[patch_index]
        coords = patch_coordinates[idx]
        nuclei_count = int(patch_center_length[idx])
        non_selected_patch_info.append(
            {
                "x1": int(coords[0]),
                "y1": int(coords[1]),
                "x2": int(coords[2]),
                "y2": int(coords[3]),
                "nuclei_count": nuclei_count,
            }
        )

    return top_patch_info, non_selected_patch_info


def run_histocartography(
    image_path: str,
    top_n: int,
    svs_level: int | None,
    grid_size: int,
    nuclei_threshold: int,
) -> dict[str, Any]:
    image, actual_level = load_input_image(image_path, svs_level)
    try:
        selected, non_selected = extract_patches(
            image=image,
            top_n=top_n,
            grid_size=grid_size,
            nuclei_threshold=nuclei_threshold,
        )
    finally:
        image.close()

    return {
        "success": bool(selected),
        "image_path": str(Path(image_path).resolve()),
        "slide_level": int(actual_level),
        "selected_patches": selected,
        "non_selected_patches": non_selected,
    }
