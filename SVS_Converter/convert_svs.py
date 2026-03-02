import argparse
import os
from typing import Optional, Tuple

import openslide
from PIL import Image


SUPPORTED_OUTPUT_FORMATS = {"jpg", "jpeg", "tif", "tiff"}


def _get_converter_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _normalize_output_format(output_format: str) -> str:
    fmt = output_format.lower().strip()
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt == "tiff":
        fmt = "tif"
    if fmt not in {"jpg", "tif"}:
        raise ValueError(
            f"Unsupported output format '{output_format}'. Supported: jpg, tif"
        )
    return fmt


def _default_output_path(svs_path: str, output_format: str, slide_level: int) -> str:
    base_name = os.path.splitext(os.path.basename(svs_path))[0]
    converter_root = _get_converter_root()

    if output_format == "jpg":
        output_dir = os.path.join(converter_root, "jpg_outputs")
        extension = "jpg"
    else:
        output_dir = os.path.join(converter_root, "tif_outputs")
        extension = "tif"

    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{base_name}_level{slide_level}.{extension}")


def _validate_slide_level(slide: openslide.OpenSlide, slide_level: int) -> None:
    if slide_level < 0 or slide_level >= slide.level_count:
        raise ValueError(
            f"Invalid slide level {slide_level}. Valid range: 0 to {slide.level_count - 1}"
        )


def _read_svs_level_image(svs_path: str, slide_level: int) -> Image.Image:
    if not os.path.exists(svs_path):
        raise FileNotFoundError(f"SVS file not found: {svs_path}")

    slide = openslide.OpenSlide(svs_path)
    try:
        _validate_slide_level(slide, slide_level)
        width, height = slide.level_dimensions[slide_level]
        image = slide.read_region((0, 0), slide_level, (width, height)).convert("RGB")
        return image
    finally:
        slide.close()


def convert_svs_to_tif(
    svs_path: str,
    slide_level: int = 0,
    output_path: Optional[str] = None,
    compression: str = "tiff_lzw",
) -> str:
    """
    Convert an SVS to TIFF by extracting a specific pyramid level.

    Args:
        svs_path: Path to input .svs file.
        slide_level: Pyramid level to extract.
        output_path: Optional explicit output path.
        compression: TIFF compression method for PIL.

    Returns:
        Output file path.
    """
    image = _read_svs_level_image(svs_path, slide_level)

    if output_path is None:
        output_path = _default_output_path(svs_path, "tif", slide_level)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    image.save(output_path, format="TIFF", compression=compression)
    print(f"Saved TIFF: {output_path}")
    return output_path


def convert_svs_to_jpg(
    svs_path: str,
    slide_level: int = 0,
    output_path: Optional[str] = None,
    quality: int = 95,
) -> str:
    """
    Convert an SVS to JPG by extracting a specific pyramid level.

    Args:
        svs_path: Path to input .svs file.
        slide_level: Pyramid level to extract.
        output_path: Optional explicit output path.
        quality: JPG quality (1-95 typically used).

    Returns:
        Output file path.
    """
    image = _read_svs_level_image(svs_path, slide_level)

    if output_path is None:
        output_path = _default_output_path(svs_path, "jpg", slide_level)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    image.save(output_path, format="JPEG", quality=quality)
    print(f"Saved JPG: {output_path}")
    return output_path


def convert_svs(
    svs_path: str,
    output_format: str = "jpg",
    slide_level: int = 0,
    output_path: Optional[str] = None,
    jpg_quality: int = 95,
    tif_compression: str = "tiff_lzw",
) -> str:
    """
    Convert an SVS to JPG or TIFF at a selected pyramid level.

    Args:
        svs_path: Path to input .svs file.
        output_format: One of jpg, jpeg, tif, tiff.
        slide_level: Pyramid level to extract.
        output_path: Optional explicit output path.
        jpg_quality: JPEG quality if output is JPG.
        tif_compression: TIFF compression if output is TIFF.

    Returns:
        Output file path.
    """
    fmt = _normalize_output_format(output_format)

    if fmt == "jpg":
        return convert_svs_to_jpg(
            svs_path=svs_path,
            slide_level=slide_level,
            output_path=output_path,
            quality=jpg_quality,
        )

    return convert_svs_to_tif(
        svs_path=svs_path,
        slide_level=slide_level,
        output_path=output_path,
        compression=tif_compression,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SVS to JPG or TIFF by extracting a selected pyramid level."
    )
    parser.add_argument(
        "svs_path",
        nargs="?",
        default="svs_examples/19.svs",
        help="Path to input SVS file (default: svs_examples/19.svs)",
    )
    parser.add_argument(
        "--format",
        "-f",
        default="jpg",
        choices=sorted(SUPPORTED_OUTPUT_FORMATS),
        help="Output format: jpg/jpeg/tif/tiff (default: jpg)",
    )
    parser.add_argument(
        "--level",
        "-l",
        type=int,
        default=2,
        help="Pyramid level to extract (default: 2)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional output file path (default: auto-generated)",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality when format is jpg/jpeg (default: 95)",
    )
    parser.add_argument(
        "--tif-compression",
        default="tiff_lzw",
        help="TIFF compression when format is tif/tiff (default: tiff_lzw)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = convert_svs(
        svs_path=args.svs_path,
        output_format=args.format,
        slide_level=args.level,
        output_path=args.output,
        jpg_quality=args.jpg_quality,
        tif_compression=args.tif_compression,
    )
    print(f"Conversion complete: {output_path}")


if __name__ == "__main__":
    main()
