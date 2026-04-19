from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .runtime import extract_patches, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local CHIEF patch extraction.")
    parser.add_argument("--config", required=True, help="Path to tools/chief config YAML")
    parser.add_argument("--image-path", required=True, help="Path to source WSI/image")
    parser.add_argument("--top-k", type=int, required=True, help="Number of CHIEF patches to return")
    parser.add_argument("--out", required=True, help="Path to write JSON output")
    parser.add_argument(
        "--valid-bounds-json",
        help="Optional JSON override for valid bounds, e.g. [[0,600,1600,1000]]",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    chief_root = Path(os.environ.get("CHIEF_REPO_DIR", config["chief_repo_dir"])).resolve()
    model_dir = Path(os.environ.get("CHIEF_MODEL_DIR", config["model_dir"])).resolve()

    valid_bounds = config.get("valid_bounds") or [[0, 600, 1600, 1000]]
    if args.valid_bounds_json:
        valid_bounds = json.loads(args.valid_bounds_json)

    result = extract_patches(
        chief_root=chief_root,
        model_dir=model_dir,
        image_path=args.image_path,
        top_k=args.top_k,
        patch_size=int(config.get("patch_size", 224)),
        slide_level=int(config.get("slide_level", 2)),
        anatomical_label=int(config.get("anatomical_label", 1)),
        valid_bounds=valid_bounds,
    )

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
