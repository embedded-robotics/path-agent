from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .runtime import load_config, resolve_path, run_histocartography


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local histocartography patch extraction.")
    parser.add_argument("--config", required=True, help="Path to tools/histocartography config YAML")
    parser.add_argument("--image-path", required=True, help="Path to source WSI/image")
    parser.add_argument("--top-n", type=int, help="Number of selected HC patches to return")
    parser.add_argument("--out", required=True, help="Path to write JSON output")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    config_dir = config_path.parent
    checkpoint_dir = resolve_path(
        os.environ.get("HISTOCARTOGRAPHY_CHECKPOINT_DIR", config.get("checkpoint_dir")),
        config_dir,
    )
    pretrained_data = os.environ.get(
        "HISTOCARTOGRAPHY_PRETRAINED_DATA",
        str(config.get("pretrained_data", "pannuke")),
    )
    result = run_histocartography(
        image_path=args.image_path,
        top_n=int(args.top_n or config.get("top_n", 3)),
        svs_level=config.get("svs_level"),
        grid_size=int(config.get("grid_size", 3)),
        nuclei_threshold=int(config.get("nuclei_threshold", 5)),
        checkpoint_dir=checkpoint_dir,
        pretrained_data=pretrained_data,
    )

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
