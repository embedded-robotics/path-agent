from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime import load_config, run_histocartography


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local histocartography patch extraction.")
    parser.add_argument("--config", required=True, help="Path to tools/histocartography config YAML")
    parser.add_argument("--image-path", required=True, help="Path to source WSI/image")
    parser.add_argument("--top-n", type=int, help="Number of selected HC patches to return")
    parser.add_argument("--out", required=True, help="Path to write JSON output")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_histocartography(
        image_path=args.image_path,
        top_n=int(args.top_n or config.get("top_n", 3)),
        svs_level=config.get("svs_level"),
        grid_size=int(config.get("grid_size", 3)),
        nuclei_threshold=int(config.get("nuclei_threshold", 5)),
    )

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
