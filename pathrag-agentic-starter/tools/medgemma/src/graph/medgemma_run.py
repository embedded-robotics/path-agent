from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.graph.runtime import infer_one, load_yaml, runtime_settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--question", required=True)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    settings = runtime_settings(cfg, args.model)

    image_dir = Path(args.image_folder).resolve()
    images = sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]
    )[: args.k]

    captions = []
    for img in images:
        caption = infer_one(settings, img, args.question)
        captions.append(caption.strip())

    answer = " ".join(c for c in captions if c).strip()
    payload = {
        "answer": answer,
        "captions": captions,
        "model": settings.model_id,
    }

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
