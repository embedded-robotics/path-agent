from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.graph.runtime import infer_one, load_yaml


def resolve_image_path(image_root: Path, raw_image: str) -> Path:
    p = Path(raw_image)
    return p if p.is_absolute() else (image_root / raw_image)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--answers-file", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    model_id = args.model or cfg["models"]["medgemma_repo"]
    max_new_tokens = int(cfg.get("generation", {}).get("max_new_tokens", 160))
    temperature = float(cfg.get("generation", {}).get("temperature", 0.0))

    qfile = Path(args.question_file).resolve()
    image_root = Path(args.image_folder).resolve()
    afile = Path(args.answers_file).resolve()
    afile.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with qfile.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    outputs = []
    for row in rows:
        prompt = row.get("text") or row.get("prompt") or row.get("question") or ""
        image_path = resolve_image_path(image_root, str(row["image"]))
        text = infer_one(model_id, image_path, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        outputs.append(
            {
                "question_id": row.get("question_id"),
                "image": str(row.get("image", "")),
                "text": text.strip(),
                "model_id": model_id,
            }
        )

    with afile.open("w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
