from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.graph.runtime import infer_one, load_yaml, runtime_settings


def _requests(path: Path) -> list[dict]:
    rows, seen = [], set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid request JSONL at line {line_number}") from exc
        for field in ("request_id", "patch_id", "task_type", "image", "prompt"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Request line {line_number} requires non-empty {field}")
        if row["task_type"] not in {"roi", "contribution"} or row["request_id"] in seen:
            raise ValueError(f"Invalid or duplicate request_id at line {line_number}")
        seen.add(row["request_id"])
        rows.append(row)
    if not rows:
        raise ValueError("Request JSONL contains no requests")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model")
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--answers-file", required=True)
    args = parser.parse_args()
    settings = runtime_settings(load_yaml(args.config), args.model)
    root, answers_path = Path(args.image_folder).resolve(), Path(args.answers_file).resolve()
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    answers = []
    for request in _requests(Path(args.question_file).resolve()):
        image = Path(request["image"])
        text = infer_one(settings, image if image.is_absolute() else root / image, request["prompt"])
        answers.append({"request_id": request["request_id"], "patch_id": request["patch_id"], "task_type": request["task_type"], "text": text.strip(), "model_id": settings.model_id, "precision": settings.precision, "quantization": settings.quantization})
    with answers_path.open("w", encoding="utf-8") as handle:
        for answer in answers:
            handle.write(json.dumps(answer, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
