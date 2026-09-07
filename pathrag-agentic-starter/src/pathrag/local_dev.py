from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from dotenv import load_dotenv


SUPPORTED_MODES = ("answer", "description")
LOCAL_DEV_STAGE4_BACKENDS = ("stub", "medgemma")
_MISSING = object()


class LocalDevError(ValueError):
    """Actionable input or configuration error for the local-dev runner."""


def _require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LocalDevError(f"{location} must be a JSON object")
    return value


def _require_rows(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise LocalDevError(f"{location} must be a JSON array")
    if not value:
        raise LocalDevError(f"{location} must contain at least one patch")
    return value


def _validate_coordinate(value: Any, patch_index: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LocalDevError(f"patch[{patch_index}].{field} must be a finite integer")
    return value


def _validate_bbox(value: Any, patch_index: int) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise LocalDevError(f"patch[{patch_index}].bbox must contain exactly four coordinates")
    bbox = [
        _validate_coordinate(coordinate, patch_index, f"bbox[{coordinate_index}]")
        for coordinate_index, coordinate in enumerate(value)
    ]
    x1, y1, x2, y2 = bbox
    if x2 <= x1:
        raise LocalDevError(f"patch[{patch_index}].bbox must satisfy x2 > x1")
    if y2 <= y1:
        raise LocalDevError(f"patch[{patch_index}].bbox must satisfy y2 > y1")
    return bbox


def _validate_score(value: Any, patch_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LocalDevError(f"patch[{patch_index}].score must be a finite number")
    score = float(value)
    if not math.isfinite(score):
        raise LocalDevError(f"patch[{patch_index}].score must be a finite number")
    return score


def _normalize_canonical_rows(rows: Any, location: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(_require_rows(rows, location)):
        row = _require_object(value, f"patch[{index}]")
        patch_id_value = row.get("id")
        if not isinstance(patch_id_value, str) or not patch_id_value.strip():
            raise LocalDevError(f"patch[{index}].id must be a non-empty string")
        patch_id = patch_id_value.strip()
        if patch_id in seen_ids:
            raise LocalDevError(f"patch[{index}].id duplicates patch ID {patch_id!r}")
        if "bbox" not in row:
            raise LocalDevError(f"patch[{index}].bbox is required")
        if "score" not in row:
            raise LocalDevError(f"patch[{index}].score is required")
        normalized.append(
            {
                "id": patch_id,
                "bbox": _validate_bbox(row["bbox"], index),
                "score": _validate_score(row["score"], index),
            }
        )
        seen_ids.add(patch_id)
    return normalized


def _normalize_chief_rows(rows: Any, location: str) -> list[dict[str, Any]]:
    source_rows = _require_rows(rows, location)
    normalized: list[dict[str, Any]] = []
    for index, value in enumerate(source_rows):
        row = _require_object(value, f"patch[{index}]")
        coordinates: list[int] = []
        for field in ("x1", "y1", "x2", "y2"):
            if field not in row:
                raise LocalDevError(f"patch[{index}].{field} is required")
            coordinates.append(_validate_coordinate(row[field], index, field))
        bbox = _validate_bbox(coordinates, index)
        normalized.append(
            {
                "id": f"CP{index}",
                "bbox": bbox,
                "score": float(len(source_rows) - index),
            }
        )
    return normalized


def normalize_patch_document(document: Any) -> list[dict[str, Any]]:
    """Normalize one of the explicitly supported saved-patch JSON shapes."""
    if isinstance(document, list):
        return _normalize_canonical_rows(document, "patch list")
    if not isinstance(document, dict):
        raise LocalDevError("patch JSON must be an array or supported wrapper object")
    if "patches" in document:
        return _normalize_canonical_rows(document["patches"], "patches")
    if "fuse" in document:
        fuse = _require_object(document["fuse"], "fuse")
        if "patches" not in fuse:
            raise LocalDevError("fuse.patches is required")
        return _normalize_canonical_rows(fuse["patches"], "fuse.patches")
    if "chief_patch_coords" in document:
        return _normalize_chief_rows(document["chief_patch_coords"], "chief_patch_coords")
    if "questions" in document:
        raise LocalDevError(
            "full evaluation sheets are not supported; extract one question's "
            "chief_patch_coords array"
        )
    raise LocalDevError(
        "unsupported patch JSON shape; expected a patch array, patches, "
        "fuse.patches, or chief_patch_coords"
    )


def load_patches_json(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except json.JSONDecodeError as exc:
        raise LocalDevError(
            f"invalid JSON in {source}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise LocalDevError(f"could not read patch JSON {source}: {exc}") from exc
    return normalize_patch_document(document)


def select_patches(
    patches: Sequence[dict[str, Any]], top_k: int
) -> tuple[list[dict[str, Any]], str | None]:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise LocalDevError("top_k must be an integer greater than or equal to 1")
    if not patches:
        raise LocalDevError("at least one validated patch is required")
    selected_count = min(top_k, len(patches))
    message = None
    if top_k > len(patches):
        message = (
            f"Requested top_k={top_k}, but only {len(patches)} patches are available; "
            f"using all {len(patches)} patches."
        )
    return list(patches[:selected_count]), message


def validate_run_options(
    image_path: str, question: str, mode: str, max_rounds: int
) -> tuple[str, str]:
    image_text = str(image_path).strip()
    if not image_text:
        raise LocalDevError("image_path must be a non-empty path or logical identifier")
    question_text = str(question).strip()
    if not question_text:
        raise LocalDevError("question must be non-empty")
    if mode not in SUPPORTED_MODES:
        raise LocalDevError("mode must be 'answer' or 'description'")
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
        raise LocalDevError(
            "max_rounds must be an integer greater than or equal to 1; "
            "the current graph always executes one critique round"
        )
    return image_text, question_text


def _emit_stage3_warnings(image_path: str, stage4_backend: str = "stub") -> None:
    warnings.warn(
        "Local-dev Stage 3 is placeholder-only: predict_tissue() hashes the image-path "
        "string without inspecting image contents, and captions come from the packaged "
        "tissue_caps.yml bank. These results have no medical meaning.",
        UserWarning,
        stacklevel=2,
    )
    if not Path(image_path).exists():
        warnings.warn(
            f"Image path does not exist; continuing with the logical path for placeholder "
            f"Stage 3 and {stage4_backend} Stage 4: {image_path}",
            UserWarning,
            stacklevel=2,
        )


@contextmanager
def _local_dev_environment(stage4_backend: str) -> Iterator[None]:
    overrides: dict[str, str | None] = {
        "PATHRAG_STAGE4_BACKEND": stage4_backend,
        "PATHRAG_STAGE4_REMOTE_URL": None,
        "PATHRAG_USE_RETRIEVER_API": "0",
    }
    previous = {name: os.environ.get(name, _MISSING) for name in overrides}
    try:
        for name, value in overrides.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is _MISSING:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)


def _require_openai_key() -> None:
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise LocalDevError(
            "OPENAI_API_KEY is required for the existing Stage 5 critique and "
            "Stage 7 fusion calls; set it in the environment or project .env file"
        )


def _load_production_graph() -> Any:
    """Import the production graph lazily and reuse its module-level app."""
    langgraph_app = importlib.import_module("pathrag.agents.langgraph_app")
    return langgraph_app.app


def _thread_id(value: str | None) -> str:
    if value is None:
        return f"local-dev-{uuid.uuid4().hex[:12]}"
    normalized = value.strip()
    if not normalized:
        raise LocalDevError("thread_id must be non-empty when provided")
    return normalized


def _build_result(
    final_state: dict[str, Any],
    *,
    patches_json: Path,
    image_path: str,
    output_path: Path,
    question: str,
    patches: list[dict[str, Any]],
    requested_top_k: int,
    max_rounds: int,
    mode: str,
    stage4_backend: str,
    thread_id: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "run": {
            "backend": stage4_backend,
            "thread_id": thread_id,
            "input_paths": {
                "patches_json": str(patches_json),
                "image_path": image_path,
            },
            "output_path": str(output_path),
            "top_k": len(patches),
            "requested_top_k": requested_top_k,
            "effective_top_k": len(patches),
            "max_rounds": max_rounds,
            "mode": mode,
        },
        "question": question,
        "subpath_label": final_state.get("subpath_label", ""),
        "full_captions": final_state.get("full_captions", []),
        "patches": patches,
        "roi_useful": final_state.get("roi_useful", []),
        "roi_desc": final_state.get("roi_desc", []),
        "patch_summaries": final_state.get("patch_summaries", []),
        "chosen_idx": final_state.get("chosen_idx", []),
        "final_answer": final_state.get("final_answer", ""),
    }
    for field in ("round_ix", "dynamic_k"):
        if field in final_state:
            result[field] = final_state[field]
    return result


def run_local_dev(
    *,
    patches_json: str | Path,
    image_path: str,
    question: str,
    top_k: int = 3,
    max_rounds: int = 1,
    mode: str = "answer",
    stage4_backend: str = "stub",
    out: str | Path = "artifacts/local-dev/result.json",
    thread_id: str | None = None,
    graph_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    source_path = Path(patches_json).expanduser().resolve()
    output_path = Path(out).expanduser().resolve()
    if output_path == source_path:
        raise LocalDevError("output path must not overwrite the input patch JSON file")

    patches = load_patches_json(source_path)
    selected, clamp_message = select_patches(patches, top_k)
    image_text, question_text = validate_run_options(
        image_path, question, mode, max_rounds
    )
    if stage4_backend not in LOCAL_DEV_STAGE4_BACKENDS:
        raise LocalDevError(
            "stage4_backend must be one of: " + ", ".join(LOCAL_DEV_STAGE4_BACKENDS)
        )
    resolved_thread_id = _thread_id(thread_id)
    _emit_stage3_warnings(image_text, stage4_backend)
    if clamp_message:
        warnings.warn(clamp_message, UserWarning, stacklevel=2)
    load_dotenv()
    _require_openai_key()

    initial_state = {
        "image_path": image_text,
        "question": question_text,
        "mode": mode,
        "top_k": len(selected),
        "max_rounds": max_rounds,
        "round_ix": 0,
        "patches": selected,
    }
    with _local_dev_environment(stage4_backend):
        graph = graph_factory() if graph_factory is not None else _load_production_graph()
        final_state = graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": resolved_thread_id}},
        )
    if not isinstance(final_state, dict):
        raise LocalDevError("graph execution did not return a complete final state object")

    result = _build_result(
        final_state,
        patches_json=source_path,
        image_path=image_text,
        output_path=output_path,
        question=question_text,
        patches=selected,
        requested_top_k=top_k,
        max_rounds=max_rounds,
        mode=mode,
        stage4_backend=stage4_backend,
        thread_id=resolved_thread_id,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Output: {output_path}")
    print(f"Selected patch IDs: {[patch['id'] for patch in selected]}")
    print(f"Chosen indices: {result['chosen_idx']}")
    print(f"Final answer: {result['final_answer']}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run PathRAG from precomputed patches with placeholder Stage 3, "
            "a selected local Stage 4 backend, and existing OpenAI-backed Stages 5 and 7."
        )
    )
    parser.add_argument("--patches-json", required=True, help="Saved patch JSON input")
    parser.add_argument("--image-path", required=True, help="Image path or logical identifier")
    parser.add_argument("--question", required=True, help="Pathology question")
    parser.add_argument("--top-k", type=int, default=3, help="Maximum patches to retain")
    parser.add_argument("--max-rounds", type=int, default=1, help="Critique rounds (minimum 1)")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default="answer")
    parser.add_argument(
        "--stage4-backend",
        choices=LOCAL_DEV_STAGE4_BACKENDS,
        default="stub",
        help="Stage 4 backend for this runner (default: stub)",
    )
    parser.add_argument("--thread-id", help="Optional LangGraph thread ID")
    parser.add_argument("--out", default="artifacts/local-dev/result.json")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_local_dev(
            patches_json=args.patches_json,
            image_path=args.image_path,
            question=args.question,
            top_k=args.top_k,
            max_rounds=args.max_rounds,
            mode=args.mode,
            stage4_backend=args.stage4_backend,
            out=args.out,
            thread_id=args.thread_id,
        )
    except LocalDevError as exc:
        parser.error(str(exc))
