import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_ROOT = REPO_ROOT / "pathrag-agentic-starter"
STARTER_SRC = STARTER_ROOT / "src"
LAUNCH_CWD = Path.cwd()
if str(STARTER_SRC) not in sys.path:
    sys.path.insert(0, str(STARTER_SRC))

os.chdir(STARTER_ROOT)

from pathrag.agents.langgraph_app import build_graph  # noqa: E402


DEFAULT_SHEET = Path(__file__).with_name("imroze_question_sheet.json").resolve()
DEFAULT_OUT = Path(__file__).with_name("imroze_langgraph_answers.json").resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def patch_rows_to_state_patches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    for i, row in enumerate(rows or []):
        patches.append(
            {
                "id": f"CP{i}",
                "bbox": [
                    int(row["x1"]),
                    int(row["y1"]),
                    int(row["x2"]),
                    int(row["y2"]),
                ],
                "score": float(len(rows) - i),
            }
        )
    return patches


def run_question(
    app: Any,
    image_path: str,
    question: str,
    patches: list[dict[str, Any]],
    top_k: int,
    max_rounds: int,
    thread_id: str,
) -> dict[str, Any]:
    init = {
        "image_path": image_path,
        "question": question,
        "mode": "answer",
        "top_k": top_k,
        "max_rounds": max_rounds,
        "round_ix": 0,
        "patches": patches,
    }

    merged: dict[str, Any] = {}
    for state in app.stream(init, config={"configurable": {"thread_id": thread_id}}):
        if isinstance(state, dict):
            merged.update(state)

    if not isinstance(merged, dict):
        return {
            "final_answer": "",
            "subpath_label": "",
            "full_captions": [],
            "chosen_idx": [],
            "patch_summaries": [],
            "roi_desc": [],
        }

    fuse = merged.get("fuse", merged)
    identify = merged.get("identify", merged)
    rerank = merged.get("rerank", merged)
    roi = merged.get("stage4", merged)
    return {
        "final_answer": fuse.get("final_answer", merged.get("final_answer", "")),
        "subpath_label": identify.get("subpath_label", merged.get("subpath_label", "")),
        "full_captions": identify.get("full_captions", merged.get("full_captions", [])),
        "chosen_idx": rerank.get("chosen_idx", merged.get("chosen_idx", [])),
        "patch_summaries": roi.get("patch_summaries", merged.get("patch_summaries", [])),
        "roi_desc": roi.get("roi_desc", merged.get("roi_desc", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LangGraph on Imroze evaluation questions using saved CHIEF patches."
    )
    parser.add_argument("--sheet", default=str(DEFAULT_SHEET), help="Path to imroze_question_sheet.json")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Path to write model answers JSON")
    parser.add_argument("--case-id", help="Run only one case_id")
    parser.add_argument("--question-id", type=int, help="Run only one question_id")
    parser.add_argument("--top-k", type=int, default=3, help="Stage 6/7 top_k")
    parser.add_argument("--max-rounds", type=int, default=1, help="Critique rounds")
    args = parser.parse_args()

    sheet_arg = Path(args.sheet)
    out_arg = Path(args.out)
    sheet_path = sheet_arg if sheet_arg.is_absolute() else (LAUNCH_CWD / sheet_arg)
    out_path = out_arg if out_arg.is_absolute() else (LAUNCH_CWD / out_arg)
    sheet_path = sheet_path.resolve()
    out_path = out_path.resolve()
    sheet = load_json(sheet_path)
    rows = sheet.get("questions", [])
    if args.case_id:
        rows = [r for r in rows if r.get("case_id") == args.case_id]
    if args.question_id is not None:
        rows = [r for r in rows if int(r.get("question_id", -1)) == args.question_id]

    app = build_graph()
    answers: list[dict[str, Any]] = []

    for row in rows:
        patches = patch_rows_to_state_patches(row.get("chief_patch_coords") or [])
        thread_id = f"{row['case_id']}-q{row['question_id']}"
        result = run_question(
            app=app,
            image_path=row["image_path"],
            question=row["question"],
            patches=patches,
            top_k=args.top_k,
            max_rounds=args.max_rounds,
            thread_id=thread_id,
        )
        answers.append(
            {
                "case_id": row["case_id"],
                "question_id": row["question_id"],
                "category": row["category"],
                "image_path": row["image_path"],
                "question": row["question"],
                "expected_answer": row["expected_answer"],
                "patch_count": len(patches),
                "model_answer": result["final_answer"],
                "subpath_label": result["subpath_label"],
                "full_captions": result["full_captions"],
                "chosen_idx": result["chosen_idx"],
                "patch_summaries": result["patch_summaries"],
                "roi_desc": result["roi_desc"],
            }
        )
        print(
            f"{row['case_id']} Q{row['question_id']}: "
            f"patches={len(patches)} chosen={result['chosen_idx']}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"answers": answers}, f, indent=2)
        f.write("\n")
    print(f"Saved answers to {out_path}")


if __name__ == "__main__":
    main()
