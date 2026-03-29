import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SHEET = Path(__file__).with_name("imroze_question_sheet.json")
DEFAULT_TEMPLATE = Path(__file__).with_name("imroze_answers_template.json")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def build_template(sheet: dict[str, Any]) -> dict[str, Any]:
    answers: list[dict[str, Any]] = []
    for row in sheet["questions"]:
        answers.append(
            {
                "case_id": row["case_id"],
                "question_id": row["question_id"],
                "question": row["question"],
                "expected_answer": row["expected_answer"],
                "model_answer": row.get("model_answer", ""),
                "grading": row.get("grading", {"score": None, "notes": "", "matched_expected": None}),
            }
        )
    return {"answers": answers}


def merge_answers(sheet: dict[str, Any], updates: dict[str, Any]) -> int:
    by_key = {
        (row["case_id"], row["question_id"]): row
        for row in sheet["questions"]
    }
    merged = 0
    for answer in updates.get("answers", []):
        key = (answer["case_id"], answer["question_id"])
        row = by_key.get(key)
        if row is None:
            continue
        if "model_answer" in answer:
            row["model_answer"] = answer["model_answer"]
        if "grading" in answer and isinstance(answer["grading"], dict):
            row["grading"].update(answer["grading"])
        merged += 1
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or merge answer records for Imroze question sheets.")
    parser.add_argument("--sheet", default=str(DEFAULT_SHEET), help="Path to imroze_question_sheet.json")
    parser.add_argument("--template-out", default=str(DEFAULT_TEMPLATE), help="Where to write a blank answer template")
    parser.add_argument("--emit-template", action="store_true", help="Write an answer template derived from the question sheet")
    parser.add_argument("--answers", help="Path to JSON answer updates to merge into the sheet")
    parser.add_argument("--out", help="Output path for merged question sheet (defaults to in-place)")
    args = parser.parse_args()

    sheet_path = Path(args.sheet)
    sheet = load_json(sheet_path)

    if args.emit_template:
        template = build_template(sheet)
        write_json(Path(args.template_out), template)
        print(f"Wrote template to {args.template_out}")

    if args.answers:
        updates = load_json(Path(args.answers))
        merged = merge_answers(sheet, updates)
        out_path = Path(args.out) if args.out else sheet_path
        write_json(out_path, sheet)
        print(f"Merged {merged} answer rows into {out_path}")

    if not args.emit_template and not args.answers:
        parser.error("use --emit-template and/or --answers")


if __name__ == "__main__":
    main()
