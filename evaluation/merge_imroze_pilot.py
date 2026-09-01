import json
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parent
PILOT_PATH = EVAL_DIR / "imroze_pilot_answers_template.json"
SHEET_PATH = EVAL_DIR / "imroze_question_sheet.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main() -> None:
    pilot = load_json(PILOT_PATH)
    sheet = load_json(SHEET_PATH)

    pilot_answers = pilot.get("answers", [])
    if len(pilot_answers) != 6:
        raise SystemExit(f"Expected 6 pilot answers, found {len(pilot_answers)}")

    by_key = {
        (row["case_id"], row["question_id"]): row
        for row in sheet["questions"]
    }

    merged = 0
    completed = 0
    for answer in pilot_answers:
        key = (answer["case_id"], answer["question_id"])
        row = by_key.get(key)
        if row is None:
            continue
        row["model_answer"] = answer.get("model_answer", "")
        grading = answer.get("grading", {})
        if isinstance(grading, dict):
            row["grading"].update(grading)
        merged += 1
        if row["model_answer"].strip():
            completed += 1

    write_json(SHEET_PATH, sheet)
    print(f"Merged {merged} pilot rows into {SHEET_PATH}")
    print(f"Filled answers: {completed}/6")


if __name__ == "__main__":
    main()
