import json
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parent
SHEET_PATH = EVAL_DIR / "imroze_question_sheet.json"
PILOT_KEYS = [
    ("07_level2", 1),
    ("07_level2", 5),
    ("07_level2", 8),
    ("19_level2", 1),
    ("19_level2", 6),
    ("19_level2", 9),
]


def main() -> None:
    with SHEET_PATH.open("r", encoding="utf-8") as f:
        sheet = json.load(f)

    by_key = {
        (row["case_id"], row["question_id"]): row
        for row in sheet["questions"]
    }

    for case_id, question_id in PILOT_KEYS:
        row = by_key.get((case_id, question_id))
        if row is None:
            continue
        grading = row.get("grading") or {}
        print(f"{case_id} Q{question_id} [{row['category']}]")
        print(f"Question: {row['question']}")
        print(f"Expected: {row['expected_answer']}")
        print(f"Model: {row.get('model_answer', '')}")
        print(
            "Grade:",
            f"score={grading.get('score')}",
            f"matched_expected={grading.get('matched_expected')}",
        )
        notes = grading.get("notes", "")
        if notes:
            print(f"Notes: {notes}")
        print()


if __name__ == "__main__":
    main()
