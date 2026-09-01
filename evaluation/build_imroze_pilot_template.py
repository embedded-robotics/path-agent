import json
from pathlib import Path


SHEET_PATH = Path(__file__).with_name("imroze_question_sheet.json")
OUT_PATH = Path(__file__).with_name("imroze_pilot_answers_template.json")
PILOT_KEYS = {
    ("07_level2", 1),
    ("07_level2", 5),
    ("07_level2", 8),
    ("19_level2", 1),
    ("19_level2", 6),
    ("19_level2", 9),
}


def main() -> None:
    with SHEET_PATH.open("r", encoding="utf-8") as f:
        sheet = json.load(f)

    answers = []
    for row in sheet["questions"]:
        key = (row["case_id"], row["question_id"])
        if key not in PILOT_KEYS:
            continue
        answers.append(
            {
                "case_id": row["case_id"],
                "question_id": row["question_id"],
                "category": row["category"],
                "question": row["question"],
                "expected_answer": row["expected_answer"],
                "model_answer": "",
                "grading": {
                    "score": None,
                    "notes": "",
                    "matched_expected": None,
                },
            }
        )

    payload = {"answers": answers}
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(answers)} pilot questions to {OUT_PATH}")


if __name__ == "__main__":
    main()
