import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULTS = Path(__file__).with_name("imroze_vqa_results.json")
DEFAULT_JSON_OUT = Path(__file__).with_name("imroze_question_sheet.json")
DEFAULT_MD_OUT = Path(__file__).with_name("imroze_question_sheet.md")


def load_results(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_question_rows(results_doc: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in results_doc["results"]:
        response = case.get("response") or {}
        patch_info = response.get("patch_extraction_info") or {}
        chief = response.get("chief_response") or {}
        saved = response.get("saved_patches_info") or {}
        for q in case.get("questions", []):
            rows.append(
                {
                    "case_id": case["case_id"],
                    "image_path": case["image_path"],
                    "source_image": case.get("source_image"),
                    "question_id": q["question_id"],
                    "category": q["category"],
                    "question": q["question"],
                    "expected_answer": q["expected_answer"],
                    "process_status_code": case.get("status_code"),
                    "process_success": response.get("success"),
                    "selected_patch_count": patch_info.get("total_patches_extracted"),
                    "selected_patch_coords": patch_info.get("coordinates"),
                    "chief_patch_count": len(chief.get("top_k_patches") or []),
                    "chief_patch_coords": chief.get("top_k_patches"),
                    "saved_patch_files": saved.get("patch_files"),
                    "visualization_file": saved.get("visualization_file"),
                    "heatmap_file": saved.get("heatmap_file"),
                    "model_answer": "",
                    "grading": {
                        "score": None,
                        "notes": "",
                        "matched_expected": None,
                    },
                }
            )
    return rows


def write_json(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump({"questions": rows}, f, indent=2)
        f.write("\n")


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Imroze Question Sheet", ""]
    current_case = None
    for row in rows:
        if row["case_id"] != current_case:
            current_case = row["case_id"]
            lines.extend(
                [
                    f"## {row['case_id']}",
                    f"Image: `{row['image_path']}`",
                    f"Visualization: `{row['visualization_file']}`",
                    f"Saved patches: {', '.join(row['saved_patch_files'] or [])}",
                    "",
                ]
            )
        lines.extend(
            [
                f"### Q{row['question_id']} ({row['category']})",
                f"Question: {row['question']}",
                f"Expected: {row['expected_answer']}",
                "Model answer:",
                "",
                "Score:",
                "",
                "Notes:",
                "",
            ]
        )
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build question-level evaluation sheets from Imroze preprocessing results.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS), help="Path to imroze_vqa_results.json")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT), help="Path to write JSON question sheet")
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT), help="Path to write Markdown question sheet")
    args = parser.parse_args()

    rows = build_question_rows(load_results(Path(args.results)))
    write_json(rows, Path(args.json_out))
    write_markdown(rows, Path(args.md_out))
    print(f"Wrote {len(rows)} question rows")
    print(f"JSON: {args.json_out}")
    print(f"Markdown: {args.md_out}")


if __name__ == "__main__":
    main()
