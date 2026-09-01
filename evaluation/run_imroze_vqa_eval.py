import argparse
import json
from pathlib import Path
from typing import Any

import requests


DEFAULT_CASES = Path(__file__).with_name("imroze_vqa_cases.json")
DEFAULT_OUT = Path(__file__).with_name("imroze_vqa_results.json")


def load_cases(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_case(api_url: str, case: dict[str, Any], top_n: int, top_k: int, save_patches: bool) -> dict[str, Any]:
    payload = {
        "image_path": case["image_path"],
        "svs_level": 2,
        "top_n": top_n,
        "save_patches": save_patches,
        "patch_size": 224,
        "anatomical_label": 1,
        "top_k": top_k,
    }
    response = requests.post(f"{api_url.rstrip('/')}/process", json=payload, timeout=1800)
    return {
        "case_id": case["case_id"],
        "image_path": case["image_path"],
        "source_image": case.get("source_image"),
        "payload": payload,
        "status_code": response.status_code,
        "response": response.json() if response.content else None,
        "questions": case["questions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Imroze JPG test cases through the combined API.")
    parser.add_argument("--api-url", default="http://localhost:8003", help="Combined API base URL")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="Path to JSON case file")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Path to write JSON results")
    parser.add_argument("--top-n", type=int, default=3, help="Top N histocartography patches")
    parser.add_argument("--top-k", type=int, default=5, help="Top K CHIEF patches")
    parser.add_argument("--save-patches", action="store_true", help="Ask combined API to save patch files")
    args = parser.parse_args()

    cases_doc = load_cases(Path(args.cases))
    results = []
    for case in cases_doc["cases"]:
        result = run_case(
            api_url=args.api_url,
            case=case,
            top_n=args.top_n,
            top_k=args.top_k,
            save_patches=args.save_patches,
        )
        results.append(result)
        print(f"{case['case_id']}: HTTP {result['status_code']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
        f.write("\n")
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
