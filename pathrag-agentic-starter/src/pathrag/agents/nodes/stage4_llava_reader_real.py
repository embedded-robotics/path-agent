# src/pathrag/agents/nodes/stage4_llava_reader.py
from __future__ import annotations
import os, json, glob
from pathlib import Path
from typing import List, Dict, Any

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL: {path}")
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    if not out:
        raise ValueError(f"No records found in {path}")
    return out

def _merge_answers_with_questions(answers: List[Dict[str, Any]], questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map LLaVA-Med answers (question_id, text) to filenames using questions.jsonl (question_id -> image)."""
    q_by_id = {q.get("question_id"): q for q in questions}
    merged = []
    for r in answers:
        qid = r.get("question_id")
        q = q_by_id.get(qid, {})
        merged.append({
            "image": q.get("image"),
            "question": r.get("prompt") or q.get("text"),
            "answer": r.get("text"),
            "model": r.get("model_id"),
            "qid": qid,
        })
    return merged

def _filename_from_patch(p) -> str | None:
    # Handle dataclass/object or dict
    if isinstance(p, dict):
        val = p.get("filename") or p.get("path") or p.get("file") or p.get("name")
        return os.path.basename(str(val)) if val else None
    for attr in ("filename", "path", "file", "name"):
        if hasattr(p, attr):
            val = getattr(p, attr)
            if val:
                return os.path.basename(str(val))
    return None

def stage4_llava_reader(state: dict) -> dict:
    """
    Stage-4 reader: load LLaVA-Med outputs from JSONL and fill patch_captions.
    Respects either:
      - answers_merged.jsonl → records with {image, question, answer, ...}
      - answers.jsonl (raw) + patch_questions.jsonl → auto-merged
    """
    # where the artifacts live; you can override with env vars
    ans_path = Path(os.environ.get("STAGE4_ANSWERS_JSONL", "/content/answers_merged.jsonl"))
    raw_ans_path = Path(os.environ.get("STAGE4_RAW_ANSWERS_JSONL", "/content/answers.jsonl"))
    q_path = Path(os.environ.get("STAGE4_QUESTIONS_JSONL", "/content/patch_questions.jsonl"))

    records: List[Dict[str, Any]] = []
    if ans_path.exists():
        records = _load_jsonl(ans_path)  # contains image/question/answer already
    else:
        # fall back to raw answers + questions
        answers = _load_jsonl(raw_ans_path)
        questions = _load_jsonl(q_path)
        records = _merge_answers_with_questions(answers, questions)

    # Build filename -> answer map
    by_name: Dict[str, str] = {}
    model_name = None
    for r in records:
        img = r.get("image")
        ans = r.get("answer")
        if img and isinstance(ans, str):
            by_name[os.path.basename(img)] = ans
        if r.get("model"):
            model_name = r.get("model")

    # Align to the graph's patch ordering if present
    patches = state.get("patches") or []
    captions: List[str] = []
    if patches:
        for p in patches:
            fname = _filename_from_patch(p)
            if fname:
                captions.append(by_name.get(fname, "(no answer)"))
            else:
                captions.append("(no filename on patch)")
    else:
        # fall back: lexicographic order of files in the patch_dir
        patch_dir = Path(state.get("patch_dir", "."))
        files = sorted(
            [os.path.basename(x) for x in glob.glob(str(patch_dir / "*"))
             if x.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
        )
        captions = [by_name.get(f, "(no answer)") for f in files]

    state["patch_captions"] = captions
    state["stage4_backend"] = "llava-med"
    if model_name:
        state["stage4_model"] = model_name
    state["stage4_records"] = records  # keep for debugging/traceability
    return state
