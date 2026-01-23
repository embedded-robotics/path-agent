from __future__ import annotations
import subprocess, json, os, pathlib
from typing import List

# Env flag: real LLaVA (1) vs stub (0)
USE_LLAVA = os.environ.get("USE_LLAVA", "0") == "1"


class LlavaMedClient:
    """
    Wrapper around LLaVA-Med's JSONL eval, with a stub fallback.

    Env:
      USE_LLAVA   -> "1" to call real LLaVA-Med via CLI, "0" to use stub mode
      LLMED_REPO  -> (real mode only) absolute path to cloned LLaVA-Med repo
      LLMED_MODEL -> (real mode only) HF id or local model path,
                     e.g. "microsoft/llava-med-v1.5-mistral-7b"
    """

    def __init__(self, repo: str | None = None, model: str | None = None):
        # Even in stub mode we compute these, but only enforce them when USE_LLAVA=1
        self.repo = pathlib.Path(repo or os.environ.get("LLMED_REPO", "")).resolve()
        self.model = model or os.environ.get("LLMED_MODEL", "")

        if USE_LLAVA:
            if not self.repo.exists():
                raise RuntimeError(
                    "Real LLaVA mode enabled (USE_LLAVA=1) but LLMED_REPO "
                    "does not exist. Set LLMED_REPO to your LLaVA-Med repo path."
                )
            if not self.model:
                raise RuntimeError(
                    "Real LLaVA mode enabled (USE_LLAVA=1) but LLMED_MODEL "
                    "is empty. Set it to your HF model id or local checkpoint path."
                )

    def _run(self, question_file: str, image_folder: str, answers_file: str):
        """
        Call the official LLaVA-Med CLI (llava.eval.model_vqa) inside the repo.
        Only used when USE_LLAVA=1.
        """
        cmd = [
            "python",
            "-m", "llava.eval.model_vqa",
            "--model-path", self.model,
            "--question-file", question_file,
            "--image-folder", image_folder,
            "--answers-file", answers_file,
        ]
        subprocess.run(cmd, check=True, cwd=str(self.repo))

    def ask_batch(self, question_jsonl: str, image_folder: str, out_jsonl: str) -> List[str]:
        """
        Run a batch of questions.

        - If USE_LLAVA=1:
            * call LLaVA-Med CLI and return real answers from out_jsonl.
        - If USE_LLAVA=0:
            * skip CLI, generate cheap stub answers, and still write out_jsonl
              so downstream code (Stage 4/5/6/7) sees the same JSONL structure.
        """

        # --- Stub mode: used for local dev, no heavy dependencies ---
        if not USE_LLAVA:
            records: list[dict] = []
            with open(question_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))

            texts: List[str] = []
            for rec in records:
                prompt = rec.get("text") or rec.get("prompt") or rec.get("question") or ""
                # Simple deterministic stub: echo part of the prompt
                stub_answer = f"[llava-stub] {prompt[:120]}"
                texts.append(stub_answer)

            # Write stub outputs so the rest of the pipeline can still read JSONL
            with open(out_jsonl, "w", encoding="utf-8") as f:
                for rec, ans in zip(records, texts):
                    out = {
                        "question_id": rec.get("question_id"),
                        "image": rec.get("image"),
                        "text": ans,
                    }
                    f.write(json.dumps(out) + "\n")

            return texts

        # --- Real LLaVA-Med CLI mode: used on Colab / full setup ---
        self._run(question_jsonl, image_folder, out_jsonl)
        texts: List[str] = []
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj.get("text", "").strip())
        return texts
