from __future__ import annotations
import subprocess, json, os, pathlib
from typing import List

class LlavaMedClient:
    """
    Wrapper around LLaVA-Med's JSONL eval.
    Env:
      LLMED_REPO  -> absolute path to cloned LLaVA-Med repo
      LLMED_MODEL -> HF id or local model path, e.g. "microsoft/llava-med-v1.5"
    """
    def __init__(self, repo: str | None = None, model: str | None = None):
        self.repo = pathlib.Path(repo or os.environ.get("LLMED_REPO", "")).resolve()
        self.model = model or os.environ.get("LLMED_MODEL", "")
        if not self.repo.exists():
            raise RuntimeError("Set LLMED_REPO to your LLaVA-Med repo path")
        if not self.model:
            raise RuntimeError("Set LLMED_MODEL to a valid model id/path")

    def _run(self, question_file: str, image_folder: str, answers_file: str):
        cmd = [
            "python", "-m", "llava.eval.model_vqa",
            "--model-path", self.model,
            "--question-file", question_file,
            "--image-folder", image_folder,
            "--answers-file", answers_file,
        ]
        subprocess.run(cmd, check=True, cwd=str(self.repo))

    def ask_batch(self, question_jsonl: str, image_folder: str, out_jsonl: str) -> List[str]:
        self._run(question_jsonl, image_folder, out_jsonl)
        texts: List[str] = []
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj.get("text", "").strip())
        return texts
