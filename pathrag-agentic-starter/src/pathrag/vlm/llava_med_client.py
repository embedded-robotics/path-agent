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
      LLMED_CONV_MODE -> conversation template; defaults to "mistral_instruct"
      LLMED_LOAD_8BIT -> "1" to request 8-bit load
      LLMED_LOAD_4BIT -> "1" to request 4-bit load
    """

    def __init__(self, repo: str | None = None, model: str | None = None):
        # Even in stub mode we compute these, but only enforce them when USE_LLAVA=1
        self.repo = pathlib.Path(repo or os.environ.get("LLMED_REPO", "")).resolve()
        self.model = model or os.environ.get("LLMED_MODEL", "")
        self.conv_mode = os.environ.get("LLMED_CONV_MODE", "mistral_instruct")
        self.load_8bit = os.environ.get("LLMED_LOAD_8BIT", "0") == "1"
        self.load_4bit = os.environ.get("LLMED_LOAD_4BIT", "0") == "1"
        # Do not resolve the venv python symlink: resolving collapses it to the
        # base interpreter and drops the virtualenv context.
        self.python = pathlib.Path(
            os.environ.get("LLMED_PYTHON", str(self.repo / ".venv" / "bin" / "python"))
        )

        if USE_LLAVA:
            if not self.repo.exists():
                raise RuntimeError(
                    "Real LLaVA mode enabled (USE_LLAVA=1) but LLMED_REPO "
                    "does not exist. Set LLMED_REPO to your LLaVA-Med repo path."
                )
            if not self.python.exists():
                raise RuntimeError(
                    "Real LLaVA mode enabled (USE_LLAVA=1) but LLMED_PYTHON "
                    f"does not exist: {self.python}"
                )
            if not self.model:
                raise RuntimeError(
                    "Real LLaVA mode enabled (USE_LLAVA=1) but LLMED_MODEL "
                    "is empty. Set it to your HF model id or local checkpoint path."
                )

    def _run(self, question_file: str, image_folder: str, answers_file: str):
        # 🔹 Make ALL paths absolute, based on the current (PathRAG) working dir
        qfile = str(pathlib.Path(question_file).resolve())
        img_folder = str(pathlib.Path(image_folder).resolve())
        afile = str(pathlib.Path(answers_file).resolve())

        cmd = [
            str(self.python), "-m", "llava.eval.model_vqa",
            "--model-path", self.model,
            "--conv-mode", self.conv_mode,
            "--question-file", qfile,
            "--image-folder", img_folder,
            "--answers-file", afile,
        ]
        if self.load_8bit:
            cmd.append("--load-8bit")
        if self.load_4bit:
            cmd.append("--load-4bit")

        # Run from inside the LLaVA-Med repo so `llava.*` imports work
        result = subprocess.run(
            cmd,
            check=False,
            cwd=str(self.repo),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "LLaVA-Med command failed.\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )


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
