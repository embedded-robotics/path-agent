from __future__ import annotations

import json
import os
import pathlib
import subprocess
from typing import List


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


class MedGemmaClient:
    """
    Local wrapper around the reconstructed tools/medgemma batch CLI.

    Env:
      MEDGEMMA_TOOL_DIR -> absolute path to the medgemma tool root
      MEDGEMMA_PYTHON   -> optional explicit interpreter inside that tool venv
      MEDGEMMA_MODEL    -> optional HF model id override
      MEDGEMMA_CONFIG   -> optional config file override
    """

    def __init__(self, tool_dir: str | None = None, model: str | None = None):
        default_tool_dir = _REPO_ROOT / "tools" / "medgemma"
        self.tool_dir = pathlib.Path(
            tool_dir or os.environ.get("MEDGEMMA_TOOL_DIR", str(default_tool_dir))
        ).resolve()
        self.python = pathlib.Path(
            os.environ.get("MEDGEMMA_PYTHON", str(self.tool_dir / ".venv" / "bin" / "python"))
        )
        self.model = model or os.environ.get("MEDGEMMA_MODEL", "")
        self.config = pathlib.Path(
            os.environ.get("MEDGEMMA_CONFIG", str(self.tool_dir / "config" / "default.yaml"))
        )

        if not self.tool_dir.exists():
            raise RuntimeError(
                "MedGemma tool directory does not exist. "
                f"Expected: {self.tool_dir}"
            )
        if not self.python.exists():
            raise RuntimeError(
                "MedGemma interpreter does not exist. "
                f"Expected: {self.python}"
            )
        if not self.config.exists():
            raise RuntimeError(
                "MedGemma config does not exist. "
                f"Expected: {self.config}"
            )

    def _run(self, question_file: str, image_folder: str, answers_file: str) -> None:
        qfile = str(pathlib.Path(question_file).resolve())
        img_folder = str(pathlib.Path(image_folder).resolve())
        afile = str(pathlib.Path(answers_file).resolve())

        cmd = [
            str(self.python), "-m", "src.graph.medgemma_eval",
            "--config", str(self.config),
            "--question-file", qfile,
            "--image-folder", img_folder,
            "--answers-file", afile,
        ]
        if self.model:
            cmd.extend(["--model", self.model])

        result = subprocess.run(
            cmd,
            check=False,
            cwd=str(self.tool_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "MedGemma command failed.\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

    def ask_batch(self, question_jsonl: str, image_folder: str, out_jsonl: str) -> List[str]:
        self._run(question_jsonl, image_folder, out_jsonl)
        texts: List[str] = []
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj.get("text", "").strip())
        return texts
