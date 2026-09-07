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

        try:
            result = subprocess.run(
                cmd, check=False, cwd=str(self.tool_dir), capture_output=True, text=True,
                timeout=int(os.environ.get("MEDGEMMA_SUBPROCESS_TIMEOUT_SECONDS", "600")),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("MedGemma subprocess timed out; no fallback backend was used.") from exc
        if result.returncode != 0:
            if "out of memory" in (result.stderr or "").lower():
                raise RuntimeError("MedGemma CUDA out of memory; reduce precision, quantize, or lower request size.")
            raise RuntimeError(
                "MedGemma command failed.\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

    @staticmethod
    def _validated_answers(requests: list[dict], answers_file: str) -> list[dict]:
        expected = {str(row["request_id"]): row for row in requests}
        answers: dict[str, dict] = {}
        provenance: tuple[str, str, str] | None = None
        try:
            lines = pathlib.Path(answers_file).read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise RuntimeError("MedGemma did not produce an answers JSONL file.") from exc
        for line_number, line in enumerate(lines, 1):
            try:
                answer = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"MedGemma produced malformed answers JSONL at line {line_number}.") from exc
            request_id = answer.get("request_id")
            if not isinstance(request_id, str) or request_id not in expected:
                raise RuntimeError("MedGemma produced an unknown response request_id.")
            if request_id in answers:
                raise RuntimeError("MedGemma produced a duplicate response request_id.")
            request = expected[request_id]
            if answer.get("patch_id") != request.get("patch_id") or answer.get("task_type") != request.get("task_type"):
                raise RuntimeError("MedGemma response patch/task metadata does not match its request.")
            if not isinstance(answer.get("text"), str) or not answer["text"].strip():
                raise RuntimeError("MedGemma produced an empty answer text.")
            if not isinstance(answer.get("model_id"), str) or not answer["model_id"].strip():
                raise RuntimeError("MedGemma response is missing model_id provenance.")
            if answer.get("precision") not in {"auto", "bf16", "fp16"}:
                raise RuntimeError("MedGemma response is missing valid precision provenance.")
            if answer.get("quantization") not in {"none", "8bit", "4bit"}:
                raise RuntimeError("MedGemma response is missing valid quantization provenance.")
            answer_provenance = (answer["model_id"], answer["precision"], answer["quantization"])
            if provenance is not None and answer_provenance != provenance:
                raise RuntimeError("MedGemma responses have inconsistent model/precision/quantization provenance.")
            provenance = answer_provenance
            answers[request_id] = answer
        if len(answers) != len(expected):
            raise RuntimeError("MedGemma responses are missing one or more request IDs.")
        return [answers[str(request["request_id"])] for request in requests]

    def ask_structured_batch(self, requests: list[dict], request_jsonl: str, answers_jsonl: str) -> list[dict]:
        if not requests:
            raise RuntimeError("MedGemma batch requires at least one request.")
        self._run(request_jsonl, ".", answers_jsonl)
        return self._validated_answers(requests, answers_jsonl)

    def ask_batch(self, question_jsonl: str, image_folder: str, out_jsonl: str) -> List[str]:
        self._run(question_jsonl, image_folder, out_jsonl)
        texts: List[str] = []
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj.get("text", "").strip())
        return texts
