from __future__ import annotations

import json
import os
import pathlib
import subprocess
from typing import Any, List, Sequence


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


class ChiefClient:
    """
    Local wrapper around the tools/chief CLI.

    Env:
      CHIEF_TOOL_DIR -> tool root containing config/default.yaml
      CHIEF_PYTHON   -> interpreter for the CHIEF tool venv
      CHIEF_CONFIG   -> optional config override
    """

    def __init__(self, tool_dir: str | None = None):
        default_tool_dir = _REPO_ROOT / "tools" / "chief"
        self.tool_dir = pathlib.Path(
            tool_dir or os.environ.get("CHIEF_TOOL_DIR", str(default_tool_dir))
        ).resolve()
        self.python = pathlib.Path(
            os.environ.get("CHIEF_PYTHON", str(self.tool_dir / ".venv" / "bin" / "python"))
        )
        self.config = pathlib.Path(
            os.environ.get("CHIEF_CONFIG", str(self.tool_dir / "config" / "default.yaml"))
        )

        if not self.tool_dir.exists():
            raise RuntimeError(
                "CHIEF tool directory does not exist. "
                f"Expected: {self.tool_dir}"
            )
        if not self.python.exists():
            raise RuntimeError(
                "CHIEF interpreter does not exist. "
                f"Expected: {self.python}"
            )
        if not self.config.exists():
            raise RuntimeError(
                "CHIEF config does not exist. "
                f"Expected: {self.config}"
            )

    def extract_top_k_patches(
        self,
        image_path: str,
        top_k: int,
        valid_bounds: Sequence[Sequence[int]] | None = None,
    ) -> dict[str, Any]:
        image_path = str(pathlib.Path(image_path).resolve())
        out_path = self.tool_dir / "artifacts" / "answer" / "chief_out.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd: List[str] = [
            str(self.python), "-m", "src.graph.chief_eval",
            "--config", str(self.config),
            "--image-path", image_path,
            "--top-k", str(int(top_k)),
            "--out", str(out_path),
        ]
        if valid_bounds:
            cmd.extend(["--valid-bounds-json", json.dumps(valid_bounds)])

        result = subprocess.run(
            cmd,
            check=False,
            cwd=str(self.tool_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "CHIEF command failed.\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        with out_path.open("r", encoding="utf-8") as f:
            return json.load(f)
