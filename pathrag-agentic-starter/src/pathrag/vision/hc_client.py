from __future__ import annotations

import json
import os
import pathlib
import subprocess
from typing import Any


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


class HistocartographyClient:
    """
    Local wrapper around the tools/histocartography CLI.

    Env:
      HISTOCARTOGRAPHY_TOOL_DIR -> tool root containing config/default.yaml
      HISTOCARTOGRAPHY_PYTHON   -> interpreter for the HC tool env
      HISTOCARTOGRAPHY_CONFIG   -> optional config override
    """

    def __init__(self, tool_dir: str | None = None):
        default_tool_dir = _REPO_ROOT / "tools" / "histocartography"
        self.tool_dir = pathlib.Path(
            tool_dir or os.environ.get("HISTOCARTOGRAPHY_TOOL_DIR", str(default_tool_dir))
        ).resolve()
        self.python = pathlib.Path(
            os.environ.get(
                "HISTOCARTOGRAPHY_PYTHON",
                str(self.tool_dir / ".venv" / "bin" / "python"),
            )
        )
        self.config = pathlib.Path(
            os.environ.get("HISTOCARTOGRAPHY_CONFIG", str(self.tool_dir / "config" / "default.yaml"))
        )

        if not self.tool_dir.exists():
            raise RuntimeError(f"Histocartography tool directory does not exist: {self.tool_dir}")
        if not self.python.exists():
            raise RuntimeError(f"Histocartography interpreter does not exist: {self.python}")
        if not self.config.exists():
            raise RuntimeError(f"Histocartography config does not exist: {self.config}")

    def extract_top_patches(self, image_path: str, top_n: int) -> dict[str, Any]:
        image_path = str(pathlib.Path(image_path).resolve())
        out_path = self.tool_dir / "artifacts" / "answer" / "hc_out.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.python),
            "-m",
            "src.graph.hc_eval",
            "--config",
            str(self.config),
            "--image-path",
            image_path,
            "--top-n",
            str(int(top_n)),
            "--out",
            str(out_path),
        ]

        result = subprocess.run(
            cmd,
            check=False,
            cwd=str(self.tool_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Histocartography command failed.\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        with out_path.open("r", encoding="utf-8") as f:
            return json.load(f)
