from __future__ import annotations
import os, json, subprocess
from pathlib import Path

def _resolve_tool_dir() -> Path:
    # Prefer explicit env var
    if "MEDGEMMA_TOOL_DIR" in os.environ:
        return Path(os.environ["MEDGEMMA_TOOL_DIR"]).expanduser().resolve()
    # Fallback: assume <repo-root>/tools/medgemma
    # __file__ = <repo>/src/pathrag/agents/nodes/medgemma_reasoner.py
    # parents[4] = <repo-root>
    return Path(__file__).resolve().parents[4] / "tools" / "medgemma"

def medgemma_reasoner(state: dict) -> dict:
    question     = state["question"]
    image_folder = state["patch_dir"]
    k            = int(state.get("k", 3))
    timeout_s    = int(state.get("timeout_s", 1200))

    tool_dir = _resolve_tool_dir()
    py  = tool_dir / ".venv/bin/python"
    cfg = tool_dir / "config/default.yaml"
    out = tool_dir / "artifacts" / "answer" / "out.json"

    if not py.exists():
        raise FileNotFoundError(f"MedGemma venv not found: {py}\nSet MEDGEMMA_TOOL_DIR or create the venv there.")

    env = os.environ.copy()
    # env["LOAD_4BIT"] = "1"  # enable if VRAM is tight

    cmd = [
        str(py), "-m", "src.graph.medgemma_run",
        "--config", str(cfg),
        "--question", question,
        "--image-folder", str(image_folder),
        "--out", str(out),
        "--k", str(k),
    ]

    r = subprocess.run(
    cmd,
    env=env,
    capture_output=True,
    text=True,
    timeout=timeout_s,
    cwd=str(tool_dir),
    )

    if r.returncode != 0:
        raise RuntimeError(f"[MedGemma tool failed]\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")

    payload = json.loads(out.read_text())
    state["stage4_answer"]   = payload.get("answer", "")
    state["stage4_captions"] = payload.get("captions", [])
    state["stage4_model"]    = payload.get("model", "google/gemma-2b-it")
    return state
