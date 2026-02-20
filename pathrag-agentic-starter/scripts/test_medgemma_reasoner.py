# scripts/test_medgemma_reasoner.py
import os, json
from pathlib import Path
from pathrag.agents.nodes.medgemma_reasoner import medgemma_reasoner

repo_root = Path(__file__).resolve().parents[1]
default_medgemma = repo_root / "tools" / "medgemma"
os.environ.setdefault("MEDGEMMA_TOOL_DIR", str(default_medgemma))
state = {
  "question": "Summarize pathology findings from these patches.",
  "patch_dir": str(default_medgemma / "data" / "images"),
  "k": 3
}
out = medgemma_reasoner(state)
print(json.dumps({
  "answer": out["stage4_answer"][:200],
  "captions": out["stage4_captions"][:3],
  "model": out["stage4_model"]
}, indent=2))
