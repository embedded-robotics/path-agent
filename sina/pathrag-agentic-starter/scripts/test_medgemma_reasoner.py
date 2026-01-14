# scripts/test_medgemma_reasoner.py
import os, json
from pathrag.agents.nodes.medgemma_reasoner import medgemma_reasoner

os.environ.setdefault("MEDGEMMA_TOOL_DIR", "/home/sina/projects/tools/medgemma")
state = {
  "question": "Summarize pathology findings from these patches.",
  "patch_dir": "/home/sina/projects/tools/medgemma/data/images",
  "k": 3
}
out = medgemma_reasoner(state)
print(json.dumps({
  "answer": out["stage4_answer"][:200],
  "captions": out["stage4_captions"][:3],
  "model": out["stage4_model"]
}, indent=2))
