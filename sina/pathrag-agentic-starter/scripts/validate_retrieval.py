import sys, os
sys.path.append("src"); os.environ["PYTHONPATH"]="src"

from pathrag.retrieval.store import load_bank, captions_for_label
from pathrag.siteid_client import CANON_LABELS

bank = load_bank()
missing = [lbl for lbl in CANON_LABELS if lbl not in bank]
if missing:
    print("WARNING: missing YAML entries:", missing)
else:
    print("All canonical labels present in YAML.")

for lbl in CANON_LABELS[:3]:
    print(lbl, "->", captions_for_label(lbl, 2))
