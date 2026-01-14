from __future__ import annotations
from typing import Dict

# Canonical labels (exactly match tissue_caps.yml keys)
CANON_LABELS = [
    "head_neck","breast","skin","lung","gastrointestinal","genitourinary",
    "gynecologic","soft_tissue","hematolymphoid",
]

# Optional synonyms → canonical
LABEL_SYNONYMS = {
    "gi": "gastrointestinal",
    "gu": "genitourinary",
    "heme": "hematolymphoid",
    "head&neck": "head_neck",
    "head-neck": "head_neck",
    "head neck": "head_neck",
}

def _canonicalize(label: str) -> str:
    k = (label or "").lower().strip()
    return LABEL_SYNONYMS.get(k, k)

def predict_tissue(image_path: str) -> Dict:
    """
    Deterministic placeholder until the real model lands:
    hashes the file name into one of the canonical labels.
    Swap this with Imroze's model or service later.
    """
    idx = sum(map(ord, image_path.lower())) % len(CANON_LABELS)
    raw = CANON_LABELS[idx]
    label = _canonicalize(raw)
    return {"label": label, "probs": {label: 0.65}, "attn_map_path": None}
