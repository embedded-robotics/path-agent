from __future__ import annotations
from typing import List, Dict, Any
import pathlib, yaml

# Global in-memory cache (lazy-loaded)
_BANK: Dict[str, list] | None = None

def _bank_path() -> pathlib.Path:
    # resolves relative to this file so it works in Colab and local
    here = pathlib.Path(__file__).resolve().parent
    return here / "banks" / "tissue_caps.yml"

def load_bank(path: str | None = None) -> Dict[str, list]:
    """Load the YAML caption bank once into memory."""
    global _BANK
    if _BANK is not None:
        return _BANK
    p = pathlib.Path(path) if path else _bank_path()
    if not p.exists():
        # safe default if file is missing
        _BANK = {}
        return _BANK
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # normalize keys to lowercase, lists to list[str]
    norm = {str(k).lower(): [str(x).strip() for x in (v or [])] for k, v in data.items()}
    _BANK = norm
    return _BANK

def captions_for_label(label: str, top_m: int = 5) -> List[str]:
    """Return top-M captions for a canonical label; fallback if missing."""
    bank = load_bank(None)
    key = (label or "").lower().strip()
    caps = bank.get(key, [])
    if not caps:
        return [f"{label}: no bank entry yet — use general pathology heuristics."]
    return caps[:top_m]
