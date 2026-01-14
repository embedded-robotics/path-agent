from collections import Counter
import re
from typing import List, Tuple

WORD = re.compile(r"""[A-Za-z0-9_]+""")


def _tokens(text: str):
    return [t.lower() for t in WORD.findall(text or "")]


def overlap_score(q: str, s: str) -> float:
    q_tokens = Counter(_tokens(q))
    s_tokens = Counter(_tokens(s))
    # simple overlap: sum of min counts normalized by query length
    inter = sum((q_tokens & s_tokens).values())
    denom = sum(q_tokens.values()) or 1
    return inter / denom


def rank(question: str, sentences: List[str], topk: int = 10) -> List[Tuple[int, float, str]]:
    scored = [(i, overlap_score(question, s), s) for i, s in enumerate(sentences)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topk]
