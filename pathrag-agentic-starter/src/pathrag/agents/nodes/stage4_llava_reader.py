from __future__ import annotations

from typing import Dict, Any


def stage4_llava_reader(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    STUB implementation for local dev.

    Original design:
      - Read precomputed LLaVA-Med answers from /content/answers.jsonl
      - Attach them to the state for later stages.

    For local development:
      - We don't have /content/answers.jsonl.
      - LLaVA is either stubbed (USE_LLAVA=0) or called per-patch in later stages.
      - So we just mark Stage 4 as 'skipped' and pass the state through.

    On Colab / full setup:
      - You can restore the original implementation or write a real version that
        reads answers.jsonl and populates state["llava_answers"].
    """
    # You can log something if you want:
    print("[stage4_llava_reader] STUB: skipping Stage 4 (no /content/answers.jsonl).")

    # Optionally, mark in state that this stage was skipped
    state["stage4_status"] = "skipped_local"

    return state
