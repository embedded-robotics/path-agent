"""
File: src/pathrag/agents/langgraph_app.py

Purpose
-------
LangGraph orchestration for the Path-RAG pipeline. This graph wires the seven stages:
  (1) tiling, (2) HC/CHEIF ranking & fusion, (3) labeling + retrieval,
  (4) ROI & patch agents, (5) critique loop, (6) question-aware re-rank,
  (7) final fusion.

Design notes
------------
- State (TypedDict) is the single source of truth passed between nodes.
- Nodes are small, idempotent functions: read fields from state → write results back.
- We use MemorySaver() to enable streaming, checkpointing, and replay.
  => Always pass a `thread_id` when calling `app.stream(...)` from the runner.

Swap policy
-----------
Keep node signatures and state keys stable. Replace internals of the tool functions
(in pathrag.agents.tools) as you bring real models/services online.
"""

from __future__ import annotations

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
# at top with others
from pathrag.agents.nodes.stage4_llava_reader import stage4_llava_reader


# Domain tools (stage implementations; currently mocked in tools.py)
from pathrag.agents.tools import (
    Patch,
    # Stage 1–2
    tile_image, histocartography_rank, cheif_rank, common_patches,
    # Stage 3
    identify_subpathology, retrieve_subpath_captions,
    # Stage 4
    roi_agent_describe, patch_agent_contribution,
    # Stage 5
    critique_round,
    # Stage 6
    rerank_for_question,
    # Stage 7
    fuse_answer,
)

# (Optional) lightweight logger; safe to remove if you prefer silent runs
try:
    from pathrag.utils.logging import get_logger
    logger = get_logger("pathrag.langgraph")
except Exception:  # pragma: no cover
    class _Null:
        def info(self, *_a, **_k): pass
    logger = _Null()


# ======================
# State schema (contract)
# ======================
class PathRAGState(TypedDict):
    # Inputs
    image_path: str          # path to image / WSI entry point
    question: str            # user question or "describe" for caption mode
    mode: str                # "answer" | "description"
    top_k: int               # K for fused/common patches and final selection
    max_rounds: int          # max critique rounds
    round_ix: int            # current critique round (incremented by node)

    # Stage 1–2 artifacts
    tiles: List[Dict[str, Any]]       # dense grid (Patch as dict)
    hc_rank: List[Dict[str, Any]]     # HC-ranked patches (Patch as dict)
    cheif_rank: List[Dict[str, Any]]  # CHEIF-ranked patches (Patch as dict)
    patches: List[Dict[str, Any]]     # fused Top-K consensus (Patch as dict)

    # Stage 3 artifacts
    subpath_label: str                # normalized label (e.g., "scc")
    full_captions: List[str]          # short knowledge snippets

    # Stage 4 artifacts (aligned by index with `patches`)
    roi_useful: List[bool]
    roi_desc: List[str]
    patch_summaries: List[str]

    # Stage 6 artifacts
    chosen_idx: List[int]             # indices into `patches` for final K

    # Stage 7 artifact
    final_answer: str


# =======================
# Stage nodes (pure-ish)
# =======================

# ---- Stages 1–2: tiling, HC/CHEIF, fused Top-K ----
def n_tile_and_rank(state: PathRAGState) -> PathRAGState:
    """
    Stage 1–2: produce consensus Top-K patches.

    Pre:
      - state["image_path"] is valid
      - state["top_k"] is set

    Post:
      - tiles, hc_rank, cheif_rank are populated (lists of Patch-as-dict)
      - patches contains fused Top-K consensus (Patch-as-dict, len == top_k)
    """
    logger.info("Stage 1–2 start")
    tiles = tile_image(state["image_path"])
    hc = histocartography_rank(state["image_path"], tiles)
    ch = cheif_rank(state["image_path"], tiles)
    fused = common_patches(hc, ch, state["top_k"])

    state["tiles"] = [t.__dict__ for t in tiles]
    state["hc_rank"] = [p.__dict__ for p in hc]
    state["cheif_rank"] = [p.__dict__ for p in ch]
    state["patches"] = [p.__dict__ for p in fused]
    logger.info(f"Stage 1–2 done: fused={len(state['patches'])}")
    return state


# ---- Stage 3: labeling + retrieval ----
def n_identify_and_retrieve(state: PathRAGState) -> PathRAGState:
    """
    Stage 3: identify sub-pathology and retrieve textual knowledge.

    Pre:
      - image_path exists
    Post:
      - subpath_label (str), full_captions (list[str])
    """
    logger.info("Stage 3 start")
    label = identify_subpathology(state["image_path"])
    caps = retrieve_subpath_captions(label, top_m=5)
    state["subpath_label"] = label
    state["full_captions"] = caps
    logger.info(f"Stage 3 done: label={label}, caps={len(caps)}")
    return state


# ---- Stage 4: ROI + patch agents (per-patch) ----
def n_roi_and_patch_agents(state: PathRAGState) -> PathRAGState:
    """
    Stage 4: for each fused patch, run ROI agent and patch contribution agent.

    Pre:
      - patches (Top-K) from Stage 2
      - question, full_captions from Stage 3

    Post:
      - roi_useful, roi_desc, patch_summaries (aligned to `patches`)
    """
    logger.info("Stage 4 start")
    patches = [Patch(**p) for p in state["patches"]]
    roi_useful, roi_desc, patch_summ = [], [], []

    for p in patches:
        roi = roi_agent_describe(p, state["question"])
        roi_useful.append(bool(roi["useful"]))
        roi_desc.append(str(roi["description"]))
        patch_summ.append(
            patch_agent_contribution(p, state["question"], state["full_captions"])
        )

    state["roi_useful"] = roi_useful
    state["roi_desc"] = roi_desc
    state["patch_summaries"] = patch_summ
    logger.info(f"Stage 4 done: patches={len(patches)}")
    return state


# ---- Stage 5: critique loop (one round) ----
def n_critique_round(state: PathRAGState) -> PathRAGState:
    """
    Stage 5: apply exactly one critique pass to patch summaries.

    Pre:
      - patch_summaries, roi_desc populated
      - round_ix < max_rounds

    Post:
      - patch_summaries updated
      - round_ix incremented by 1
    """
    logger.info(f"Stage 5 round {state['round_ix']} start")
    state["patch_summaries"] = critique_round(
        state["patch_summaries"], state["roi_desc"], state["question"], state["round_ix"]
    )
    state["round_ix"] += 1
    logger.info(f"Stage 5 round done → round_ix={state['round_ix']}")
    return state


def route_more_critiques(state: PathRAGState) -> str:
    """
    Router for Stage 5 loop.
    Return the next node key:
      - "critique" to continue another round
      - "rerank" to exit loop and proceed
    """
    return "critique" if state["round_ix"] < state["max_rounds"] else "rerank"


# ---- Stage 6: question-aware Top-K selection ----
def n_rerank_and_choose(state: PathRAGState) -> PathRAGState:
    """
    Stage 6: re-rank summaries against the question and choose Top-K indices.

    Pre:
      - patches, patch_summaries aligned
      - top_k set

    Post:
      - chosen_idx: list[int] (indices into `patches`)
    """
    logger.info("Stage 6 start")
    patches = [Patch(**p) for p in state["patches"]]
    idx = rerank_for_question(
        patches, state["patch_summaries"], state["question"], k=state["top_k"]
    )
    state["chosen_idx"] = idx
    logger.info(f"Stage 6 done: chosen_idx={idx}")
    return state


# ---- Stage 7: final fusion (AI pathologist) ----
def n_fuse(state: PathRAGState) -> PathRAGState:
    """
    Stage 7: synthesize final answer (or description) from chosen patches.

    Pre:
      - chosen_idx computed
      - subpath_label, patch_summaries set
    Post:
      - final_answer: str
    """
    logger.info("Stage 7 start")
    patches = [Patch(**p) for p in state["patches"]]
    chosen = [(patches[i], state["patch_summaries"][i]) for i in state["chosen_idx"]]
    state["final_answer"] = fuse_answer(
        state["question"], state["subpath_label"], chosen, mode=state["mode"]
    )
    logger.info("Stage 7 done")
    return state


# =========================
# Build and compile the DAG
# =========================
def build_graph():
    """
    Build the Path-RAG LangGraph and compile with in-memory checkpoints.

    Usage:
        app = build_graph()
        init = { ... all required keys ... }
        for step in app.stream(init, config={"configurable": {"thread_id": "run-1"}}):
            last = step
        print(last["fuse"]["final_answer"])
    """
    g = StateGraph(PathRAGState)

    # Register nodes
    g.add_node("tile_rank", n_tile_and_rank)           # Stages 1–2
    g.add_node("identify", n_identify_and_retrieve)    # Stage 3
    g.add_node("stage4", stage4_llava_reader)  # stage3 -> stage4 -> stage5
    g.add_node("critique", n_critique_round)           # Stage 5 (loop)
    g.add_node("rerank", n_rerank_and_choose)          # Stage 6
    g.add_node("fuse", n_fuse)                         # Stage 7

    # Wire edges
    g.add_edge(START, "tile_rank")
    g.add_edge("tile_rank", "identify")
    g.add_edge("identify", "stage4")
    g.add_edge("stage4", "critique")

    # Loop: critique → (critique | rerank)
    g.add_conditional_edges(
        "critique",
        route_more_critiques,
        {"critique": "critique", "rerank": "rerank"},
    )

    g.add_edge("rerank", "fuse")
    g.add_edge("fuse", END)

    # Checkpointer enables stream/replay (requires a thread_id at runtime)
    return g.compile(checkpointer=MemorySaver())
