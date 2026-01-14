from pathrag.agents.langgraph_app import build_graph

def _get(node: dict | None, key: str, default=None):
    return node.get(key) if isinstance(node, dict) else default

if __name__ == "__main__":
    app = build_graph()
    init = {
        "image_path": "sample_he.png",
        "question": "What are a few well-developed cell nests with?",
        "mode": "answer",
        "top_k": 3,
        "max_rounds": 1,
        "round_ix": 0,
    }

    last = None
    # MemorySaver requires a thread_id in config.configurable
    for s in app.stream(init, config={"configurable": {"thread_id": "run-1"}}):
        last = s

    # Handle both shapes: last["node_name"][key] OR last[key]
    fuse_state     = _get(last, "fuse", last)          # final stage (preferred)
    identify_state = _get(last, "identify", last)      # Stage 3
    tile_state     = _get(last, "tile_rank", last)     # Stages 1–2

    final_answer   = _get(fuse_state, "final_answer") or _get(last, "final_answer")
    patches        = _get(tile_state, "patches") or _get(last, "patches")
    captions       = _get(identify_state, "full_captions") or _get(last, "full_captions")
    label          = _get(identify_state, "subpath_label") or _get(last, "subpath_label")

    print("FINAL ANSWER:", final_answer)
    print("PATCHES:", patches)           # fused Top-K from Stages 1–2
    print("CAPTIONS:", captions)         # retrieval from Stage 3

    # Friendly summary
    print("\n=== SUMMARY ===")
    print("Site label:", label)
    if isinstance(captions, list):
        for c in captions:
            print(" -", c)
