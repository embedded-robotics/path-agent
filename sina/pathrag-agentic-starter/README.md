# PathRAG Agentic Starter

A GitHub-ready, minimal starter for building an **agentic architecture** around **Path-RAG** using two frameworks:

- **LangGraph (LangChain)** — production-grade, stateful graph orchestration.
- **AutoGen** — fast multi-agent prototyping (planner/critic/executor).

This repo ships with:
- A clean Python package layout (`src/`).
- Mock **tools** that simulate HistoCartography, LLaVA-Med, critique/re-rank/fusion, and a text **reasoner**.
- Two runnable templates:
  - `scripts/run_langgraph.py`
  - `scripts/run_autogen.py`

> Swap the mocked tool functions with your real implementations (REST calls or Python libs). :contentReference[oaicite:0]{index=0}

---

## Quickstart

```bash
# 1) Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Set your API key (if you plan to call OpenAI for the reasoner)
cp src/pathrag/workflows/config/.env.example .env
# then edit .env and add OPENAI_API_KEY=...

# 4) Run either template
python scripts/run_langgraph.py

### Expected output
FINAL ANSWER: [mocked] keratinization (example)
``` :contentReference[oaicite:1]{index=1}

---

## Repo structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── pathrag/
│       ├── __init__.py
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── langgraph_app.py     # LangGraph template
│       │   ├── autogen_app.py       # AutoGen template
│       │   └── tools.py             # Mocked tools (swap with real ones)
│       ├── workflows/
│       │   ├── config/
│       │   │   └── .env.example
│       │   └── examples/
│       └── utils/
│           ├── __init__.py
│           └── logging.py
├── scripts/
│   ├── run_langgraph.py
│   └── run_autogen.py
└── tests/
    └── test_smoke.py
``` :contentReference[oaicite:2]{index=2}

---

## Path-RAG LangGraph pipeline (stages & knobs)

The LangGraph app wires the full 7-stage Path-RAG pipeline:

1. **Stages 1–2 — tiling + HC/CHEIF fusion**  
   - Tiles the image, ranks patches with HistoCartography and CHEIF, and produces a **consensus Top-K** patch set.  
   - Controlled by `state["top_k"]`. Only these K patches are carried forward.   

2. **Stage 3 — labeling + retrieval**  
   - Identifies a sub-pathology label (e.g., `"scc"`) and retrieves short textual snippets (`full_captions`) for that label. :contentReference[oaicite:4]{index=4}  

3. **Stage 4 — ROI & patch agents**  
   - For each of the K patches:  
     - An ROI agent decides if the patch is useful and explains *why*.  
     - A patch contribution agent produces a short summary (`patch_summaries[i]`) conditioned on the question + retrieved captions.   

4. **Stage 5 — critique loop**  
   - A critique node refines the patch summaries once per loop iteration.  
   - Loop routing is controlled by:
     - `max_rounds`: maximum critique passes allowed.
     - `round_ix`: current critique round (incremented inside the node).
   - The router `route_more_critiques` decides whether to:
     - go back to `"critique"` (another round), or  
     - exit to `"rerank"` (Stage 6) once `round_ix >= max_rounds`.   

5. **Stage 6 — question-aware selection**  
   - Re-scores all K patch summaries against the question and returns `chosen_idx` — the indices of patches to feed into the final fusion step.  
   - `top_k` is also used here to control how many patches you keep in the final subset.   

6. **Stage 7 — fusion (AI pathologist)**  
   - Combines:
     - the question,
     - the sub-pathology label,
     - the selected patches (`chosen_idx` + their summaries),
   - and produces `final_answer`. :contentReference[oaicite:8]{index=8}  

### Key state fields

The LangGraph state is defined as a `TypedDict` called `PathRAGState` and is the **contract** between all nodes:   

- **Inputs**
  - `image_path: str` — path to the slide / image.
  - `question: str` — VQA question, or `"describe"` for captioning mode.
  - `mode: str` — `"answer"` or `"description"`.
  - `top_k: int` — **K = number of fused patches** we keep and reason over for this question (Stages 1–2, 4, 5, 6 all operate on this set).
  - `max_rounds: int` — maximum number of critique passes in Stage 5.
  - `round_ix: int` — current critique round (start at 0).

- **Artifacts**
  - `tiles`, `hc_rank`, `cheif_rank`, `patches` — tiling and ranking outputs (patches stored as dicts).
  - `subpath_label`, `full_captions` — Stage-3 outputs.
  - `roi_useful`, `roi_desc`, `patch_summaries` — Stage-4 outputs aligned to `patches`.
  - `chosen_idx` — indices of selected patches after Stage-6.
  - `final_answer` — final long-form answer / description (Stage-7).

### Minimal example: running the graph in Python

```python
from pathrag.agents.langgraph_app import build_graph

app = build_graph()

init_state = {
    "image_path": "examples/sample_slide.png",
    "question": "What are the main pathological findings?",
    "mode": "answer",      # or "description"
    "top_k": 3,            # K patches to carry through the pipeline
    "max_rounds": 1,       # critique passes in Stage-5
    "round_ix": 0,         # must start at 0
}

final_step = None
for step in app.stream(init_state, config={"configurable": {"thread_id": "demo-1"}}):
    final_step = step

print(final_step["fuse"]["final_answer"])

---

## Notes
- This starter avoids heavy dependencies and keeps the orchestrators **CPU‑light**.
- For production, consider:
  - **Queues** (Redis/Rabbit) between orchestrator and GPU workers.
  - **Tracing** (LangSmith or OpenTelemetry).
  - **HITL** pause nodes in LangGraph or approvals via an outer workflow (e.g., n8n).