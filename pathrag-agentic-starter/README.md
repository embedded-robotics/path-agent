# PathRAG Agentic Starter

A GitHub-ready, minimal starter for building an **agentic architecture** around **Path-RAG** using two frameworks:

- **LangGraph (LangChain)** — production-grade, stateful graph orchestration.
- **AutoGen** — fast multi-agent prototyping (planner/critic/executor).

This repo ships with:
- A clean Python package layout (`src/`).
- Tool adapters for:
  - local histocartography patch extraction
  - local CHIEF patch refinement
  - Stage 3 local caption bank or retriever API
  - Stage 4 backend selection between local `medgemma` and `llava-med`
  - Stage 5 / Stage 7 OpenAI critique and fusion
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

## Local-First Runtime

The default runtime is now local-first:

- Stage 1: local histocartography patch extraction (`tools/histocartography`)
- Stage 2: local CHIEF refinement (`tools/chief`)
- Stage 3: local caption bank by default, optional retriever API
- Stage 4: local `medgemma` or `llava-med`
- Stage 5 / Stage 7: OpenAI-backed critique and fusion

The old CHIEF Docker + combined API path is no longer the intended local runtime.

### Stage 1–2 local environments

Histocartography and CHIEF each run in isolated Python environments and are called as local subprocess tools.

Required env knobs:

```bash
export HISTOCARTOGRAPHY_PYTHON=/home/sina/projects/path-agent/pathrag-agentic-starter/tools/histocartography/.venv/bin/python
export CHIEF_PYTHON=/home/sina/projects/path-agent/pathrag-agentic-starter/tools/chief/.venv/bin/python
```

Notes:
- The histocartography business logic now lives under `pathrag-agentic-starter/tools/histocartography`.
- The dedicated histocartography env is expected under:
  - `pathrag-agentic-starter/tools/histocartography/.venv`
- Default Linux/XDG data locations are now:
  - CHIEF repo: `~/.local/share/pathrag/chief/repo`
  - CHIEF weights: `~/.local/share/pathrag/chief/model_weight`
  - Histocartography checkpoints: `~/.local/share/pathrag/histocartography/checkpoints`
- These locations are configured in the tool YAML files and can be overridden via env vars.

### Stage 4 backends

Stage 4 can now run with either a local MedGemma backend or an LLaVA-Med backend.

#### Local MedGemma

Use this when you want a fully local fallback:

```bash
export PATHRAG_STAGE4_BACKEND=medgemma
unset PATHRAG_STAGE4_REMOTE_URL
export MEDGEMMA_TOOL_DIR=/home/sina/projects/path-agent/pathrag-agentic-starter/tools/medgemma
```

Notes:
- the MedGemma tool lives under `tools/medgemma`
- it uses its own `.venv`
- current default model is `google/medgemma-4b-it`

#### Remote LLaVA-Med

Use this when you have a stronger GPU machine or Colab serving LLaVA-Med:

```bash
export PATHRAG_STAGE4_BACKEND=llava-med
export PATHRAG_STAGE4_REMOTE_URL=https://YOUR-STAGE4-ENDPOINT
```

Notes:
- remote LLaVA-Med is currently the stronger Stage 4 backend on the tested Imroze case
- local 12 GB GPUs were not reliable for `microsoft/llava-med-v1.5-mistral-7b` fp16 loading
- the remote endpoint is expected to expose:
  - `POST /describe_roi`
  - `POST /patch_contribution`

Colab notebook used for the remote LLaVA-Med path:
- https://colab.research.google.com/drive/18UIsFA5Bq4qYgT1ZY-t85PUfIXE6Eqpq#scrollTo=ERx7apvRyF4W

### Runtime env knobs

- `HISTOCARTOGRAPHY_PYTHON`: interpreter used by the local histocartography tool
- `HISTOCARTOGRAPHY_TOOL_DIR`: optional override for `tools/histocartography`
- `HISTOCARTOGRAPHY_CONFIG`: optional histocartography config override
- `HISTOCARTOGRAPHY_CHECKPOINT_DIR`: optional override for nuclei model checkpoints
- `HISTOCARTOGRAPHY_PRETRAINED_DATA`: optional nuclei model choice (`pannuke` or `monusac`)
- `CHIEF_PYTHON`: interpreter used by the local CHIEF tool
- `CHIEF_TOOL_DIR`: optional override for `tools/chief`
- `CHIEF_CONFIG`: optional CHIEF config override
- `CHIEF_REPO_DIR`: optional override for the CHIEF source checkout
- `CHIEF_MODEL_DIR`: optional override for CHIEF weight files
- `PATHRAG_USE_RETRIEVER_API`: `1` to use external caption retriever
- `PATHRAG_RETRIEVER_API_URL`: Retriever API endpoint.
- `PATHRAG_HTTP_TIMEOUT`: HTTP timeout (seconds)
- `PATHRAG_STAGE4_BACKEND`: `llava-med` or `medgemma`
- `PATHRAG_STAGE4_REMOTE_URL`: remote Stage 4 base URL for LLaVA-Med
- `PATHRAG_STAGE4_USE_CAPTIONS`: `0` disables Stage 3 captions inside Stage 4 prompts
- `MEDGEMMA_TOOL_DIR`: path to the local MedGemma tool directory

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

1. **Stages 1–2 — local HC + local CHIEF**  
   - Tiles the image into the same uniform `3x3` grid used in Imroze’s original combined API.
   - Runs local histocartography logic to rank those grid cells by nuclei density.
   - Converts the selected HC patches into `valid_bounds`.
   - Runs local CHIEF only inside those bounds and carries forward the final CHIEF `Top-K` patches.
   - Controlled by `state["top_k"]`. Only these K final CHIEF patches are carried forward.   

2. **Stage 3 — labeling + retrieval**  
   - Identifies a sub-pathology label (e.g., `"scc"`) and retrieves short textual snippets (`full_captions`) for that label. :contentReference[oaicite:4]{index=4}  

3. **Stage 4 — ROI & patch agents**  
   - For each of the K patches:  
     - An ROI agent decides if the patch is useful and explains *why*.  
     - A patch contribution agent produces a short summary (`patch_summaries[i]`) conditioned on the question and the selected Stage 4 backend.  
   - Backend options:
     - `llava-med`: local or remote LLaVA-Med
     - `medgemma`: local MedGemma tool
   - Stage 3 captions can be disabled for Stage 4 via `PATHRAG_STAGE4_USE_CAPTIONS=0`.   

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
     - the retrieved Stage 3 captions as weak context,
     - the selected patches (`chosen_idx` + their summaries),
   - and produces `final_answer`.
   - Current prompt design treats visible patch evidence as primary and Stage 3 context as secondary. :contentReference[oaicite:8]{index=8}  

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
  - `tiles`, `hc_rank`, `cheif_rank`, `patches` — Stage 1–2 outputs (patches stored as dicts).
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
```

## Evaluation runner

There is a dedicated evaluation runner for the Imroze sheet that reuses saved CHIEF patch coordinates and starts from Stage 3 onward:

```bash
cd /home/sina/projects/path-agent

export PATHRAG_STAGE4_BACKEND=llava-med
export PATHRAG_STAGE4_REMOTE_URL=https://YOUR-STAGE4-ENDPOINT
export OPENAI_API_KEY=...

pathrag-agentic-starter/.venv/bin/python evaluation/run_imroze_langgraph_eval.py \
  --case-id 07_level2 \
  --question-id 1 \
  --out pathrag-agentic-starter/evaluation/imroze_langgraph_smoke_real.json
```

Local MedGemma variant:

```bash
cd /home/sina/projects/path-agent

export PATHRAG_STAGE4_BACKEND=medgemma
unset PATHRAG_STAGE4_REMOTE_URL
export MEDGEMMA_TOOL_DIR=/home/sina/projects/path-agent/pathrag-agentic-starter/tools/medgemma
export OPENAI_API_KEY=...

pathrag-agentic-starter/.venv/bin/python evaluation/run_imroze_langgraph_eval.py \
  --case-id 07_level2 \
  --question-id 1 \
  --out pathrag-agentic-starter/evaluation/imroze_langgraph_smoke_medgemma.json
```

Notes:
- the runner changes into `pathrag-agentic-starter` internally to import the app
- relative `--sheet` and `--out` paths are now resolved from the shell launch directory
- the evaluation runner reuses saved patches, so it skips Stage 1–2 entirely
- current tested quality result:
  - remote `llava-med` outperforms local `medgemma` on the current Imroze smoke case

---

## Notes
- This starter avoids heavy dependencies and keeps the orchestrators **CPU‑light**.
- For production, consider:
  - **Queues** (Redis/Rabbit) between orchestrator and GPU workers.
  - **Tracing** (LangSmith or OpenTelemetry).
  - **HITL** pause nodes in LangGraph or approvals via an outer workflow (e.g., n8n).
