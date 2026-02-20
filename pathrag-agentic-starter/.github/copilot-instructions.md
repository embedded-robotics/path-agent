# Copilot Instructions for PathRAG Agentic Starter

This guide enables AI coding agents to be productive in this codebase. It summarizes architecture, workflows, and conventions specific to this project.

## Architecture Overview
- **Agentic design**: Two agent orchestration frameworks are provided:
  - `src/pathrag/agents/langgraph_app.py`: LangGraph (LangChain) template for stateful, production-grade orchestration.
  - `src/pathrag/agents/autogen_app.py`: AutoGen template for rapid multi-agent prototyping.
- **Tools**: All external integrations (HistoCartography, LLaVA-Med, Reasoner) are mocked in `src/pathrag/agents/tools.py`. Replace these with real implementations (REST calls, Python libs).
- **Workflows**: Configurations and examples live in `src/pathrag/workflows/config/` and `src/pathrag/workflows/examples/`.
- **Utilities**: Logging and shared helpers are in `src/pathrag/utils/`.

## Developer Workflows
- **Setup**:
  - Create a Python virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
  - Install dependencies: `pip install -r requirements.txt`
  - Set API keys in `.env` (see `.env.example` for template)
- **Run agents**:
  - LangGraph: `python scripts/run_langgraph.py`
  - AutoGen: `python scripts/run_autogen.py`
- **Testing**:
  - Run smoke tests: `pytest tests/test_smoke.py`
- **Debugging**:
  - All tool calls are mocked; swap with real endpoints for integration debugging.
  - Logging is handled via `src/pathrag/utils/logging.py`.

## Project-Specific Patterns
- **Mocked tool pattern**: All external service calls are abstracted in `tools.py` with clear return types. Example:
  ```python
  def run_histocartography(...):
      return {"is_he": True, "patches": ["<img>::patchA", ...]}
  ```
- **Agent orchestration**: Each agent template is self-contained. Extend by adding new nodes or modifying the graph structure in `langgraph_app.py` or agent roles in `autogen_app.py`.
- **Configuration**: Use `.env` for secrets; do not hardcode API keys.

## Integration Points
- **HistoCartography**: Replace mock in `tools.py` with your pipeline or HTTP client.
- **LLaVA-Med**: Replace mock in `tools.py` with local inference or REST endpoint.
- **Reasoner**: Uses OpenAI by default; swap for Anthropic/local LLM as needed.

## Conventions
- **Source code**: All Python code is under `src/pathrag/`.
- **Scripts**: Entry points for running agents are in `scripts/`.
- **Tests**: All tests are in `tests/`.
- **No heavy dependencies**: Designed to be CPU-light for orchestration; GPU workloads should be offloaded.

## Example: Adding a Real Tool
- To integrate a real HistoCartography service:
  1. Edit `src/pathrag/agents/tools.py` and replace the mock with your implementation.
  2. Ensure the return type matches the expected dict structure.
  3. Update agent logic in `langgraph_app.py` or `autogen_app.py` if needed.

---
For more details, see `README.md` and the source files referenced above.
