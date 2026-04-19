# CHIEF Local Tool

Local subprocess wrapper for the CHIEF heatmap patch extractor.

This tool does not vendor the original CHIEF model code or weights. It expects
an existing CHIEF checkout on disk and runs the extractor locally without Docker.

Defaults:
- `CHIEF_REPO_DIR=/home/sina/chief-image-file/CHIEF`
- `CHIEF_MODEL_DIR=/home/sina/chief-image-file/CHIEF/model_weight`

Recommended env:

```bash
export CHIEF_TOOL_DIR=/home/sina/projects/path-agent/pathrag-agentic-starter/tools/chief
export CHIEF_REPO_DIR=/home/sina/chief-image-file/CHIEF
export CHIEF_MODEL_DIR=/home/sina/chief-image-file/CHIEF/model_weight
export CHIEF_PYTHON=/path/to/chief/env/bin/python
```

The current `valid_bounds` handling is intentionally a placeholder that mirrors
Imroze's example-style usage. The real bounds provider should be wired in later.
