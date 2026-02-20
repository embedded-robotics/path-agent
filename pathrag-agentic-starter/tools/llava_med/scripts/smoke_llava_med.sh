#!/usr/bin/env bash
set -euo pipefail
: "${LLAVA_MED_SMOKE_CMD:?Set LLAVA_MED_SMOKE_CMD to the LLaVA‑Med smoke command}"
echo "[Smoke] Executing: ${LLAVA_MED_SMOKE_CMD}"
eval "${LLAVA_MED_SMOKE_CMD}"
echo "[Smoke] Done. Check artifacts/answer/smoke.jsonl"
