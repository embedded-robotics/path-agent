# PathRAG + MedGemma Local Tool

This tool is the local Stage 4 backend for smaller GPU or CPU-adjacent setups.

It is designed to mirror the JSONL-style interface used by `LlavaMedClient`, so
the main PathRAG pipeline can switch between:

- `llava-med` for heavier GPU machines or remote Colab service
- `medgemma` for local execution

## Expected setup

```bash
cd pathrag-agentic-starter/tools/medgemma
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional environment variables:

```bash
export MEDGEMMA_MODEL="google/medgemma-1.5-4b-it"  # next authorized smoke-test override
export HF_HOME="$HOME/.cache/huggingface"
```

`MEDGEMMA_MICROBATCH_SIZE` currently accepts only `1`: the tool intentionally
uses sequential single-item inference for 12 GB VRAM safety. Larger
microbatches are reserved for a future measured optimization.

## Batch CLI

```bash
.venv/bin/python -m src.graph.medgemma_eval \
  --config config/default.yaml \
  --question-file /abs/path/to/questions.jsonl \
  --image-folder /abs/path/to/image/root \
  --answers-file /abs/path/to/answers.jsonl
```

## Compatibility CLI

Older wrappers in this repo expect:

```bash
.venv/bin/python -m src.graph.medgemma_run \
  --config config/default.yaml \
  --question "Summarize pathology findings from these patches." \
  --image-folder data/images \
  --out artifacts/answer/out.json \
  --k 3
```

That compatibility entry point is also included here.
