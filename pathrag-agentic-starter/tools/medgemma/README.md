# PathRAG + MedGemma Local Tool

This tool is the local Stage 4 backend for smaller GPU or CPU-adjacent setups.

It is designed to mirror the JSONL-style interface used by `LlavaMedClient`, so
the main PathRAG pipeline can switch between:

- `llava-med` for heavier GPU machines or remote Colab service
- `medgemma` for local execution

The global PathRAG Stage 4 backend remains `llava-med`. When callers explicitly
select `PATHRAG_STAGE4_BACKEND=medgemma`, this tool defaults to
`google/medgemma-1.5-4b-it` with 4-bit NF4 (double quantization), BF16 compute,
96 output tokens, and microbatch size 1. These settings were engineering
smoke-tested—not medical-quality validated—on an RTX 4070 SUPER 12 GB GPU: a
one-patch/two-prompt adapter run and a two-patch/four-prompt batch run observed
approximately 8.7 GB peak VRAM.

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
# All are optional overrides of config/default.yaml.
export MEDGEMMA_MODEL="another-authorized-model-id"
export MEDGEMMA_PRECISION="fp16"              # auto, bf16, or fp16
export MEDGEMMA_QUANTIZATION="8bit"           # none, 8bit, or 4bit
export MEDGEMMA_MAX_NEW_TOKENS=128
export MEDGEMMA_SUBPROCESS_TIMEOUT_SECONDS=600
export HF_HOME="$HOME/.cache/huggingface"
```

The tool is cache-first: Hugging Face uses an existing local cache and downloads
the gated model when missing if authorization and network access are available.
For strictly cached execution, set `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. A 4-bit or 8-bit selection requires `bitsandbytes` in
this isolated `tools/medgemma` environment. The YAML no longer advertises a
`cache_dir` setting because cache selection is controlled by Hugging Face (for
example, `HF_HOME`) rather than that unused configuration field.

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
