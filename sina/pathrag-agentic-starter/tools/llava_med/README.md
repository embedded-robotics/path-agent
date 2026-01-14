# PathRAG + LLaVA‑Med Starter (RTX 4070 Super)

This pack sets sane defaults for **LLaVA‑Med installation/usage** and wires the knobs you confirmed:

- **Artifacts**: `artifacts/answer/smoke.jsonl` (smoke), `artifacts/last_run.json` (full run)
- **Pins**: `transformers==4.36.2`, `accelerate==0.21.0`, `timm==0.9.12`, `einops==0.6.1`, `einops-exts==0.0.4`, `tokenizers>=0.15,<0.16`, `sentencepiece==0.1.99`, `httpx==0.24.0`
- **K (patches)**: default **3** (flag to 6)
- **Critique loop**: default **1** (flag to 0)
- **H&E gate**: default nuclei threshold **>= 5** (configurable)
- **Re‑rank**: **lexical** first; embeddings can be slotted later

> NOTE: Install PyTorch CUDA wheels separately per your CUDA version. Suggested command below uses cu124.

## Quickstart

```bash
# 0) System prep (Ubuntu 22.04+; CUDA 12.x)
sudo apt-get update -y
# optional quality-of-life packages
sudo apt-get install -y git git-lfs build-essential python3-venv

# 1) Create venv
python3 -m venv .venv
source .venv/bin/activate

# 2) Torch (CUDA 12.4 wheels) — adjust if needed
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# 3) Core deps (pinned to match LLaVA‑Med quirks)
pip install -r requirements.txt

# 4) (One-time) HF login for gated models
huggingface-cli login  # paste your token

# 5) Set the CLI command for LLaVA‑Med smoke test (model path may vary)
export LLAVA_MED_SMOKE_CMD='python -m llava.eval.generate --model-path lamm-mit/llava-med-v1.5 --question-file data/smoke.jsonl --answers-file artifacts/answer/smoke.jsonl --temperature 0.0'

# 6) Run smoke
make smoke

# 7) Run full graph (Stage-4 real VLM) when ready
make run
```

### OOM tips (4070 Super 12GB)
- Lower input/patch resolution (config: `stage4.vis.max_size`)
- Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`
- If still tight, switch to **MedGemma-2B‑IT** (config: `models.medgemma_repo`) and keep graph identical.

