SHELL := /bin/bash

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

export PYTORCH_CUDA_ALLOC_CONF ?= $(shell python3 -c 'print("max_split_size_mb:128")')

.PHONY: venv install smoke run env

venv:
	python3 -m venv $(VENV)
	@echo "Run: source $(VENV)/bin/activate"

install:
	$(PIP) install --upgrade pip
	# Install torch separately (per README)
	$(PIP) install -r requirements.txt

smoke:
	@if [ -z "$$LLAVA_MED_SMOKE_CMD" ]; then \
	  echo "Set LLAVA_MED_SMOKE_CMD to your LLaVA‑Med smoke command."; \
	  echo "Example:"; \
	  echo "  export LLAVA_MED_SMOKE_CMD='python -m llava.eval.generate --model-path lamm-mit/llava-med-v1.5 --question-file data/smoke.jsonl --answers-file artifacts/answer/smoke.jsonl --temperature 0.0'"; \
	  exit 2; \
	fi
	@echo "Running smoke…"
	bash -lc "$$LLAVA_MED_SMOKE_CMD"
	@echo "Smoke output at artifacts/answer/smoke.jsonl"

run:
	$(PY) -m src.graph.stage4_vlm --config config/default.yaml
