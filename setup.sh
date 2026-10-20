#!/usr/bin/env bash
# One-shot setup on macOS / Linux.  Run from the project folder:  bash setup.sh
set -e
cd "$(dirname "$0")"

echo "== python env"
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
[ -f .env ] || cp .env.example .env

echo "== ollama"
if ! command -v ollama >/dev/null; then
  echo "Ollama not found. Install from https://ollama.com/download then re-run."; exit 1
fi
if ! curl -s localhost:11434/api/tags >/dev/null; then
  echo "starting ollama serve in background"; (ollama serve >/dev/null 2>&1 &); sleep 3
fi
# the model pool (see assistant/config.py) + embedder.
# 8 GB RAM box: small-model set, ~7 GB total, one-time download.
for m in nomic-embed-text qwen2.5-coder:3b llama3.2:3b moondream llama3.2:1b; do
  ollama list | grep -q "^$m" || ollama pull "$m"
done

echo "== build Model A (intent classifier) and Model C index"
python scripts/make_intent_dataset.py
python scripts/train_intent.py
python scripts/build_index.py

echo "== offline tests"
ASSISTANT_FORCE_PROVIDER=mock ASSISTANT_EMBEDDER=tfidf python -m pytest -q tests

echo
echo "done.  next:"
echo "  source .venv/bin/activate"
echo "  streamlit run app.py                 # demo"
echo "  python scripts/run_ablation.py       # full ablation on Ollama (~20-40 min)"
