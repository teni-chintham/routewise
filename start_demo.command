#!/bin/bash
# Double-click to start the Context-Aware AI Assistant demo.
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
source .venv/bin/activate
if ! curl -s localhost:11434/api/tags >/dev/null; then
  echo "Starting Ollama..."
  (ollama serve >/dev/null 2>&1 &)
  sleep 4
fi
echo "Opening the demo at http://localhost:8501  (close this window or press Ctrl+C to stop)"
streamlit run app.py --server.port 8501
