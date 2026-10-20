#!/bin/bash
# Final evaluation, one double-click.  Takes ~6-7 h on an 8 GB M3.  Safe to stop and re-run:
# every answer is saved as it is produced and reused on the next run.
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
mkdir -p .runlogs
export ASSISTANT_SPEND_CAP=5   # paid cloud models stop once real spend reaches $5
LOG=.runlogs/final_run.log
STATUS=.runlogs/final_status.txt
step() { echo "$(date '+%H:%M:%S') $*" | tee -a "$STATUS"; }
: > "$STATUS"
step "START"

# ---- python env
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install -q -r requirements.txt || { step "FAIL pip install"; exit 1; }

# ---- ollama
if ! curl -s localhost:11434/api/tags >/dev/null; then
  step "starting Ollama"
  (open -a Ollama 2>/dev/null || (ollama serve >/dev/null 2>&1 &)); sleep 8
fi
curl -s localhost:11434/api/tags >/dev/null || { step "FAIL Ollama not running. Open the Ollama app and double-click again."; exit 1; }
for m in nomic-embed-text qwen2.5-coder:3b llama3.2:3b llama3.2:1b moondream; do
  ollama list | grep -q "^$m" || { step "pulling $m"; ollama pull "$m"; }
done

# ---- cloud key
if ! grep -qE '^(GEMINI|ANTHROPIC|OPENAI)_API_KEY=..' .env 2>/dev/null; then
  echo; echo "No cloud API key in .env. Cloud models (and the cost comparison) will be skipped."
  read -p "Continue with local models only? [y/N] " yn; [ "$yn" = y ] || exit 1
fi

{
step "preflight: one call per model"
python scripts/preflight.py || { step "FAIL preflight (see above)"; exit 1; }
export ASSISTANT_DISABLE="$(cat .runlogs/disabled_models.txt 2>/dev/null)"
[ -n "$ASSISTANT_DISABLE" ] && step "cloud models left out (failed preflight): $ASSISTANT_DISABLE"

step "intent classifier + document index"
python scripts/train_intent.py && python scripts/build_index.py || { step "FAIL training/index"; exit 1; }

step "offline tests"
python -m pytest -q tests || { step "FAIL tests"; exit 1; }

step "phrasing study"
python scripts/run_phrasing_study.py

step "EVAL: 400 public tasks x every model (longest step)"
python -u scripts/run_eval.py || { step "FAIL eval"; exit 1; }

step "REPEAT: cloud models again on the test split (stability check)"
python -u scripts/run_repeat.py || step "repeat step failed (not critical)"

step "ABLATION: baseline/A/B/C/D + privacy check"
python -u scripts/run_ablation.py --privacy-check || { step "FAIL ablation"; exit 1; }

step "figures"
python scripts/make_figures.py

step "app screenshots (optional)"
pip install -q playwright && python -m playwright install chromium >/dev/null 2>&1 && {
  ASSISTANT_HOME="$PWD/.runlogs/demo_home" streamlit run app.py --server.headless true --server.port 8501 >/dev/null 2>&1 & SPID=$!
  sleep 10; python scripts/make_screenshots_final.py || echo "screenshots failed (not critical)"
  kill $SPID 2>/dev/null
} || echo "playwright not available, screenshots skipped"
step "DONE"
} 2>&1 | tee -a "$LOG"
echo; echo "Finished. Results in results/final/. You can close this window."
