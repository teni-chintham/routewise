# Context-Aware Personal AI Assistant

An open-source, local-first assistant that sends each query to the model **measured** to be best for it,
remembers you across sessions, and answers from your own documents. Your memory and documents stay on
your computer.

BCSE497J Project-I, VIT Vellore. Nischay N, Tejas K M, Chintham Tenidhar Reddy. Guide: Prof. Mary Mekala A.

## What it does

1. **Bring any model.** Add it to `models.yaml`: a local Ollama model, any OpenAI-compatible API
   (OpenAI, Groq, OpenRouter, LM Studio, vLLM ...), Anthropic or Gemini.
2. **Score it.** `python -m assistant calibrate --model <name>` runs it on 200 public benchmark tasks
   (MBPP, GSM8K, Natural Questions, IFEval, images) and writes a **scorecard**: accuracy per task type
   with a 95% confidence interval, median latency on your machine, and real cost per query from token counts.
3. **Route with the scores.** Each query is classified (coding, writing, search, reasoning, multimodal)
   and every eligible model gets a **Routing Utility Score**; the highest wins. Defined in
   [docs/SCORING.md](docs/SCORING.md). A premium model is used only where it is measurably worth its
   price, which is what cuts API spend.
4. **Local first.** For coding and arithmetic the assistant first answers on your computer and keeps the
   answer if a check passes that needs no reference answer (the code passes the `assert` example in your
   prompt; two local models agree on the number). Otherwise it escalates to the RUS choice.
5. **Keep data local.** Memory (SQLite), the document index and embeddings live in `~/.context-assistant`.
   A prompt that carries your memory or documents is only sent to local models (configurable). Every
   request that leaves the computer is recorded in an egress audit log, without its text.

| Build | Routing | Memory | RAG |
|---|---|---|---|
| Baseline | | | |
| Model A | ✓ | | |
| Model B | | ✓ | |
| Model C | | | ✓ |
| Model D (full) | ✓ | ✓ | ✓ |

## Results (200 unseen public test tasks)

| Policy | Accuracy (95% CI) | Cost per 1,000 queries | Answered on the laptop |
|---|---|---|---|
| Routed, balanced weights | 89.5% (84.5–93.0) | $0.127 | 0% |
| Local-first cascade | 88% (82.8–91.8) | $0.105 | 28.5% |
| Always Claude Sonnet 5.5 | 89% (83.9–92.6) | $2.506 | 0% |
| Always Claude Haiku 5.5 | 88.5% (83.3–92.2) | $0.169 | 0% |
| Always Gemini 3.1 Flash-Lite | 86% (80.5–90.1) | $0.170 | 0% |
| Always llama3.2-3b (best local) | 48% (41.2–54.9) | $0.000 | 100% |

Routing vs Sonnet: exact McNemar p = 1.0 (no difference) at 95% lower cost. Privacy check: with every
cloud model available, 0 of 54 requests carrying memory left the computer. Full write-up in the report.

## Quick start (macOS / Linux)

```bash
bash setup.sh                          # venv, dependencies, Ollama models, classifier, index, tests
source .venv/bin/activate
cp .env.example .env                   # add API keys only for cloud models you enable
python -m assistant models             # what is registered, what is calibrated
python -m assistant route "write a python function to merge two sorted lists"
streamlit run app.py                   # the demo, with the decision inspector
```

Add a model and score it:

```bash
ollama pull qwen2.5:7b
python -m assistant add-model qwen2.5-7b --provider ollama --model qwen2.5:7b --local
python -m assistant calibrate --model qwen2.5-7b      # ~20-40 min for a local 7B model
```

## Documents

| | |
|---|---|
| Final report (VIT format) | [PDF](docs/report/BCSE497J_Final_Report.pdf) · [Word](docs/report/BCSE497J_Final_Report.docx) |
| Final review deck | [PDF](docs/review/Final_Review_Deck.pdf) · [PowerPoint](docs/review/Final_Review_Deck.pptx) |
| Q&A defence sheet | [PDF](docs/review/QA_Defence_Sheet.pdf) · [Word](docs/review/QA_Defence_Sheet.docx) |
| Scoring definitions | [docs/SCORING.md](docs/SCORING.md) |
| Diagrams (UML, DFD, architecture) | [docs/diagrams](docs/diagrams) (sources in `docs/diagrams/src`) |
| Review 2 material (superseded) | [docs/review2](docs/review2), [results/review2](results/review2) |

## Reproduce the paper numbers

```bash
./run_final.command          # or: python scripts/run_eval.py && python scripts/run_ablation.py --privacy-check
python scripts/make_figures.py
PLANTUML_JAR=/path/to/plantuml.jar python scripts/make_diagrams.py   # needs Graphviz and Java
```

Every answer is saved in `results/final/raw.jsonl` with its grade and the reason, so any number can be
audited back to individual answers. `python scripts/run_eval.py --analyse-only` re-grades and recomputes
everything from the saved answers without calling a model.

## Layout

```
models.yaml               model registry (add yours here)
assistant/
  registry.py             loads models.yaml
  providers.py            Ollama / OpenAI-compatible / Anthropic / Gemini / mock; token counts; egress audit
  scorecard.py            per-model measurements + Wilson intervals
  router.py               Routing Utility Score
  cascade.py              local-first checks (code vs the prompt's assert; two local models agree)
  intent.py               TF-IDF + calibrated linear SVM intent classifier
  memory.py, rag.py       local memory and retrieval
  pipeline.py             one pipeline, three switches (the ablation)
  bench/                  graders and the benchmark runner
  __main__.py             command line
scripts/
  build_public_bench.py   samples the 400-task public benchmark (seed 2026)
  run_eval.py             scorecards from calibration, routing evaluated on test
  run_ablation.py         baseline / A / B / C / D, privacy check
  make_figures.py         every chart, from results/final/*.json
  make_diagrams.py        architecture, DFD (Yourdon), Gantt; calls diagrams_uml.py
  diagrams_uml.py         UML use case, class (parsed from the code), sequence, activity (PlantUML)
docs/SCORING.md           every metric defined
results/final/            the final run; results/review2/ is the superseded Review 2 run
```

## Data and licences

Benchmarks: MBPP (CC-BY-4.0), GSM8K (MIT), Natural Questions (CC-BY-SA-3.0), IFEval (Apache-2.0);
the 80 images are generated by `scripts/build_public_bench.py`. Code: MIT licence (see LICENSE).
