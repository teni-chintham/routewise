"""Central configuration.  Everything that shapes a routing decision is here or in models.yaml.

Where data lives
    Personal data (memory, document index, scorecards measured on this machine, the cost ledger
    and the egress audit log) lives in ASSISTANT_HOME, default ~/.context-assistant.  It never
    leaves the computer except as part of a prompt sent to a cloud model the user enabled, and
    every such request is recorded in egress.jsonl.

    Research artefacts (benchmarks, results, the reference scorecards from the published run)
    live in the repository.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:  # settings and API keys from .env (never committed)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:  # pragma: no cover
    pass
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
REGISTRY_PATH = Path(os.getenv("ASSISTANT_REGISTRY", ROOT / "models.yaml"))
REFERENCE_SCORECARDS = ROOT / "scorecards"  # measured on the reference machine, shipped with the repo

HOME = Path(os.getenv("ASSISTANT_HOME", Path.home() / ".context-assistant")).expanduser()
SCORECARD_DIR = Path(os.getenv("ASSISTANT_SCORECARDS", HOME / "scorecards"))
EGRESS_LOG = HOME / "egress.jsonl"
SPEND_FILE = HOME / "spend_usd.txt"  # running total of real spend on paid (non-free-tier) cloud models
SPEND_CAP_USD = float(os.environ["ASSISTANT_SPEND_CAP"]) if os.getenv("ASSISTANT_SPEND_CAP") else None
LEDGER = HOME / "ledger.jsonl"

for _d in (MODELS_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)

INTENTS = ["coding", "writing", "search", "reasoning", "multimodal"]

# ---------------------------------------------------------------------------
# Providers.  "mock" is a deterministic fake model used by the offline test-suite.
# ---------------------------------------------------------------------------
PROVIDER_OVERRIDE = os.getenv("ASSISTANT_FORCE_PROVIDER") or None  # "mock" only: every model becomes the offline mock
if PROVIDER_OVERRIDE not in (None, "mock"):
    raise ValueError("ASSISTANT_FORCE_PROVIDER can only be 'mock' (models choose their own provider in models.yaml)")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDER = os.getenv("ASSISTANT_EMBEDDER", "ollama")  # ollama | tfidf  (both local)
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
USD_TO_INR = float(os.getenv("USD_TO_INR", "88.0"))

# ---------------------------------------------------------------------------
# Routing Utility Score (RUS), see docs/SCORING.md
#
#   RUS(m, i) = w_a * A(m, i)
#             + w_c * (1 - min(C(m, i) / C_max, 1))
#             + w_l * (1 - min(L(m, i) / L_max, 1))
#
#   A  lower bound of the 95% Wilson interval of measured accuracy on intent i
#   C  mean measured cost per query in USD (0 for local models)
#   L  median measured latency in seconds on this machine
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {"accuracy": 0.6, "cost": 0.2, "latency": 0.2}
COST_BUDGET_USD = float(os.getenv("ASSISTANT_COST_BUDGET", "0.002"))   # C_max: most you would pay for one query
LATENCY_BUDGET_S = float(os.getenv("ASSISTANT_LATENCY_BUDGET", "30"))  # L_max: longest you would wait
CONFIDENCE_Z = 1.96  # 95% Wilson interval

# Privacy: when the prompt carries personal memory or private document passages, only local
# models are eligible ("local").  "allow" lets cloud models see it (still logged in egress.jsonl).
PRIVATE_CONTEXT_POLICY = os.getenv("ASSISTANT_PRIVATE_CONTEXT", "local")

# The model a non-routing variant always uses (ablation baseline, B and C)
FIXED_MODEL = os.getenv("ASSISTANT_FIXED_MODEL", "llama3.2-3b")

# ---------------------------------------------------------------------------
# Memory (Model B)
# ---------------------------------------------------------------------------
MEMORY_DB = os.getenv("ASSISTANT_MEMORY_DB", str(HOME / "memory.db"))
MEMORY_TOP_K = 5
SESSION_SUMMARY_EVERY = 6  # turns

# ---------------------------------------------------------------------------
# RAG (Model C)
# ---------------------------------------------------------------------------
CORPUS_DIR = Path(os.getenv("ASSISTANT_CORPUS", DATA_DIR / "corpus"))
INDEX_PATH = Path(os.getenv("ASSISTANT_INDEX", MODELS_DIR / "rag_index.pkl"))
CHUNK_WORDS = 120
CHUNK_OVERLAP = 30
RAG_TOP_K = 4
RAG_MIN_SIM = 0.15


def ensure_home() -> Path:
    HOME.mkdir(parents=True, exist_ok=True)
    SCORECARD_DIR.mkdir(parents=True, exist_ok=True)
    return HOME
