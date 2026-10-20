"""Design diagrams for the report, generated from the code.

    architecture        Graphviz block diagram of the components and where they run
    DFD level 0 and 1   Yourdon-DeMarco notation, following pipeline.Assistant.chat step by step
    gantt               the project's real dates, planned items marked
    UML                 use case, class (parsed from the code), sequence and activity diagrams:
                        see scripts/diagrams_uml.py

Needs: graphviz (`dot`), Java and PlantUML (PLANTUML_JAR, default tools/plantuml.jar).
Usage: python scripts/make_diagrams.py [--out docs/diagrams]
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FONT = "Inter"
BLUE, BLUE_L, ORANGE_L, GREEN_L, GREY_L, INK = "#2a78d6", "#e8f1fc", "#fdeee6", "#e3f6ee", "#f1f0ec", "#0b0b0b"


def dot(src: str, out: Path):
    subprocess.run(["dot", "-Tpng", "-Gdpi=220", "-o", str(out)], input=src.encode(), check=True)
    print("  ", out.name)


# --------------------------------------------------------------------------- architecture + DFD
def architecture(out: Path):
    src = f'''digraph G {{
graph [rankdir=LR, fontname="{FONT}", nodesep=0.3, ranksep=0.55, newrank=true];
node [shape=box, style="rounded,filled", fillcolor="white", color="#52514e", fontname="{FONT}", fontsize=10];
edge [color="#52514e", fontname="{FONT}", fontsize=8.5, fontcolor="#52514e"];
user [label="User query\\n(+ optional image)", fillcolor="{GREY_L}"];
answer [label="Answer + decision table\\n(model, RUS, cost,\\nlocal or cloud)", fillcolor="{GREY_L}"];
cloud [label="Cloud models\\n(any API in models.yaml)", fillcolor="{ORANGE_L}"];
calib [label="Calibration (once per model)\\npublic benchmark, 200 tasks", style="rounded,dashed"];
subgraph cluster_dev {{ label="This computer (ASSISTANT_HOME)"; fontsize=11; style="rounded,dashed"; color="{BLUE}"; fontcolor="{BLUE}";
  intent [label="Intent classifier\\nTF-IDF + calibrated SVM", fillcolor="{BLUE_L}"];
  mem    [label="Memory (Model B)\\nSQLite: turns, facts, summaries", fillcolor="{GREEN_L}"];
  rag    [label="Retriever (Model C)\\nlocal embeddings, top-k", fillcolor="{GREEN_L}"];
  cards  [label="Scorecards\\nmeasured accuracy, cost, latency", shape=cylinder, fillcolor="{GREY_L}"];
  reg    [label="models.yaml\\nmodel registry", shape=note, fillcolor="{GREY_L}"];
  router [label="Router (Model A)\\nRouting Utility Score", fillcolor="{BLUE_L}"];
  ctx    [label="Context builder\\nsystem prompt + citations"];
  casc   [label="Local-first check (coding, maths)\\ncode passes the prompt's assert /\\ntwo local models agree", fillcolor="{GREEN_L}", style="rounded,filled,dashed"];
  local  [label="Local models\\n(Ollama)", fillcolor="{GREEN_L}"];
  audit  [label="Egress log +\\ncost ledger", shape=note, fillcolor="{GREY_L}"];
}}
{{rank=same; intent; mem; rag}}
{{rank=same; router; ctx}}
{{rank=same; local; cloud}}
user -> intent; user -> mem; user -> rag;
calib -> cards [style=dashed, label="writes"];
intent -> router [label="task type"];
cards -> router [label="A, C, L"]; reg -> router [label="eligibility"];
mem -> router [label="private context?", style=dashed];
mem -> ctx [label="facts"]; rag -> ctx [label="passages"];
router -> local [label="chosen model"];
router -> casc [label="coding / maths", style=dashed]; casc -> local [label="try first", style=dashed];
casc -> cloud [label="check failed:\\nescalate", style=dashed];
router -> cloud [label="chosen model\\n(no private context)"];
ctx -> local [label="prompt"]; ctx -> cloud [label="prompt", style=dashed];
local -> answer; cloud -> answer;
cloud -> audit [style=dotted, arrowhead=none, label="logged"];
}}'''
    dot(src, out / "architecture.png")


def dfd(out: Path):
    """Yourdon-DeMarco notation: external entities are rectangles, processes are numbered circles,
    data stores are open-ended boxes (parallel lines) with an ID, every arrow is a named data flow.
    Level 1 follows pipeline.Assistant.chat step by step; file names are the real paths in config.py."""
    ent = f'shape=box, style=filled, fillcolor="{GREY_L}", penwidth=1.4'
    proc = f'shape=circle, style=filled, fillcolor="{BLUE_L}", penwidth=1.2, fixedsize=true'

    def store(sid, name):
        return (f'{sid} [shape=none, margin=0, label=<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">'
                f'<TR><TD SIDES="TBR" BGCOLOR="#f1f0ec"><B>{sid}</B></TD><TD SIDES="TB" BGCOLOR="#f1f0ec">{name}</TD></TR>'
                f'</TABLE>>];')

    l0 = f'''digraph G {{ graph [rankdir=LR, fontname="{FONT}", nodesep=0.6, ranksep=1.1];
edge [fontname="{FONT}", fontsize=9, color="#52514e", fontcolor="#333333"];
node [fontname="{FONT}", fontsize=10];
U [label="User", {ent}];
K [label="Contributor", {ent}];
P [label="0\\nContext-Aware\\nPersonal AI\\nAssistant", {proc}, width=1.9];
L [label="Local model runtime\\n(Ollama)", {ent}];
C [label="Cloud model API", {ent}];
U -> P [label="question, image,\\nweights, build choice"];
P -> U [label="answer, sources,\\nrouting decision, cost"];
K -> P [label="model entry,\\ncalibrate / evaluate"];
P -> K [label="scorecards,\\nevaluation results"];
P -> L [label="prompt (may carry\\nmemory, passages)"];
L -> P [label="reply, token counts"];
P -> C [label="prompt without\\nprivate context"];
C -> P [label="reply, token counts"];
{{rank=same; U; K}}
{{rank=same; L; C}}
}}'''
    dot(l0, out / "dfd_level0.png")

    l1 = f'''digraph G {{ graph [rankdir=LR, fontname="{FONT}", nodesep=0.35, ranksep=0.75, newrank=true];
edge [fontname="{FONT}", fontsize=8.5, color="#52514e", fontcolor="#333333"];
node [fontname="{FONT}", fontsize=9.5];
U [label="User", {ent}];
L [label="Local model\\nruntime (Ollama)", {ent}];
C [label="Cloud model API", {ent}];
p1 [label="1.0\\nRecall\\nmemory", {proc}, width=1.15];
p2 [label="2.0\\nRetrieve\\npassages", {proc}, width=1.15];
p3 [label="3.0\\nClassify\\nintent", {proc}, width=1.15];
p4 [label="4.0\\nRoute\\nby RUS", {proc}, width=1.15];
p5 [label="5.0\\nTry local\\nfirst", {proc}, width=1.15];
p6 [label="6.0\\nGenerate\\nanswer", {proc}, width=1.15];
p7 [label="7.0\\nRemember\\nturn", {proc}, width=1.15];
p8 [label="8.0\\nLog cost", {proc}, width=1.15];
{store("D1", "memory.db (facts, summaries, turns)")}
{store("D2", "rag_index.pkl (document chunks)")}
{store("D3", "intent_clf.joblib")}
{store("D4", "scorecards/*.json")}
{store("D5", "models.yaml")}
{store("D6", "egress.jsonl")}
{store("D7", "ledger.jsonl")}
U -> p1 [label="question"]; U -> p2 [label="question"]; U -> p3 [label="question, image?"];
D1 -> p1 [label="facts, summaries,\\nrecent turns"];
D2 -> p2 [label="chunks"];
D3 -> p3 [label="classifier"];
p1 -> p4 [label="private?"]; p2 -> p4 [label="private?"];
p3 -> p4 [label="task type"];
D4 -> p4 [label="A, C, L per model"]; D5 -> p4 [label="eligibility"];
p4 -> p5 [label="coding / maths"];
p5 -> L [label="local prompt"]; L -> p5 [label="local reply"];
p4 -> p6 [label="chosen model"];
p5 -> p6 [label="kept answer\\nor escalate"];
p1 -> p6 [label="facts, history"]; p2 -> p6 [label="passages"];
p6 -> L [label="prompt"]; L -> p6 [label="reply, tokens"];
p6 -> C [label="prompt (no\\nprivate context)"]; C -> p6 [label="reply, tokens"];
p6 -> D6 [label="size + hash\\nof cloud prompt"];
p6 -> U [label="answer, sources,\\ndecision"];
p6 -> p7 [label="turn"]; p7 -> D1 [label="turn, new facts,\\nsummary"];
p6 -> p8 [label="tokens, cost,\\nlatency"]; p8 -> D7 [label="ledger row"];
{{rank=same; p1; p2; p3}}
{{rank=same; L; C}}
}}'''
    dot(l1, out / "dfd_level1.png")


# --------------------------------------------------------------------------- gantt (real dates)
TIMELINE = [  # (task, start, end, status)  dates from the team's records
    ("Topic selection, literature survey", "2026-06-16", "2026-07-28", "done"),
    ("Review 1", "2026-07-29", "2026-07-29", "milestone"),
    ("Perplexity study, novelty revision", "2026-08-06", "2026-08-28", "done"),
    ("v1 build: routing, memory, RAG, ablation", "2026-08-28", "2026-09-23", "done"),
    ("Review 2", "2026-09-30", "2026-09-30", "milestone"),
    ("Measured routing: registry, scorecards, RUS", "2026-10-20", "2026-10-20", "done"),
    ("Public benchmark, final evaluation, ablation", "2026-10-20", "2026-10-20", "done"),
    ("Report, deck, defence sheet", "2026-10-20", "2026-10-20", "done"),
    ("Defence preparation and rehearsal", "2026-10-21", "2026-11-01", "planned"),
    ("Final review", "2026-11-02", "2026-11-02", "planned-milestone"),
]


def gantt(out: Path, timeline=TIMELINE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for k, (name, s, e, st) in enumerate(timeline):
        s, e = dt.date.fromisoformat(s), dt.date.fromisoformat(e)
        y = len(timeline) - 1 - k
        if "milestone" in st:
            ax.plot(s, y, "D", ms=9, color=BLUE if st == "milestone" else "white", mec=BLUE, mew=1.5)
        else:
            w = max((e - s).days, 1)
            ax.barh(y, w, left=s, height=0.55, color=BLUE if st == "done" else "white", edgecolor=BLUE, lw=1.3)
    ax.set_yticks(range(len(timeline)), [t[0] for t in reversed(timeline)], fontsize=8.5)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="#e6e5e0")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Project timeline (filled = done, outline = planned)", loc="left", fontsize=11, fontweight="bold")
    fig.savefig(out / "gantt.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("   gantt.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "diagrams"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    architecture(out)
    dfd(out)
    gantt(out)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import diagrams_uml  # UML: use case, class, sequence, activity (needs Java + PlantUML)
    diagrams_uml.main(out)


if __name__ == "__main__":
    sys.exit(main())
