"""UML diagrams in standard notation, drawn from the code.

    use case    UML 2.5: actors outside the system boundary, use cases inside, associations as plain
                lines, <<include>> and <<extend>> as dashed open arrows (extend carries its condition),
                generalisation as a hollow triangle.  Two diagrams, one per primary actor.
    class       PlantUML, generated from assistant/*.py with `ast`: every class, attribute type,
                operation signature and relationship shown exists in the code.
    sequence    PlantUML, one call to pipeline.Assistant.chat with opt/alt fragments for each switch.
    activity    PlantUML, the same call as a flow with swimlanes and decision nodes.

Needs Java and plantuml.jar (set PLANTUML_JAR, default tools/plantuml.jar).
"""
from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
JAR = Path(os.getenv("PLANTUML_JAR", ROOT / "tools" / "plantuml.jar"))
INK, LINE, FILL, NAVY = "#111111", "#333333", "#EEF4FC", "#1F4E79"


# =========================================================================== use case (matplotlib, exact layout)
class UseCase:
    def __init__(self, title, size, boundary):
        plt.rcParams["font.family"] = "DejaVu Sans"
        self.W, self.H = size
        self.fig, self.ax = plt.subplots(figsize=(self.W / 10, self.H / 10))
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(0, self.H)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        x0, y0, x1, y1 = boundary
        self.ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=NAVY, lw=1.4))
        self.ax.text((x0 + x1) / 2, y1 - 2.2, title, ha="center", va="top", fontsize=11, fontweight="bold", color=INK)
        self.uc, self.act = {}, {}

    def case(self, key, x, y, text, w=24, h=7):
        self.ax.add_patch(Ellipse((x, y), w, h, fc=FILL, ec=NAVY, lw=1.1, zorder=2))
        self.ax.text(x, y, text, ha="center", va="center", fontsize=8.2, color=INK, zorder=3)
        self.uc[key] = (x, y, w / 2, h / 2)

    def actor(self, key, x, y, name, stereo=None):
        a = self.ax
        a.add_patch(plt.Circle((x, y + 4.6), 1.4, fill=False, lw=1.3, color=INK))
        a.plot([x, x], [y + 3.2, y - 0.6], color=INK, lw=1.3)
        a.plot([x - 2.4, x + 2.4], [y + 2.1, y + 2.1], color=INK, lw=1.3)
        a.plot([x, x - 2], [y - 0.6, y - 3.6], color=INK, lw=1.3)
        a.plot([x, x + 2], [y - 0.6, y - 3.6], color=INK, lw=1.3)
        label = (f"«{stereo}»\n" if stereo else "") + name
        a.text(x, y - 5, label, ha="center", va="top", fontsize=8.6, color=INK)
        self.act[key] = (x, y + 1)

    def _edge_point(self, key, tx, ty):
        """Point on the ellipse boundary of use case `key` facing (tx, ty)."""
        x, y, a, b = self.uc[key]
        dx, dy = tx - x, ty - y
        t = 1 / ((dx / a) ** 2 + (dy / b) ** 2) ** 0.5
        return x + dx * t, y + dy * t

    def _end(self, key, toward):
        if key in self.act:
            x, y = self.act[key]
            dx, dy = toward[0] - x, toward[1] - y
            d = (dx * dx + dy * dy) ** 0.5
            return x + dx / d * 3.2, y + dy / d * 3.2
        return self._edge_point(key, *toward)

    def assoc(self, a, b):
        pa = self.act.get(a) or self.uc[a][:2]
        pb = self.act.get(b) or self.uc[b][:2]
        s, e = self._end(a, pb), self._end(b, pa)
        self.ax.plot([s[0], e[0]], [s[1], e[1]], color=LINE, lw=0.9, zorder=1)

    def dep(self, a, b, kind, cond=None, lab_t=0.5, lab_dx=0.0, lab_dy=0.0):
        """Dashed open arrow a -> b labelled <<kind>> (and [cond] for extend)."""
        pa, pb = self.uc[a][:2], self.uc[b][:2]
        s, e = self._edge_point(a, *pb), self._edge_point(b, *pa)
        self.ax.annotate("", xy=e, xytext=s, zorder=1,
                         arrowprops=dict(arrowstyle="->,head_length=0.7,head_width=0.35", ls=(0, (4, 3)), color=LINE, lw=0.9))
        mx, my = s[0] + (e[0] - s[0]) * lab_t + lab_dx, s[1] + (e[1] - s[1]) * lab_t + lab_dy
        txt = f"«{kind}»" + (f"\n[{cond}]" if cond else "")
        self.ax.text(mx, my, txt, ha="center", va="center", fontsize=7, color="#444444", style="italic", zorder=4,
                     bbox=dict(fc="white", ec="none", pad=0.6))

    def general(self, child, parent):
        (cx, cy), (px, py) = self.act[child], self.act[parent]
        s, e = (cx, cy + 6.8), (px, py - 9.5)
        self.ax.annotate("", xy=e, xytext=s, arrowprops=dict(arrowstyle="-|>,head_length=1.2,head_width=0.6",
                                                            fc="white", ec=INK, lw=1.0))

    def save(self, path):
        self.fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(self.fig)


def use_case_user(out: Path):
    """Primary actor: User (app.py tabs and sidebar, `python -m assistant chat/route`)."""
    import math
    d = UseCase("Context-Aware Personal AI Assistant", (182, 108), (24, 14, 152, 105))
    d.actor("user", 9, 53, "User")
    d.actor("local", 170, 62, "Local model\nruntime (Ollama)", "system")
    d.actor("cloud", 170, 28, "Cloud model API", "system")
    # what the user does: on an arc around the actor, so every association is a clear radial line
    col1 = [("weights", "Set routing weights,\nlocal-first, privacy policy", 27), ("build", "Choose build\n(baseline, A, B, C, D)", 24),
            ("inspect", "Inspect routing decision\n(intent, RUS table, sources)", 27), ("ask", "Ask a question", 24),
            ("compare", "Compare models or\nbuilds on one prompt", 24), ("session", "Start new session /\nwipe memory", 24),
            ("logs", "View cost ledger and\negress audit log", 24)]
    for (key, text, w), ang in zip(col1, (50, 34, 17, 0, -17, -34, -50)):
        r = math.radians(ang)
        d.case(key, 15 + 42 * math.cos(r), 54 + 42 * math.sin(r), text, w=w)
    # behaviour inside "Ask a question" (column 2) and inside routing (column 3)
    d.case("image", 98, 93, "Attach an image")
    d.case("memory", 98, 81, "Recall memory and\nstore the turn")
    d.case("rag", 98, 69, "Retrieve document\npassages")
    d.case("route", 98, 54, "Route by RUS")
    d.case("generate", 98, 39, "Generate answer")
    d.case("log", 98, 24, "Log cost and egress")
    d.case("intent", 134, 76, "Classify intent")
    d.case("local_first", 134, 61, "Try local first\n(check, else escalate)")
    for k, _, _ in col1:
        d.assoc("user", k)
    d.dep("image", "ask", "extend", "image attached", lab_t=0.3, lab_dx=-5.5)
    d.dep("memory", "ask", "extend", "build B or D", lab_t=0.42, lab_dx=1)
    d.dep("rag", "ask", "extend", "build C or D", lab_t=0.42, lab_dx=1.5, lab_dy=0.5)
    d.dep("route", "ask", "extend", "build A or D", lab_t=0.5, lab_dy=3)
    d.dep("ask", "generate", "include", lab_t=0.55, lab_dy=-2.2)
    d.dep("ask", "log", "include", lab_t=0.55, lab_dx=-3)
    d.dep("route", "intent", "include", lab_t=0.5, lab_dx=-4, lab_dy=1)
    d.dep("local_first", "route", "extend", "local first on and\ncoding or maths", lab_t=0.45, lab_dy=-5.5)
    d.assoc("generate", "local")
    d.assoc("generate", "cloud")
    d.assoc("local_first", "local")
    d.save(out / "use_case_user.png")


def use_case_contributor(out: Path):
    """Primary actor: Contributor (models.yaml, `add-model`, `calibrate`, scripts/run_eval.py, run_ablation.py)."""
    d = UseCase("Context-Aware Personal AI Assistant", (168, 70), (24, 3, 134, 67))
    d.actor("user", 9, 52, "User")
    d.actor("contrib", 9, 24, "Contributor")
    d.actor("local", 154, 42, "Local model\nruntime (Ollama)", "system")
    d.actor("cloud", 154, 14, "Cloud model API", "system")
    d.general("contrib", "user")
    d.case("register", 46, 52, "Register a model\n(models.yaml / add-model)", w=28)
    d.case("calibrate", 46, 36, "Calibrate a model\n(write its scorecard)", w=28)
    d.case("evaluate", 46, 12, "Run evaluation\nand ablation", w=28)
    d.case("grade", 76, 24, "Grade answers on the\npublic benchmark", w=26)
    d.case("ask_gen", 112, 24, "Generate answers\nto benchmark tasks", w=26)
    for k in ("register", "calibrate", "evaluate"):
        d.assoc("contrib", k)
    d.dep("calibrate", "grade", "include", lab_t=0.5, lab_dx=-2.5, lab_dy=1.2)
    d.dep("evaluate", "grade", "include", lab_t=0.5, lab_dx=-2.5, lab_dy=-1.2)
    d.dep("calibrate", "ask_gen", "include", lab_t=0.62, lab_dy=1.6)
    d.dep("evaluate", "ask_gen", "include", lab_t=0.62, lab_dy=-1.6)
    d.assoc("ask_gen", "local")
    d.assoc("ask_gen", "cloud")
    d.save(out / "use_case_contributor.png")


# =========================================================================== PlantUML helpers
def plantuml(src: str, out_png: Path, keep_src: Path | None = None):
    src_dir = keep_src or out_png.parent
    src_dir.mkdir(parents=True, exist_ok=True)
    f = src_dir / (out_png.stem + ".puml")
    f.write_text(src)
    subprocess.run(["java", "-DPLANTUML_LIMIT_SIZE=16384", "-jar", str(JAR), "-tpng", "-Sdpi=200", "-o", str(out_png.parent.resolve()), str(f)],
                   check=True, capture_output=True)
    print("  ", out_png.name)


SKIN = """skinparam shadowing false
skinparam defaultFontName "DejaVu Sans"
skinparam defaultFontSize 12
skinparam classAttributeIconSize 0
skinparam ArrowColor #333333
skinparam classBorderColor #1F4E79
skinparam classBackgroundColor #FFFFFF
skinparam classHeaderBackgroundColor #EEF4FC
skinparam stereotypeCBackgroundColor #EEF4FC
skinparam packageBorderColor #8A8984
skinparam sequenceParticipantBorderColor #1F4E79
skinparam sequenceParticipantBackgroundColor #EEF4FC
skinparam sequenceLifeLineBorderColor #8A8984
skinparam sequenceGroupBorderColor #1F4E79
skinparam sequenceGroupHeaderFontColor #1F4E79
skinparam activityBorderColor #1F4E79
skinparam activityBackgroundColor #EEF4FC
skinparam activityDiamondBorderColor #1F4E79
skinparam activityDiamondBackgroundColor #FFFFFF
skinparam swimlaneBorderColor #8A8984
skinparam noteBackgroundColor #FFFBEA
skinparam noteBorderColor #B8A86A
"""


# =========================================================================== class diagram from the code
# Attribute types for attributes assigned in __init__ without an annotation, read from the code that
# assigns them.  Every key is checked against the attributes `ast` finds, so a renamed or removed
# attribute fails the build instead of leaving the diagram stale.
INIT_TYPES = {
    "Assistant": {"routing": "bool", "use_memory": "bool", "use_rag": "bool", "weights": "dict[str, float]",
                  "specs": "dict[str, ModelSpec]", "cards": "dict[str, dict]", "policy": "str",
                  "pool": "list[str] | None", "fixed_model": "str", "clf": "IntentClassifier | None",
                  "memory": "Memory | None", "retriever": "Retriever | None", "ledger": "bool",
                  "cascade": "bool", "session_id": "str"},
    "Memory": {"db": "sqlite3.Connection", "user_id": "str", "embedder": "str | None"},
    "Retriever": {"index": "dict"},
    "IntentClassifier": {"pipe": "sklearn Pipeline", "classes": "list[str]"},
}
MODULES = ["pipeline", "cascade", "router", "scorecard", "registry", "providers", "intent", "memory", "rag",
           "bench/runner", "bench/graders"]
MODULE_FUNCS = {"cascade", "router", "scorecard", "registry", "providers", "bench/runner", "bench/graders"}


def _args(fn: ast.FunctionDef, drop=("self", "cls"), limit=80):
    """Parameters with their types; keyword-only parameters are listed after '*'.  Very long
    signatures keep the types of the positional parameters only (types may be omitted in UML)."""
    def fmt(args, typed):
        return [f"{a.arg}: {ast.unparse(a.annotation)}" if typed and a.annotation is not None else a.arg
                for a in args if a.arg not in drop]
    pos, kw = fn.args.args, fn.args.kwonlyargs
    full = ", ".join(fmt(pos, True) + (["*"] + fmt(kw, True) if kw else []))
    if len(full) <= limit:
        return full
    return ", ".join(fmt(pos, True) + (["*"] + fmt(kw, False) if kw else []))


def _parse(path: Path):
    tree = ast.parse(path.read_text())
    classes, funcs = [], []
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            bases = [ast.unparse(b) for b in n.bases]
            fields, methods, init_attrs = [], [], []
            for b in n.body:
                if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name):
                    fields.append((b.target.id, ast.unparse(b.annotation)))
                elif isinstance(b, ast.FunctionDef):
                    if b.name == "__init__":
                        for s in ast.walk(b):
                            if (isinstance(s, ast.Attribute) and isinstance(s.value, ast.Name) and s.value.id == "self"
                                    and isinstance(s.ctx, ast.Store) and s.attr not in init_attrs):
                                init_attrs.append(s.attr)
                    elif not b.name.startswith("_"):
                        static = any(isinstance(d, ast.Name) and d.id == "classmethod" for d in b.decorator_list)
                        ret = f": {ast.unparse(b.returns)}" if b.returns is not None else ""
                        methods.append(("{static} " if static else "") + f"+{b.name}({_args(b)}){ret}")
            if init_attrs:
                types = INIT_TYPES.get(n.name, {})
                missing = set(init_attrs) - set(types)
                stale = set(types) - set(init_attrs)
                if missing or stale:
                    raise SystemExit(f"class diagram out of date for {n.name}: missing {missing}, stale {stale}")
                fields += [(a, types[a]) for a in init_attrs]
            classes.append((n.name, bases, fields, methods))
        elif isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
            ret = f": {ast.unparse(n.returns)}" if n.returns is not None else ""
            funcs.append(f"+{n.name}({_args(n, drop=())}){ret}")
    return classes, funcs


def _class_block(mods, funcs_for):
    lines, seen = [], set()
    for m in mods:
        classes, funcs = _parse(ROOT / "assistant" / f"{m}.py")
        pkg = m.replace("/", ".")
        lines.append(f'package "{pkg}.py" {{')
        for name, bases, fields, methods in classes:
            seen.add(name)
            exc = any(b.endswith("Error") for b in bases)
            lines.append(f"  class {name}{' <<exception>>' if exc else ''} {{")
            lines += [f"    +{a}: {t}" for a, t in fields]
            lines += [f"    {mm}" for mm in methods]
            lines.append("  }")
        if funcs and m in funcs_for:
            key = f"M_{pkg.replace('.', '_')}"
            seen.add(key)
            lines.append(f'  class "{pkg}" as {key} <<module>> {{')
            lines += [f"    {{static}} {f}" for f in funcs]
            lines.append("  }")
        lines.append("}")
    return lines, seen


HEAD = ["set separator none", "hide circle", "hide empty members", "skinparam packageStyle rectangle", "skinparam nodesep 30",
        "skinparam ranksep 45", "skinparam linetype polyline"]


def class_diagram(out: Path, src_dir: Path):
    """Runtime classes, in two diagrams that each fit a page.  (a) the pipeline and what it owns;
    (b) routing and model access.  A class drawn in the other diagram appears here by name only."""
    lines, _ = _class_block(["pipeline", "intent", "memory", "rag"], set())
    ext = ['class RouteDecision', 'class ModelSpec', 'class "router" as M_router <<module>>',
           'class "cascade" as M_cascade <<module>>', 'class "providers" as M_providers <<module>>']
    rel = [
        'Assistant *-- "0..1" IntentClassifier : clf',
        'Assistant *-- "0..1" Memory : memory',
        'Assistant *-- "0..1" Retriever : retriever',
        'Assistant --> "1..*" ModelSpec : specs',
        'Assistant ..> Response : <<create>>',
        'Response --> "0..1" IntentResult : intent',
        'Response --> "0..1" RouteDecision : decision',
        'Response --> "0..*" MemoryHit : memory_hits',
        'Response --> "0..*" Passage : passages',
        'Memory ..> MemoryHit : <<create>>',
        'Retriever ..> Passage : <<create>>',
        'IntentClassifier ..> IntentResult : <<create>>',
        'Assistant ..> M_router : route()',
        'Assistant ..> M_cascade : check_code(),\\ncheck_agreement()',
        'Assistant ..> M_providers : generate()',
    ]
    src = ["@startuml class_diagram", SKIN, *HEAD, "left to right direction", *lines, *ext, *rel, "@enduml"]
    plantuml("\n".join(src), out / "class_diagram.png", src_dir)

    lines, _ = _class_block(["router", "cascade", "registry", "scorecard", "providers"],
                            {"router", "cascade", "registry", "scorecard", "providers"})
    rel = [
        'M_cascade ..> CascadeStep : <<create>>',
        'M_cascade ..> M_bench_graders : grade()\\n(runs the code check)',
        'M_router ..> RouteDecision : <<create>>',
        'M_router ..> M_scorecard : load_all()',
        'M_router ..> M_registry : load()',
        'M_router ..> ModelSpec : eligibility',
        'M_registry ..> ModelSpec : <<create>>\\nfrom models.yaml',
        'M_providers ..> GenResult : <<create>>',
        'M_providers ..> ModelSpec : reads',
        'QuotaExhausted --|> ProviderError',
    ]
    ext = ['class "bench.graders" as M_bench_graders <<module>>']
    src = ["@startuml class_diagram_routing", SKIN, *HEAD, "left to right direction", *lines, *ext, *rel, "@enduml"]
    plantuml("\n".join(src), out / "class_diagram_routing.png", src_dir)


def class_diagram_bench(out: Path, src_dir: Path):
    """Benchmark side: how scorecards are produced."""
    lines, seen = _class_block(["bench/runner", "bench/graders", "scorecard"],
                               {"bench/runner", "bench/graders", "scorecard"})
    rel = ['M_bench_runner ..> M_bench_graders : grade()', 'M_bench_runner ..> M_scorecard : build(), save()']
    ext = ['class "providers" as M_providers <<module>>',
           'M_bench_runner ..> M_providers : generate()']
    src = ["@startuml class_diagram_bench", SKIN, *HEAD, *lines, *ext, *rel, "@enduml"]
    plantuml("\n".join(src), out / "class_diagram_bench.png", src_dir)


# =========================================================================== sequence diagram
SEQUENCE = """@startuml sequence
SKIN
autonumber
hide footbox
actor User
boundary "app.py\\n(Streamlit)" as UI
participant ":Assistant" as A
participant ":Memory" as M
participant ":Retriever" as R
participant ":IntentClassifier" as I
participant "router" as RT
participant "cascade" as C
participant "providers" as P
participant "Model backend\\n(Ollama / cloud API)" as B

User -> UI : type question (+ image)
UI -> A : chat(query, image)
activate A
opt memory on (builds B, D)
  A -> M : recall(query)
  M --> A : hits: list[MemoryHit]
end
opt RAG on (builds C, D)
  A -> R : retrieve(query)
  R --> A : passages: list[Passage]
end
opt memory on
  A -> M : recent_turns(session_id)
  M --> A : history (earlier turns)
end
note over A : private = hits or passages or history
opt routing on (builds A, D)
  A -> I : predict(query)
  I --> A : IntentResult
  A -> RT : route(intent, weights, image, private)
  note right of RT : RUS for every eligible model.\\nPrivate context: local models only.
  RT --> A : RouteDecision (chosen model)
end
opt local first on, intent is coding or reasoning, no image\\n(coding: only if the prompt has an assert)
  A -> P : generate(local model(s), prompt)
  P -> B : prompt
  B --> P : reply
  P --> A : GenResult
  A -> C : check_code(query, answer) / check_agreement(answers)
  C --> A : CascadeStep(kept_local)
end
opt answer not kept local (or local first off)
  A -> P : generate(chosen or fixed model, system prompt, messages, image)
  activate P
  opt chosen model is a cloud model
    P -> P : spend cap, rate limit,\\nappend to egress log (size + hash)
  end
  P -> B : prompt
  B --> P : reply + token counts
  P --> A : GenResult (text, seconds, tokens, cost)
  deactivate P
end
opt passages used but no [n] marker
  A -> A : attach_sources(answer, passages)
end
opt memory on
  A -> M : add_turn(user), add_turn(assistant)
  note right of M : extract_facts() on the user turn;\\nsession summary every N turns
end
A -> A : append to cost ledger
A --> UI : Response(answer, model, decision, cost, left_device)
deactivate A
UI --> User : answer + Inspector
@enduml
"""

ACTIVITY = """@startuml activity
SKIN
|User|
start
:Type a question\\n(optionally attach an image);
|Assistant (this computer)|
if (Memory on?) then (yes)
  :Recall facts and summaries\\nsimilar to the question;
else (no)
endif
if (RAG on?) then (yes)
  :Retrieve top-k passages\\nfrom the user's documents;
else (no)
endif
if (Memory on?) then (yes)
  :Load recent turns of this session;
else (no)
endif
:private context = any memory hit,\\npassage or earlier turn;
if (Routing on?) then (yes)
  :Classify intent;
  if (Image attached?) then (yes)
    :Intent = multimodal;
  else (no)
  endif
  :Score every eligible model with RUS\\n(private context: local models only);
  :Pick the highest RUS;
else (no)
  :Use the fixed model;
endif
if (Routing on, local first on,\\ncoding or maths, no image?) then (yes)
  if (Intent?) then (coding)
    if (Prompt has an assert?) then (yes)
      :Local coder (Ollama) writes the code;
      :Run the code against\\nthe prompt's assert;
    else (no)
    endif
  else (maths)
    :Two local models (Ollama) answer;
    :Compare their final numbers;
  endif
else (no)
endif
if (Local answer passed its check?) then (yes)
  :Keep the local answer;
else (no)
  |Model backend|
  :Chosen (or fixed) model answers;
  note right
    a cloud model only when
    the query has no private
    context; logged in the
    egress log
  end note
  |Assistant (this computer)|
endif
if (Passages used but no [n] marker?) then (yes)
  :Append the sources used;
else (no)
endif
if (Memory on?) then (yes)
  :Store the turn;\\nextract facts from the user's message;
else (no)
endif
:Append to cost ledger;
|User|
:Read the answer and the\\nrouting decision (Inspector);
stop
@enduml
"""


def sequence(out: Path, src_dir: Path):
    plantuml(SEQUENCE.replace("SKIN", SKIN), out / "sequence.png", src_dir)


def activity(out: Path, src_dir: Path):
    plantuml(ACTIVITY.replace("SKIN", SKIN), out / "activity.png", src_dir)


def main(out: Path):
    src = out / "src"
    use_case_user(out)
    use_case_contributor(out)
    print("   use_case_user.png, use_case_contributor.png")
    class_diagram(out, src)
    class_diagram_bench(out, src)
    sequence(out, src)
    activity(out, src)


if __name__ == "__main__":
    main(ROOT / "docs" / "diagrams")
