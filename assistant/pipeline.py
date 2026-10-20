"""One pipeline, three switches.  The switches ARE the ablation.

    Assistant(routing=True,  memory=False, rag=False)  -> Model A
    Assistant(routing=False, memory=True,  rag=False)  -> Model B
    Assistant(routing=False, memory=False, rag=True)   -> Model C
    Assistant(routing=True,  memory=True,  rag=True)   -> Model D

Flow per query:
    recall memory (if memory) -> retrieve passages (if rag)
    -> classify intent + score every eligible model with RUS (if routing, else fixed model)
       (private context present => only local models are eligible, by default)
    -> build system prompt -> generate -> store turn (if memory) -> append to cost ledger
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field

from . import cascade as cascade_mod
from . import config, providers, registry, scorecard
from .intent import IntentClassifier, IntentResult
from .memory import Memory, MemoryHit
from .rag import Passage, Retriever, format_context
from .router import RouteDecision, route

VARIANTS = {
    "A": dict(routing=True, memory=False, rag=False),
    "B": dict(routing=False, memory=True, rag=False),
    "C": dict(routing=False, memory=False, rag=True),
    "D": dict(routing=True, memory=True, rag=True),
    "baseline": dict(routing=False, memory=False, rag=False),
}

BASE_SYSTEM = (
    "You are a helpful personal assistant. Answer concisely and directly. "
    "If the answer is not known from the conversation or the provided context, say you do not know "
    "rather than guessing."
)


@dataclass
class Response:
    answer: str
    model: str
    intent: IntentResult | None
    decision: RouteDecision | None
    memory_hits: list[MemoryHit]
    passages: list[Passage]
    latency_s: float
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    private_context: bool = False
    left_device: bool = False
    cascade_note: str = ""
    sources_added: bool = False
    system_prompt: str = field(repr=False, default="")


CITATION = re.compile(r"\[\d+\]")
_WORD = re.compile(r"[a-z0-9]+")


def attach_sources(answer: str, passages: list[Passage], min_overlap: float = 0.5) -> tuple[str, bool]:
    """Make every grounded answer traceable to the user's documents.

    Small local models often answer correctly from the retrieved passages but drop the [n] markers
    the system prompt asks for.  If the answer has no marker, the passages it actually draws on (at
    least half of its content words appear in the passage; at most two) are listed after it.  An
    answer that uses no passage (for example "I don't know") gets nothing.  Returns (answer, added).
    """
    if not passages or CITATION.search(answer):
        return answer, False
    words = {w for w in _WORD.findall(answer.lower()) if len(w) > 3}
    if not words:
        return answer, False
    scored = sorted(((len(words & set(_WORD.findall(p.text.lower()))) / len(words), k, p)
                     for k, p in enumerate(passages, 1)), key=lambda x: (-x[0], x[1]))
    used = sorted((k, p) for ov, k, p in scored[:2] if ov >= min_overlap)
    if not used:
        return answer, False
    return answer + "\n\nSources: " + "; ".join(f"[{k}] {p.doc}" for k, p in used), True


class Assistant:
    def __init__(self, routing: bool, memory: bool, rag: bool, *, weights=None, embedder=None, memory_db=None,
                 user_id="default", fixed_model=None, specs=None, cards=None, policy=None, pool=None,
                 ledger: bool = True, cascade: bool = False):
        self.routing, self.use_memory, self.use_rag = routing, memory, rag
        self.weights = weights or config.DEFAULT_WEIGHTS
        self.specs = specs or registry.load()
        self.cards = cards if cards is not None else scorecard.load_all()
        self.policy = policy or config.PRIVATE_CONTEXT_POLICY
        self.pool = pool
        self.fixed_model = fixed_model or config.FIXED_MODEL
        if self.fixed_model not in self.specs:
            raise ValueError(f"fixed model {self.fixed_model!r} is not in the registry")
        self.clf = IntentClassifier() if routing else None
        if memory and memory_db is None:
            config.ensure_home()
        self.memory = Memory(memory_db, user_id=user_id, embedder=embedder) if memory else None
        self.retriever = Retriever() if rag else None
        self.ledger = ledger
        self.cascade = cascade
        self.session_id = uuid.uuid4().hex[:8]

    @classmethod
    def variant(cls, name: str, **kw) -> "Assistant":
        return cls(**VARIANTS[name], **kw)

    def new_session(self):
        self.session_id = uuid.uuid4().hex[:8]

    def chat(self, query: str, image: str | None = None) -> Response:
        hits = self.memory.recall(query) if self.use_memory else []
        passages = self.retriever.retrieve(query) if self.use_rag else []
        history = self.memory.recent_turns(self.session_id) if self.use_memory else []
        private = bool(hits or passages or history)

        intent = decision = None
        model = self.fixed_model
        if self.routing:
            intent = self.clf.predict(query)
            if image and intent.intent != "multimodal":
                intent = IntentResult("multimodal", 1.0, {**intent.probs, "multimodal": 1.0})
            decision = route(intent.intent, self.weights, cards=self.cards, specs=self.specs, image=bool(image),
                             private=private, policy=self.policy, pool=self.pool)
            model = decision.chosen or self.fixed_model

        system = self._system_prompt(hits, passages)
        messages = history + [{"role": "user", "content": query}]
        note, extra_s = "", 0.0
        if self.routing and self.cascade and not image and intent and intent.intent in ("coding", "reasoning"):
            kept = self._try_local(intent.intent, query, system, messages)
            if kept:
                model, g, note = kept
            else:
                note, extra_s = self._last_note, self._last_secs
        if not note.startswith("kept"):
            spec = self.specs[model]
            g = providers.generate(spec, system, messages, image=image, private=private, raise_errors=False)
            g.seconds = round(g.seconds + extra_s, 3)
        spec = self.specs[model]
        answer = g.text if not g.error else f"[{model} failed: {g.error}]"
        added = False
        if passages and not g.error:
            answer, added = attach_sources(answer, passages)

        if self.use_memory:
            self.memory.add_turn(self.session_id, "user", query)
            self.memory.add_turn(self.session_id, "assistant", answer)
        resp = Response(answer, model, intent, decision, hits, passages, g.seconds, g.cost_usd, g.tokens_in,
                        g.tokens_out, private, not spec.local, note, added, system)
        if self.ledger:
            self._log(resp)
        return resp

    def _try_local(self, intent: str, query: str, system: str, messages: list[dict]):
        """Local-first cascade (assistant/cascade.py): answer locally, check without any reference answer,
        keep it if the check passes.  Returns (model, GenResult, note) or None to escalate."""
        self._last_note, self._last_secs = "", 0.0
        needed = [cascade_mod.CODER] if intent == "coding" else list(cascade_mod.REASONERS)
        if any(m not in self.specs for m in needed):
            return None
        if intent == "coding" and not cascade_mod.prompt_asserts(query):
            self._last_note = "escalated: no example test in the prompt to check against"
            return None
        outs = {m: providers.generate(self.specs[m], system, messages, raise_errors=False) for m in needed}
        secs = sum(o.seconds for o in outs.values())
        step = (cascade_mod.check_code(query, outs[cascade_mod.CODER].text) if intent == "coding"
                else cascade_mod.check_agreement({m: o.text for m, o in outs.items()}))
        if step.kept_local:
            g = outs[step.model]
            g.seconds = round(secs, 3)
            return step.model, g, f"kept on this computer: {step.reason}"
        self._last_note, self._last_secs = f"escalated: {step.reason}", secs
        return None

    def _log(self, r: Response):
        try:
            config.ensure_home()
            with config.LEDGER.open("a") as f:
                f.write(json.dumps({"ts": round(time.time(), 3), "model": r.model,
                                    "intent": r.intent.intent if r.intent else None, "cost_usd": r.cost_usd,
                                    "tokens_in": r.tokens_in, "tokens_out": r.tokens_out,
                                    "latency_s": r.latency_s, "left_device": r.left_device,
                                    "private_context": r.private_context}) + "\n")
        except OSError:
            pass

    def _system_prompt(self, hits: list[MemoryHit], passages: list[Passage]) -> str:
        parts = [BASE_SYSTEM]
        if hits:
            parts.append("What you remember about this user:\n" + "\n".join(f"- {h.text}" for h in hits))
        if passages:
            parts.append(
                "Retrieved context. Answer ONLY from it and cite passages as [1], [2] etc. "
                "If it does not contain the answer, say so.\n" + format_context(passages)
            )
        return "\n\n".join(parts)
