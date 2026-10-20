"""Generation and embedding providers.

    generate(spec, system, messages, image=None, private=False) -> GenResult
        text, seconds, tokens_in, tokens_out, cost_usd, error

Backends: ollama (local), any OpenAI-compatible API, Anthropic, Gemini, and a deterministic mock.
Token counts come from each API's own usage field, so cost is measured, not estimated.

Every call to a non-local model is appended to the egress audit log (ASSISTANT_HOME/egress.jsonl):
time, model, characters sent, whether personal context was included, and a SHA-256 of the
payload.  The log stores no prompt text, so the audit trail itself leaks nothing.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

from . import config
from .registry import ModelSpec

try:  # optional .env support
    from dotenv import load_dotenv
    load_dotenv(config.ROOT / ".env")
except Exception:  # pragma: no cover
    pass


class ProviderError(RuntimeError):
    pass


@dataclass
class GenResult:
    text: str
    seconds: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: str | None = None


# --------------------------------------------------------------------------- helpers
def _b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _key(spec: ModelSpec) -> str:
    k = os.getenv(spec.api_key_env or "", "")
    if not k:
        raise ProviderError(f"{spec.name}: set {spec.api_key_env} in .env")
    return k


_last_call: dict[str, float] = {}
_rate_lock = threading.Lock()


def _respect_rpm(spec: ModelSpec):
    if not spec.rpm:
        return
    with _rate_lock:
        gap = 60.0 / spec.rpm
        wait = _last_call.get(spec.name, 0) + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call[spec.name] = time.time()


class QuotaExhausted(ProviderError):
    pass


_exhausted: set[str] = set()


def _post(url, *, retries=4, **kw):
    for attempt in range(retries + 1):
        r = requests.post(url, **kw)
        if r.status_code == 429 and re.search(r"per.?day|PerDay|daily|insufficient_quota|credit balance", r.text, re.I):
            raise QuotaExhausted(f"quota exhausted: {r.text[:160]}")
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(min(60, 5 * 2 ** attempt))
            continue
        return r
    return r


# --------------------------------------------------------------------------- backends
# Each returns (text, tokens_in, tokens_out)
def _ollama(spec, system, messages, image, max_tokens):
    msgs = [{"role": "system", "content": system}] + messages
    if image:
        msgs[-1] = dict(msgs[-1], images=[_b64(image)])
    opts = {"temperature": 0, "seed": 7, "num_predict": max_tokens,
            **{k: v for k, v in spec.options.items() if k != "min_output_tokens"}}
    r = requests.post(f"{config.OLLAMA_HOST}/api/chat", timeout=900,
                      json={"model": spec.model, "messages": msgs, "stream": False, "options": opts})
    if r.status_code != 200:
        raise ProviderError(f"Ollama {r.status_code}: {r.text[:200]}")
    j = r.json()
    return j["message"]["content"].strip(), j.get("prompt_eval_count", 0), j.get("eval_count", 0)


def _openai(spec, system, messages, image, max_tokens):
    base = (spec.base_url or "https://api.openai.com/v1").rstrip("/")
    msgs = [{"role": "system", "content": system}] + [dict(m) for m in messages]
    if image:
        msgs[-1]["content"] = [{"type": "text", "text": msgs[-1]["content"]},
                               {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(image)}"}}]
    opts = {k: v for k, v in spec.options.items() if k != "min_output_tokens"}
    body = {"model": spec.model, "messages": msgs, "max_completion_tokens": max_tokens}
    if not opts.pop("no_temperature", False):  # reasoning models (gpt-5 family) only accept the default temperature
        body["temperature"] = 0
    if opts.pop("legacy_max_tokens", False):   # some OpenAI-compatible servers only know max_tokens
        body["max_tokens"] = body.pop("max_completion_tokens")
    r = _post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {_key(spec)}"}, timeout=180,
              json={**body, **opts})
    if r.status_code != 200:
        raise ProviderError(f"{spec.name} {r.status_code}: {r.text[:200]}")
    j = r.json()
    u = j.get("usage", {})
    return j["choices"][0]["message"]["content"].strip(), u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def _anthropic(spec, system, messages, image, max_tokens):
    msgs = [dict(m) for m in messages]
    if image:
        msgs[-1]["content"] = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(image)}},
                               {"type": "text", "text": msgs[-1]["content"]}]
    opts = {k: v for k, v in spec.options.items() if k != "min_output_tokens"}
    body = {"model": spec.model, "system": system, "messages": msgs, "max_tokens": max_tokens}
    if not opts.pop("no_temperature", False):  # newer Claude models reject the temperature parameter
        body["temperature"] = 0
    r = _post("https://api.anthropic.com/v1/messages", timeout=180,
              headers={"x-api-key": _key(spec), "anthropic-version": "2023-06-01"}, json={**body, **opts})
    if r.status_code != 200:
        raise ProviderError(f"{spec.name} {r.status_code}: {r.text[:200]}")
    j = r.json()
    u = j.get("usage", {})
    return "".join(b.get("text", "") for b in j["content"]).strip(), u.get("input_tokens", 0), u.get("output_tokens", 0)


def _gemini(spec, system, messages, image, max_tokens):
    contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in messages]
    if image:
        contents[-1]["parts"].append({"inline_data": {"mime_type": "image/png", "data": _b64(image)}})
    gen_cfg = {"temperature": 0, "maxOutputTokens": max_tokens, **spec.options.get("generationConfig", {})}
    r = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{spec.model}:generateContent",
              headers={"x-goog-api-key": _key(spec)}, timeout=180,
              json={"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
                    "generationConfig": gen_cfg})
    if r.status_code != 200:
        raise ProviderError(f"{spec.name} {r.status_code}: {r.text[:200]}")
    j = r.json()
    u = j.get("usageMetadata", {})
    parts = j.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    out = u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)  # thinking is billed as output
    return text, u.get("promptTokenCount", 0), out


def _mock(spec, system, messages, image, max_tokens):
    """Deterministic stand-in for tests: answers only from context the pipeline injected
    (memory facts or retrieved passages); with no usable context it says it does not know."""
    question = messages[-1]["content"].lower()
    q_words = set(re.findall(r"[a-z0-9]+", question)) - _STOP
    best, best_overlap, best_cite = None, 0, ""
    for line in system.splitlines():
        line = line.strip(" -•")
        if not line or line.endswith(":") or line.lower().startswith(("you are", "rules", "if the", "cite", "answer")):
            continue
        cite = ""
        m = re.match(r"(\[\d+\]) \([^)]*\) ", line)
        if m:
            cite, line = m.group(1) + " ", line[m.end():]
        for sent in re.split(r"(?<=[.!?])\s+", line):
            words = set(re.findall(r"[a-z0-9]+", sent.lower())) - _STOP
            overlap = len(q_words & words)
            if overlap > best_overlap:
                best, best_overlap, best_cite = sent, overlap, cite
    text = best_cite + best if best and best_overlap >= 1 else "I don't have that information in my context."
    return text, len(system.split()) + len(question.split()), len(text.split())


_STOP = set("the a an is are was were of to in on for and or what which who how do does did my me i you your it this "
            "that with at by from as be can please tell about".split())

_BACKENDS = {"ollama": _ollama, "openai": _openai, "anthropic": _anthropic, "gemini": _gemini, "mock": _mock}


# --------------------------------------------------------------------------- public API
def _audit(spec: ModelSpec, system: str, messages: list[dict], image: str | None, private: bool):
    payload = json.dumps({"system": system, "messages": messages}, sort_keys=True)
    rec = {"ts": round(time.time(), 3), "model": spec.name, "provider": spec.provider, "chars": len(payload),
           "image": bool(image), "private_context": private,
           "sha256": hashlib.sha256(payload.encode()).hexdigest()}
    try:
        config.ensure_home()
        with config.EGRESS_LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


_spend_lock = threading.Lock()


def _spent_usd() -> float:
    try:
        return float(config.SPEND_FILE.read_text())
    except (OSError, ValueError):
        return 0.0


def _add_spend(usd: float):
    with _spend_lock:
        config.ensure_home()
        config.SPEND_FILE.write_text(f"{_spent_usd() + usd:.8f}")


def generate(spec: ModelSpec, system: str, messages: list[dict], image: str | None = None, *,
             private: bool = False, max_tokens: int = 1024, raise_errors: bool = True) -> GenResult:
    backend = config.PROVIDER_OVERRIDE or spec.provider
    paid = backend != "mock" and not spec.local and not spec.free_tier
    # models that think before answering spend output tokens on thinking; give them room so the answer
    # is not cut off (set per model in models.yaml)
    max_tokens = max(max_tokens, int(spec.options.get("min_output_tokens", 0)))
    stop = None
    if spec.name in _exhausted:
        stop = f"{spec.name}: quota exhausted earlier in this run"
    elif paid and config.SPEND_CAP_USD is not None and _spent_usd() >= config.SPEND_CAP_USD:
        stop = f"spend cap ${config.SPEND_CAP_USD} reached (spent ${_spent_usd():.2f}); paid models stopped"
    if stop:
        if raise_errors:
            raise ProviderError(stop)
        return GenResult("", 0.0, error=stop)
    if image and not spec.vision and backend != "mock":
        err = f"{spec.name} does not accept images"
        if raise_errors:
            raise ProviderError(err)
        return GenResult("", 0.0, error=err)
    if backend != "mock" and not spec.local:
        _respect_rpm(spec)
        _audit(spec, system, messages, image, private)
    t0 = time.time()
    try:
        text, tin, tout = _BACKENDS[backend](spec, system, messages, image, max_tokens)
    except Exception as e:  # noqa: BLE001
        if isinstance(e, QuotaExhausted):
            _exhausted.add(spec.name)
        if raise_errors:
            raise
        return GenResult("", round(time.time() - t0, 3), error=str(e)[:300])
    secs = round(time.time() - t0, 3)
    cost = spec.cost_usd(tin or 0, tout or 0)
    if paid:
        _add_spend(cost)
    return GenResult(text, secs, int(tin or 0), int(tout or 0), cost)


# --------------------------------------------------------------------------- embeddings (always local)
_hash_vec = None


def _tfidf_embed(texts: list[str]) -> np.ndarray:
    """Fit-free lexical embedding (hashed char n-grams, L2 normed)."""
    global _hash_vec
    from sklearn.feature_extraction.text import HashingVectorizer
    if _hash_vec is None:
        _hash_vec = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** 18, norm="l2",
                                      alternate_sign=False)
    return _hash_vec.transform(texts).toarray().astype(np.float32)


def _ollama_embed(texts: list[str]) -> np.ndarray:
    r = requests.post(f"{config.OLLAMA_HOST}/api/embed",
                      json={"model": config.OLLAMA_EMBED_MODEL, "input": texts}, timeout=300)
    if r.status_code != 200:
        raise ProviderError(f"Ollama embed {r.status_code}: {r.text[:300]}")
    v = np.array(r.json()["embeddings"], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


def embed(texts: list[str], embedder: str | None = None) -> np.ndarray:
    embedder = embedder or config.EMBEDDER
    if embedder == "ollama":
        return _ollama_embed(texts)
    return _tfidf_embed(texts)


def ollama_available() -> bool:
    try:
        return requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


def ollama_models() -> set[str]:
    try:
        return {m["name"] for m in requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3).json()["models"]}
    except Exception:
        return set()


GEMINI_PRICES = {  # USD per 1M tokens (in, out), list prices Oct 2026; used only when preflight swaps the model id
    "gemini-2.5-flash-lite": (0.10, 0.40), "gemini-2.5-flash": (0.30, 2.50), "gemini-3-flash": (0.50, 3.00),
    "gemini-3.1-flash-lite": (0.25, 1.50), "gemini-3.5-flash-lite": (0.30, 2.50), "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.6-flash": (0.75, 3.75), "gemini-3.7-flash": (0.75, 3.75),
}


def gemini_models(spec: ModelSpec) -> list[str]:
    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                     headers={"x-goog-api-key": _key(spec)}, timeout=30)
    r.raise_for_status()
    return [m["name"].split("/", 1)[1] for m in r.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])]
