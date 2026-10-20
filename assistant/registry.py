"""Model registry: the list of models the assistant may use, read from models.yaml."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from . import config

PROVIDERS = {"ollama", "openai", "anthropic", "gemini", "mock"}


@dataclass
class ModelSpec:
    name: str
    provider: str
    model: str
    local: bool = False
    vision: bool = False
    price_in: float = 0.0   # USD / 1M input tokens
    price_out: float = 0.0  # USD / 1M output tokens
    base_url: str | None = None
    api_key_env: str | None = None
    rpm: int | None = None
    enabled: bool = True
    free_tier: bool = False  # cloud model used through a no-billing free tier
    options: dict = field(default_factory=dict)

    def cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.price_in + tokens_out * self.price_out) / 1e6

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, {}, [])}


def load(path: Path | None = None, include_disabled: bool = False) -> dict[str, ModelSpec]:
    """ASSISTANT_DISABLE (comma list) switches models off for one run, e.g. ones that failed preflight."""
    import os
    path = Path(path or config.REGISTRY_PATH)
    off = {x.strip() for x in os.getenv("ASSISTANT_DISABLE", "").split(",") if x.strip()}
    free_only = os.getenv("ASSISTANT_FREE_ONLY", "") == "1"  # zero-spend guard: never call a paid cloud model
    raw = yaml.safe_load(path.read_text()) or {}
    out: dict[str, ModelSpec] = {}
    for e in raw.get("models", []):
        spec = ModelSpec(**e)
        if spec.provider not in PROVIDERS:
            raise ValueError(f"{spec.name}: unknown provider {spec.provider!r}, use one of {sorted(PROVIDERS)}")
        if spec.name in out:
            raise ValueError(f"duplicate model name {spec.name!r} in {path}")
        if spec.local and (spec.price_in or spec.price_out):
            raise ValueError(f"{spec.name}: a local model has no per-token price")
        if free_only and not spec.local and not spec.free_tier:
            continue
        if (spec.enabled and spec.name not in off) or include_disabled:
            out[spec.name] = spec
    return out


def add(spec: ModelSpec, path: Path | None = None) -> None:
    path = Path(path or config.REGISTRY_PATH)
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    raw = raw or {}
    models = [m for m in raw.get("models", []) if m.get("name") != spec.name]
    models.append(spec.to_dict())
    raw["models"] = models
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
