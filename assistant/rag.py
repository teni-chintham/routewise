"""Model C: retrieval-augmented generation.

build_index()  chunk every file in data/corpus, embed, pickle to models/
Retriever      cosine top-k over the chunks, returns passages with source ids
                so the generator can cite [1], [2], ...
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, providers


@dataclass
class Passage:
    doc: str
    chunk_id: int
    text: str
    score: float


def chunk_text(text: str, words: int = config.CHUNK_WORDS, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    toks = text.split()
    out, i = [], 0
    while i < len(toks):
        out.append(" ".join(toks[i:i + words]))
        if i + words >= len(toks):
            break
        i += words - overlap
    return out


def build_index(corpus_dir: Path | None = None, index_path: Path | None = None, embedder: str | None = None) -> dict:
    corpus_dir = corpus_dir or config.CORPUS_DIR
    index_path = index_path or config.INDEX_PATH
    chunks, meta = [], []
    for f in sorted(corpus_dir.glob("*.md")) + sorted(corpus_dir.glob("*.txt")):
        text = re.sub(r"\s+", " ", f.read_text())
        for j, ch in enumerate(chunk_text(text)):
            chunks.append(ch)
            meta.append((f.name, j))
    if not chunks:
        raise RuntimeError(f"no documents in {corpus_dir}")
    embs = providers.embed(chunks, embedder)
    index = {"chunks": chunks, "meta": meta, "embs": embs, "embedder": embedder or config.EMBEDDER}
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_bytes(pickle.dumps(index))
    return {"documents": len(set(m[0] for m in meta)), "chunks": len(chunks), "dim": int(embs.shape[1])}


class Retriever:
    def __init__(self, index_path: Path | None = None):
        index_path = index_path or config.INDEX_PATH
        if not index_path.exists():
            raise FileNotFoundError(f"{index_path} missing. Run: python scripts/build_index.py")
        self.index = pickle.loads(index_path.read_bytes())

    def retrieve(self, query: str, top_k: int = config.RAG_TOP_K, min_sim: float = config.RAG_MIN_SIM) -> list[Passage]:
        q = providers.embed([query], self.index["embedder"])[0]
        sims = self.index["embs"] @ q
        order = np.argsort(-sims)[:top_k]
        return [
            Passage(self.index["meta"][i][0], self.index["meta"][i][1], self.index["chunks"][i], float(sims[i]))
            for i in order if sims[i] >= min_sim
        ]


def format_context(passages: list[Passage]) -> str:
    return "\n".join(f"[{k + 1}] ({p.doc}) {p.text}" for k, p in enumerate(passages))
