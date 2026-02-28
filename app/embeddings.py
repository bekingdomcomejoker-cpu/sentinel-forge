"""
Pure Python TF-IDF + cosine similarity engine.
No sklearn. No torch. No transformers.

Vocabulary and centroids built ONCE at startup.
Per-request cost: tokenize + sparse dict dot product only.
Safe for 4GB Android / Termux.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

# ── Stopwords ────────────────────────────────────────────────────────────────

_STOP = frozenset({
    "a","an","the","is","it","in","on","at","to","for","of","and","or",
    "but","i","you","we","they","he","she","my","your","our","not","no",
    "be","do","have","with","this","that","are","was","were","will","can",
    "would","could","should","may","might","am","been","being","has","had",
    "does","did","so","if","as","by","from","about","more","than","just",
    "all","up","out","its","also","get","me","him","her","us","them",
})

# ── Tokenizer ─────────────────────────────────────────────────────────────────

def _tok(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]+", text.lower())
            if t not in _STOP and len(t) > 1]

# ── Sparse TF-IDF primitives ──────────────────────────────────────────────────

def _tf(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    c = Counter(tokens)
    n = len(tokens)
    return {t: v / n for t, v in c.items()}

def _build_idf(docs: list[list[str]]) -> dict[str, float]:
    N = len(docs)
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((N + 1) / (f + 1)) + 1.0 for t, f in df.items()}

def _vec(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = _tf(tokens)
    return {t: tf[t] * idf[t] for t in tf if t in idf}

def _norm(v: dict[str, float]) -> float:
    return math.sqrt(sum(x * x for x in v.values())) if v else 0.0

def _cos(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(av * b[t] for t, av in a.items() if t in b)
    n = _norm(a) * _norm(b)
    return dot / n if n > 1e-10 else 0.0

def _centroid(vecs: list[dict[str, float]]) -> dict[str, float]:
    if not vecs:
        return {}
    merged: dict[str, float] = {}
    n = len(vecs)
    for v in vecs:
        for t, val in v.items():
            merged[t] = merged.get(t, 0.0) + val / n
    nm = _norm(merged)
    return {t: v / nm for t, v in merged.items()} if nm > 1e-10 else merged

# ── Singleton engine ──────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Build once with EmbeddingEngine.build(groups).
    Access anywhere with EmbeddingEngine.get().
    """
    _instance: Optional["EmbeddingEngine"] = None

    def __init__(self, idf: dict[str, float], centroids: dict[str, dict[str, float]]) -> None:
        self._idf = idf
        self._centroids = centroids

    @classmethod
    def build(cls, groups: dict[str, list[str]]) -> "EmbeddingEngine":
        """Build IDF + centroids from named phrase groups. Call once at startup."""
        if cls._instance is not None:
            return cls._instance
        all_texts = [t for phrases in groups.values() for t in phrases]
        tokenized = [_tok(t) for t in all_texts]
        idf = _build_idf(tokenized)
        centroids = {}
        for name, phrases in groups.items():
            vecs = [_vec(_tok(p), idf) for p in phrases]
            centroids[name] = _centroid(vecs)
        cls._instance = cls(idf, centroids)
        return cls._instance

    @classmethod
    def get(cls) -> "EmbeddingEngine":
        if cls._instance is None:
            raise RuntimeError("EmbeddingEngine not built. Call build() at startup.")
        return cls._instance

    def embed(self, text: str) -> dict[str, float]:
        return _vec(_tok(text), self._idf)

    def cosine(self, text: str, group: str) -> float:
        v = self.embed(text)
        c = self._centroids.get(group, {})
        return round(_cos(v, c), 4)

    def top3_mean(self, texts: list[str], group: str) -> float:
        sims = sorted([self.cosine(t, group) for t in texts], reverse=True)
        top = sims[:3]
        return round(sum(top) / len(top), 4) if top else 0.0
