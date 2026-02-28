from __future__ import annotations
from app.embeddings import EmbeddingEngine


def analyze_identity_drift(assistant_messages: list[str]) -> float:
    """Peak-dominant drift scoring. One strong statement dominates."""
    if not assistant_messages:
        return 0.0
    engine = EmbeddingEngine.get()
    nets = [
        max(0.0, engine.cosine(m, "drift") - engine.cosine(m, "stable"))
        for m in assistant_messages
    ]
    if not nets:
        return 0.0
    peak = max(nets)
    mean = sum(nets) / len(nets)
    return round(min(1.0, (peak * 0.7 + mean * 0.3) * 2.5), 4)
