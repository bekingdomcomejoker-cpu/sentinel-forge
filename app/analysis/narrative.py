from __future__ import annotations
from app.embeddings import EmbeddingEngine
from app.corpus import PHASE_CLUSTERS


def analyze_narrative_phase(messages: list[tuple[str, str]]) -> dict:
    """Position-weighted narrative phase classification."""
    if not messages:
        return {"phase": "Curiosity", "confidence": 0.0}
    engine = EmbeddingEngine.get()
    n = len(messages)
    scores: dict[str, float] = {p: 0.0 for p in PHASE_CLUSTERS}
    for i, (_, content) in enumerate(messages):
        if not content.strip():
            continue
        w = 0.5 + (i / max(n - 1, 1))  # 0.5 → 1.5 linear ramp
        for phase in PHASE_CLUSTERS:
            scores[phase] += engine.cosine(content, f"phase_{phase}") * w
    total = sum(scores.values())
    if total < 1e-10:
        return {"phase": "Curiosity", "confidence": 0.0}
    norm = {p: v / total for p, v in scores.items()}
    dominant = max(norm, key=lambda p: norm[p])
    return {"phase": dominant, "confidence": round(norm[dominant], 4)}
