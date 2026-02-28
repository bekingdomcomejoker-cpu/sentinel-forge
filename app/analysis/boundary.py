from __future__ import annotations
from app.embeddings import EmbeddingEngine
from app.corpus import VECTOR_CLUSTERS

_THRESHOLD = 0.22


def analyze_boundary_pressure(user_messages: list[str]) -> dict:
    if not user_messages:
        return {"score": 0.0, "vectors": []}
    engine = EmbeddingEngine.get()
    score = engine.top3_mean(user_messages, "boundary")
    vectors = [
        name for name in VECTOR_CLUSTERS
        if max(engine.cosine(m, f"vec_{name}") for m in user_messages) >= _THRESHOLD
    ]
    return {"score": score, "vectors": vectors}
