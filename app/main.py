"""
Sentinel Forge — AI Influence & Boundary Intelligence Platform
Ultra-light build: pure Python TF-IDF, no ML frameworks.
"""
from __future__ import annotations

import uuid

from fastapi import FastAPI

from app.analysis.boundary import analyze_boundary_pressure
from app.analysis.identity import analyze_identity_drift
from app.analysis.narrative import analyze_narrative_phase
from app.analysis.policy import analyze_policy_consistency
from app.analysis.scoring import classify_risk, compute_alignment_risk, compute_escalation_index
from app.corpus import ALL_GROUPS
from app.embeddings import EmbeddingEngine
from app.models import log_result
from app.schemas import AnalysisRequest, AnalysisResult, BatchRequest, BatchResult, Role

app = FastAPI(
    title="Sentinel Forge",
    description="AI Influence, Boundary & Alignment Intelligence Platform",
    version="1.0.0",
)


@app.on_event("startup")
def startup() -> None:
    """Build TF-IDF vocabulary and all cluster centroids once at startup."""
    EmbeddingEngine.build(ALL_GROUPS)


def _run_analysis(request: AnalysisRequest) -> AnalysisResult:
    transcript = request.transcript
    user_msgs = [m.content for m in transcript if m.role == Role.user]
    asst_msgs = [m.content for m in transcript if m.role == Role.assistant]
    pairs = [(m.role.value, m.content) for m in transcript]

    boundary = analyze_boundary_pressure(user_msgs)
    narrative = analyze_narrative_phase(pairs)
    drift = analyze_identity_drift(asst_msgs)
    policy = analyze_policy_consistency(asst_msgs)

    escalation = compute_escalation_index(boundary["score"], drift, narrative["phase"])
    risk = compute_alignment_risk(escalation, policy)
    classification = classify_risk(risk)

    result = AnalysisResult(
        id=str(uuid.uuid4()),
        label=request.label,
        boundary_pressure_score=boundary["score"],
        narrative_phase=narrative["phase"],
        narrative_confidence=narrative["confidence"],
        manipulation_vectors=boundary["vectors"],
        identity_drift_score=drift,
        policy_consistency_delta=policy,
        escalation_index=escalation,
        alignment_risk=risk,
        summary_classification=classification,
        turn_count=len(transcript),
        user_turn_count=len(user_msgs),
        assistant_turn_count=len(asst_msgs),
    )
    log_result(result.model_dump())
    return result


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "sentinel-forge"}


@app.post("/analyze", response_model=AnalysisResult, tags=["analysis"])
def analyze(request: AnalysisRequest) -> AnalysisResult:
    return _run_analysis(request)


@app.post("/analyze/batch", response_model=BatchResult, tags=["analysis"])
def analyze_batch(request: BatchRequest) -> BatchResult:
    results = [_run_analysis(r) for r in request.transcripts]
    mean = sum(r.alignment_risk for r in results) / len(results)
    return BatchResult(
        results=results,
        batch_size=len(results),
        mean_alignment_risk=round(mean, 4),
    )
