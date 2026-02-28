from enum import Enum
from pydantic import BaseModel, Field, model_validator


class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Message(BaseModel):
    role: Role
    content: str = Field(..., min_length=1)


class AnalysisRequest(BaseModel):
    transcript: list[Message] = Field(..., min_length=1)
    label: str | None = None

    @model_validator(mode="after")
    def needs_content(self) -> "AnalysisRequest":
        roles = {m.role for m in self.transcript}
        if Role.user not in roles and Role.assistant not in roles:
            raise ValueError("Need at least one user or assistant message")
        return self


class BatchRequest(BaseModel):
    transcripts: list[AnalysisRequest] = Field(..., min_length=1, max_length=50)


class AnalysisResult(BaseModel):
    id: str
    label: str | None
    boundary_pressure_score: float
    narrative_phase: str
    narrative_confidence: float
    manipulation_vectors: list[str]
    identity_drift_score: float
    policy_consistency_delta: float
    escalation_index: float
    alignment_risk: float
    summary_classification: str
    turn_count: int
    user_turn_count: int
    assistant_turn_count: int


class BatchResult(BaseModel):
    results: list[AnalysisResult]
    batch_size: int
    mean_alignment_risk: float
