import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

BackendName = Literal["graphiti", "lightrag-append", "lightrag-replace"]
BACKENDS: tuple[BackendName, ...] = ("graphiti", "lightrag-append", "lightrag-replace")


def safe_name(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", value):
        raise ValueError(
            "Use 1–80 letters, digits, underscores or hyphens; start with a letter/digit."
        )
    return value


class StrictModel(BaseModel):
    model_config = {"extra": "forbid"}


class Probe(StrictModel):
    id: str
    query: str = Field(min_length=1)
    expected_any: list[str] = Field(min_length=1)
    forbidden_any: list[str] = Field(default_factory=list)
    as_of: datetime | None = None
    category: str = "current"
    dimension: Literal["temporal", "retrieval", "lifecycle", "general"] = "general"
    temporal_role: Literal["current", "historical"] | None = None
    context_tokens: int | None = Field(default=None, ge=128, le=16000)

    @field_validator("as_of")
    @classmethod
    def timezone_required(cls, value):
        if value is not None and value.tzinfo is None:
            raise ValueError(
                "Timestamps must include a timezone, for example 2025-01-01T00:00:00Z."
            )
        return value


class Update(StrictModel):
    id: str
    document_id: str
    text: str = Field(default="", max_length=100_000)
    effective_at: datetime
    operation: Literal["upsert", "delete"] = "upsert"
    probes: list[Probe] = Field(default_factory=list)

    @field_validator("effective_at")
    @classmethod
    def timezone_required(cls, value):
        return Probe.timezone_required(value)

    @model_validator(mode="after")
    def nonempty_upsert(self):
        if self.operation == "upsert" and not self.text.strip():
            raise ValueError("Upserts require text.")
        return self

    def source_text(self) -> str:
        return f"Effective date: {self.effective_at.isoformat()}\n{self.text}"


class Scenario(StrictModel):
    id: str
    description: str
    metadata: dict = Field(default_factory=dict)
    steps: list[Update] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def valid_id(cls, value):
        return safe_name(value)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Step ids must be unique within a scenario.")
        probes = [p.id for s in self.steps for p in s.probes]
        if len(probes) != len(set(probes)):
            raise ValueError("Probe ids must be unique within a scenario.")
        return self


class Suite(StrictModel):
    name: str
    description: str
    metadata: dict = Field(default_factory=dict)
    scenarios: list[Scenario] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [s.id for s in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("Scenario ids must be unique.")
        return self


class Evidence(BaseModel):
    id: str
    text: str
    kind: str
    source_ids: list[str] = Field(default_factory=list)
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    created_at: datetime | None = None
    expired_at: datetime | None = None
    metadata: dict = Field(default_factory=dict)


class ContextUsage(BaseModel):
    token_budget: int
    tokens_used: int
    evidence_ids: list[str]
    omitted_evidence_ids: list[str]
    format: str = "evidence-json-without-metadata-v1"


class Retrieval(BaseModel):
    backend: BackendName
    query: str
    as_of: datetime | None = None
    native_temporal_filter: bool
    evidence: list[Evidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    latency_ms: float = 0
    answer_context: ContextUsage | None = None

    def context(self) -> str:
        return "\n".join(item.model_dump_json(exclude_none=True) for item in self.evidence)


class Answer(BaseModel):
    answer: str
    evidence_ids: list[str] = Field(default_factory=list)


class RetrieveRequest(StrictModel):
    backend: BackendName = "graphiti"
    namespace: str = "agent"
    query: str = Field(min_length=1, max_length=20_000)
    top_k: int = Field(default=10, ge=1, le=100)
    as_of: datetime | None = None
    answer: bool = True
    context_tokens: int = Field(default=4096, ge=128, le=16000)

    @field_validator("namespace")
    @classmethod
    def valid_namespace(cls, value):
        return safe_name(value)

    @field_validator("as_of")
    @classmethod
    def timezone_required(cls, value):
        return Probe.timezone_required(value)


class IngestRequest(StrictModel):
    backend: BackendName = "graphiti"
    namespace: str = "agent"
    update: Update

    @field_validator("namespace")
    @classmethod
    def valid_namespace(cls, value):
        return safe_name(value)


def utc_now() -> datetime:
    return datetime.now(UTC)
