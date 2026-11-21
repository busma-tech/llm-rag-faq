"""Pydantic v2 request and response schemas for the FAQ RAG HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm_rag_faq.config import DEFAULT_COURSE, DEFAULT_MAX_QUERY_CHARS


class ChatRequest(BaseModel):
    """Payload for ``POST /api/chat``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=DEFAULT_MAX_QUERY_CHARS,
        description="User question to answer against the FAQ index.",
    )
    course: str | None = Field(
        default=DEFAULT_COURSE,
        description="Course slug to filter by, or 'all' / null to search all courses.",
    )
    num_results: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional override for number of FAQ documents retrieved.",
    )
    enable_fallback: bool = Field(
        default=True,
        description="Enable query-rewrite fallback when the initial answer is 'I don't know.'.",
    )

    @field_validator("query")
    @classmethod
    def _strip_and_validate_query(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("query must not be empty.")
        return clean

    @field_validator("course")
    @classmethod
    def _normalize_and_validate_course(cls, value: str | None) -> str | None:
        if value is None:
            return None
        clean = value.strip()
        if not clean or clean.lower() == "all":
            return None
        if "/" in clean or "\\" in clean or ".." in clean or "\x00" in clean:
            raise ValueError(f"Invalid course slug {value!r}: path traversal is not allowed.")
        return clean


class SourceDocument(BaseModel):
    """Single FAQ source document returned alongside a RAG answer."""

    id: str | None = None
    course: str | None = None
    section: str = ""
    question: str = ""
    answer: str = ""


class ChatResponse(BaseModel):
    """Structured answer payload returned by ``POST /api/chat``."""

    answer: str
    sources: list[SourceDocument]
    rewritten_query: str | None = None
    used_fallback: bool = False
    course: str | None = None


class CoursesResponse(BaseModel):
    """Available course slugs and default course returned by ``GET /api/courses``."""

    default_course: str | None
    courses: list[str]


class HealthResponse(BaseModel):
    """Service health and runtime metadata returned by ``GET /api/health``."""

    status: Literal["ok"] = "ok"
    provider: str
    default_course: str | None
    num_documents: int
