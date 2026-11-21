"""Service lifecycle, provider wiring, and thread-safe RAG execution for the HTTP API."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_rag_faq.config import DEFAULT_CACHE_DIR, DEFAULT_CLOUD_MODEL, DEFAULT_COURSE
from llm_rag_faq.data import load_faq_data
from llm_rag_faq.llm import AnthropicClient, LLMClient, LocalGGUFClient
from llm_rag_faq.rag import RAGBase, SearchIndex
from llm_rag_faq.retrieval import build_index

from .schemas import ChatRequest, ChatResponse, CoursesResponse, HealthResponse, SourceDocument

logger = logging.getLogger(__name__)

DEFAULT_CORS_ORIGINS: tuple[str, ...] = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)
DEFAULT_STATIC_DIR = Path("frontend/dist")


@dataclass(frozen=True)
class APISettings:
    """Immutable runtime settings for the FastAPI FAQ RAG service."""

    provider: str = "auto"
    model: str = DEFAULT_CLOUD_MODEL
    model_path: str | None = None
    cache_dir: str = DEFAULT_CACHE_DIR
    course: str | None = None
    default_course: str | None = DEFAULT_COURSE
    local_files_only: bool = False
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    static_dir: Path | None = DEFAULT_STATIC_DIR

    @classmethod
    def from_env(cls) -> APISettings:
        """Build ``APISettings`` from ``FAQ_*`` environment variables."""
        raw_course = (os.getenv("FAQ_COURSE") or "").strip()
        course: str | None = None if not raw_course or raw_course.lower() == "all" else raw_course

        raw_default = (os.getenv("FAQ_DEFAULT_COURSE") or "").strip()
        if raw_default:
            default_course: str | None = None if raw_default.lower() == "all" else raw_default
        else:
            default_course = course if course is not None else DEFAULT_COURSE

        raw_origins = (os.getenv("FAQ_CORS_ORIGINS") or "").strip()
        cors_origins = (
            tuple(origin.strip() for origin in raw_origins.split(",") if origin.strip())
            if raw_origins
            else DEFAULT_CORS_ORIGINS
        )

        raw_static = (os.getenv("FAQ_STATIC_DIR") or "").strip()
        static_dir = Path(raw_static) if raw_static else DEFAULT_STATIC_DIR

        local_only = (os.getenv("FAQ_LOCAL_FILES_ONLY") or "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        model_path = (os.getenv("FAQ_MODEL_PATH") or "").strip() or None

        return cls(
            provider=(os.getenv("FAQ_PROVIDER") or "auto").strip().lower(),
            model=(os.getenv("FAQ_MODEL") or DEFAULT_CLOUD_MODEL).strip(),
            model_path=model_path,
            cache_dir=(os.getenv("FAQ_CACHE_DIR") or DEFAULT_CACHE_DIR).strip(),
            course=course,
            default_course=default_course,
            local_files_only=local_only,
            cors_origins=cors_origins,
            static_dir=static_dir,
        )


def _create_llm_client(settings: APISettings) -> tuple[LLMClient, str]:
    """Instantiate the configured local or cloud ``LLMClient`` and return its provider label."""
    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()

    if settings.provider == "anthropic":
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Set it in .env or use provider='local'."
            )
        return AnthropicClient(model=settings.model, api_key=api_key), f"anthropic:{settings.model}"

    if settings.provider not in {"auto", "local"}:
        raise ValueError(
            f"Unsupported provider {settings.provider!r}. Expected 'auto', 'local', or 'anthropic'."
        )

    use_local = settings.provider == "local" or (
        settings.provider == "auto" and (settings.model_path is not None or not api_key)
    )
    if use_local:
        return (
            LocalGGUFClient(
                model_path=settings.model_path,
                cache_dir=settings.cache_dir,
                local_files_only=settings.local_files_only,
            ),
            "local-gguf",
        )

    return AnthropicClient(model=settings.model, api_key=api_key), f"anthropic:{settings.model}"


def _extract_courses(documents: Sequence[dict[str, Any]]) -> list[str]:
    """Return sorted unique non-empty course slugs present in ``documents``."""
    return sorted({str(doc["course"]).strip() for doc in documents if doc.get("course")})


def _to_source_document(doc: dict[str, Any]) -> SourceDocument:
    """Convert a raw FAQ document dictionary to a ``SourceDocument`` schema."""
    raw_id = doc.get("id")
    raw_course = doc.get("course")
    return SourceDocument(
        id=str(raw_id) if raw_id is not None else None,
        course=str(raw_course) if raw_course is not None else None,
        section=str(doc.get("section") or ""),
        question=str(doc.get("question") or ""),
        answer=str(doc.get("answer") or ""),
    )


class FAQChatService:
    """Thread-safe service wrapping a shared FAQ search index and LLM client."""

    def __init__(
        self,
        *,
        index: SearchIndex,
        llm_client: Any,
        courses: Sequence[str],
        num_documents: int,
        provider_name: str = "unknown",
        default_course: str | None = DEFAULT_COURSE,
        model: str = DEFAULT_CLOUD_MODEL,
        owns_client: bool = False,
    ) -> None:
        self.index = index
        self.llm_client = llm_client
        self.courses = sorted({c.strip() for c in courses if c.strip()})
        self.num_documents = num_documents
        self.provider_name = provider_name
        self.default_course = (
            default_course
            if (default_course is None or default_course in self.courses)
            else (self.courses[0] if self.courses else None)
        )
        self.model = model
        self._owns_client = owns_client
        self._lock = threading.Lock()

    def __enter__(self) -> FAQChatService:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying LLM client if owned by this service."""
        if not self._owns_client:
            return
        close_fn = getattr(self.llm_client, "close", None)
        if callable(close_fn):
            close_fn()

    def health(self) -> HealthResponse:
        """Return current service health metadata."""
        return HealthResponse(
            provider=self.provider_name,
            default_course=self.default_course,
            num_documents=self.num_documents,
        )

    def list_courses(self) -> CoursesResponse:
        """Return available courses and the default course slug."""
        return CoursesResponse(
            default_course=self.default_course,
            courses=list(self.courses),
        )

    def answer(self, request: ChatRequest) -> ChatResponse:
        """Run a stateless RAG lookup for ``request`` under a thread lock."""
        if request.course is not None and self.courses and request.course not in self.courses:
            raise ValueError(
                f"Unknown course {request.course!r}. Available courses: {self.courses}"
            )

        assistant = RAGBase(
            index=self.index,
            llm_client=self.llm_client,
            course=request.course,
            model=self.model,
            num_results=request.num_results,
            enable_fallback=request.enable_fallback,
        )

        with self._lock:
            result = assistant.rag_with_sources(request.query, num_results=request.num_results)

        return ChatResponse(
            answer=result.answer,
            sources=[_to_source_document(doc) for doc in result.sources],
            rewritten_query=result.rewritten_query,
            used_fallback=result.used_fallback,
            course=request.course,
        )


def build_chat_service(settings: APISettings | None = None) -> FAQChatService:
    """Load FAQ documents, build the hybrid index, and initialize the LLM client."""
    cfg = settings if settings is not None else APISettings.from_env()
    logger.info(
        "Initializing FAQChatService (course=%s, provider=%s, cache_dir=%s)",
        cfg.course or "all",
        cfg.provider,
        cfg.cache_dir,
    )
    documents = load_faq_data(course=cfg.course, cache_dir=cfg.cache_dir)
    index = build_index(documents, cache_dir=cfg.cache_dir)
    llm_client, provider_name = _create_llm_client(cfg)
    courses = _extract_courses(documents)
    logger.info(
        "FAQChatService ready: %d documents across %d courses using %s",
        len(documents),
        len(courses),
        provider_name,
    )
    return FAQChatService(
        index=index,
        llm_client=llm_client,
        courses=courses,
        num_documents=len(documents),
        provider_name=provider_name,
        default_course=cfg.default_course,
        model=cfg.model,
        owns_client=True,
    )
