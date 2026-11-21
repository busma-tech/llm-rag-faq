"""FastAPI + Pydantic HTTP API layer for ``llm_rag_faq``."""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
