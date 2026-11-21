"""RAG pipeline, prompt templates, and result models."""

from __future__ import annotations

from .pipeline import RAGBase, RAGResult, SearchIndex
from .prompts import (
    INSTRUCTIONS,
    PROMPT_TEMPLATE,
    REWRITE_INSTRUCTIONS,
    escape_delimiter_tags,
    render_prompt,
)

__all__ = [
    "INSTRUCTIONS",
    "PROMPT_TEMPLATE",
    "REWRITE_INSTRUCTIONS",
    "RAGBase",
    "RAGResult",
    "SearchIndex",
    "escape_delimiter_tags",
    "render_prompt",
]
