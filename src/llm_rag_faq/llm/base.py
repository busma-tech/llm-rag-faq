"""Shared protocol for generative LLM clients."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for generative LLM clients used by the RAG pipeline."""

    def generate(
        self,
        system: str | None,
        prompt: str,
        max_tokens: int = ...,
    ) -> str: ...
