"""LLM client protocols and provider implementations."""

from __future__ import annotations

from .anthropic_client import AnthropicClient, extract_anthropic_text
from .base import LLMClient
from .local_gguf import LocalGGUFClient

__all__ = [
    "AnthropicClient",
    "LLMClient",
    "LocalGGUFClient",
    "extract_anthropic_text",
]
