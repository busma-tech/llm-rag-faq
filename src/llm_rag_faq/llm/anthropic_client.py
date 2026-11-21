"""Thin Anthropic Messages API wrapper implementing ``LLMClient``."""

from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from llm_rag_faq.config import DEFAULT_CLOUD_MAX_TOKENS, DEFAULT_CLOUD_MODEL


def extract_anthropic_text(response: Any) -> str:
    """Extract and join text content blocks from an Anthropic Messages response."""
    content_blocks = getattr(response, "content", None) or []
    text_parts = [
        str(block.text)
        for block in content_blocks
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    return "\n".join(text_parts).strip()


class AnthropicClient:
    """Adapter around ``anthropic.Anthropic`` conforming to ``LLMClient``."""

    def __init__(
        self,
        model: str = DEFAULT_CLOUD_MODEL,
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        """Initialize the Anthropic client wrapper.

        Args:
            model: Anthropic model identifier.
            api_key: Optional API key (ignored when ``client`` is supplied).
            client: Optional pre-initialized ``Anthropic``-compatible client.
                When supplied, ``close()`` does not close the external client.
        """
        self.model = model
        self._owns_client = client is None
        self._closed = False
        if client is not None:
            self.client: Any = client
        elif api_key is not None:
            self.client = Anthropic(api_key=api_key)
        else:
            self.client = Anthropic()

    def __enter__(self) -> AnthropicClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying Anthropic client if owned by this wrapper."""
        if not self._closed:
            self._closed = True
            if self._owns_client and hasattr(self.client, "close"):
                self.client.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("AnthropicClient has been closed.")

    def generate(
        self,
        system: str | None,
        prompt: str,
        max_tokens: int = DEFAULT_CLOUD_MAX_TOKENS,
    ) -> str:
        """Generate a text response via the Anthropic Messages API."""
        self._ensure_open()
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a str, got {type(prompt).__name__}")
        if system is not None and not isinstance(system, str):
            raise TypeError(
                f"system must be a str or None, got {type(system).__name__}"
            )
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None and system.strip():
            kwargs["system"] = system.strip()

        response = self.client.messages.create(**kwargs)
        return extract_anthropic_text(response)
