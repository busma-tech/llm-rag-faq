"""RAG orchestration pipeline combining retrieval and LLM generation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from llm_rag_faq.config import (
    DEFAULT_BOOST,
    DEFAULT_CLOUD_MAX_TOKENS,
    DEFAULT_CLOUD_MODEL,
    DEFAULT_CLOUD_NUM_RESULTS,
    DEFAULT_COURSE,
    DEFAULT_LOCAL_MAX_TOKENS,
    DEFAULT_LOCAL_NUM_RESULTS,
    DEFAULT_MAX_QUERY_CHARS,
    DEFAULT_REWRITE_MAX_TOKENS,
    NO_ANSWER_RESPONSE,
)
from llm_rag_faq.llm.anthropic_client import AnthropicClient, extract_anthropic_text
from llm_rag_faq.llm.base import LLMClient
from llm_rag_faq.llm.local_gguf import LocalGGUFClient

from .prompts import (
    INSTRUCTIONS,
    PROMPT_TEMPLATE,
    REWRITE_INSTRUCTIONS,
    escape_delimiter_tags,
    render_prompt,
)

_IDK_RE = re.compile(
    r'^(?:sorry[,!]?\s+|unfortunately[,!]?\s+)?["\']?'
    r"i\s+(?:don't|do\s+not)\s+know\.?"
    r'["\']?$',
    flags=re.IGNORECASE,
)


def _is_idk_response(answer: str) -> bool:
    """Return True when ``answer`` is a standalone 'I don't know' refusal."""
    return bool(_IDK_RE.match(answer.strip()))


def _doc_fingerprint(doc: dict[str, Any]) -> Any:
    """Return a hashable key identifying an FAQ document for deduplication."""
    doc_id = doc.get("id")
    if doc_id is not None:
        return ("id", doc.get("course"), doc_id)
    return (
        "fields",
        doc.get("course") or "",
        doc.get("section") or "",
        doc.get("question") or "",
        doc.get("answer") or "",
    )


class SearchIndex(Protocol):
    """Minimal protocol required for retrieval indices passed to ``RAGBase``."""

    def search(
        self,
        query: str,
        num_results: int = 5,
        boost_dict: dict[str, float] | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class RAGResult:
    """Structured output returned by ``RAGBase.rag_with_sources``."""

    answer: str
    sources: list[dict[str, Any]]
    rewritten_query: str | None = None
    used_fallback: bool = False


class RAGBase:
    """Retrieval-Augmented Generation assistant over an FAQ search index.

    Supports both clients implementing ``LLMClient`` (such as ``LocalGGUFClient``
    and ``AnthropicClient``) and raw Anthropic SDK clients exposing
    ``messages.create(...)``.
    """

    def __init__(
        self,
        index: SearchIndex,
        llm_client: Any = None,
        instructions: str = INSTRUCTIONS,
        prompt_template: str = PROMPT_TEMPLATE,
        course: str | None = DEFAULT_COURSE,
        model: str = DEFAULT_CLOUD_MODEL,
        num_results: int | None = None,
        max_tokens: int | None = None,
        boost_dict: dict[str, float] | None = None,
        enable_fallback: bool = True,
        max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
    ) -> None:
        """Initialize the RAG assistant.

        Args:
            index: Search index implementing
                ``search(query, num_results, boost_dict, filter_dict)``.
            llm_client: Generative client implementing ``LLMClient`` or raw
                Anthropic client with ``.messages.create(...)``. When ``None``,
                instantiates and owns a default ``LocalGGUFClient()``.
            instructions: System instructions for grounded QA generation.
            prompt_template: Template containing ``{context}`` and ``{question}``
                placeholders.
            course: Course slug filter passed to ``index.search``, or ``None``
                to search across all indexed courses.
            model: Anthropic model identifier (used on the raw Anthropic API path).
            num_results: Default number of documents retrieved per query.
                Defaults to ``3`` for local clients and ``5`` for cloud clients.
            max_tokens: Default generation token limit. Defaults to ``512`` for
                local clients and ``1024`` for cloud clients.
            boost_dict: Default lexical boost weights. Defaults to ``DEFAULT_BOOST``.
            enable_fallback: If ``True``, retries once with a rewritten keyword
                query when initial retrieval misses or returns "I don't know.".
            max_query_chars: Maximum allowed length for incoming queries.

        Raises:
            ValueError: If numeric configuration arguments are invalid.
        """
        if max_query_chars <= 0:
            raise ValueError(f"max_query_chars must be > 0, got {max_query_chars}")

        self.index = index
        self._owns_client = llm_client is None
        self.llm_client: Any = LocalGGUFClient() if llm_client is None else llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.course = course
        self.model = model
        self.boost_dict = (
            dict(DEFAULT_BOOST) if boost_dict is None else dict(boost_dict)
        )
        self.enable_fallback = enable_fallback
        self.max_query_chars = max_query_chars

        is_local = self._is_local_client()
        default_k = (
            DEFAULT_LOCAL_NUM_RESULTS if is_local else DEFAULT_CLOUD_NUM_RESULTS
        )
        default_max_tok = (
            DEFAULT_LOCAL_MAX_TOKENS if is_local else DEFAULT_CLOUD_MAX_TOKENS
        )

        self.num_results = default_k if num_results is None else num_results
        self.max_tokens = default_max_tok if max_tokens is None else max_tokens
        if self.num_results <= 0:
            raise ValueError(f"num_results must be > 0, got {self.num_results}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")

    def __enter__(self) -> RAGBase:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying LLM client if it was created by this ``RAGBase``."""
        if self._owns_client and hasattr(self.llm_client, "close"):
            self.llm_client.close()

    def _has_generate_api(self) -> bool:
        return isinstance(self.llm_client, LLMClient) and callable(
            getattr(self.llm_client, "generate", None)
        )

    def _is_local_client(self) -> bool:
        return self._has_generate_api() and not isinstance(
            self.llm_client, AnthropicClient
        )

    def _validate_query(self, query: str) -> str:
        if not isinstance(query, str):
            raise TypeError(f"query must be a str, got {type(query).__name__}")
        clean = query.strip()
        if not clean:
            raise ValueError("query must not be empty.")
        if len(clean) > self.max_query_chars:
            raise ValueError(
                f"query length ({len(clean)}) exceeds max_query_chars "
                f"({self.max_query_chars})."
            )
        return clean

    def build_context(self, search_results: Sequence[dict[str, Any]]) -> str:
        """Format retrieved FAQ documents into a plain-text context block."""
        lines: list[str] = []
        for doc in search_results:
            section = str(doc.get("section") or "")
            question = str(doc.get("question") or "")
            answer = str(doc.get("answer") or "")
            lines.append(section)
            lines.append("Q: " + question)
            lines.append("A: " + answer)
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(
        self,
        query: str,
        search_results: Sequence[dict[str, Any]],
    ) -> str:
        """Render ``self.prompt_template`` with escaped context and question."""
        return render_prompt(
            context=self.build_context(search_results),
            question=query,
            template=self.prompt_template,
        )

    def search(
        self,
        query: str,
        num_results: int | None = None,
        boost_dict: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the underlying FAQ index using configured course and boost settings."""
        k = self.num_results if num_results is None else num_results
        effective_boost = self.boost_dict if boost_dict is None else boost_dict
        filter_dict = {"course": self.course} if self.course is not None else {}

        return self.index.search(
            query,
            num_results=k,
            boost_dict=effective_boost,
            filter_dict=filter_dict,
        )

    def llm(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Invoke the configured local or Anthropic LLM client.

        Args:
            prompt: User prompt text.
            system: Optional system instruction override (defaults to
                ``self.instructions``).
            max_tokens: Optional generation token limit override (defaults to
                ``self.max_tokens``).

        Returns:
            Model text output (empty string if no text blocks were returned).
        """
        effective_system = self.instructions if system is None else system
        effective_max_tokens = (
            self.max_tokens if max_tokens is None else max_tokens
        )

        if isinstance(self.llm_client, LLMClient) and callable(
            getattr(self.llm_client, "generate", None)
        ):
            return self.llm_client.generate(
                system=effective_system,
                prompt=prompt,
                max_tokens=effective_max_tokens,
            )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": effective_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if effective_system is not None and effective_system.strip():
            kwargs["system"] = effective_system
        response = self.llm_client.messages.create(**kwargs)
        return extract_anthropic_text(response)

    def rewrite_query(self, query: str) -> str:
        """Rewrite a user question into concise FAQ search keywords."""
        clean_query = self._validate_query(query)
        safe_question = escape_delimiter_tags(clean_query, "question")
        rewrite_prompt = f"<question>\n{safe_question}\n</question>"
        rewritten = self.llm(
            rewrite_prompt,
            system=REWRITE_INSTRUCTIONS,
            max_tokens=DEFAULT_REWRITE_MAX_TOKENS,
        )
        return rewritten.strip().strip('"\'')

    def rag_with_sources(
        self,
        query: str,
        num_results: int | None = None,
    ) -> RAGResult:
        """Run end-to-end RAG and return the answer along with retrieved sources."""
        clean_query = self._validate_query(query)
        k = self.num_results if num_results is None else num_results

        search_results = self.search(clean_query, num_results=k)
        if search_results:
            answer = self.llm(self.build_prompt(clean_query, search_results))
        else:
            answer = NO_ANSWER_RESPONSE

        if not self.enable_fallback or not _is_idk_response(answer):
            return RAGResult(
                answer=answer,
                sources=search_results,
                rewritten_query=None,
                used_fallback=False,
            )

        # Agentic fallback: rewrite query and re-search once if initial retrieval missed
        alt_query = self.rewrite_query(clean_query)
        if not alt_query or alt_query.lower() == clean_query.lower():
            return RAGResult(
                answer=answer,
                sources=search_results,
                rewritten_query=alt_query or None,
                used_fallback=True,
            )

        alt_results = self.search(alt_query, num_results=k)
        seen = {_doc_fingerprint(d) for d in search_results}
        has_new_docs = any(_doc_fingerprint(d) not in seen for d in alt_results)

        if not alt_results or not has_new_docs:
            return RAGResult(
                answer=answer,
                sources=search_results,
                rewritten_query=alt_query,
                used_fallback=True,
            )

        retry_answer = self.llm(self.build_prompt(clean_query, alt_results))
        return RAGResult(
            answer=retry_answer,
            sources=alt_results,
            rewritten_query=alt_query,
            used_fallback=True,
        )

    def rag(self, query: str, num_results: int | None = None) -> str:
        """Run end-to-end RAG and return the final answer string."""
        return self.rag_with_sources(query, num_results=num_results).answer
