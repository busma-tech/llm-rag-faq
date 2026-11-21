"""Shared offline test doubles and FAQ fixtures for ``llm_rag_faq`` tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray

SAMPLE_FAQ_DOCS: list[dict[str, Any]] = [
    {
        "id": "llm-1",
        "course": "llm-zoomcamp",
        "section": "General Course-related",
        "question": "How do I run Ollama locally with Docker?",
        "answer": "Mount the ollama volume and pull llama3 inside the container.",
    },
    {
        "id": "llm-2",
        "course": "llm-zoomcamp",
        "section": "Module 1: Intro",
        "question": "Which OpenAI API key is required for homework?",
        "answer": "You can use OpenAI or an open-source local model via Ollama.",
    },
    {
        "id": "llm-3",
        "course": "llm-zoomcamp",
        "section": "Module 2: Vector Search",
        "question": "What is hybrid search and RRF?",
        "answer": "Hybrid search combines minsearch TF-IDF with fastembed embeddings using RRF.",
    },
    {
        "id": "llm-4",
        "course": "llm-zoomcamp",
        "section": "Module 3: Evaluation",
        "question": "How are RAG retrieval metrics computed?",
        "answer": "Hit rate and MRR measure whether relevant FAQ documents appear near the top.",
    },
    {
        "id": "ml-1",
        "course": "ml-zoomcamp",
        "section": "General",
        "question": "How do I install Docker for ML Zoomcamp?",
        "answer": "Install Docker Desktop or Docker Engine and verify with docker run hello-world.",
    },
    {
        "id": "ml-2",
        "course": "ml-zoomcamp",
        "section": "Deployment",
        "question": "How do I serve a scikit-learn model with Flask?",
        "answer": "Wrap predict_proba in a Flask endpoint and package it with Gunicorn.",
    },
]


class FakeEmbedder:
    """Deterministic offline ``Embedder`` double producing non-unit float32 vectors."""

    def __init__(self, dim: int = 4, scale: float = 1.0) -> None:
        self.dim = dim
        self.scale = scale
        self.passage_calls: list[tuple[list[str], dict[str, Any]]] = []
        self.query_calls: list[tuple[list[str], dict[str, Any]]] = []
        self.last_batch_size: int | None = None

    def _encode(self, text: str) -> NDArray[np.float32]:
        lower = text.lower()
        vec = np.zeros(self.dim, dtype=np.float32)
        if self.dim >= 1:
            vec[0] = float(lower.count("ollama") + lower.count("docker") + 1) * 2.0
        if self.dim >= 2:
            vec[1] = float(lower.count("hybrid") + lower.count("rrf") + lower.count("search")) * 3.0
        if self.dim >= 3:
            vec[2] = float(lower.count("flask") + lower.count("scikit") + lower.count("ml")) * 4.0
        if self.dim >= 4:
            vec[3] = float(lower.count("eval") + lower.count("mrr") + lower.count("openai")) * 2.5
        return vec * np.float32(self.scale)

    def passage_embed(
        self,
        texts: Iterable[str],
        **kwargs: Any,
    ) -> Iterable[NDArray[np.float32]]:
        text_list = list(texts)
        self.passage_calls.append((text_list, dict(kwargs)))
        batch_size = kwargs.get("batch_size")
        self.last_batch_size = int(batch_size) if batch_size is not None else None
        return [self._encode(text) for text in text_list]

    def query_embed(
        self,
        query: str | Iterable[str],
        **kwargs: Any,
    ) -> Iterable[NDArray[np.float32]]:
        query_list = [query] if isinstance(query, str) else list(query)
        self.query_calls.append((query_list, dict(kwargs)))
        return [self._encode(item) for item in query_list]


class FakeLlama:
    """Offline double for ``llama_cpp.Llama`` with byte-level tokenization."""

    def __init__(
        self,
        responses: Sequence[str | dict[str, Any]] | None = None,
        stream_chunks: Sequence[Sequence[str | dict[str, Any]]] | None = None,
        default_content: str = "Fake local response.",
        finish_reason: str = "stop",
    ) -> None:
        self._responses: list[str | dict[str, Any]] = (
            list(responses) if responses is not None else []
        )
        self._stream_chunks: list[list[str | dict[str, Any]]] = (
            [list(seq) for seq in stream_chunks] if stream_chunks is not None else []
        )
        self.default_content = default_content
        self.finish_reason = finish_reason
        self.calls: list[dict[str, Any]] = []
        self.tokenize_calls: list[dict[str, Any]] = []
        self.closed: bool = False
        self.close_count: int = 0

    def tokenize(
        self,
        text: bytes,
        add_bos: bool = True,
        special: bool = False,
    ) -> list[int]:
        self.tokenize_calls.append({"text": text, "add_bos": add_bos, "special": special})
        return list(text)

    def detokenize(
        self,
        tokens: Sequence[int],
        prev_tokens: Sequence[int] | None = None,
        special: bool = False,
    ) -> bytes:
        del prev_tokens, special
        return bytes(tokens)

    def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float = 0.2,
        repeat_penalty: float = 1.0,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "repeat_penalty": repeat_penalty,
                "stop": list(stop) if stop is not None else [],
                "seed": seed,
                "stream": stream,
                **kwargs,
            }
        )
        if stream:
            chunks = self._stream_chunks.pop(0) if self._stream_chunks else ["Fake ", "stream."]

            def _iter_chunks() -> Iterator[dict[str, Any]]:
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        yield chunk
                    else:
                        yield {"choices": [{"delta": {"content": chunk}}]}

            return _iter_chunks()

        item: str | dict[str, Any] = (
            self._responses.pop(0) if self._responses else self.default_content
        )
        if isinstance(item, dict):
            return item
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": item},
                    "finish_reason": self.finish_reason,
                }
            ]
        }

    def close(self) -> None:
        self.closed = True
        self.close_count += 1


class _FakeMessages:
    """Messages namespace double for ``FakeAnthropic``."""

    def __init__(self, parent: FakeAnthropic) -> None:
        self._parent = parent

    def create(self, **kwargs: Any) -> Any:
        self._parent.calls.append(dict(kwargs))
        if self._parent._responses:
            item = self._parent._responses.pop(0)
        else:
            item = self._parent.default_text

        if isinstance(item, str):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=item)])
        if isinstance(item, list):
            return SimpleNamespace(content=item)
        return item


class FakeAnthropic:
    """Offline double for ``anthropic.Anthropic`` (deliberately lacks ``.generate``)."""

    def __init__(
        self,
        responses: Sequence[Any] | None = None,
        default_text: str = "Fake Anthropic response.",
    ) -> None:
        self._responses: list[Any] = list(responses) if responses is not None else []
        self.default_text = default_text
        self.calls: list[dict[str, Any]] = []
        self.closed: bool = False
        self.close_count: int = 0
        self.messages = _FakeMessages(self)

    def close(self) -> None:
        self.closed = True
        self.close_count += 1


class FakeSearchIndex:
    """Configurable ``SearchIndex`` double recording all ``search`` calls."""

    def __init__(
        self,
        batches: Sequence[Sequence[dict[str, Any]]] | None = None,
        fn: Callable[..., list[dict[str, Any]]] | None = None,
    ) -> None:
        self._batches: list[list[dict[str, Any]]] = (
            [list(batch) for batch in batches] if batches is not None else []
        )
        self._fn = fn
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        num_results: int = 5,
        boost_dict: dict[str, float] | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "query": query,
                "num_results": num_results,
                "boost_dict": boost_dict,
                "filter_dict": filter_dict,
            }
        )
        if self._fn is not None:
            return self._fn(query, num_results, boost_dict, filter_dict)
        if self._batches:
            return self._batches.pop(0)
        return []
