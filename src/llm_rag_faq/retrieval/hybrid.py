"""Hybrid lexical (minsearch) and semantic (fastembed) FAQ search with RRF."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from fastembed import TextEmbedding
from minsearch import Index
from numpy.typing import NDArray

from llm_rag_faq.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BOOST,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MIN_CANDIDATES,
    DEFAULT_RRF_K,
    DEFAULT_THREADS,
)


@runtime_checkable
class Embedder(Protocol):
    """Structural protocol matching ``fastembed.TextEmbedding``."""

    def passage_embed(
        self,
        texts: Iterable[str],
        **kwargs: Any,
    ) -> Iterable[NDArray[Any]]: ...

    def query_embed(
        self,
        query: str | Iterable[str],
        **kwargs: Any,
    ) -> Iterable[NDArray[Any]]: ...


def _l2_normalize(vectors: NDArray[Any]) -> NDArray[np.float32]:
    """L2-normalize a 1D vector or 2D array of row vectors as float32."""
    arr: NDArray[np.float32] = np.asarray(vectors, dtype=np.float32)
    if arr.size == 0:
        return arr
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        if norm > 0.0:
            normalized: NDArray[np.float32] = arr / norm
            return normalized
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    out = np.zeros_like(arr, dtype=np.float32)
    np.divide(arr, norms, out=out, where=norms > 0.0)
    return out


def _format_document(doc: dict[str, Any]) -> str:
    """Format an FAQ document dictionary into a single passage string."""
    section = doc.get("section") or ""
    question = doc.get("question") or ""
    answer = doc.get("answer") or ""
    return f"{section} | {question} | {answer}"


def _matches_filter(doc: dict[str, Any], filter_dict: dict[str, Any]) -> bool:
    """Return True if ``doc`` satisfies all keyword predicates in ``filter_dict``."""
    for field, expected in filter_dict.items():
        value = doc.get(field)
        if isinstance(expected, list | tuple | set):
            if value not in expected:
                return False
        elif value != expected:
            return False
    return True


def _rrf_fuse(
    ranked_lists: Sequence[Sequence[int]],
    k: int = DEFAULT_RRF_K,
    num_results: int = 5,
) -> list[int]:
    """Merge ranked lists of document indices using 1-based Reciprocal Rank Fusion."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_idx in enumerate(ranked, start=1):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank)

    return sorted(scores, key=lambda idx: scores[idx], reverse=True)[:num_results]


class HybridFAQSearch:
    """Combines lexical search (minsearch) and semantic vector search (fastembed) via RRF.

    Each FAQ document is indexed in a ``minsearch.Index`` over the ``question``,
    ``section``, and ``answer`` text fields (with ``course`` as a keyword field)
    and embedded into an L2-normalized dense vector via ``fastembed.TextEmbedding``.

    Attributes:
        documents: Shallow-copied list of indexed FAQ documents.
        lexical_index: Fitted TF-IDF lexical search index.
        embedder: Embedding model used for passage and query vectors.
        doc_vectors: L2-normalized 2D float32 array of shape ``(N, dim)``
            aligned by index with ``self.documents`` (or shape ``(0, 0)`` when
            ``self.documents`` is empty).
        rrf_k: Smoothing constant ``k`` used in Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        documents: Sequence[dict[str, Any]],
        course: str | None = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        threads: int = DEFAULT_THREADS,
        rrf_k: int = DEFAULT_RRF_K,
        cache_dir: str | Path | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        """Initialize lexical and semantic indices over FAQ documents.

        Args:
            documents: Sequence of FAQ document dictionaries.
            course: Optional course slug to filter ``documents`` before fitting
                and embedding. When ``None`` (the default), all supplied
                documents are indexed.
            model_name: FastEmbed model identifier used when ``embedder`` is not
                supplied.
            batch_size: Batch size passed to ``embedder.passage_embed``.
            threads: Number of ONNX runtime threads when constructing a default
                ``TextEmbedding`` instance.
            rrf_k: Smoothing constant for Reciprocal Rank Fusion (default ``60``).
            cache_dir: Optional directory for caching normalized document
                embeddings as ``.npy`` files keyed by model name and content.
            embedder: Optional pre-initialized ``Embedder`` instance to reuse.
        """
        if course is not None:
            self.documents = [d for d in documents if d.get("course") == course]
        else:
            self.documents = list(documents)

        self.rrf_k = rrf_k
        self.model_name = model_name

        self.lexical_index = Index(
            text_fields=["question", "section", "answer"],
            keyword_fields=["course"],
        )
        self.lexical_index.fit(self.documents)

        self.embedder: Embedder = (
            embedder
            if embedder is not None
            else TextEmbedding(model_name=model_name, threads=threads)
        )
        self.doc_vectors = self._build_doc_vectors(
            batch_size=batch_size,
            cache_dir=cache_dir,
        )

    def _build_doc_vectors(
        self,
        batch_size: int,
        cache_dir: str | Path | None,
    ) -> NDArray[np.float32]:
        """Compute (or load from cache) L2-normalized document embeddings."""
        if not self.documents:
            return np.empty((0, 0), dtype=np.float32)

        texts = [_format_document(d) for d in self.documents]

        cache_file: Path | None = None
        if cache_dir is not None:
            digest = hashlib.sha256()
            digest.update(self.model_name.encode("utf-8"))
            for text in texts:
                digest.update(b"\0")
                digest.update(text.encode("utf-8"))
            cache_file = (
                Path(cache_dir) / f"embeddings_{digest.hexdigest()[:16]}.npy"
            )
            if cache_file.exists():
                cached = np.load(cache_file, allow_pickle=False)
                if (
                    isinstance(cached, np.ndarray)
                    and cached.ndim == 2
                    and cached.shape[0] == len(texts)
                    and cached.shape[1] > 0
                ):
                    return _l2_normalize(cached)

        raw_vectors = self.embedder.passage_embed(texts, batch_size=batch_size)
        vectors = _l2_normalize(np.asarray(list(raw_vectors), dtype=np.float32))

        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, vectors)

        return vectors

    def _semantic_search(
        self,
        query: str,
        num_results: int,
        filter_dict: dict[str, Any],
    ) -> list[int]:
        """Return up to ``num_results`` document indices ranked by cosine similarity."""
        if not self.documents or num_results <= 0:
            return []

        if filter_dict:
            mask = np.fromiter(
                (_matches_filter(d, filter_dict) for d in self.documents),
                dtype=bool,
                count=len(self.documents),
            )
            valid_indices = np.flatnonzero(mask)
            if valid_indices.size == 0:
                return []
        else:
            valid_indices = np.arange(len(self.documents))

        q_vec = _l2_normalize(next(iter(self.embedder.query_embed([query]))))
        valid_scores = self.doc_vectors[valid_indices] @ q_vec

        k = min(num_results, valid_indices.size)
        if k < valid_indices.size:
            top_local = np.argpartition(-valid_scores, k - 1)[:k]
            top_local = top_local[np.argsort(-valid_scores[top_local])]
        else:
            top_local = np.argsort(-valid_scores)

        return [int(valid_indices[i]) for i in top_local]

    def search(
        self,
        query: str,
        num_results: int = 5,
        boost_dict: dict[str, float] | None = None,
        filter_dict: dict[str, Any] | None = None,
        candidate_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run hybrid lexical + semantic search and merge rankings with RRF.

        Args:
            query: User search query.
            num_results: Maximum number of fused documents to return (must be
                ``>= 0``).
            boost_dict: Per-field boost weights for lexical search. Defaults to
                ``DEFAULT_BOOST`` only when ``None`` (an explicit ``{}`` disables
                boosting).
            filter_dict: Keyword filters applied consistently to both lexical
                and semantic retrieval.
            candidate_k: Optional candidate pool size per retriever. Defaults to
                ``max(num_results * 2, DEFAULT_MIN_CANDIDATES)``.

        Returns:
            Top FAQ document dictionaries ordered by descending RRF score.

        Raises:
            ValueError: If ``num_results < 0``.
        """
        if num_results < 0:
            raise ValueError(f"num_results must be >= 0, got {num_results}")
        if num_results == 0 or not self.documents:
            return []

        effective_boost = (
            dict(DEFAULT_BOOST) if boost_dict is None else boost_dict
        )
        effective_filter = {} if filter_dict is None else filter_dict
        k_candidates = (
            candidate_k
            if candidate_k is not None
            else max(num_results * 2, DEFAULT_MIN_CANDIDATES)
        )

        # Lexical candidates (keyed by canonical document index via output_ids=True)
        lex_hits = self.lexical_index.search(
            query,
            num_results=k_candidates,
            boost_dict=effective_boost,
            filter_dict=effective_filter,
            output_ids=True,
        )
        lex_ids = [int(hit["_id"]) for hit in lex_hits]

        # Semantic candidates
        sem_ids = self._semantic_search(
            query=query,
            num_results=k_candidates,
            filter_dict=effective_filter,
        )

        top_indices = _rrf_fuse(
            [lex_ids, sem_ids],
            k=self.rrf_k,
            num_results=num_results,
        )
        return [self.documents[idx] for idx in top_indices]


def build_index(
    documents: Sequence[dict[str, Any]],
    course: str | None = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    **kwargs: Any,
) -> HybridFAQSearch:
    """Build a ``HybridFAQSearch`` index from a sequence of FAQ documents.

    Args:
        documents: FAQ document dictionaries to index.
        course: Optional course slug to filter ``documents`` before indexing.
            Defaults to ``None`` so pre-filtered documents from any course are
            preserved as-is.
        model_name: FastEmbed model identifier.
        **kwargs: Additional keyword arguments forwarded to ``HybridFAQSearch``.

    Returns:
        Initialized ``HybridFAQSearch`` instance.
    """
    return HybridFAQSearch(
        documents,
        course=course,
        model_name=model_name,
        **kwargs,
    )
