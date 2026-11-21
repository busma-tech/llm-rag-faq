"""Unit tests for ``llm_rag_faq.retrieval.hybrid``."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from minsearch.filters import FilterValidationError
from numpy.typing import NDArray

from llm_rag_faq.config import DEFAULT_BOOST, DEFAULT_MIN_CANDIDATES
from llm_rag_faq.retrieval import Embedder, HybridFAQSearch, build_index
from llm_rag_faq.retrieval.hybrid import (
    _format_document,
    _l2_normalize,
    _matches_filter,
    _rrf_fuse,
)
from tests.conftest import SAMPLE_FAQ_DOCS, FakeEmbedder


class _UniformEmbedder:
    """Embedder returning identical vectors so lexical boosting breaks ties."""

    def passage_embed(
        self, texts: Iterable[str], **kwargs: Any
    ) -> Iterable[NDArray[np.float32]]:
        del kwargs
        return [np.array([1.0, 0.0], dtype=np.float32) for _ in texts]

    def query_embed(
        self, query: str | Iterable[str], **kwargs: Any
    ) -> Iterable[NDArray[np.float32]]:
        del kwargs
        items = [query] if isinstance(query, str) else list(query)
        return [np.array([1.0, 0.0], dtype=np.float32) for _ in items]


class TestRetrievalHelpers(unittest.TestCase):
    def test_fake_embedder_satisfies_protocol(self) -> None:
        self.assertIsInstance(FakeEmbedder(), Embedder)

    def test_l2_normalize_1d_2d_zero_and_empty(self) -> None:
        norm_1d = _l2_normalize(np.array([3.0, 4.0], dtype=np.float64))
        self.assertEqual(norm_1d.dtype, np.float32)
        np.testing.assert_allclose(norm_1d, np.array([0.6, 0.8], dtype=np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(norm_1d)), 1.0, places=6)

        zero_1d = _l2_normalize(np.array([0.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(zero_1d, np.array([0.0, 0.0], dtype=np.float32))
        self.assertFalse(np.isnan(zero_1d).any())

        norm_2d = _l2_normalize(
            np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        )
        self.assertEqual(norm_2d.dtype, np.float32)
        np.testing.assert_allclose(
            norm_2d, np.array([[0.6, 0.8], [0.0, 0.0]], dtype=np.float32)
        )

        empty = _l2_normalize(np.empty((0, 0), dtype=np.float32))
        self.assertEqual(empty.shape, (0, 0))
        self.assertEqual(empty.dtype, np.float32)

    def test_format_document_missing_and_none_fields(self) -> None:
        self.assertEqual(_format_document({}), " |  | ")
        self.assertEqual(
            _format_document({"section": None, "question": "Q?", "answer": None}),
            " | Q? | ",
        )

    def test_matches_filter_scalar_sequence_empty_string_and_multifield(self) -> None:
        doc_llm = {"course": "llm-zoomcamp", "section": "S1"}
        doc_empty = {"course": "", "section": "S2"}

        self.assertTrue(_matches_filter(doc_llm, {"course": "llm-zoomcamp"}))
        self.assertFalse(_matches_filter(doc_llm, {"course": "ml-zoomcamp"}))

        self.assertTrue(_matches_filter(doc_llm, {"course": ["a", "llm-zoomcamp"]}))
        self.assertTrue(_matches_filter(doc_llm, {"course": ("a", "llm-zoomcamp")}))
        self.assertTrue(_matches_filter(doc_llm, {"course": {"a", "llm-zoomcamp"}}))

        self.assertTrue(_matches_filter(doc_empty, {"course": ""}))
        self.assertFalse(_matches_filter(doc_llm, {"course": ""}))

        self.assertTrue(
            _matches_filter(
                doc_llm, {"course": "llm-zoomcamp", "section": ["S1", "S2"]}
            )
        )
        self.assertFalse(
            _matches_filter(
                doc_llm, {"course": "llm-zoomcamp", "section": "Other"}
            )
        )

    def test_rrf_fuse_1_based_scoring_and_limit(self) -> None:
        # With k=60: doc 20 gets 1/62 + 1/61, doc 10 gets 1/61, doc 30 gets 1/62
        fused = _rrf_fuse([[10, 20], [20, 30]], k=60, num_results=5)
        self.assertEqual(fused, [20, 10, 30])
        self.assertEqual(_rrf_fuse([[10, 20], [20, 30]], k=60, num_results=2), [20, 10])

        # k=0 directly proves 1-based ranking (0-based would raise ZeroDivisionError)
        fused_k0 = _rrf_fuse([[0, 1], [1, 2]], k=0, num_results=3)
        self.assertEqual(fused_k0, [1, 0, 2])


class TestHybridFAQSearchConstructionAndEmptyCorpus(unittest.TestCase):
    def test_course_none_vs_specific_course_and_shallow_copy(self) -> None:
        docs = list(SAMPLE_FAQ_DOCS)
        idx_all = HybridFAQSearch(docs, course=None, embedder=FakeEmbedder())
        docs.append(
            {
                "course": "extra",
                "section": "S",
                "question": "Q",
                "answer": "A",
            }
        )
        self.assertEqual(len(idx_all.documents), len(SAMPLE_FAQ_DOCS))

        idx_llm = HybridFAQSearch(
            SAMPLE_FAQ_DOCS, course="llm-zoomcamp", embedder=FakeEmbedder()
        )
        self.assertEqual(len(idx_llm.documents), 4)
        self.assertTrue(all(d["course"] == "llm-zoomcamp" for d in idx_llm.documents))

    def test_build_index_defaults_to_course_none_and_forwards_kwargs(self) -> None:
        emb = FakeEmbedder()
        idx = build_index(SAMPLE_FAQ_DOCS, embedder=emb, rrf_k=40, batch_size=8)
        self.assertEqual(len(idx.documents), len(SAMPLE_FAQ_DOCS))
        self.assertEqual(idx.rrf_k, 40)
        self.assertEqual(emb.last_batch_size, 8)

    def test_doc_vectors_are_l2_normalized_float32(self) -> None:
        idx = HybridFAQSearch(SAMPLE_FAQ_DOCS, embedder=FakeEmbedder(dim=4, scale=3.0))
        self.assertEqual(idx.doc_vectors.shape, (len(SAMPLE_FAQ_DOCS), 4))
        self.assertEqual(idx.doc_vectors.dtype, np.float32)
        norms = np.linalg.norm(idx.doc_vectors, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(SAMPLE_FAQ_DOCS)), atol=1e-6)

    def test_empty_corpus_shape_and_search(self) -> None:
        for docs, course in (([], None), (SAMPLE_FAQ_DOCS, "unknown-course")):
            with self.subTest(docs_len=len(docs), course=course):
                emb = FakeEmbedder()
                idx = HybridFAQSearch(docs, course=course, embedder=emb)
                self.assertEqual(idx.documents, [])
                self.assertEqual(idx.doc_vectors.shape, (0, 0))
                self.assertEqual(idx.doc_vectors.dtype, np.float32)
                self.assertEqual(emb.passage_calls, [])
                self.assertEqual(idx.search("docker"), [])
                self.assertEqual(emb.query_calls, [])

    def test_missing_optional_doc_fields(self) -> None:
        sparse_docs: list[dict[str, Any]] = [
            {"question": "How to enroll?"},
            {"answer": "See syllabus", "section": None, "question": None},
        ]
        idx = HybridFAQSearch(sparse_docs, embedder=FakeEmbedder())
        hits = idx.search("enroll")
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0]["question"], "How to enroll?")


class TestHybridFAQSearchQueryAndFiltering(unittest.TestCase):
    def test_num_results_negative_raises_value_error(self) -> None:
        idx = HybridFAQSearch(SAMPLE_FAQ_DOCS, embedder=FakeEmbedder())
        with self.assertRaisesRegex(ValueError, "num_results must be >= 0"):
            idx.search("docker", num_results=-1)

    def test_num_results_zero_returns_empty_without_embedding(self) -> None:
        emb = FakeEmbedder()
        idx = HybridFAQSearch(SAMPLE_FAQ_DOCS, embedder=emb)
        self.assertEqual(idx.search("docker", num_results=0), [])
        self.assertEqual(emb.query_calls, [])

    def test_num_results_over_10_and_candidate_pool_sizing(self) -> None:
        docs = [
            {
                "id": f"d-{i}",
                "course": "llm-zoomcamp",
                "section": "General",
                "question": f"How to use docker container {i}?",
                "answer": f"Docker command example {i}.",
            }
            for i in range(15)
        ]
        idx = HybridFAQSearch(docs, embedder=FakeEmbedder())

        with (
            patch.object(
                idx.lexical_index, "search", wraps=idx.lexical_index.search
            ) as lex_spy,
            patch.object(
                idx, "_semantic_search", wraps=idx._semantic_search
            ) as sem_spy,
        ):
            res12 = idx.search("docker", num_results=12)
            self.assertEqual(len(res12), 12)
            self.assertEqual(lex_spy.call_args.kwargs["num_results"], 24)
            self.assertEqual(sem_spy.call_args.kwargs["num_results"], 24)

            res3 = idx.search("docker", num_results=3)
            self.assertEqual(len(res3), 3)
            self.assertEqual(
                lex_spy.call_args.kwargs["num_results"], DEFAULT_MIN_CANDIDATES
            )
            self.assertEqual(
                sem_spy.call_args.kwargs["num_results"], DEFAULT_MIN_CANDIDATES
            )

            res_custom = idx.search("docker", num_results=3, candidate_k=7)
            self.assertEqual(len(res_custom), 3)
            self.assertEqual(lex_spy.call_args.kwargs["num_results"], 7)
            self.assertEqual(sem_spy.call_args.kwargs["num_results"], 7)

    def test_filter_dict_consistent_across_lexical_and_semantic(self) -> None:
        docs = [
            {
                "id": "llm",
                "course": "llm-zoomcamp",
                "section": "General",
                "question": "Docker setup for LLM",
                "answer": "Run docker compose up.",
            },
            {
                "id": "ml",
                "course": "ml-zoomcamp",
                "section": "General",
                "question": "Docker setup for ML",
                "answer": "Run docker build.",
            },
            {
                "id": "empty",
                "course": "",
                "section": "General",
                "question": "Docker setup for uncategorized",
                "answer": "Run docker ps.",
            },
        ]
        idx = HybridFAQSearch(docs, embedder=FakeEmbedder())

        self.assertEqual(
            [d["id"] for d in idx.search("docker", filter_dict={"course": "ml-zoomcamp"})],
            ["ml"],
        )
        multi_list = idx.search(
            "docker", filter_dict={"course": ["llm-zoomcamp", "ml-zoomcamp"]}
        )
        self.assertEqual({d["id"] for d in multi_list}, {"llm", "ml"})

        self.assertEqual(
            [d["id"] for d in idx.search("docker", filter_dict={"course": ("ml-zoomcamp",)})],
            ["ml"],
        )
        self.assertEqual(
            [d["id"] for d in idx.search("docker", filter_dict={"course": {"llm-zoomcamp"}})],
            ["llm"],
        )
        self.assertEqual(
            [d["id"] for d in idx.search("docker", filter_dict={"course": ""})],
            ["empty"],
        )
        self.assertEqual(idx.search("docker", filter_dict={"course": []}), [])
        self.assertEqual(
            {d["id"] for d in idx.search("docker", filter_dict=None)},
            {"llm", "ml", "empty"},
        )
        self.assertEqual(
            {d["id"] for d in idx.search("docker", filter_dict={})},
            {"llm", "ml", "empty"},
        )

        # No empty-course doc in SAMPLE_FAQ_DOCS -> filter {"course": ""} yields []
        idx_sample = HybridFAQSearch(SAMPLE_FAQ_DOCS, embedder=FakeEmbedder())
        self.assertEqual(idx_sample.search("docker", filter_dict={"course": ""}), [])

        with self.assertRaises(FilterValidationError):
            idx.search("docker", filter_dict={"section": "General"})

    def test_boost_dict_none_vs_empty_dict(self) -> None:
        docs = [
            {
                "id": "q-match",
                "course": "llm-zoomcamp",
                "section": "Other",
                "question": "zephyr setup guide",
                "answer": "Follow standard steps.",
            },
            {
                "id": "a-match",
                "course": "llm-zoomcamp",
                "section": "Other",
                "question": "General setup guide",
                "answer": "Use zephyr zephyr zephyr for faster runs.",
            },
        ]
        idx = HybridFAQSearch(docs, embedder=_UniformEmbedder())
        baseline_boost = dict(DEFAULT_BOOST)

        with patch.object(
            idx.lexical_index, "search", wraps=idx.lexical_index.search
        ) as lex_spy:
            res_default = idx.search("zephyr", boost_dict=None)
            self.assertEqual(lex_spy.call_args.kwargs["boost_dict"], DEFAULT_BOOST)
            self.assertEqual(res_default[0]["id"], "q-match")
            self.assertEqual(DEFAULT_BOOST, baseline_boost)

            res_empty = idx.search("zephyr", boost_dict={})
            self.assertEqual(lex_spy.call_args.kwargs["boost_dict"], {})
            self.assertEqual(res_empty[0]["id"], "a-match")

    def test_hybrid_rrf_ranking_combines_lexical_and_semantic(self) -> None:
        docs = [
            {
                "id": "lex-only",
                "course": "llm-zoomcamp",
                "section": "Misc",
                "question": "alpha topic",
                "answer": "alpha details",
            },
            {
                "id": "both-top",
                "course": "llm-zoomcamp",
                "section": "Hybrid",
                "question": "alpha hybrid search",
                "answer": "hybrid search and rrf details",
            },
            {
                "id": "sem-only",
                "course": "llm-zoomcamp",
                "section": "Search",
                "question": "vector topic",
                "answer": "hybrid search rrf",
            },
        ]
        idx = HybridFAQSearch(docs, embedder=FakeEmbedder())
        results = idx.search("alpha hybrid search rrf", num_results=3)
        self.assertEqual(results[0]["id"], "both-top")


class TestHybridFAQSearchVectorCaching(unittest.TestCase):
    @staticmethod
    def _expected_cache_file(
        cache_dir: Path,
        documents: list[dict[str, Any]],
        model_name: str,
    ) -> Path:
        digest = hashlib.sha256()
        digest.update(model_name.encode("utf-8"))
        for doc in documents:
            digest.update(b"\0")
            digest.update(_format_document(doc).encode("utf-8"))
        return cache_dir / f"embeddings_{digest.hexdigest()[:16]}.npy"

    def test_npy_cache_miss_writes_and_cache_hit_skips_passage_embed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "vec_cache"
            emb1 = FakeEmbedder(dim=4, scale=2.0)
            idx1 = HybridFAQSearch(
                SAMPLE_FAQ_DOCS, cache_dir=cache_dir, embedder=emb1
            )
            self.assertEqual(len(emb1.passage_calls), 1)
            npy_files = list(cache_dir.glob("embeddings_*.npy"))
            self.assertEqual(len(npy_files), 1)

            emb2 = FakeEmbedder(dim=4, scale=2.0)
            idx2 = HybridFAQSearch(
                SAMPLE_FAQ_DOCS, cache_dir=cache_dir, embedder=emb2
            )
            self.assertEqual(emb2.passage_calls, [])
            self.assertEqual(idx2.doc_vectors.dtype, np.float32)
            np.testing.assert_allclose(
                idx2.doc_vectors, idx1.doc_vectors, rtol=1e-6, atol=1e-6
            )

    def test_npy_cache_hit_normalizes_unnormalized_cached_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            model_name = "test-model"
            cache_file = self._expected_cache_file(
                cache_dir, SAMPLE_FAQ_DOCS, model_name
            )
            unnorm = np.tile(
                np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32),
                (len(SAMPLE_FAQ_DOCS), 1),
            )
            np.save(cache_file, unnorm)

            emb = FakeEmbedder()
            idx = HybridFAQSearch(
                SAMPLE_FAQ_DOCS,
                model_name=model_name,
                cache_dir=cache_dir,
                embedder=emb,
            )
            self.assertEqual(emb.passage_calls, [])
            norms = np.linalg.norm(idx.doc_vectors, axis=1)
            np.testing.assert_allclose(norms, np.ones(len(SAMPLE_FAQ_DOCS)), atol=1e-6)

    def test_npy_cache_rejects_pickled_object_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            model_name = "test-model"
            cache_file = self._expected_cache_file(
                cache_dir, SAMPLE_FAQ_DOCS, model_name
            )
            obj_arr = np.array([{"bad": 1}] * len(SAMPLE_FAQ_DOCS), dtype=object)
            np.save(cache_file, obj_arr, allow_pickle=True)

            with self.assertRaises(ValueError):
                HybridFAQSearch(
                    SAMPLE_FAQ_DOCS,
                    model_name=model_name,
                    cache_dir=cache_dir,
                    embedder=FakeEmbedder(),
                )

    def test_npy_cache_mismatched_shape_recomputes_and_overwrites(self) -> None:
        for bad_shape in ((len(SAMPLE_FAQ_DOCS) + 1, 4), (4,)):
            with self.subTest(bad_shape=bad_shape), tempfile.TemporaryDirectory() as tmp:
                cache_dir = Path(tmp)
                model_name = "test-model"
                cache_file = self._expected_cache_file(
                    cache_dir, SAMPLE_FAQ_DOCS, model_name
                )
                np.save(cache_file, np.ones(bad_shape, dtype=np.float32))

                emb = FakeEmbedder(dim=4)
                idx = HybridFAQSearch(
                    SAMPLE_FAQ_DOCS,
                    model_name=model_name,
                    cache_dir=cache_dir,
                    embedder=emb,
                )
                self.assertEqual(len(emb.passage_calls), 1)
                self.assertEqual(idx.doc_vectors.shape, (len(SAMPLE_FAQ_DOCS), 4))
                reloaded = np.load(cache_file, allow_pickle=False)
                self.assertEqual(reloaded.shape, (len(SAMPLE_FAQ_DOCS), 4))

    def test_npy_cache_corrupt_bytes_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            model_name = "test-model"
            cache_file = self._expected_cache_file(
                cache_dir, SAMPLE_FAQ_DOCS, model_name
            )
            cache_file.write_bytes(b"not a valid npy payload")

            with self.assertRaises((ValueError, OSError)):
                HybridFAQSearch(
                    SAMPLE_FAQ_DOCS,
                    model_name=model_name,
                    cache_dir=cache_dir,
                    embedder=FakeEmbedder(),
                )
