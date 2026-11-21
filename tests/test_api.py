"""Offline unit tests for the FastAPI + Pydantic API layer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from conftest import SAMPLE_FAQ_DOCS, FakeEmbedder, FakeLlama, FakeSearchIndex
from llm_rag_faq.api import create_app
from llm_rag_faq.api.service import APISettings, FAQChatService
from llm_rag_faq.config import DEFAULT_MAX_QUERY_CHARS
from llm_rag_faq.llm import LocalGGUFClient
from llm_rag_faq.retrieval import HybridFAQSearch


class TestFastAPILayer(unittest.TestCase):
    """Offline tests for /api/health, /api/courses, /api/chat, and lifespan."""

    def _make_service(
        self,
        *,
        responses: list[str] | None = None,
        owns_client: bool = False,
    ) -> tuple[FAQChatService, FakeLlama]:
        fake_llama = FakeLlama(responses=responses or ["Mount the ollama volume and pull llama3."])
        llm_client = LocalGGUFClient(llm=fake_llama)
        index = HybridFAQSearch(SAMPLE_FAQ_DOCS, embedder=FakeEmbedder())
        service = FAQChatService(
            index=index,
            llm_client=llm_client,
            courses=["llm-zoomcamp", "ml-zoomcamp"],
            num_documents=len(SAMPLE_FAQ_DOCS),
            provider_name="local-gguf",
            default_course="llm-zoomcamp",
            owns_client=owns_client,
        )
        return service, fake_llama

    def test_health_and_courses_endpoints(self) -> None:
        service, _ = self._make_service()
        app = create_app(service=service, settings=APISettings(static_dir=None))

        with TestClient(app) as client:
            health = client.get("/api/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(
                health.json(),
                {
                    "status": "ok",
                    "provider": "local-gguf",
                    "default_course": "llm-zoomcamp",
                    "num_documents": len(SAMPLE_FAQ_DOCS),
                },
            )

            courses = client.get("/api/courses")
            self.assertEqual(courses.status_code, 200)
            self.assertEqual(
                courses.json(),
                {
                    "default_course": "llm-zoomcamp",
                    "courses": ["llm-zoomcamp", "ml-zoomcamp"],
                },
            )

    def test_chat_returns_answer_and_filtered_sources(self) -> None:
        service, _ = self._make_service(
            responses=["Mount the ollama volume and pull llama3 inside the container."]
        )
        app = create_app(service=service, settings=APISettings(static_dir=None))

        with TestClient(app) as client:
            resp = client.post(
                "/api/chat",
                json={"query": "How do I run Ollama locally?", "course": "llm-zoomcamp"},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("ollama", data["answer"].lower())
            self.assertEqual(data["course"], "llm-zoomcamp")
            self.assertFalse(data["used_fallback"])
            self.assertIsNone(data["rewritten_query"])
            self.assertGreater(len(data["sources"]), 0)
            self.assertTrue(all(src["course"] == "llm-zoomcamp" for src in data["sources"]))

    def test_chat_all_courses_normalizes_to_none(self) -> None:
        service, _ = self._make_service(responses=["Wrap predict_proba in a Flask endpoint."])
        app = create_app(service=service, settings=APISettings(static_dir=None))

        with TestClient(app) as client:
            resp = client.post(
                "/api/chat",
                json={"query": "How do I serve a scikit-learn model with Flask?", "course": "all"},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIsNone(data["course"])
            courses_returned = {src["course"] for src in data["sources"]}
            self.assertIn("ml-zoomcamp", courses_returned)

    def test_chat_fallback_rewrite_fields(self) -> None:
        index = FakeSearchIndex(
            batches=[
                [SAMPLE_FAQ_DOCS[0]],
                [SAMPLE_FAQ_DOCS[2]],
            ]
        )
        fake_llama = FakeLlama(
            responses=[
                "I don't know.",
                "hybrid search RRF",
                "Hybrid search combines minsearch TF-IDF with fastembed embeddings.",
            ]
        )
        llm_client = LocalGGUFClient(llm=fake_llama)
        service = FAQChatService(
            index=index,
            llm_client=llm_client,
            courses=["llm-zoomcamp"],
            num_documents=len(SAMPLE_FAQ_DOCS),
            provider_name="local-gguf",
            default_course="llm-zoomcamp",
        )
        app = create_app(service=service, settings=APISettings(static_dir=None))

        with TestClient(app) as client:
            resp = client.post(
                "/api/chat",
                json={"query": "Tell me about RRF?", "course": "llm-zoomcamp"},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["used_fallback"])
            self.assertEqual(data["rewritten_query"], "hybrid search RRF")
            self.assertIn("Hybrid search", data["answer"])

    def test_chat_validation_and_unknown_course(self) -> None:
        service, _ = self._make_service()
        app = create_app(service=service, settings=APISettings(static_dir=None))

        with TestClient(app) as client:
            # Empty / whitespace query -> 422
            self.assertEqual(client.post("/api/chat", json={"query": "   "}).status_code, 422)
            # Overly long query -> 422
            too_long = "a" * (DEFAULT_MAX_QUERY_CHARS + 1)
            self.assertEqual(client.post("/api/chat", json={"query": too_long}).status_code, 422)
            # Path traversal course slug -> 422
            self.assertEqual(
                client.post("/api/chat", json={"query": "hi", "course": "../etc"}).status_code,
                422,
            )
            # Unknown extra field -> 422
            self.assertEqual(
                client.post("/api/chat", json={"query": "hi", "extra_field": 1}).status_code,
                422,
            )
            # Invalid num_results -> 422
            self.assertEqual(
                client.post("/api/chat", json={"query": "hi", "num_results": 0}).status_code,
                422,
            )
            # Unknown valid course slug -> 400
            unknown = client.post(
                "/api/chat",
                json={"query": "hi", "course": "nonexistent-course"},
            )
            self.assertEqual(unknown.status_code, 400)

    def test_lifespan_ownership_closes_only_app_owned_service(self) -> None:
        injected_service, injected_llama = self._make_service(owns_client=True)
        app_injected = create_app(service=injected_service, settings=APISettings(static_dir=None))
        with TestClient(app_injected) as client:
            self.assertEqual(client.get("/api/health").status_code, 200)
        self.assertFalse(injected_llama.closed)

        owned_service, owned_llama = self._make_service(owns_client=True)
        with patch("llm_rag_faq.api.app.build_chat_service", return_value=owned_service):
            app_owned = create_app(settings=APISettings(static_dir=None))
            with TestClient(app_owned) as client:
                self.assertEqual(client.get("/api/health").status_code, 200)
            self.assertTrue(owned_llama.closed)

    def test_static_spa_serving_when_dist_present(self) -> None:
        service, _ = self._make_service()
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            (dist_dir / "index.html").write_text(
                "<html><body>FAQ UI</body></html>",
                encoding="utf-8",
            )
            app = create_app(service=service, settings=APISettings(static_dir=dist_dir))
            with TestClient(app) as client:
                root = client.get("/")
                self.assertEqual(root.status_code, 200)
                self.assertIn("FAQ UI", root.text)
                missing_api = client.get("/api/nonexistent")
                self.assertEqual(missing_api.status_code, 404)


if __name__ == "__main__":
    unittest.main()
