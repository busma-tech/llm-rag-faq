"""Unit tests for ``llm_rag_faq.data.faq_loader``."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import requests

from llm_rag_faq.config import DEFAULT_COURSE, DEFAULT_TIMEOUT, DOCS_URL, URL_PREFIX
from llm_rag_faq.data import load_faq_data
from llm_rag_faq.data.faq_loader import _resolve_course_url, _validate_course_slug


def _make_response(
    payload: Any,
    status_error: Exception | None = None,
) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.side_effect = status_error
    return resp


def _make_session(url_map: dict[str, MagicMock]) -> MagicMock:
    session = MagicMock()
    session.get.side_effect = lambda url, timeout=None: url_map[url]
    return session


class TestResolveCourseUrl(unittest.TestCase):
    def test_relative_path(self) -> None:
        self.assertEqual(
            _resolve_course_url("json/llm-zoomcamp.json"),
            f"{URL_PREFIX}json/llm-zoomcamp.json",
        )

    def test_leading_slashes_stripped(self) -> None:
        self.assertEqual(
            _resolve_course_url("/json/llm-zoomcamp.json"),
            f"{URL_PREFIX}json/llm-zoomcamp.json",
        )
        self.assertEqual(
            _resolve_course_url("///a.json"),
            f"{URL_PREFIX}a.json",
        )

    def test_absolute_http_and_https_urls_unchanged(self) -> None:
        self.assertEqual(
            _resolve_course_url("http://example.com/a.json"),
            "http://example.com/a.json",
        )
        self.assertEqual(
            _resolve_course_url("https://cdn.example.com/b.json"),
            "https://cdn.example.com/b.json",
        )

    def test_custom_prefix(self) -> None:
        self.assertEqual(
            _resolve_course_url("/sub/faq.json", prefix="https://example.org/base/"),
            "https://example.org/base/sub/faq.json",
        )


class TestValidateCourseSlug(unittest.TestCase):
    def test_valid_slugs_pass(self) -> None:
        for slug in ("llm-zoomcamp", "mlops_zoomcamp", "de-zoomcamp-2025"):
            with self.subTest(slug=slug):
                self.assertEqual(_validate_course_slug(slug), slug)

    def test_rejects_traversal_and_separators(self) -> None:
        bad_slugs = ["../etc/passwd", "a/b", "a\\b", "foo..bar", "bad\x00slug"]
        for slug in bad_slugs:
            with (
                self.subTest(slug=slug),
                self.assertRaisesRegex(ValueError, "path traversal is not allowed"),
            ):
                _validate_course_slug(slug)


class TestLoadFaqDataHttpAndSelection(unittest.TestCase):
    def test_specific_course_fetches_only_matching_course(self) -> None:
        catalog = [
            {"course": "llm-zoomcamp", "path": "json/llm.json"},
            {"course": "ml-zoomcamp", "path": "https://example.com/ml.json"},
        ]
        llm_docs = [{"course": "llm-zoomcamp", "question": "Q1", "answer": "A1"}]
        ml_docs = [{"course": "ml-zoomcamp", "question": "Q2", "answer": "A2"}]
        sess = _make_session(
            {
                DOCS_URL: _make_response(catalog),
                f"{URL_PREFIX}json/llm.json": _make_response(llm_docs),
                "https://example.com/ml.json": _make_response(ml_docs),
            }
        )

        docs = load_faq_data(course="llm-zoomcamp", session=sess)

        self.assertEqual(docs, llm_docs)
        fetched_urls = [call.args[0] for call in sess.get.call_args_list]
        self.assertEqual(fetched_urls, [DOCS_URL, f"{URL_PREFIX}json/llm.json"])

    def test_default_course_used_when_course_omitted(self) -> None:
        catalog = [
            {"course": DEFAULT_COURSE, "path": "json/default.json"},
            {"course": "other-course", "path": "json/other.json"},
        ]
        default_docs = [{"course": DEFAULT_COURSE, "question": "Q1"}]
        sess = _make_session(
            {
                DOCS_URL: _make_response(catalog),
                f"{URL_PREFIX}json/default.json": _make_response(default_docs),
            }
        )

        docs = load_faq_data(session=sess)

        self.assertEqual(docs, default_docs)
        fetched_urls = [call.args[0] for call in sess.get.call_args_list]
        self.assertEqual(fetched_urls, [DOCS_URL, f"{URL_PREFIX}json/default.json"])

    def test_course_none_fetches_and_concatenates_all_courses(self) -> None:
        catalog = [
            {"course": "llm-zoomcamp", "path": "/json/llm.json"},
            {"course": "ml-zoomcamp", "path": "https://example.com/ml.json"},
        ]
        llm_docs = [{"course": "llm-zoomcamp", "question": "Q1"}]
        ml_docs = [{"course": "ml-zoomcamp", "question": "Q2"}]
        sess = _make_session(
            {
                DOCS_URL: _make_response(catalog),
                f"{URL_PREFIX}json/llm.json": _make_response(llm_docs),
                "https://example.com/ml.json": _make_response(ml_docs),
            }
        )

        docs = load_faq_data(course=None, session=sess)

        self.assertEqual(docs, llm_docs + ml_docs)
        fetched_urls = [call.args[0] for call in sess.get.call_args_list]
        self.assertEqual(
            fetched_urls,
            [DOCS_URL, f"{URL_PREFIX}json/llm.json", "https://example.com/ml.json"],
        )

    def test_unknown_course_raises_value_error_listing_available(self) -> None:
        catalog = [
            {"course": "ml-zoomcamp", "path": "json/ml.json"},
            {"course": "de-zoomcamp", "path": "json/de.json"},
        ]
        sess = _make_session({DOCS_URL: _make_response(catalog)})

        with self.assertRaises(ValueError) as ctx:
            load_faq_data(course="nonexistent", session=sess)

        msg = str(ctx.exception)
        self.assertIn("Unknown course 'nonexistent'", msg)
        self.assertIn("['de-zoomcamp', 'ml-zoomcamp']", msg)

    def test_missing_or_empty_path_raises_key_error(self) -> None:
        for entry in (
            {"course": "llm-zoomcamp"},
            {"course": "llm-zoomcamp", "path": ""},
        ):
            with self.subTest(entry=entry):
                sess = _make_session({DOCS_URL: _make_response([entry])})
                with self.assertRaisesRegex(KeyError, "missing 'path'"):
                    load_faq_data(course="llm-zoomcamp", session=sess)

    def test_http_error_propagates_and_skips_cache_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"

            # Catalog HTTP error
            sess_cat = _make_session(
                {DOCS_URL: _make_response([], status_error=requests.HTTPError("500"))}
            )
            with self.assertRaises(requests.HTTPError):
                load_faq_data(
                    course="llm-zoomcamp",
                    session=sess_cat,
                    cache_dir=cache_dir,
                )
            self.assertFalse((cache_dir / "faq_llm-zoomcamp.json").exists())

            # Course URL HTTP error
            catalog = [{"course": "llm-zoomcamp", "path": "json/llm.json"}]
            sess_course = _make_session(
                {
                    DOCS_URL: _make_response(catalog),
                    f"{URL_PREFIX}json/llm.json": _make_response(
                        [], status_error=requests.HTTPError("404")
                    ),
                }
            )
            with self.assertRaises(requests.HTTPError):
                load_faq_data(
                    course="llm-zoomcamp",
                    session=sess_course,
                    cache_dir=cache_dir,
                )
            self.assertFalse((cache_dir / "faq_llm-zoomcamp.json").exists())

    def test_traversal_slug_raises_before_http_or_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sess = MagicMock()
            with self.assertRaises(ValueError):
                load_faq_data(course="../secret", session=sess, cache_dir=tmp)
            sess.get.assert_not_called()


class TestLoadFaqDataSessionAndTimeout(unittest.TestCase):
    def test_external_session_reused_and_not_closed_on_success_or_error(self) -> None:
        catalog = [{"course": "llm-zoomcamp", "path": "json/llm.json"}]
        sess = _make_session(
            {
                DOCS_URL: _make_response(catalog),
                f"{URL_PREFIX}json/llm.json": _make_response([{"question": "Q"}]),
            }
        )
        load_faq_data(course="llm-zoomcamp", session=sess)
        sess.close.assert_not_called()

        with self.assertRaises(ValueError):
            load_faq_data(course="missing", session=sess)
        sess.close.assert_not_called()

    def test_none_session_creates_and_closes_owned_session(self) -> None:
        catalog = [{"course": "llm-zoomcamp", "path": "json/llm.json"}]
        owned_ok = _make_session(
            {
                DOCS_URL: _make_response(catalog),
                f"{URL_PREFIX}json/llm.json": _make_response([{"question": "Q"}]),
            }
        )
        with patch("llm_rag_faq.data.faq_loader.requests.Session", return_value=owned_ok):
            docs = load_faq_data(course="llm-zoomcamp", session=None)
        self.assertEqual(docs, [{"question": "Q"}])
        owned_ok.close.assert_called_once_with()

        owned_err = _make_session({DOCS_URL: _make_response(catalog)})
        with (
            patch("llm_rag_faq.data.faq_loader.requests.Session", return_value=owned_err),
            self.assertRaises(ValueError),
        ):
            load_faq_data(course="missing", session=None)
        owned_err.close.assert_called_once_with()

        owned_http_err = _make_session(
            {DOCS_URL: _make_response([], status_error=requests.HTTPError("503"))}
        )
        with (
            patch(
                "llm_rag_faq.data.faq_loader.requests.Session",
                return_value=owned_http_err,
            ),
            self.assertRaises(requests.HTTPError),
        ):
            load_faq_data(course="llm-zoomcamp", session=None)
        owned_http_err.close.assert_called_once_with()

    def test_timeout_propagates_default_and_custom(self) -> None:
        catalog = [{"course": "llm-zoomcamp", "path": "json/llm.json"}]
        url_map = {
            DOCS_URL: _make_response(catalog),
            f"{URL_PREFIX}json/llm.json": _make_response([]),
        }
        sess_default = _make_session(url_map)
        load_faq_data(course="llm-zoomcamp", session=sess_default)
        for call in sess_default.get.call_args_list:
            self.assertEqual(call.kwargs["timeout"], DEFAULT_TIMEOUT)

        sess_custom = _make_session(url_map)
        load_faq_data(course="llm-zoomcamp", timeout=7.5, session=sess_custom)
        for call in sess_custom.get.call_args_list:
            self.assertEqual(call.kwargs["timeout"], 7.5)


class TestLoadFaqDataDiskCache(unittest.TestCase):
    def test_cache_miss_writes_json_and_cache_hit_skips_http(self) -> None:
        catalog = [{"course": "llm-zoomcamp", "path": "json/llm.json"}]
        payload = [{"course": "llm-zoomcamp", "question": "Q1", "answer": "A1"}]
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "nested" / "cache"
            sess1 = _make_session(
                {
                    DOCS_URL: _make_response(catalog),
                    f"{URL_PREFIX}json/llm.json": _make_response(payload),
                }
            )
            docs1 = load_faq_data(
                course="llm-zoomcamp",
                session=sess1,
                cache_dir=cache_dir,
            )
            self.assertEqual(docs1, payload)
            cache_file = cache_dir / "faq_llm-zoomcamp.json"
            self.assertTrue(cache_file.is_file())
            self.assertEqual(json.loads(cache_file.read_text(encoding="utf-8")), payload)

            sess2 = MagicMock()
            docs2 = load_faq_data(
                course="llm-zoomcamp",
                session=sess2,
                cache_dir=cache_dir,
            )
            self.assertEqual(docs2, payload)
            sess2.get.assert_not_called()

            with patch("llm_rag_faq.data.faq_loader.requests.Session") as session_cls:
                docs3 = load_faq_data(
                    course="llm-zoomcamp",
                    session=None,
                    cache_dir=cache_dir,
                )
            self.assertEqual(docs3, payload)
            session_cls.assert_not_called()

    def test_course_none_uses_faq_all_json_cache(self) -> None:
        catalog = [{"course": "llm-zoomcamp", "path": "json/llm.json"}]
        payload = [{"course": "llm-zoomcamp", "question": "All Q"}]
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            sess1 = _make_session(
                {
                    DOCS_URL: _make_response(catalog),
                    f"{URL_PREFIX}json/llm.json": _make_response(payload),
                }
            )
            docs1 = load_faq_data(course=None, session=sess1, cache_dir=cache_dir)
            self.assertEqual(docs1, payload)
            self.assertTrue((cache_dir / "faq_all.json").is_file())

            sess2 = MagicMock()
            docs2 = load_faq_data(course=None, session=sess2, cache_dir=cache_dir)
            self.assertEqual(docs2, payload)
            sess2.get.assert_not_called()

    def test_invalid_json_in_cache_raises_json_decode_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "faq_llm-zoomcamp.json").write_text("{bad json", encoding="utf-8")
            sess = MagicMock()
            with self.assertRaises(json.JSONDecodeError):
                load_faq_data(course="llm-zoomcamp", session=sess, cache_dir=cache_dir)
            sess.get.assert_not_called()

    def test_non_list_json_in_cache_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "faq_llm-zoomcamp.json").write_text('{"not": "a list"}', encoding="utf-8")
            sess = MagicMock()
            with self.assertRaisesRegex(ValueError, "Expected JSON list"):
                load_faq_data(course="llm-zoomcamp", session=sess, cache_dir=cache_dir)
            sess.get.assert_not_called()

    def test_symlink_outside_cache_dir_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            outside = root / "outside.json"
            outside.write_text("[]", encoding="utf-8")
            (cache_dir / "faq_evil.json").symlink_to(outside)

            sess = MagicMock()
            with self.assertRaisesRegex(ValueError, "resolves outside cache directory"):
                load_faq_data(course="evil", session=sess, cache_dir=cache_dir)
            sess.get.assert_not_called()
