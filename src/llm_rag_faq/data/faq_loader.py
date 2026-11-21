"""DataTalks.Club FAQ fetching and JSON disk caching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from llm_rag_faq.config import (
    DEFAULT_COURSE,
    DEFAULT_TIMEOUT,
    DOCS_URL,
    URL_PREFIX,
)


def _resolve_course_url(path: str, prefix: str = URL_PREFIX) -> str:
    """Resolve a catalog entry path against the FAQ base URL."""
    if path.startswith(("http://", "https://")):
        return path
    return urljoin(prefix, path.lstrip("/"))


def _validate_course_slug(course: str) -> str:
    """Validate that a course slug cannot escape the cache directory."""
    if "/" in course or "\\" in course or ".." in course or "\x00" in course:
        raise ValueError(f"Invalid course slug {course!r}: path traversal is not allowed")
    return course


def load_faq_data(
    course: str | None = DEFAULT_COURSE,
    timeout: float = DEFAULT_TIMEOUT,
    session: requests.Session | None = None,
    cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch FAQ documents from DataTalks.Club for one course or all courses.

    Downloads the course catalog from ``DOCS_URL``, selects matching entries
    (or all entries when ``course is None``), and fetches and concatenates the
    corresponding course FAQ JSON payloads.

    Args:
        course: Course slug to load (e.g., ``"llm-zoomcamp"``), or ``None`` to
            load all courses in the catalog.
        timeout: Per-request HTTP timeout in seconds.
        session: Optional ``requests.Session`` to reuse across HTTP requests.
        cache_dir: Optional directory for caching fetched FAQ JSON on disk.

    Returns:
        List of FAQ document dictionaries.

    Raises:
        ValueError: If ``course`` contains path traversal characters, is not
            ``None`` and matches no entry in the course catalog, or a cached
            payload is not a JSON list.
        KeyError: If a matched catalog entry is missing a ``"path"`` field.
        requests.HTTPError: If any HTTP request returns a non-2xx status code.
        requests.RequestException: On network, timeout, or connection errors.
    """
    if course is not None:
        _validate_course_slug(course)

    cache_file: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        slug = course if course is not None else "all"
        cache_file = cache_path / f"faq_{slug}.json"
        if cache_file.resolve().parent != cache_path.resolve():
            raise ValueError(f"Invalid course slug {course!r}: resolves outside cache directory")
        if cache_file.exists():
            cached: Any = json.loads(cache_file.read_text(encoding="utf-8"))
            if not isinstance(cached, list):
                raise ValueError(
                    f"Expected JSON list in cache file {cache_file}, got {type(cached).__name__}"
                )
            return list(cached)

    owns_session = session is None
    http = session if session is not None else requests.Session()
    try:
        response = http.get(DOCS_URL, timeout=timeout)
        response.raise_for_status()
        courses_raw = response.json()

        if course is None:
            selected = courses_raw
        else:
            selected = [c for c in courses_raw if c.get("course") == course]
            if not selected:
                available = sorted(c["course"] for c in courses_raw if "course" in c)
                raise ValueError(f"Unknown course {course!r}. Available courses: {available}")

        documents: list[dict[str, Any]] = []
        for entry in selected:
            path = entry.get("path")
            if not path:
                raise KeyError(f"Course catalog entry is missing 'path': {entry!r}")
            course_url = _resolve_course_url(path)
            course_response = http.get(course_url, timeout=timeout)
            course_response.raise_for_status()
            documents.extend(course_response.json())
    finally:
        if owns_session:
            http.close()

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(documents), encoding="utf-8")

    return documents
