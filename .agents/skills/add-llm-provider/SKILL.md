---
name: add-llm-provider
description: >-
  Use when adding a new LLM backend/provider to llm-rag-faq, implementing
  LLMClient, wiring it into the CLI, and adding offline unit tests.
---

# Add an LLM Provider

Follow this checklist when adding a new LLM client implementation to `llm-rag-faq`.

## Steps

1. **Create the client module** under `src/llm_rag_faq/llm/<provider>_client.py`:
   - Start with `from __future__ import annotations`.
   - Implement the `LLMClient` protocol from `src/llm_rag_faq/llm/base.py`:
     `generate(self, system: str | None, prompt: str, max_tokens: int = ...) -> str`.
   - Accept an optional pre-built SDK client in `__init__` (constructor DI) and track `self._owns_client = client is None`.
   - Implement `close()`, `__enter__()`, and `__exit__()` so only owned resources are closed.
   - Validate inputs fail-fast (`TypeError` / `ValueError`) and use `logging.getLogger(__name__)` (never `print()`).
   - Import only from `llm_rag_faq.config` or sibling `llm` modules (never `data`, `retrieval`, `rag`, `cli`, or root `llm_rag_faq`).

2. **Add default constants** (model ID, token limits, etc.) in `src/llm_rag_faq/config.py`.

3. **Export from the `llm` subpackage**:
   - Re-export the class in `src/llm_rag_faq/llm/__init__.py` and keep `__all__` sorted.
   - Only if the client should be part of the public facade:
     - Re-export it in `src/llm_rag_faq/__init__.py` (keep `__all__` sorted).
     - Update `EXPECTED_PUBLIC_EXPORTS` in `tests/test_architecture.py`.
     - If the new provider introduces a third-party SDK, add that SDK module name to `FORBIDDEN_RAG_SDKS` in `tests/test_architecture.py` and `[tool.importlinter.contracts]` in `pyproject.toml` so `rag/` stays isolated.

4. **Wire into the CLI** (`src/llm_rag_faq/cli.py`):
   - Add the provider name to `--provider` choices in `build_parser()`.
   - Instantiate it in `_create_llm_client()`, reading any API key from `os.getenv(...)` and failing fast with a clear `RuntimeError` if required and missing.
   - Document any new env var in `.env.example` (never `.env`) and `README.md`.

5. **Add offline unit tests**:
   - Add a deterministic fake SDK double in `tests/conftest.py`.
   - Create `tests/llm/test_<provider>_client.py` using `unittest.TestCase` covering happy path, input validation, empty/malformed responses, and `close()` / context-manager ownership.

6. **Verify**:
   - Run `make check` (or `python -m unittest discover -s tests -v` if offline without dev deps).
