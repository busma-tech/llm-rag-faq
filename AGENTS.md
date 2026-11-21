# AGENTS.md — `llm-rag-faq`

Hybrid lexical + semantic FAQ RAG assistant for DataTalks.Club courses (`src/` layout, Python 3.12, `uv`).

## Quick Commands

| Task | Command |
|---|---|
| Full CI quality suite | `make check` (`lint` + `typecheck` + `arch` + `test`) |
| Offline tests (zero dev deps) | `python -m unittest discover -s tests -v` |
| Architecture checks only | `make arch` or `python -m unittest tests/test_architecture.py -v` |
| Auto-format & fix lint | `make format` |
| Run CLI | `uv run llm-rag-faq "Can I join late?" --show-sources` |

> If you edit dependencies in `pyproject.toml`, `uv.lock` must be refreshed with `uv lock` in an online environment before CI (`uv sync --locked --group dev`).

## Architecture & Hard Boundaries (Option B)

Strict one-way layers, machine-enforced by `import-linter` (`pyproject.toml`) and `tests/test_architecture.py`:

| Layer | Allowed Internal Imports | Forbidden |
|---|---|---|
| `config` | *(none)* | Any `llm_rag_faq.*` module |
| `data` | `config` | `retrieval`, `llm`, `rag`, `cli`, root facade |
| `retrieval` | `config` | `data`, `llm`, `rag`, `cli`, root facade |
| `llm` | `config` | `data`, `retrieval`, `rag`, `cli`, root facade |
| `rag` | `config`, `llm` | `data`, `retrieval`, `cli`, root facade, and raw SDKs (`requests`, `minsearch`, `fastembed`, `llama_cpp`, `anthropic`, `huggingface_hub`) |
| `cli` | `config`, `data`, `retrieval`, `llm`, `rag` | Root facade `llm_rag_faq` (prevents import cycles) |
| `__init__` (facade) | `data`, `retrieval`, `llm`, `rag` | Must export exact sorted 7-symbol `__all__` (`tests/test_architecture.py`) |

- `rag/` interacts with retrieval only via the `SearchIndex` `Protocol` in `src/llm_rag_faq/rag/pipeline.py`.
- Use absolute imports across layers (`from llm_rag_faq.config import ...`) and single-dot relative imports only within the same subpackage (`from .prompts import ...`). Never use `..` cross-layer relative imports.

## Key Code & Security Invariants

- **Every `.py` file** starts with `from __future__ import annotations`.
- **Typing & style**: Python 3.12 built-in generics (`str | None`, `list[...]`), keyword-only args (`*`) for optional flags, Google-style docstrings, Ruff (`line-length = 100`), `mypy src` clean.
- **No mutable default args**: Default to `None` and make a defensive copy (`dict(...)`).
- **Resource lifecycle**: Classes owning native/HTTP resources (`LocalGGUFClient`, `AnthropicClient`, `RAGBase`) implement `__enter__`, `__exit__`, `close()`, and track ownership (`_owns_client` / `owns_session`) — close only what you created.
- **Logging**: Library modules use `logging.getLogger(__name__)` with `%s`/`%d` placeholders; only `cli.py` calls `print()`.
- **Offline tests**: Write tests with `unittest.TestCase` (compatible with both `unittest` and `pytest`) and reuse the offline doubles (`FakeEmbedder`, `FakeLlama`, `FakeAnthropic`, `FakeSearchIndex`) in `tests/conftest.py`. Never make real network or model calls in unit tests.
- **Security**:
  - Keep `allow_pickle=False` on all `np.load(...)` cache reads.
  - Preserve pinned Hugging Face GGUF revision SHA and wildcard rejection (`*?[`) in `local_gguf.py`.
  - Preserve XML `<context>`/`<question>` delimiter escaping, single-pass `render_prompt()`, ChatML control-token sanitization, and course-slug path-traversal validation.

## Documentation Routing (Read On Demand)

Do not preload all docs; read the relevant file only when working in that area:

- **Layer design, retrieval/RRF, protocols, local 1.5B LLM strategy**: `docs/system-architecture.md`
- **Python conventions, typing, decorators, ownership, testing patterns**: `docs/python-standards-and-patterns.md`
- **Security hardening history, prompt injection, supply-chain pinning**: `docs/code-review-notes.md`
- **Jupyter notebook cells, shared cache, OOM prevention**: `docs/notebook-usage-guide.md`

## Paths Not to Touch or Commit

- `.env` — contains local secrets (`ANTHROPIC_API_KEY`, `HF_TOKEN`); never read, print, or commit it. Use `.env.example` for template changes.
- `.cache/` — local Hugging Face GGUF weights, `.npy` embeddings, and cached FAQ JSON; never edit or commit.
- `.venv/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/` — generated tooling artifacts.
