---
name: pre-pr-check
description: >-
  Use before opening a PR or finalizing a non-trivial change in llm-rag-faq to
  run quality gates, check documentation drift, and verify security invariants.
---

# Pre-PR Verification Checklist

Run through these checks before concluding a feature, refactor, or bug fix.

## 1. Automated Quality Gates

Run the full CI pipeline:

```bash
make check
```

This executes `lint` (`ruff check` + `ruff format --check`), `typecheck` (`mypy src`), `arch` (`lint-imports` + `tests/test_architecture.py`), and `test` (`pytest -v`).

If dev dependencies are unavailable in an offline environment, at minimum run:

```bash
python -m unittest discover -s tests -v
```

## 2. Security & Invariant Spot-Check

If your changes touched prompts, retrieval, caching, HTTP loading, or local GGUF loading, verify against `docs/code-review-notes.md`:
- No `np.load(..., allow_pickle=True)`.
- No unescaped `<context>` / `<question>` delimiters or multi-pass `.format()` / chained `.replace()` prompt rendering.
- No unpinned Hugging Face model revisions or wildcard GGUF filenames.
- No path traversal in cache file paths (`_validate_course_slug` + `.resolve().parent` check).
- No secrets committed (`.env` stays untracked; new env vars added to `.env.example` only).

## 3. Documentation & Lockfile Drift

- Check `git status` and `git diff --stat`:
  - If CLI flags changed in `src/llm_rag_faq/cli.py`, update the CLI table in `README.md`.
  - If files/modules were added or removed, update the Repository Tree in `README.md` (and `AGENTS.md` if layer rules or docs changed).
  - If `pyproject.toml` dependencies changed, confirm `uv.lock` was refreshed with `uv lock` (or note that CI requires an online `uv lock`).
