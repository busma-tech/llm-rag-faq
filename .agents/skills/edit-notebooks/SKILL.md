---
name: edit-notebooks
description: >-
  Use when creating, editing, or updating Jupyter notebooks under notebooks/ or
  keeping docs/notebook-usage-guide.md in sync with package changes.
---

# Edit & Sync Jupyter Notebooks

Follow these rules whenever modifying notebooks in `notebooks/`.

## Rules

1. **Read context first**: Check `docs/notebook-usage-guide.md` before changing cells in `notebooks/02_hybrid_rag_demo.ipynb`.
2. **Use the public facade only**: Production demo notebooks (like `02_hybrid_rag_demo.ipynb`) must import exclusively from `llm_rag_faq` (`from llm_rag_faq import ...`), not internal submodules.
3. **Share the repo-root cache**: Keep `CACHE_DIR = "../.cache"` so notebooks reuse FAQ JSON, `.npy` embeddings, and GGUF weights cached by the CLI instead of re-downloading.
4. **Prevent native memory leaks / OOM**:
   - Use `with RAGBase(...) as assistant:` or explicitly call `close()` when replacing `LocalGGUFClient` instances.
   - Avoid loading multiple local GGUF models simultaneously in the same kernel.
5. **Keep secrets and noisy outputs out of git**:
   - Never hardcode `ANTHROPIC_API_KEY` or `HF_TOKEN`; load via `dotenv.load_dotenv("../.env")`.
   - Avoid committing huge cell outputs, stack traces, or environment-specific absolute paths.
6. **Keep docs in sync**:
   - If you add, remove, or reorder cells, or change public API usage in `02_hybrid_rag_demo.ipynb`, update `docs/notebook-usage-guide.md` to match.
   - If you add a new notebook, list it in `README.md`.
