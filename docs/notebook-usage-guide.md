# Jupyter Notebook Usage Guide — `llm-rag-faq`

This guide shows how to use the `llm_rag_faq` public facade (`src/llm_rag_faq/__init__.py`) inside the repository's Jupyter notebooks:

- [notebooks/01_rag_exploration.ipynb](../notebooks/01_rag_exploration.ipynb) — initial exploratory RAG notebook (`requests`, `minsearch`, and Anthropic).
- [notebooks/02_hybrid_rag_demo.ipynb](../notebooks/02_hybrid_rag_demo.ipynb) — end-to-end hybrid RAG demo using `llm_rag_faq`.

---

## Prerequisites

1. Select the project virtual environment kernel (`llm-rag-faq (3.12.x)`) in Jupyter.
2. *(Optional, for Anthropic cloud API)* Copy `.env.example` to `.env` at the repository root and set `ANTHROPIC_API_KEY`. If omitted, you can run fully offline/locally with `LocalGGUFClient`.

> **Tip:** `.gitignore` already ignores `.cache/` at the repository root. Because notebooks execute with `notebooks/` as their working directory, set `CACHE_DIR = "../.cache"` so notebook runs share the root `.cache/` directory with the CLI (`DEFAULT_CACHE_DIR` in `config.py`, which uses `.cache` from repo root).

---

## Quick Start (Cell-by-Cell)

### Cell 1 — Load Environment & Imports

`load_dotenv()` walks parent directories automatically when run inside `notebooks/`, picking up the repository-root `.env`.

```python
import os
from dotenv import load_dotenv

load_dotenv()

from llm_rag_faq import (
    AnthropicClient,
    LocalGGUFClient,
    RAGBase,
    build_index,
    load_faq_data,
)
```

> **Note:** Raw `from anthropic import Anthropic` is also supported directly by `RAGBase` for backwards convenience (as used in `02_hybrid_rag_demo.ipynb` Cell 3).

---

### Cell 2 — Fetch FAQ Data & Build the Hybrid Index (with Disk Caching)

Passing `cache_dir="../.cache"` caches both the downloaded FAQ JSON and the SHA-256 content-hashed embeddings (`.npy`), making notebook restarts instant and saving memory.

```python
CACHE_DIR = "../.cache"
COURSE = "llm-zoomcamp"  # Or e.g. "mlops-zoomcamp", or None for ALL courses

documents = load_faq_data(course=COURSE, cache_dir=CACHE_DIR)
index = build_index(documents, cache_dir=CACHE_DIR)

print(f"Indexed {len(index.documents)} documents (vector matrix: {index.doc_vectors.shape})")
```

> **Note:** `build_index(documents)` defaults to `course=None`, so whatever course(s) you loaded with `load_faq_data(...)` will be indexed without being silently dropped.

---

### Cell 3 — Initialize the LLM Client & `RAGBase`

Choose **Option A** (Anthropic Cloud API), **Option B** (Local GGUF on CPU), or **Option C** (automatic fallback from Anthropic to Local GGUF).

#### Option A: Anthropic Claude API (`AnthropicClient`)

```python
if "llm_client" in globals() and hasattr(llm_client, "close"):
    llm_client.close()

api_key = os.getenv("ANTHROPIC_API_KEY")
assert api_key, "Set ANTHROPIC_API_KEY in .env"

llm_client = AnthropicClient(model="claude-haiku-5-5", api_key=api_key)
assistant = RAGBase(
    index=index,
    llm_client=llm_client,
    course=COURSE,
)
```

> **Resource ownership note:** `RAGBase.close()` only closes an `LLMClient` that it created internally (`llm_client=None`). When you pass an external `AnthropicClient`, raw `Anthropic(...)`, or `LocalGGUFClient`, close that client explicitly (`llm_client.close()` / `local_client.close()`).

#### Option B: Local GGUF Model (Offline-Capable Once Cached)

```python
# Release previous model first if re-running this cell to avoid OOM!
if "assistant" in globals():
    assistant.close()
if "local_client" in globals():
    local_client.close()

local_client = LocalGGUFClient(
    # Pinned Qwen2.5-1.5B-Instruct Q4_K_M by default; loads offline once cached
    n_ctx=4096,
    n_threads=2,            # Keep low (2-4) in small containers / Codespaces
    repeat_penalty=1.1,
    local_files_only=False, # Set True once downloaded to guarantee zero network calls
)

assistant = RAGBase(
    index=index,
    llm_client=local_client,
    course=COURSE,
    num_results=3,          # Smaller context keeps CPU inference fast
)
```

#### Option C: Automatic Selection (Cloud if Key Present, Else Local)

```python
if "assistant" in globals():
    assistant.close()
if "llm_client" in globals() and hasattr(llm_client, "close"):
    llm_client.close()

api_key = os.getenv("ANTHROPIC_API_KEY")
llm_client = (
    AnthropicClient(model="claude-haiku-5-5", api_key=api_key)
    if api_key
    else None
)

# When llm_client=None, RAGBase creates and owns a LocalGGUFClient internally
assistant = RAGBase(
    index=index,
    llm_client=llm_client,
    course=COURSE,
)
```

---

### Cell 4 — Ask a Question (`rag`)

```python
question = "Where do I drop my questions during a live workshop so the speakers actually see them?"
answer = assistant.rag(question)
print(answer)
```

---

### Cell 5 — Inspect Sources & Fallback Behavior (`rag_with_sources`)

Use `rag_with_sources()` to see **which FAQ entries were retrieved** and **whether the query-rewrite fallback ran**—invaluable for debugging RAG quality in notebooks.

```python
q = "I never got an email after signing up—am I actually enrolled in the class?"
result = assistant.rag_with_sources(q)

print("Answer:\n", result.answer)
print("\nUsed fallback:", result.used_fallback)
if result.rewritten_query:
    print("Rewritten query:", result.rewritten_query)

print("\nRetrieved Sources:")
for i, doc in enumerate(result.sources, start=1):
    print(f"  [{i}] ({doc.get('course')} | {doc.get('section')}) {doc.get('question')}")
```

---

## Common Notebook Workflows

### 1. Debug Retrieval Without Calling the LLM

Test hybrid search directly before spending API credits or CPU generation time:

```python
q = "Where do I drop my questions during a live workshop?"
hits = assistant.search(q, num_results=5)

for i, doc in enumerate(hits, start=1):
    print(f"[{i}] ({doc.get('id')}) {doc.get('question')}")
```

You can also experiment with lexical boosts (including disabling boosts with `{}`):

```python
# Emphasize question matches heavily
hits_boosted = assistant.search(q, num_results=5, boost_dict={"question": 5.0, "section": 0.2})

# Disable lexical boosting completely
hits_unboosted = assistant.search(q, num_results=5, boost_dict={})
```

---

### 2. Query Across All Courses (Multi-Course RAG)

Pass `course=None` to both `load_faq_data()` and `RAGBase()`:

```python
all_docs = load_faq_data(course=None, cache_dir="../.cache")
all_index = build_index(all_docs, cache_dir="../.cache")

multi_assistant = RAGBase(
    index=all_index,
    llm_client=llm_client,
    course=None,  # No course filter applied during search
)

res = multi_assistant.rag_with_sources("Can I get a certificate in self-paced mode?")
print(res.answer)
for d in res.sources:
    print("-", d.get("course"), "|", d.get("question"))
```

You can also filter to a subset of courses directly on the index:

```python
subset_hits = all_index.search(
    "homework deadline",
    num_results=5,
    filter_dict={"course": ["llm-zoomcamp", "mlops-zoomcamp"]},
)
```

---

### 3. Stream Tokens from `LocalGGUFClient`

CPU generation can take 10–30+ seconds. Stream tokens live in a notebook cell so you see output immediately:

```python
q = "I just discovered the course. Can I join now?"
hits = assistant.search(q, num_results=3)
prompt = assistant.build_prompt(q, hits)

for token in local_client.generate_stream(system=assistant.instructions, prompt=prompt):
    print(token, end="", flush=True)
print()
```

---

### 4. One-Shot Context Manager (Automatic Memory Cleanup)

When running quick experiments with the local GGUF model, use a `with` block so native C++ model memory is freed as soon as the cell finishes:

```python
with RAGBase(index=index, llm_client=None, course=COURSE) as temp_assistant:
    print(temp_assistant.rag("Can I submit homework after the deadline?"))
# Underlying LocalGGUFClient is automatically closed here
```

---

### 5. Disable Fallback Rewrite (Strictly 1 LLM Call per Query)

If you want faster/cheaper evaluation runs without the agentic query-rewrite retry:

```python
fast_assistant = RAGBase(
    index=index,
    llm_client=llm_client,
    course=COURSE,
    enable_fallback=False,
)
print(fast_assistant.rag("What is the weather in Paris?"))  # -> "I don't know." in 1 call
```

---

## CLI & Automated Verification Alternative

You can also query the RAG pipeline from the terminal or verify the package offline/in CI:

```bash
# CLI one-shot query (runs from repo root using .cache)
uv run llm-rag-faq "Can I join late?" --show-sources

# Offline unit & AST architecture tests (100 tests, zero dev dependencies required)
python -m unittest discover -s tests -v

# Full quality & CI suite (once `uv sync --group dev` is installed)
make check
```

---

## Troubleshooting in Notebooks / Codespaces

| Symptom | Cause | Fix |
|---|---|---|
| **Jupyter kernel crashes (OOM)** when running `build_index` + `LocalGGUFClient` | Re-running cells creates multiple ~1 GB `Llama` instances alongside ONNX `fastembed`, or `batch_size` is too high for container RAM | Always call `local_client.close()` / `assistant.close()` before re-initializing; use `cache_dir="../.cache"` and smaller `batch_size=16` in `build_index(documents, batch_size=16, cache_dir="../.cache")` |
| **`ValueError: Unknown course '...'`** | Typo in `course` slug | Check the available slugs printed in the error message, or pass `course=None` |
| **Offline run fails to download model** | First run needs the GGUF cached locally | Run once with internet (or pass `model_path="/path/to/model.gguf"`), then set `local_files_only=True` |
| **Changes to `src/llm_rag_faq/**/*.py` don't show up in notebook** | Python caches imported modules in the running kernel | Restart the kernel, or run `%load_ext autoreload` and `%autoreload 2` in Cell 1 |

---

## Related Docs

- [README](../README.md) — Project overview, CLI quickstart, and test/CI commands.
- [System Architecture](system-architecture.md) — Option B layered design, hybrid retrieval & local 1.5B strategies, and enforcement checks.
- [Code Review & Security Notes](code-review-notes.md) — Detailed before/after catalog of bugs, prompt-injection hardening, and boundary rules.
