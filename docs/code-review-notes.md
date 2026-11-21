# Code Review & Security Notes — `llm-rag-faq`

A study and revision guide documenting all bugs, security vulnerabilities, and architectural improvements identified and fixed across the Option B layered package (`src/llm_rag_faq/`):

- `src/llm_rag_faq/config.py` — centralized constants and defaults
- `src/llm_rag_faq/data/faq_loader.py` — FAQ HTTP ingestion and JSON disk cache
- `src/llm_rag_faq/retrieval/hybrid.py` — `HybridFAQSearch`, `build_index`, RRF fusion, and `.npy` cache
- `src/llm_rag_faq/llm/` (`base.py`, `local_gguf.py`, `anthropic_client.py`) — `LLMClient` protocol, hardened `LocalGGUFClient`, and `AnthropicClient`
- `src/llm_rag_faq/rag/` (`prompts.py`, `pipeline.py`) — prompt escaping/rendering and `RAGBase` orchestration
- `src/llm_rag_faq/cli.py` and `src/llm_rag_faq/__init__.py` — CLI entrypoint and 7-symbol public facade

---

## Table of Contents

1. [Data Ingestion and Hybrid Search](#1-data-ingestion-and-hybrid-search)
2. [In-Process Local GGUF and Provider Abstraction](#2-in-process-local-gguf-and-provider-abstraction)
3. [RAG Orchestration and Prompt Hardening](#3-rag-orchestration-and-prompt-hardening)
4. [Option B Layering, Architecture Guard, and Quality Tooling](#4-option-b-layering-architecture-guard-and-quality-tooling)
5. [Quick-Reference Engineering Checklist](#5-quick-reference-engineering-checklist)

---

## 1. Data Ingestion and Hybrid Search

### 1.1 `build_index()` Silently Dropped Non-Default Courses

- **Category:** Bug (Logic / API Design)
- **Location:** `src/llm_rag_faq/retrieval/hybrid.py`
- **Root Cause:** `build_index(documents)` originally forwarded only `documents` to `HybridFAQSearch(documents)`, which defaulted to `course="llm-zoomcamp"`. Loading any other course and passing it to `build_index()` filtered `self.documents` down to `[]`.

**Before:**
```python
docs = load_faq_data(course="mlops-zoomcamp")  # 249 docs
index = build_index(docs)
# HybridFAQSearch.__init__ filtered to d.get("course") == "llm-zoomcamp" -> 0 docs!
```

**After:**
```python
def build_index(
    documents: Sequence[dict[str, Any]],
    course: str | None = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    **kwargs: Any,
) -> HybridFAQSearch:
    return HybridFAQSearch(documents, course=course, model_name=model_name, **kwargs)
```

- **Takeaway:** Filter once at the boundary (or default downstream builders to `course=None`) and forward configuration kwargs rather than hiding conflicting defaults behind thin wrappers.

---

### 1.2 Empty Corpus Crashed `search()` with NumPy Dimension Mismatch

- **Category:** Bug (Crash)
- **Location:** `src/llm_rag_faq/retrieval/hybrid.py`
- **Root Cause:** When `self.documents` was empty, `np.array(list(embedder.embed([])))` created a 1D array of shape `(0,)` instead of `(0, dim)`. Calling `self.doc_vectors @ q_vec` in `search()` raised `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0`.

**Before:**
```python
self.doc_vectors = np.array(list(self.embedder.embed(texts, batch_size=32)))
# Later in search():
scores = self.doc_vectors @ q_vec  # ValueError when self.documents is []
```

**After:**
```python
# In _build_doc_vectors():
if not self.documents:
    return np.empty((0, 0), dtype=np.float32)

# In search():
if num_results == 0 or not self.documents:
    return []
```

- **Takeaway:** Always guard empty-collection edge cases before matrix algebra.

---

### 1.3 `load_faq_data(course=None)` Returned `[]` and Lacked Path-Traversal Validation

- **Category:** Bug (Inconsistent Contract) + Security (Path Traversal)
- **Location:** `src/llm_rag_faq/data/faq_loader.py`
- **Root Cause:** `if c.get("course") != course: continue` skipped every entry when `course=None`, silently returned `[]` on typos/unknown course names, and wrote `cache_dir / f"faq_{course}.json"` without validating `course` against path traversal characters (`/`, `\`, `..`, `\x00`).

**Before:**
```python
for c in courses_raw:
    if c.get("course") != course:
        continue
```

**After:**
- `_validate_course_slug(course)` rejects `/`, `\`, `..`, and `\x00` with `ValueError`, and `_cache_path()` verifies `cache_file.resolve().parent == cache_root` to defend against symlink traversal.
- `course=None` loads all courses; unknown slugs fail loudly with available courses:
```python
if course is None:
    selected = courses_raw
else:
    selected = [c for c in courses_raw if c.get("course") == course]
    if not selected:
        available = sorted(c["course"] for c in courses_raw if "course" in c)
        raise ValueError(f"Unknown course {course!r}. Available courses: {available}")
```

- **Takeaway:** Distinguish `None` ("no filter / all") from an unmatched value, fail loudly on typos, and sanitize user-supplied identifiers before building filesystem paths.

---

### 1.4 Hardcoded Candidate Pool Cap (`10`) Truncated `num_results > 10`

- **Category:** Bug (Silent Truncation)
- **Location:** `src/llm_rag_faq/retrieval/hybrid.py`
- **Root Cause:** Lexical and semantic retrievers each fetched at most 10 candidates regardless of `num_results`. When both retrievers overlapped, `search(q, num_results=15)` could return as few as 10 results.

**Before:**
```python
lex_hits = self.lexical_index.search(query, num_results=10, ...)
sem_ranked = sorted(valid_indices, key=lambda i: scores[i], reverse=True)[:10]
```

**After:**
```python
k_candidates = (
    candidate_k
    if candidate_k is not None
    else max(num_results * 2, DEFAULT_MIN_CANDIDATES)
)
```

- **Takeaway:** First-stage candidate pool sizes in hybrid/reranked search must scale with the requested `num_results`.

---

### 1.5 Inconsistent `filter_dict` Behavior Between Lexical and Semantic Search

- **Category:** Bug (Data Leakage / Retrieval Mismatch)
- **Location:** `src/llm_rag_faq/retrieval/hybrid.py`
- **Root Cause:** `minsearch` supports multi-field filters, list-valued filters (`{"course": ["a", "b"]}`), and exact matching on `""` and `None`. Semantic search only checked `if not course or d.get("course") == course`, which:
  1. Ignored all non-`course` filter keys (leaking unfiltered docs into RRF).
  2. Returned 0 semantic hits for list-valued `course` filters.
  3. Treated `{"course": ""}` as unfiltered (`not ""` is `True`).

**Before:**
```python
course = filter_dict.get("course")
valid_indices = [
    i for i, d in enumerate(self.documents)
    if not course or d.get("course") == course
]
```

**After:**
```python
def _matches_filter(doc: dict[str, Any], filter_dict: dict[str, Any]) -> bool:
    for field, expected in filter_dict.items():
        value = doc.get(field)
        if isinstance(expected, list | tuple | set):
            if value not in expected:
                return False
        elif value != expected:
            return False
    return True
```

- **Takeaway:** All retrievers in a hybrid ensemble must enforce identical filter semantics, or unfiltered results from one retriever will leak through rank fusion.

---

### 1.6 Falsy `boost_dict={}` Overridden by Default

- **Category:** Bug (Python Truthiness Pitfall)
- **Location:** `src/llm_rag_faq/retrieval/hybrid.py`
- **Root Cause:** `boost_dict = boost_dict or {"question": 3.0, "section": 0.5}` treated an explicit empty dict `{}` (meaning "disable boosting") as falsy and replaced it with the default.

**Before:**
```python
boost_dict = boost_dict or {"question": 3.0, "section": 0.5}
```

**After:**
```python
effective_boost = dict(DEFAULT_BOOST) if boost_dict is None else boost_dict
```

- **Takeaway:** Always use `is None`—not `or`—for optional parameters where falsy values (`{}`, `[]`, `0`, `""`, `False`) are meaningful.

---

### 1.7 `KeyError` on Missing Fields & Cross-Course `doc["id"]` Collisions

- **Category:** Bug (Robustness & Data Integrity)
- **Location:** `src/llm_rag_faq/retrieval/hybrid.py`
- **Root Cause:** Indexing `d["section"]`, `d["question"]`, `d["answer"]`, and `doc["id"]` directly raised `KeyError` on incomplete documents. Keying RRF by `doc["id"]` also risked collisions when indexing multiple courses.

**Before:**
```python
texts = [f"{d['section']} | {d['question']} | {d['answer']}" for d in self.documents]
# ...
doc_id = doc["id"]
doc_map[doc_id] = doc
```

**After:**
- Formatted safely with `.get(field) or ""` in `_format_document()`.
- Used `minsearch`'s `output_ids=True` (integer index into `self.documents`) as the canonical RRF key, returning the original unmodified `[self.documents[idx] for idx in top_indices]`.

- **Takeaway:** Use internal positional indices for rank fusion rather than assuming external document IDs exist and are globally unique.

---

### 1.8 Missing HTTP Timeouts, External List Aliasing, & Other Improvements

| Issue | Before | After |
|---|---|---|
| **Hanging network calls** | `requests.get(url)` (no timeout, new connection per call) | `requests.Session` + `timeout=DEFAULT_TIMEOUT` (`30.0`) + safe `urljoin(URL_PREFIX, path.lstrip("/"))` |
| **External list mutation** | `self.documents = documents` (shared reference) | `self.documents = list(documents)` (shallow copy) |
| **Negative `num_results`** | `top_ids[:-1]` returned all-but-last docs | Raises `ValueError` for `num_results < 0` |
| **Unnormalized dot product** | `self.doc_vectors @ q_vec` assumed unit vectors | Explicit `_l2_normalize()` on both doc and query vectors |
| **Asymmetric embeddings** | `embedder.embed(...)` for both docs and queries | `embedder.passage_embed(...)` and `embedder.query_embed(...)` |
| **0-based RRF rank** | `1.0 / (60 + rank)` with `rank` starting at `0` | Standard 1-based `enumerate(ranked, start=1)` in `_rrf_fuse()` |
| **Sorting whole corpus in Python** | `sorted(valid_indices, ...)` ($O(N \log N)$) | Vectorized NumPy mask + `np.argpartition` |
| **Repeated download / embedding** | Re-fetched and re-embedded on every notebook run | Optional `cache_dir` for JSON and content-hashed `.npy` embeddings (`allow_pickle=False`) |

---

## 2. In-Process Local GGUF and Provider Abstraction

### 2.1 Prompt Injection Amplified by Merging `system` into `user`

- **Category:** Security Vulnerability (Prompt Injection)
- **Location:** `src/llm_rag_faq/llm/local_gguf.py`
- **Root Cause:** Merging `system` and `prompt` into a single `{"role": "user", "content": ...}` stripped the role-level boundary that `Qwen2.5-1.5B-Instruct` (ChatML) is trained to respect, and left raw control tokens (`<|im_end|>`, `<|im_start|>`) unstripped in untrusted input.

**Before:**
```python
combined_prompt = f"{system.strip()}\n\n{prompt.strip()}"
response = self.llm.create_chat_completion(
    messages=[{"role": "user", "content": combined_prompt}],
    ...
)
```

**After:**
```python
prompt_clean = _sanitize_control_tokens(prompt.strip())
messages = []
if system_clean:
    messages.append({"role": "system", "content": system_clean})
messages.append({"role": "user", "content": prompt_clean})
```

- **Takeaway:** Preserve separate `system` and `user` roles whenever the model's chat template supports them, and strip model control tokens from untrusted text.

---

### 2.2 Unpinned Remote Model Download & Wildcard Glob Fragility

- **Category:** Security Vulnerability (Supply Chain) + Reliability Bug (Offline Failure)
- **Location:** `src/llm_rag_faq/llm/local_gguf.py`, `src/llm_rag_faq/config.py`
- **Root Cause:** `Llama.from_pretrained(repo_id=..., filename="*q4_k_m.gguf")`:
  1. Pulled from the mutable `main` branch with no pinned commit `revision` (supply-chain risk: arbitrary GGUF + embedded Jinja2 chat template).
  2. In `llama-cpp-python`, always called `HfFileSystem().ls(repo_id, recursive=True)` over the network—breaking offline execution even when the file was cached.
  3. Crashed with `ValueError("Multiple files found...")` if a second file matched the glob.
  4. Emitted the deprecated `local_dir_use_symlinks` warning.

**Before:**
```python
self.llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    filename="*q4_k_m.gguf",
    ...
)
```

**After:**
```python
resolved_path = hf_hub_download(
    repo_id=repo_id,               # DEFAULT_LOCAL_REPO_ID
    filename=filename,             # DEFAULT_LOCAL_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    revision=revision,             # DEFAULT_LOCAL_REVISION = "91cad51170dc346986eccefdc2dd33a9da36ead9"
    local_files_only=local_files_only,
)
self.llm = Llama(model_path=resolved_path, ...)
```
Wildcard filenames (`*`, `?`, `[`) are rejected with `ValueError` in `_resolve_model_path()`, and a direct local `model_path` is also supported.

- **Takeaway:** Always pin exact filenames and commit SHAs for downloaded ML artifacts so runs are reproducible, offline-capable, and protected from upstream repo changes.

---

### 2.3 Context-Window Overflow (`n_ctx=2048`) Crashed Generation

- **Category:** Bug (Runtime Crash) + DoS Risk
- **Location:** `src/llm_rag_faq/llm/local_gguf.py`, `src/llm_rag_faq/config.py`
- **Root Cause:** No token count check existed before `create_chat_completion()`. Long FAQ answers easily pushed `system + prompt + max_tokens` past `2048`, raising `ValueError` inside `llama.cpp`.

**Before:**
```python
# No token budgeting; crashes if prompt + 512 > 2048 tokens
response = self.llm.create_chat_completion(messages=..., max_tokens=max_tokens)
```

**After:**
- Raised `DEFAULT_N_CTX` to `4096` (`TEMPLATE_OVERHEAD_TOKENS = 64`).
- Added `_build_messages()` token budgeting using `self.llm.tokenize(...)` / `detokenize(...)` against `n_ctx - max_tokens - TEMPLATE_OVERHEAD_TOKENS`, truncating oversized prompts cleanly with a warning (or raising `ValueError` when `truncate_prompt=False`).

- **Takeaway:** Never pass unbounded context into a fixed-context local model without measuring tokens first.

---

### 2.4 `<think>` Tag Stripping: ReDoS, Truncation Leaks, & False Stripping

- **Category:** Security (ReDoS) + Bug (Data Loss / Leakage)
- **Location:** `src/llm_rag_faq/llm/local_gguf.py`
- **Root Cause:** `re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)`:
  1. Degraded to $O(n^2)$ backtracking on repeated unclosed `<think>` tags.
  2. Leaked raw unfinished reasoning if generation hit `max_tokens` before emitting `</think>`.
  3. Silently deleted legitimate mentions of `<think>...</think>` inside regular answers (e.g., an LLM course FAQ!).

**Before:**
```python
text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
```

**After:**
```python
# Anchored at start, handles truncated thoughts via (?:</think>|\Z), and opt-in (strip_thinking=False)
_THINK_PREFIX_RE = re.compile(r"^\s*<think>.*?(?:</think>|\Z)\s*", flags=re.DOTALL)
```

- **Takeaway:** Anchor post-processing regexes, account for `max_tokens` truncation, and don't unconditionally strip tags for non-reasoning models.

---

### 2.5 Native Memory Leaks, Thread Safety, & Null Content

| Issue | Before | After |
|---|---|---|
| **Native C++ memory leak** | No `close()` or context manager; re-running notebook cells stacked ~1 GB models in RAM | Implemented `close()`, `__enter__`, `__exit__`, and post-close `RuntimeError` guard |
| **Unsafe concurrent access** | Unsynchronized calls to non-thread-safe `llama.cpp` context | Wrapped `create_chat_completion` with `threading.Lock` |
| **`None` message content** | `re.sub(..., None)` raised `TypeError` | Safely extracts `(first.get("message") or {}).get("content") or ""` |
| **Greedy repetition loops** | `temperature=0.0` with no `repeat_penalty` | Configurable `DEFAULT_REPEAT_PENALTY = 1.1`, `DEFAULT_SEED = 42`, `_default_threads()` (`min(os.cpu_count() or 2, 4)`), `stop`, and `generate_stream()` |
| **Empty/`None` `system`** | `f"{system.strip()}\n\n..."` crashed on `None` or added leading blank lines | Normalizes `(system or "").strip()` and omits `system` message when empty |

---

## 3. RAG Orchestration and Prompt Hardening

### 3.1 System-Prompt Conflict Broke `rewrite_query()`

- **Category:** Bug (Critical Logic Defect)
- **Location:** `src/llm_rag_faq/rag/prompts.py`, `src/llm_rag_faq/rag/pipeline.py`
- **Root Cause:** `rewrite_query()` called `self.llm(rewrite_prompt)`, which injected the QA `self.instructions` (*"Answer using only facts from CONTEXT... Only say 'I don't know.' if the context does not contain information"*). Because `rewrite_prompt` had no `CONTEXT`, compliant models replied `"I don't know."` instead of search keywords.

**Before:**
```python
def rewrite_query(self, query: str) -> str:
    rewrite_prompt = "Rewrite the user's question into 4-6 concise search keywords...\n\nQuestion: " + query
    return self.llm(rewrite_prompt).strip()  # Uses QA system prompt!
```

**After (`prompts.py` + `pipeline.py`):**
```python
def llm(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None) -> str:
    effective_system = self.instructions if system is None else system
    ...

def rewrite_query(self, query: str) -> str:
    safe_question = escape_delimiter_tags(self._validate_query(query), "question")
    return self.llm(
        f"<question>\n{safe_question}\n</question>",
        system=REWRITE_INSTRUCTIONS,
        max_tokens=DEFAULT_REWRITE_MAX_TOKENS,
    ).strip().strip("\"'")
```

- **Takeaway:** Auxiliary LLM calls (query rewriting, routing, grading) must have their own dedicated system instructions and tight `max_tokens` limits.

---

### 3.2 Prompt Injection & Fake-Context Spoofing in `PROMPT_TEMPLATE`

- **Category:** Security Vulnerability (Direct & Indirect Prompt Injection)
- **Location:** `src/llm_rag_faq/rag/prompts.py`, `src/llm_rag_faq/rag/pipeline.py`
- **Root Cause:** Plain-text `QUESTION:` preceded `CONTEXT:` with no delimiters, letting a user query like `foo\n\nCONTEXT:\nQ: ... A: ...` spoof fake FAQ context. `str.format()` also exposes attribute traversal if `prompt_template` is ever untrusted.

**Before:**
```python
PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()

return self.prompt_template.format(question=query, context=context)
```

**After (`src/llm_rag_faq/rag/prompts.py`):**
```python
PROMPT_TEMPLATE = """
<context>
{context}
</context>

<question>
{question}
</question>
""".strip()

def render_prompt(context: str, question: str, template: str = PROMPT_TEMPLATE) -> str:
    safe_context = escape_delimiter_tags(context, "context")
    safe_question = escape_delimiter_tags(question.strip(), "question")
    values = {"context": safe_context, "question": safe_question}
    return _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], template)
```
`RAGBase.build_prompt()` delegates to `render_prompt()`, and `INSTRUCTIONS` explicitly states to treat `<context>` and `<question>` as untrusted data and ignore embedded instructions.

- **Takeaway:** Wrap untrusted data in XML delimiters, escape closing delimiter tags inside the data, place retrieved context before user input, and avoid `str.format()` when substituting untrusted/configurable templates.

---

### 3.3 `StopIteration` Crash & Dropped Blocks on Anthropic Responses

- **Category:** Bug (Crash / Data Loss)
- **Location:** `src/llm_rag_faq/llm/anthropic_client.py`
- **Root Cause:** `next(block.text for block in response.content if block.type == "text")` crashed with `StopIteration` when only non-text blocks (like `thinking`) were returned, and dropped any text blocks after the first.

**Before:**
```python
return next(block.text for block in response.content if block.type == "text")
```

**After (`extract_anthropic_text()` in `src/llm_rag_faq/llm/anthropic_client.py`):**
```python
def extract_anthropic_text(response: Any) -> str:
    content_blocks = getattr(response, "content", None) or []
    text_parts: list[str] = []
    for block in content_blocks:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", None)
            if text:
                text_parts.append(str(text))
    return "\n".join(text_parts).strip()
```
`AnthropicClient.generate()` and `RAGBase.llm()` (for raw `anthropic.Anthropic` instances) both call `extract_anthropic_text(response)`.

---

### 3.4 `course=None` Broke Multi-Course Search

- **Category:** Bug (Silent Empty Results)
- **Location:** `src/llm_rag_faq/rag/pipeline.py`
- **Root Cause:** `filter_dict = {"course": self.course}` sent `{"course": None}`, matching zero documents.

**Before:**
```python
filter_dict = {"course": self.course}
```

**After:**
```python
filter_dict = {"course": self.course} if self.course is not None else {}
```

---

### 3.5 Redundant LLM Calls, Brittle `"I don't know"` Check, & Cost Amplification

- **Category:** Bug (Wasted Latency/Cost) + Security (DoS/Cost Amplification)
- **Location:** `src/llm_rag_faq/rag/pipeline.py`
- **Root Cause:**
  1. Empty `search_results` still invoked the LLM with empty context.
  2. Fallback retry still invoked a 3rd LLM call even when `alt_results` was empty or identical to `search_results`.
  3. `answer.strip().lower().startswith("i don't know")` missed `"I do not know."` / `"Sorry, I don't know."` and false-positived on `"I don't know the date, but..."`.
  4. No validation on empty or huge queries (`DEFAULT_MAX_QUERY_CHARS = 2000`).

**After:**
- `_validate_query()` rejects empty or oversized queries (`> max_query_chars`).
- Skips the 1st LLM call when initial `search_results` is empty.
- Uses `_is_idk_response()` anchored regex for standalone refusal phrases.
- Uses `_doc_fingerprint()` to skip the 3rd LLM call if `alt_results` adds no new documents.
- Added `enable_fallback: bool = True` toggle and `rag_with_sources()` returning a frozen `RAGResult(answer, sources, rewritten_query=None, used_fallback=False)` dataclass.

---

### 3.6 Missing Fields, Falsy Client Check, & Protocol-Based Dispatch

| Issue | Before | After |
|---|---|---|
| **Missing/`None` doc fields** | `doc["section"]`, `"Q: " + doc["question"]` raised `KeyError`/`TypeError` | `str(doc.get(field) or "")` |
| **Falsy `llm_client` check** | `llm_client or LocalGGUFClient()` | `LocalGGUFClient() if llm_client is None else llm_client` |
| **Tight coupling to concrete class** | `isinstance(self.llm_client, LocalGGUFClient)` | `isinstance(self.llm_client, LLMClient)` protocol check + `not isinstance(self.llm_client, AnthropicClient)` for local vs. cloud defaults; `SearchIndex` protocol for retrieval |
| **Resource ownership** | No `close()` for auto-created `LocalGGUFClient` | `RAGBase` tracks `self._owns_client` and closes owned clients in `close()` / `__exit__`; `AnthropicClient` also implements `close()`, `__enter__`, `__exit__`, and `_owns_client` |

---

## 4. Option B Layering, Architecture Guard, and Quality Tooling

Splitting the original monolithic modules into the Option B layered package prevents accidental cross-layer coupling as the codebase grows:

- **Zero circular imports & strict one-way dependencies:** `config` has no internal imports; `data`, `retrieval`, and `llm` depend only on `config`; `rag` depends only on `config` and `llm` (interacting with retrieval via the `SearchIndex` protocol); `cli` wires all layers together without importing `__init__`.
- **Infrastructure SDK isolation:** `rag/` is forbidden from directly importing `requests`, `minsearch`, `fastembed`, `llama_cpp`, `anthropic`, or `huggingface_hub`.
- **Dual enforcement:**
  - **Offline stdlib AST check (`tests/test_architecture.py`):** Runs under `python -m unittest discover -s tests -v` with zero dev dependencies installed (verifies DFS acyclicity, `ALLOWED_LAYER_IMPORTS`, `FORBIDDEN_RAG_SDKS`, `llm_rag_faq.__all__`, and synthetic violation detection).
  - **CI enforcement (`make check`):** Runs `ruff check` & `ruff format --check`, `mypy src`, `lint-imports` (`[tool.importlinter]` in [pyproject.toml](../pyproject.toml)), and `pytest -v`.

---

## 5. Quick-Reference Engineering Checklist

Use this checklist when building or reviewing future RAG pipelines:

- [ ] **Filter consistency:** Do lexical, semantic, and wrapper layers handle `None`, lists, empty strings, and multi-field filters identically?
- [ ] **Empty & boundary inputs:** What happens on `documents=[]`, `search_results=[]`, `query=""`, `num_results=0/-1/50`, or `prompt > n_ctx`?
- [ ] **Python truthiness:** Are optional defaults checked with `is None` instead of `or`?
- [ ] **Prompt boundaries:** Are system instructions separated from user/context data, wrapped in escaped delimiters, and given dedicated prompts for auxiliary tasks like query rewriting?
- [ ] **Artifact & network hygiene:** Are HTTP calls using `Session` + `timeout`? Are model downloads pinned by exact filename + commit SHA and offline-capable?
- [ ] **Resource lifecycle:** Do heavy in-process models (`llama.cpp`, ONNX) expose `close()` / context managers, locks for thread safety, and disk caching for notebooks?
- [ ] **Observability & cost control:** Can you inspect retrieved sources and fallback behavior (`rag_with_sources`), and avoid redundant LLM calls when retrieval yields nothing new?
- [ ] **Layer boundaries & CI:** Do `tests/test_architecture.py` and `make check` pass with zero circular imports, strict layer boundaries, and isolated SDKs?

---

## Related Docs

- [README](../README.md) — Project overview, CLI quickstart, and test/CI commands.
- [System Architecture](system-architecture.md) — Option B layered design, hybrid retrieval & local 1.5B strategies, and enforcement checks.
- [Jupyter Notebook Usage Guide](notebook-usage-guide.md) — Ready-to-run notebook cells, shared `../.cache` configuration, streaming, and OOM troubleshooting.
