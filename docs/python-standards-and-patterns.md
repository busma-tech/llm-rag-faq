# Python Coding Standards, Approaches & Patterns in `llm-rag-faq`

This guide explains the Python coding conventions, language features, and software engineering patterns used throughout `llm-rag-faq`. It is written for developers who are new to modern Python development and uses real examples directly from this repository.

---

## Table of Contents

1. [Project Layout & Packaging (`src/` Layout & `pyproject.toml`)](#1-project-layout--packaging-src-layout--pyprojecttoml)
2. [Modules, Imports & Exports](#2-modules-imports--exports)
3. [Naming Conventions & "Private" Members (Encapsulation)](#3-naming-conventions--private-members-encapsulation)
4. [Attributes, Immutability & Safe Defaults](#4-attributes-immutability--safe-defaults)
5. [Decorators](#5-decorators)
6. [Type Hints, Protocols & Duck Typing](#6-type-hints-protocols--duck-typing)
7. [Constraints & Validation (Static, Runtime & Architectural)](#7-constraints--validation-static-runtime--architectural)
8. [Resource Lifecycle, Ownership, Concurrency & Streaming](#8-resource-lifecycle-ownership-concurrency--streaming)
9. [Error Handling, Logging & Defensive Coding](#9-error-handling-logging--defensive-coding)
10. [Dependency Injection & Offline Testing Patterns](#10-dependency-injection--offline-testing-patterns)
11. [Docstrings & Automated Quality Tooling](#11-docstrings--automated-quality-tooling)

---

## 1. Project Layout & Packaging (`src/` Layout & `pyproject.toml`)

### The `src/` Layout
All installable package code lives under `src/llm_rag_faq/` rather than directly at the repository root.

```text
llm-rag-faq/
├── pyproject.toml              # Single source of truth for metadata, deps & tools
├── Makefile                    # Developer shortcuts (lint, typecheck, arch, test)
├── src/
│   └── llm_rag_faq/            # The actual Python package
│       ├── __init__.py         # Public facade
│       ├── config.py           # Pure constants
│       ├── cli.py              # CLI entrypoint
│       ├── data/
│       ├── retrieval/
│       ├── llm/
│       └── rag/
└── tests/                      # Mirrors src/llm_rag_faq/ structure
```

**Why `src/`?**  
In Python, the current working directory is often on `sys.path`. If `llm_rag_faq/` sat at the repo root, running tests could accidentally import your unbuilt local folder instead of the installed package. Putting it inside `src/` forces tests and scripts to import the properly configured package.

### `pyproject.toml` (PEP 621)
Modern Python projects consolidate metadata, dependencies, build settings, and tool configs into a single `pyproject.toml` file (replacing older `setup.py`, `setup.cfg`, and `requirements.txt`):

- **Python version constraint:** `requires-python = ">=3.11,<3.14"` declares which interpreter versions are supported.
- **CLI script entrypoint:**
  ```toml
  [project.scripts]
  llm-rag-faq = "llm_rag_faq.cli:main"
  ```
  This registers the `llm-rag-faq` terminal command so `uv run llm-rag-faq` calls the `main()` function inside `src/llm_rag_faq/cli.py`.
- **Dependency groups:** `[dependency-groups]` separates runtime dependencies from `dev` tools (`pytest`, `ruff`, `mypy`, `import-linter`) so production installs stay lean.
- **Lockfile (`uv.lock`):** Pins exact transitive versions for reproducible builds across machines and CI.

---

## 2. Modules, Imports & Exports

### Postponed Type Evaluation (`from __future__ import annotations`)
Notice that **every `.py` file** in `src/` and `tests/` starts with:

```python
from __future__ import annotations
```

By default, Python evaluates type annotations at import/definition time. If a class references itself in a method signature (like `def __enter__(self) -> AnthropicClient:` in `src/llm_rag_faq/llm/anthropic_client.py`), Python would raise a `NameError` because `AnthropicClient` isn't finished being defined yet. `from __future__ import annotations` (PEP 563) turns annotations into strings automatically, solving forward references and keeping imports fast.

### Import Grouping & Ordering (PEP 8 / Ruff `I`)
Imports are always placed at the top of a module and organized into three alphabetical blocks separated by blank lines (enforced automatically by `ruff`):

```python
# 1. Python standard library
import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# 2. Third-party packages
import numpy as np
from fastembed import TextEmbedding
from minsearch import Index

# 3. Local project imports
from llm_rag_faq.config import DEFAULT_RRF_K, DEFAULT_THREADS
```

### Absolute vs. Relative Imports
This project uses a clear convention for internal imports:

- **Absolute imports across layers:** When importing from another layer (e.g., `config` or `llm`), we use the full package path:
  ```python
  # src/llm_rag_faq/rag/pipeline.py
  from llm_rag_faq.config import DEFAULT_BOOST, NO_ANSWER_RESPONSE
  from llm_rag_faq.llm.base import LLMClient
  ```
- **Relative imports within the same subpackage:** When a subpackage imports from a sibling file in the same folder, it uses a single leading dot (`.`):
  ```python
  # src/llm_rag_faq/rag/pipeline.py
  from .prompts import INSTRUCTIONS, PROMPT_TEMPLATE, render_prompt

  # src/llm_rag_faq/data/__init__.py
  from .faq_loader import load_faq_data
  ```
- **Never** use multi-level relative imports like `from ..retrieval import ...` to jump across layers. Our AST test (`tests/test_architecture.py`) explicitly rejects escaping or cross-layer relative imports.

### Defining Exports with `__all__` & the Facade Pattern
In Python, any folder containing an `__init__.py` file is a **package**. By default, `from pkg import *` (and IDE auto-import suggestions / docs tools) would expose every name imported in that file—including helper modules or constants you didn't mean to share.

We explicitly control each package's public API using **`__all__`**:

```python
# src/llm_rag_faq/__init__.py
from llm_rag_faq.data import load_faq_data
from llm_rag_faq.llm import AnthropicClient, LocalGGUFClient
from llm_rag_faq.rag import RAGBase, RAGResult
from llm_rag_faq.retrieval import HybridFAQSearch, build_index

__all__ = [
    "AnthropicClient",
    "HybridFAQSearch",
    "LocalGGUFClient",
    "RAGBase",
    "RAGResult",
    "build_index",
    "load_faq_data",
]
```

This implements the **Facade Pattern**:
- Users (and notebooks) only need `from llm_rag_faq import RAGBase, build_index, load_faq_data` without knowing which internal submodule each class lives in.
- Each subpackage (`data/__init__.py`, `llm/__init__.py`, `rag/__init__.py`, `retrieval/__init__.py`) also defines its own `__all__` to re-export its clean public surface.
- `tests/test_architecture.py` asserts that `llm_rag_faq.__all__` is sorted, has no duplicates, and matches the exact 7 expected symbols.
- Internal modules (including `cli.py`) **never** import from the top-level facade `llm_rag_faq`, preventing circular import loops.

### The `if __name__ == "__main__":` Guard
At the bottom of `src/llm_rag_faq/cli.py`:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

When a file is imported by another module (or a test), `__name__` is `"llm_rag_faq.cli"`, so `main()` does **not** run automatically. When executed directly (`python -m llm_rag_faq.cli`), `__name__` is `"__main__"`, and `raise SystemExit(main())` exits the process with the integer exit code (`0` for success, `1` for error) returned by `main()`.

---

## 3. Naming Conventions & "Private" Members (Encapsulation)

Python follows **PEP 8** naming conventions:

| Style | Used For | Examples in This Repo |
|---|---|---|
| `snake_case` | Functions, methods, variables, modules | `load_faq_data`, `build_index`, `faq_loader.py` |
| `PascalCase` | Classes, Protocols, Dataclasses | `RAGBase`, `HybridFAQSearch`, `LLMClient`, `RAGResult` |
| `UPPER_SNAKE_CASE` | Module-level constants | `DOCS_URL`, `DEFAULT_RRF_K`, `NO_ANSWER_RESPONSE` |
| `_leading_underscore` | Private / internal helpers, attributes, constants | `_ensure_open`, `_rrf_fuse`, `self._closed`, `_EXIT_COMMANDS` |
| `__dunder__` | Python's built-in special ("magic") methods/names | `__init__`, `__enter__`, `__exit__`, `__all__`, `__name__` |

### How "Private" Methods, Functions & Attributes Work in Python
Unlike Java, C#, or TypeScript, Python has **no `private` or `protected` keyword** enforced by the compiler. Instead, Python follows the philosophy *"we are all consenting adults"* and uses a **single leading underscore (`_`)** as a strong convention meaning: *"this is an internal implementation detail—do not call or rely on it from outside this class or module."*

Examples in `llm-rag-faq`:
- **Private module-level functions:**
  - `_validate_course_slug()` and `_resolve_course_url()` in `src/llm_rag_faq/data/faq_loader.py`
  - `_l2_normalize()`, `_format_document()`, `_matches_filter()`, `_rrf_fuse()` in `src/llm_rag_faq/retrieval/hybrid.py`
  - `_is_idk_response()` and `_doc_fingerprint()` in `src/llm_rag_faq/rag/pipeline.py`
  - `_create_llm_client()`, `_print_result()`, `_run_repl()` in `src/llm_rag_faq/cli.py`
- **Private methods on a class:**
  - `_ensure_open()`, `_tokenize()`, `_build_messages()`, `_validate_generation_args()` in `LocalGGUFClient` (`src/llm_rag_faq/llm/local_gguf.py`)
  - `_build_doc_vectors()`, `_semantic_search()` in `HybridFAQSearch` (`src/llm_rag_faq/retrieval/hybrid.py`)
  - `_has_generate_api()`, `_is_local_client()`, `_validate_query()` in `RAGBase` (`src/llm_rag_faq/rag/pipeline.py`)
- **Private instance attributes:**
  - `self._closed`, `self._lock`, `self._owns_client` — internal state that callers should not mutate directly.
- **Private module constants:**
  - `_EXIT_COMMANDS` (`cli.py`), `_CONTROL_TOKENS_RE` & `_GLOB_CHARS` (`local_gguf.py`), `_PLACEHOLDER_RE` (`prompts.py`), `_IDK_RE` (`pipeline.py`).

> **Note on double leading underscores (`__name`):**  
> In Python, `__foo` inside a class triggers *name mangling* (`_ClassName__foo`), which is meant to avoid name collisions in deep inheritance hierarchies—not for general privacy. Because this project favors composition over deep inheritance, single `_` is used consistently everywhere.

### Dunder (`__double_underscore__`) Methods
Names surrounded by double underscores are reserved for Python's **Data Model** (hooks called automatically by Python syntax):
- `__init__`: Object initializer called when instantiating a class (`RAGBase(...)`).
- `__enter__` / `__exit__`: Context manager hooks called by `with ... as ...:`.
- `__all__`: Export list used by `from module import *` and tools.

Never invent your own custom `__dunder__` names; stick to the ones defined by Python.

---

## 4. Attributes, Immutability & Safe Defaults

### Instance Attributes in `__init__`
In Python, instance attributes are created by assigning to `self.<name>` inside `__init__`. Defining **all** attributes in `__init__` (even if initialized to `False` or `None`) is a best practice because:
1. Readers can see the complete shape of an object in one place.
2. Static type checkers (`mypy`) can verify every attribute type.

When a type can't be inferred cleanly from the right-hand side, we annotate the attribute directly:
```python
# src/llm_rag_faq/retrieval/hybrid.py
self.embedder: Embedder = (
    embedder
    if embedder is not None
    else TextEmbedding(model_name=model_name, threads=threads)
)
```

### Avoiding the Mutable Default Argument Trap
One of the most famous Python gotchas is using a mutable object (`dict` or `list`) as a default parameter value:

```python
# BAD — the same dict is shared across every call!
def search(self, query: str, boost_dict: dict[str, float] = DEFAULT_BOOST): ...
```

Instead, this codebase **always** defaults optional mutable arguments to `None` and makes a fresh defensive copy inside the function:

```python
# GOOD — src/llm_rag_faq/rag/pipeline.py
def __init__(self, ..., boost_dict: dict[str, float] | None = None) -> None:
    self.boost_dict = (
        dict(DEFAULT_BOOST) if boost_dict is None else dict(boost_dict)
    )
```

This also distinguishes between *"caller omitted `boost_dict` (`None` → use `DEFAULT_BOOST`)"* and *"caller explicitly passed `{}` (disable boosting)"*, as documented in `HybridFAQSearch.search()`.

### Immutability (`frozen=True` & `frozenset`)
Immutable objects are safer because they cannot be accidentally modified after creation:
- **`@dataclass(frozen=True)`:** Used for `RAGResult` (`src/llm_rag_faq/rag/pipeline.py`) and `ImportEdge` (`tests/test_architecture.py`). Attempting `result.answer = "other"` raises `FrozenInstanceError`.
- **`frozenset`:** Used for constant membership lookups like `_EXIT_COMMANDS = frozenset({"quit", "exit", ":q"})`, `_GLOB_CHARS = frozenset("*?[")`, and `FORBIDDEN_RAG_SDKS`. Unlike a regular `set`, a `frozenset` cannot be mutated at runtime and offers `O(1)` lookup.

### Defensive Attribute Introspection (`getattr`, `hasattr`, `callable`)
When interacting with duck-typed or third-party SDK objects where an attribute might not exist, we use safe built-ins instead of assuming it's there:

```python
# src/llm_rag_faq/cli.py
close_fn = getattr(llm_client, "close", None)
if callable(close_fn):
    close_fn()

# src/llm_rag_faq/llm/anthropic_client.py
content_blocks = getattr(response, "content", None) or []
```

---

## 5. Decorators

A **decorator** (`@decorator_name`) is syntactic sugar that wraps or transforms a function or class at definition time:

```python
@decorator
class MyClass:
    ...

# is equivalent to:
class MyClass:
    ...
MyClass = decorator(MyClass)
```

This project uses two standard-library class decorators:

### 1. `@dataclass(frozen=True)` (`dataclasses`)
Used on `RAGResult` (`src/llm_rag_faq/rag/pipeline.py`) and `ImportEdge` (`tests/test_architecture.py`):

```python
@dataclass(frozen=True)
class RAGResult:
    """Structured output returned by ``RAGBase.rag_with_sources``."""

    answer: str
    sources: list[dict[str, Any]]
    rewritten_query: str | None = None
    used_fallback: bool = False
```

Instead of writing boilerplate `__init__`, `__repr__`, and `__eq__` methods by hand, `@dataclass` generates them automatically from the type-annotated fields. Adding `frozen=True` makes instances read-only.

### 2. `@runtime_checkable` (`typing`)
Used on `LLMClient` (`src/llm_rag_faq/llm/base.py`) and `Embedder` (`src/llm_rag_faq/retrieval/hybrid.py`):

```python
@runtime_checkable
class LLMClient(Protocol):
    def generate(
        self,
        system: str | None,
        prompt: str,
        max_tokens: int = ...,
    ) -> str: ...
```

By default, `typing.Protocol` is only checked by static tools like `mypy` and cannot be used with `isinstance()` at runtime. Adding `@runtime_checkable` enables `isinstance(self.llm_client, LLMClient)` in `RAGBase` (`src/llm_rag_faq/rag/pipeline.py`), letting `RAGBase` cleanly distinguish `LLMClient` implementations from raw `anthropic.Anthropic` SDK clients at runtime.

---

## 6. Type Hints, Protocols & Duck Typing

### Modern Type Hint Syntax (Python 3.10+)
Thanks to Python 3.12 (and `from __future__ import annotations`), we use clean built-in syntax rather than legacy `typing.Optional` / `typing.List` / `typing.Dict`:

- **Union / Optional:** `str | None` instead of `Optional[str]`; `str | Path | None` instead of `Union[str, Path, None]`.
- **Built-in generics:** `list[dict[str, Any]]`, `dict[str, float]`, `tuple[list[ImportEdge], list[str]]`.
- **Abstract collection types for inputs (`collections.abc`):** Parameters accept `Sequence[str]`, `Iterable[str]`, or `Sequence[dict[str, Any]]` (which match lists, tuples, generators, etc.), while return types are concrete (`list[dict[str, Any]]`, `Iterator[str]`). This follows the principle: *"be liberal in what you accept, specific in what you return."*
- **NumPy typing:** `NDArray[np.float32]` from `numpy.typing` in `src/llm_rag_faq/retrieval/hybrid.py` documents array element types.

### `Protocol` (Structural Subtyping) vs. Inheritance
In languages like Java, a class must explicitly write `implements LLMClient`. In Python, `typing.Protocol` (PEP 544) supports **structural subtyping** (type-safe duck typing): any class that defines a matching `generate(system, prompt, max_tokens) -> str` method automatically satisfies `LLMClient`—no base-class inheritance required!

Notice the `...` (`Ellipsis`) in protocol methods:
```python
# src/llm_rag_faq/rag/pipeline.py
class SearchIndex(Protocol):
    def search(
        self,
        query: str,
        num_results: int = 5,
        boost_dict: dict[str, float] | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...
```
In a `Protocol`, `...` is the standard way to say *"this is a signature specification only; implementations provide the body."*

**Why this matters architecturally:**  
Because `rag/pipeline.py` defines its own minimal `SearchIndex` protocol, `rag` never has to import `retrieval/hybrid.py` (or `minsearch` / `fastembed`). Both `HybridFAQSearch` (in production) and `FakeSearchIndex` (in `tests/conftest.py`) satisfy `SearchIndex` without coupling to each other.

---

## 7. Constraints & Validation (Static, Runtime & Architectural)

Constraints in this project are enforced at **four distinct levels**:

### 1. Function Signature Constraints (Keyword-Only Arguments `*`)
In `src/llm_rag_faq/rag/pipeline.py`:
```python
def llm(
    self,
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int | None = None,
) -> str:
```
The bare `*` means every parameter after it (`system`, `max_tokens`) **must be passed by keyword** (`self.llm(prompt, max_tokens=64)`). Calling `self.llm(prompt, "sys", 64)` raises a `TypeError`. This prevents bugs from accidentally swapping positional arguments and makes call sites self-documenting.

### 2. Runtime Validation (Fail-Fast)
Python's type hints are **not** enforced at runtime by the interpreter. For public boundaries and security-critical inputs, we validate explicitly and fail fast with clear error messages:
- **Type checks (`TypeError`):** `if not isinstance(query, str): raise TypeError(...)` in `RAGBase._validate_query()` and `LocalGGUFClient._build_messages()`.
- **Value/bounds checks (`ValueError`):** Rejecting `max_tokens <= 0`, `n_ctx <= TEMPLATE_OVERHEAD_TOKENS`, empty queries, oversized queries (`max_query_chars`), wildcard GGUF filenames (`*?[`), and path-traversal slugs (`../`, `/`, `\`, `\x00`) in `_validate_course_slug()`.
- **Filesystem checks (`FileNotFoundError`):** Verifying `model_path` exists before invoking `llama.cpp`.

### 3. Static Type & Lint Constraints (`mypy` & `ruff`)
Configured in `pyproject.toml`:
- **`mypy`** (`make typecheck`) checks type safety across `src/`, with `warn_return_any = true` and `warn_unused_ignores = true`.
- **`ruff`** (`make lint`) enforces PEP 8 style (`E`), bug/unused-name detection (`F`), import sorting (`I`), modern Python syntax upgrades (`UP`), bugbear anti-patterns (`B`), simplifications (`SIM`), Bandit security checks (`S`), and Ruff-specific rules (`RUF`). Test-only relaxations (like allowing `assert` via `S101`) are scoped strictly to `"tests/**"`.

### 4. Architectural Import Constraints (`import-linter` & AST Tests)
Layer boundaries are not just documented—they are **machine-enforced**:
- **`import-linter`** in `pyproject.toml` enforces three contracts: *Strict layered architecture*, *Peer infrastructure layers remain independent*, and *RAG orchestration must not import raw infrastructure SDKs*.
- **`tests/test_architecture.py`** parses every `.py` file using Python's built-in `ast` (Abstract Syntax Tree) module to detect circular imports (3-color DFS), layer violations, SDK leaks, and public facade drift—with zero external dependencies.

---

## 8. Resource Lifecycle, Ownership, Concurrency & Streaming

### Context Managers (`with` Statement)
Native C/C++ memory (`llama.cpp`) and HTTP connection pools (`requests.Session`, `Anthropic`) should be released deterministically instead of waiting for Python's garbage collector.

`LocalGGUFClient`, `AnthropicClient`, and `RAGBase` all implement the **Context Manager protocol** (`__enter__`, `__exit__`, and `close()`):

```python
with RAGBase(index=index, llm_client=None, course="llm-zoomcamp") as assistant:
    result = assistant.rag_with_sources("Where do I ask live workshop questions?")
# assistant.close() is guaranteed to run here, even if an exception was raised!
```

### Resource Ownership Pattern (`_owns_client` / `owns_session`)
A subtle bug in many codebases is closing a resource that the caller passed in and still wants to reuse. This project consistently tracks **ownership**:

```python
# src/llm_rag_faq/data/faq_loader.py
owns_session = session is None
http = session if session is not None else requests.Session()
try:
    ...
finally:
    if owns_session:
        http.close()
```

The same pattern appears in `AnthropicClient` (`self._owns_client = client is None`) and `RAGBase` (`self._owns_client = llm_client is None`): **if you created it, you close it; if it was injected, the caller owns its lifecycle.**

### Thread Safety (`threading.Lock`)
`llama.cpp` contexts are not safe for concurrent access from multiple threads. `LocalGGUFClient` guards all native calls (`generate`, `generate_stream`, `close`) with a `threading.Lock`:

```python
with self._lock:
    self._ensure_open()
    ...
```

### Lazy Streaming Generators with Eager Validation
In Python, any function containing `yield` becomes a **generator function**—calling it does *not* execute its body until the caller starts iterating (`for token in stream:`). That means validation errors inside a generator would be delayed until iteration begins!

`LocalGGUFClient.generate_stream()` (`src/llm_rag_faq/llm/local_gguf.py`) solves this with the **outer-validation / inner-generator pattern**:
1. The outer method `generate_stream()` validates `max_tokens`, `temperature`, `prompt`, `system`, and `_ensure_open()` **immediately**.
2. Only after validation passes does it define and return the inner generator `_stream_tokens()`, which holds `self._lock` and `yield`s text chunks lazily as `llama.cpp` produces them.

---

## 9. Error Handling, Logging & Defensive Coding

### Specific Exception Handling
Never use a bare `except:` (which catches `KeyboardInterrupt` and `SystemExit`). Catch only the specific exceptions you know how to handle, e.g.:
- In `src/llm_rag_faq/cli.py`: `except (EOFError, KeyboardInterrupt):` to exit the REPL cleanly, and `except (ValueError, KeyError, FileNotFoundError, RuntimeError, OSError) as exc:` to print a friendly error to `sys.stderr` and return exit code `1`.

### Standard Logging (`logging.getLogger(__name__)`)
Library modules should **never** call `print()` for diagnostics (only `cli.py` prints to stdout/stderr). Instead, `src/llm_rag_faq/llm/local_gguf.py` creates a module-scoped logger:

```python
logger = logging.getLogger(__name__)
...
logger.warning(
    "Truncating prompt from %d to %d tokens to fit within n_ctx=%d.",
    len(prompt_tokens),
    prompt_budget,
    self.n_ctx,
)
```

Using `%d` placeholders instead of f-strings (`f"..."`) is a Python logging best practice: the string is only formatted if the `WARNING` log level is actually enabled.

### Security & Defensive Habits Used Here
- **`pathlib.Path` over string concatenation:** Cleaner cross-platform paths (`cache_path / f"faq_{slug}.json"`) plus `.resolve().parent` verification to block path traversal (`faq_loader.py`).
- **Disable pickle on untrusted/cached NumPy files:** `np.load(cache_file, allow_pickle=False)` in `hybrid.py` prevents arbitrary code execution via malicious `.npy` files.
- **Precompiled regexes:** `re.compile(...)` at module level (`_CONTROL_TOKENS_RE`, `_THINK_PREFIX_RE`, `_PLACEHOLDER_RE`, `_IDK_RE`) avoids recompiling patterns on every call.
- **Single-pass template substitution:** `render_prompt()` in `prompts.py` uses `_PLACEHOLDER_RE.sub(...)` instead of `template.format(...)` or chained `.replace()` so `{question}` appearing inside retrieved FAQ context can never be expanded a second time.

---

## 10. Dependency Injection & Offline Testing Patterns

### Constructor Dependency Injection
Rather than hardcoding external services inside classes, every infrastructure-touching class accepts an optional pre-built dependency in `__init__`:

| Component | Optional Injected Dependency | Default When `None` |
|---|---|---|
| `load_faq_data` | `session: requests.Session \| None` | Creates and closes `requests.Session()` |
| `HybridFAQSearch` | `embedder: Embedder \| None` | Creates `fastembed.TextEmbedding(...)` |
| `LocalGGUFClient` | `llm: Llama \| None` | Downloads/loads GGUF via `llama_cpp.Llama` |
| `AnthropicClient` | `client: Any = None` | Creates `anthropic.Anthropic(...)` |
| `RAGBase` | `index: SearchIndex`, `llm_client: Any` | Creates `LocalGGUFClient()` when `llm_client=None` |

### Lightweight Test Doubles (Fakes)
Because of Dependency Injection and Protocols, `tests/conftest.py` provides deterministic, zero-network **Fakes** (`FakeEmbedder`, `FakeLlama`, `FakeAnthropic`, `FakeSearchIndex`) that record all calls for assertions and use `types.SimpleNamespace` to build lightweight mock response objects without pulling in heavy mocking frameworks.

### Dual `unittest` + `pytest` Compatibility
Tests are written using Python's built-in `unittest.TestCase` (with `self.subTest(...)` for parameterized cases). This means:
- All 100 tests run out of the box with zero dev dependencies via `python -m unittest discover -s tests -v`.
- They also run seamlessly under `pytest` (`uv run pytest -v`) in CI and `make test`.

---

## 11. Docstrings & Automated Quality Tooling

### Google-Style Docstrings
Public modules, classes, and functions use **Google-style docstrings** with clear sections (`Args:`, `Returns:`, `Raises:`, `Attributes:`) and double backticks (`` ``symbol`` ``) for code identifiers. Even small private helpers include a concise one-line docstring describing *what* they do.

### Automated Verification (`Makefile`)
Rather than relying on manual code reviews to catch style, typing, or architectural drift, `Makefile` wires every check into a single command:

```bash
make check   # Runs all four checks below in order:
make lint      # ruff check + ruff format --check
make typecheck # mypy src
make arch      # import-linter + AST architecture tests
make test      # pytest -v
```

GitHub Actions (`.github/workflows/ci.yml`) runs `make check` automatically on every push and pull request so these standards remain enforced over time.
