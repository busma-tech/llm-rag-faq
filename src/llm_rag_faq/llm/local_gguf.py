"""In-process local GGUF LLM client backed by ``llama-cpp-python``."""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from llm_rag_faq.config import (
    DEFAULT_LOCAL_FILENAME,
    DEFAULT_LOCAL_MAX_TOKENS,
    DEFAULT_LOCAL_REPO_ID,
    DEFAULT_LOCAL_REVISION,
    DEFAULT_N_BATCH,
    DEFAULT_N_CTX,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    TEMPLATE_OVERHEAD_TOKENS,
)

logger = logging.getLogger(__name__)

# Common chat-template control tokens stripped from untrusted user/context input
_CONTROL_TOKENS_RE = re.compile(
    r"<\|(?:im_start|im_end|endoftext|system|user|assistant|"
    r"eot_id|start_header_id|end_header_id)\|>"
)

# Anchored at the start of output to prevent quadratic backtracking and avoid
# deleting literal "<think>" mentions inside regular answers.
_THINK_PREFIX_RE = re.compile(r"^\s*<think>.*?(?:</think>|\Z)\s*", flags=re.DOTALL)

_GLOB_CHARS = frozenset("*?[")


def _default_threads() -> int:
    """Choose a conservative CPU thread count for local inference."""
    return min(os.cpu_count() or 2, 4)


def _sanitize_control_tokens(text: str) -> str:
    """Remove special chat-template control tokens from untrusted text."""
    return _CONTROL_TOKENS_RE.sub("", text)


def _strip_thinking_prefix(text: str) -> str:
    """Strip a leading ``<think>...</think>`` (or truncated ``<think>...``) block."""
    return _THINK_PREFIX_RE.sub("", text).strip()


def _resolve_model_path(
    repo_id: str,
    filename: str,
    revision: str | None,
    model_path: str | Path | None,
    cache_dir: str | Path | None,
    local_files_only: bool,
) -> str:
    """Resolve a local GGUF file path or download a pinned file from Hugging Face."""
    if model_path is not None:
        resolved = Path(model_path)
        if not resolved.is_file():
            raise FileNotFoundError(f"GGUF model file not found: {resolved}")
        return str(resolved)

    if any(ch in filename for ch in _GLOB_CHARS):
        raise ValueError(
            f"Wildcard filename {filename!r} is not allowed; specify an exact GGUF filename "
            "(e.g. 'qwen2.5-1.5b-instruct-q4_k_m.gguf') for deterministic and offline loading."
        )

    effective_revision = (
        revision
        if revision is not None
        else (DEFAULT_LOCAL_REVISION if repo_id == DEFAULT_LOCAL_REPO_ID else None)
    )

    return str(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=effective_revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
        )
    )


class LocalGGUFClient:
    """Runs an instruction-tuned GGUF model in-process on CPU via ``llama.cpp``.

    Supports pinned Hugging Face downloads (offline-compatible once cached),
    explicit local ``model_path`` loading, role-separated system/user messages,
    token budgeting against ``n_ctx``, thread-safe generation, and deterministic
    native memory cleanup via ``close()`` or a ``with`` block.
    """

    def __init__(
        self,
        repo_id: str = DEFAULT_LOCAL_REPO_ID,
        filename: str = DEFAULT_LOCAL_FILENAME,
        revision: str | None = None,
        model_path: str | Path | None = None,
        n_ctx: int = DEFAULT_N_CTX,
        n_threads: int | None = None,
        n_batch: int = DEFAULT_N_BATCH,
        n_gpu_layers: int = 0,
        temperature: float = DEFAULT_TEMPERATURE,
        repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
        seed: int | None = DEFAULT_SEED,
        merge_system_into_user: bool = False,
        strip_thinking: bool = False,
        sanitize_control_tokens: bool = True,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        verbose: bool = False,
        llm: Llama | None = None,
    ) -> None:
        """Initialize the local GGUF client.

        Args:
            repo_id: Hugging Face model repository ID (ignored if ``model_path``
                or ``llm`` is supplied).
            filename: Exact GGUF filename inside ``repo_id`` (no wildcards).
            revision: Git commit SHA or tag in ``repo_id``. Defaults to
                ``DEFAULT_LOCAL_REVISION`` when using ``DEFAULT_LOCAL_REPO_ID``.
            model_path: Optional path to a local ``.gguf`` file on disk.
            n_ctx: Context window size in tokens.
            n_threads: Number of CPU threads. Defaults to ``min(cpu_count, 4)``.
            n_batch: Prompt-processing batch size.
            n_gpu_layers: Number of layers to offload to GPU (``0`` for CPU-only).
            temperature: Default sampling temperature (``0.0`` for factual RAG).
            repeat_penalty: Repetition penalty to curb greedy loops on small models.
            seed: Optional RNG seed for reproducible sampling.
            merge_system_into_user: If ``True``, merges ``system`` and ``prompt``
                into one ``user`` message for models whose chat template lacks a
                ``system`` role. Defaults to ``False`` to preserve role separation.
            strip_thinking: If ``True``, removes a leading ``<think>...</think>``
                (or truncated ``<think>...``) prefix from outputs.
            sanitize_control_tokens: If ``True``, strips raw chat-template control
                tokens from ``prompt`` before sending to the model.
            cache_dir: Optional Hugging Face cache directory.
            local_files_only: If ``True``, never contact the network when resolving
                the model file.
            verbose: Forwarded to ``llama_cpp.Llama`` logging verbosity.
            llm: Optional pre-initialized ``Llama`` instance to wrap.

        Raises:
            ValueError: If numeric configuration parameters are invalid or
                ``filename`` contains wildcard characters.
            FileNotFoundError: If ``model_path`` does not exist.
        """
        if n_ctx <= TEMPLATE_OVERHEAD_TOKENS:
            raise ValueError(f"n_ctx must be > {TEMPLATE_OVERHEAD_TOKENS}, got {n_ctx}")
        if n_batch <= 0:
            raise ValueError(f"n_batch must be > 0, got {n_batch}")
        threads = _default_threads() if n_threads is None else n_threads
        if threads <= 0:
            raise ValueError(f"n_threads must be > 0, got {threads}")
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0, got {temperature}")
        if repeat_penalty <= 0.0:
            raise ValueError(f"repeat_penalty must be > 0.0, got {repeat_penalty}")

        self.n_ctx = n_ctx
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.seed = seed
        self.merge_system_into_user = merge_system_into_user
        self.strip_thinking = strip_thinking
        self.sanitize_control_tokens = sanitize_control_tokens

        self._lock = threading.Lock()
        self._closed = False

        if llm is not None:
            self.llm: Any = llm
        else:
            resolved_path = _resolve_model_path(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                model_path=model_path,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            self.llm = Llama(
                model_path=resolved_path,
                n_ctx=n_ctx,
                n_threads=threads,
                n_batch=n_batch,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )

    def __enter__(self) -> LocalGGUFClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release underlying native ``llama.cpp`` model and context memory."""
        with self._lock:
            if not self._closed:
                self._closed = True
                if hasattr(self.llm, "close"):
                    self.llm.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("LocalGGUFClient has been closed.")

    def _tokenize(self, text: str) -> list[int]:
        tokens = self.llm.tokenize(
            text.encode("utf-8"),
            add_bos=False,
            special=False,
        )
        return [int(t) for t in tokens]

    def _build_messages(
        self,
        system: str | None,
        prompt: str,
        max_tokens: int,
        truncate_prompt: bool,
    ) -> list[dict[str, str]]:
        """Validate, sanitize, token-budget, and format chat messages."""
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a str, got {type(prompt).__name__}")
        if system is not None and not isinstance(system, str):
            raise TypeError(f"system must be a str or None, got {type(system).__name__}")

        system_clean = (system or "").strip()
        prompt_clean = prompt.strip()
        if self.sanitize_control_tokens:
            prompt_clean = _sanitize_control_tokens(prompt_clean)

        input_budget = self.n_ctx - max_tokens - TEMPLATE_OVERHEAD_TOKENS
        sys_tokens = self._tokenize(system_clean) if system_clean else []
        prompt_tokens = self._tokenize(prompt_clean) if prompt_clean else []

        prompt_budget = input_budget - len(sys_tokens)
        if prompt_budget <= 0:
            raise ValueError(
                f"System prompt ({len(sys_tokens)} tokens) + max_tokens ({max_tokens}) "
                f"+ template overhead ({TEMPLATE_OVERHEAD_TOKENS}) exceeds n_ctx ({self.n_ctx})."
            )

        if len(prompt_tokens) > prompt_budget:
            if not truncate_prompt:
                raise ValueError(
                    f"Prompt length ({len(prompt_tokens)} tokens) exceeds available "
                    f"budget ({prompt_budget} tokens for n_ctx={self.n_ctx}, "
                    f"max_tokens={max_tokens})."
                )
            logger.warning(
                "Truncating prompt from %d to %d tokens to fit within n_ctx=%d.",
                len(prompt_tokens),
                prompt_budget,
                self.n_ctx,
            )
            detokenized = self.llm.detokenize(prompt_tokens[:prompt_budget])
            prompt_clean = detokenized.decode("utf-8", errors="ignore").strip()

        if self.merge_system_into_user:
            combined = f"{system_clean}\n\n{prompt_clean}" if system_clean else prompt_clean
            return [{"role": "user", "content": combined}]

        messages: list[dict[str, str]] = []
        if system_clean:
            messages.append({"role": "system", "content": system_clean})
        messages.append({"role": "user", "content": prompt_clean})
        return messages

    def _validate_generation_args(
        self,
        max_tokens: int,
        temperature: float,
        repeat_penalty: float,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
        if max_tokens >= self.n_ctx - TEMPLATE_OVERHEAD_TOKENS:
            raise ValueError(
                f"max_tokens ({max_tokens}) leaves insufficient room in n_ctx ({self.n_ctx})."
            )
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0, got {temperature}")
        if repeat_penalty <= 0.0:
            raise ValueError(f"repeat_penalty must be > 0.0, got {repeat_penalty}")

    def generate(
        self,
        system: str | None,
        prompt: str,
        max_tokens: int = DEFAULT_LOCAL_MAX_TOKENS,
        temperature: float | None = None,
        repeat_penalty: float | None = None,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
        truncate_prompt: bool = True,
    ) -> str:
        """Generate a complete response for the given system instructions and prompt.

        Args:
            system: System instruction text (may be ``None`` or empty).
            prompt: User prompt (e.g., formatted RAG question + context).
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature override (defaults to ``self.temperature``).
            repeat_penalty: Repetition penalty override (defaults to ``self.repeat_penalty``).
            stop: Optional list of stop sequences.
            seed: Optional RNG seed override (defaults to ``self.seed``).
            truncate_prompt: If ``True``, truncates ``prompt`` to fit within
                ``n_ctx`` instead of raising ``ValueError``.

        Returns:
            Generated assistant text.
        """
        eff_temp = self.temperature if temperature is None else temperature
        eff_penalty = self.repeat_penalty if repeat_penalty is None else repeat_penalty
        eff_seed = self.seed if seed is None else seed
        self._validate_generation_args(max_tokens, eff_temp, eff_penalty)

        with self._lock:
            self._ensure_open()
            messages = self._build_messages(
                system=system,
                prompt=prompt,
                max_tokens=max_tokens,
                truncate_prompt=truncate_prompt,
            )
            response: dict[str, Any] = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=eff_temp,
                repeat_penalty=eff_penalty,
                stop=list(stop) if stop is not None else [],
                seed=eff_seed,
            )

        choices = response.get("choices") or []
        if not choices:
            return ""

        first = choices[0]
        if first.get("finish_reason") == "length":
            logger.warning(
                "LocalGGUFClient output reached max_tokens=%d and may be truncated.",
                max_tokens,
            )

        content = str((first.get("message") or {}).get("content") or "")
        if self.strip_thinking:
            return _strip_thinking_prefix(content)
        return content.strip()

    def generate_stream(
        self,
        system: str | None,
        prompt: str,
        max_tokens: int = DEFAULT_LOCAL_MAX_TOKENS,
        temperature: float | None = None,
        repeat_penalty: float | None = None,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
        truncate_prompt: bool = True,
    ) -> Iterator[str]:
        """Stream generated response tokens as they are produced.

        Validates arguments and client state eagerly before returning the
        stream iterator, and holds the client lock for the duration of the
        stream so concurrent calls do not access ``llama.cpp`` simultaneously.
        """
        eff_temp = self.temperature if temperature is None else temperature
        eff_penalty = self.repeat_penalty if repeat_penalty is None else repeat_penalty
        eff_seed = self.seed if seed is None else seed
        self._validate_generation_args(max_tokens, eff_temp, eff_penalty)
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a str, got {type(prompt).__name__}")
        if system is not None and not isinstance(system, str):
            raise TypeError(f"system must be a str or None, got {type(system).__name__}")
        with self._lock:
            self._ensure_open()

        def _stream_tokens() -> Iterator[str]:
            with self._lock:
                self._ensure_open()
                messages = self._build_messages(
                    system=system,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    truncate_prompt=truncate_prompt,
                )
                stream = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=eff_temp,
                    repeat_penalty=eff_penalty,
                    stop=list(stop) if stop is not None else [],
                    seed=eff_seed,
                    stream=True,
                )
                for chunk in stream:
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content")
                    if delta:
                        yield str(delta)

        return _stream_tokens()
