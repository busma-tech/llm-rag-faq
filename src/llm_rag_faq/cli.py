"""Command-line interface for querying DataTalks.Club course FAQs."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from dotenv import load_dotenv

from llm_rag_faq.config import DEFAULT_CACHE_DIR, DEFAULT_CLOUD_MODEL, DEFAULT_COURSE
from llm_rag_faq.data import load_faq_data
from llm_rag_faq.llm import AnthropicClient, LLMClient, LocalGGUFClient
from llm_rag_faq.rag import RAGBase, RAGResult
from llm_rag_faq.retrieval import build_index

_EXIT_COMMANDS = frozenset({"quit", "exit", ":q"})


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``llm-rag-faq``."""
    parser = argparse.ArgumentParser(
        prog="llm-rag-faq",
        description="Query DataTalks.Club course FAQs with hybrid search + LLM RAG.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Question to answer (omit to start an interactive REPL).",
    )
    parser.add_argument(
        "--course",
        default=DEFAULT_COURSE,
        help=(
            f"Course slug to filter (default: {DEFAULT_COURSE!r}). "
            "Pass 'all' to search across all courses."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "local", "anthropic"],
        default="auto",
        help="LLM backend to use (default: auto).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_CLOUD_MODEL,
        help=f"Anthropic model ID (default: {DEFAULT_CLOUD_MODEL!r}).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a local .gguf model file (implies local backend under auto).",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help=f"Directory for cached FAQ JSON and embeddings (default: {DEFAULT_CACHE_DIR!r}).",
    )
    parser.add_argument(
        "--num-results",
        type=int,
        default=None,
        help="Number of FAQ passages to retrieve (default: 3 local, 5 cloud).",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print retrieved FAQ source metadata beneath the answer.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable query-rewrite fallback when the initial answer is 'I don't know.'.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Disallow network downloads when resolving a local GGUF model.",
    )
    return parser


def _create_llm_client(args: argparse.Namespace) -> LLMClient:
    """Instantiate the configured local or cloud ``LLMClient``."""
    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()

    if args.provider == "anthropic":
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Set it in .env or use --provider local."
            )
        return AnthropicClient(model=args.model, api_key=api_key)

    use_local = args.provider == "local" or (
        args.provider == "auto" and (args.model_path is not None or not api_key)
    )
    if use_local:
        return LocalGGUFClient(
            model_path=args.model_path,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )

    return AnthropicClient(model=args.model, api_key=api_key)


def _print_result(result: RAGResult, show_sources: bool) -> None:
    """Print the RAG answer and optional source diagnostics."""
    print(result.answer)
    if result.used_fallback and result.rewritten_query:
        print(f"\n[Fallback rewritten query: {result.rewritten_query}]")
    if show_sources:
        print("\nSources:")
        if not result.sources:
            print("  (none)")
            return
        for idx, doc in enumerate(result.sources, start=1):
            course = doc.get("course") or "unknown"
            section = doc.get("section") or ""
            question = doc.get("question") or ""
            print(f"  {idx}. [{course}] {section} — {question}")


def _run_repl(assistant: RAGBase, show_sources: bool) -> int:
    """Run an interactive question-answering loop until EOF or exit command."""
    print("Interactive FAQ RAG session. Type 'quit', 'exit', or ':q' to exit.")
    while True:
        try:
            query = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not query:
            continue
        if query.lower() in _EXIT_COMMANDS:
            return 0

        try:
            result = assistant.rag_with_sources(query)
        except (ValueError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue

        print()
        _print_result(result, show_sources=show_sources)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``llm-rag-faq``."""
    load_dotenv()
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    course: str | None = (
        None if args.course is None or args.course.lower() == "all" else args.course
    )

    try:
        documents = load_faq_data(course=course, cache_dir=args.cache_dir)
        index = build_index(documents, cache_dir=args.cache_dir)
        llm_client = _create_llm_client(args)
    except (ValueError, KeyError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        assistant = RAGBase(
            index=index,
            llm_client=llm_client,
            course=course,
            model=args.model,
            num_results=args.num_results,
            enable_fallback=not args.no_fallback,
        )
        if args.query is not None:
            result = assistant.rag_with_sources(args.query)
            _print_result(result, show_sources=args.show_sources)
            return 0
        return _run_repl(assistant, show_sources=args.show_sources)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        close_fn = getattr(llm_client, "close", None)
        if callable(close_fn):
            close_fn()


if __name__ == "__main__":
    raise SystemExit(main())
