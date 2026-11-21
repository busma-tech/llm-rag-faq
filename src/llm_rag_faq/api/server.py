"""CLI launcher for the FastAPI FAQ RAG server."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

import uvicorn
from dotenv import load_dotenv

from llm_rag_faq.config import DEFAULT_CACHE_DIR, DEFAULT_CLOUD_MODEL


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``llm-rag-faq-api``."""
    parser = argparse.ArgumentParser(
        prog="llm-rag-faq-api",
        description="Run the FastAPI backend for the DataTalks.Club FAQ RAG chat UI.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")
    parser.add_argument(
        "--provider",
        choices=["auto", "local", "anthropic"],
        default=None,
        help="LLM backend to use (default: auto).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Anthropic model ID (default: {DEFAULT_CLOUD_MODEL!r}).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a local .gguf model file.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=f"Directory for cached FAQ JSON and embeddings (default: {DEFAULT_CACHE_DIR!r}).",
    )
    parser.add_argument(
        "--course",
        default=None,
        help="Restrict startup indexing to a single course slug (default: all courses).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Disallow network downloads when resolving a local GGUF model.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the FastAPI server with Uvicorn."""
    load_dotenv()
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    env_overrides: dict[str, str | None] = {
        "FAQ_PROVIDER": args.provider,
        "FAQ_MODEL": args.model,
        "FAQ_MODEL_PATH": args.model_path,
        "FAQ_CACHE_DIR": args.cache_dir,
        "FAQ_COURSE": args.course,
        "FAQ_LOCAL_FILES_ONLY": "1" if args.local_files_only else None,
    }
    for key, value in env_overrides.items():
        if value is not None:
            os.environ[key] = value

    uvicorn.run(
        "llm_rag_faq.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
