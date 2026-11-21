"""FastAPI application factory for the FAQ RAG chat API and optional static UI."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .schemas import ChatRequest, ChatResponse, CoursesResponse, HealthResponse
from .service import APISettings, FAQChatService, build_chat_service

logger = logging.getLogger(__name__)


def _get_service(request: Request) -> FAQChatService:
    """Retrieve the active ``FAQChatService`` from ``app.state``."""
    service: FAQChatService | None = getattr(request.app.state, "chat_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Chat service is not initialized.")
    return service


def _mount_static_ui(app: FastAPI, static_dir: Path | None) -> None:
    """Serve built Vite assets from ``static_dir`` when the directory exists."""
    if static_dir is None:
        return
    resolved = static_dir.resolve()
    index_file = resolved / "index.html"
    if not index_file.is_file():
        return

    assets_dir = resolved / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str) -> FileResponse:
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found.")
        candidate = (resolved / full_path).resolve()
        if full_path and candidate.is_file() and resolved in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index_file)


def create_app(
    *,
    service: FAQChatService | None = None,
    settings: APISettings | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        service: Optional pre-built ``FAQChatService`` (primarily for offline
            unit tests). When supplied, the app does not close it on shutdown.
        settings: Optional ``APISettings`` used when constructing the service
            during lifespan startup. Defaults to ``APISettings.from_env()``.

    Returns:
        Configured ``FastAPI`` application instance.
    """
    cfg = settings if settings is not None else APISettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_service = service is None
        active_service = service if service is not None else build_chat_service(cfg)
        app.state.chat_service = active_service
        try:
            yield
        finally:
            if owns_service:
                active_service.close()
            app.state.chat_service = None

    app = FastAPI(
        title="llm-rag-faq API",
        description="Hybrid lexical + semantic FAQ RAG chat API for DataTalks.Club courses.",
        version="0.1.0",
        lifespan=lifespan,
    )

    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cfg.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "Accept"],
        )

    @app.get("/api/health", response_model=HealthResponse, tags=["system"])
    async def get_health(request: Request) -> HealthResponse:
        return _get_service(request).health()

    @app.get("/api/courses", response_model=CoursesResponse, tags=["faq"])
    async def get_courses(request: Request) -> CoursesResponse:
        return _get_service(request).list_courses()

    @app.post("/api/chat", response_model=ChatResponse, tags=["chat"])
    async def post_chat(payload: ChatRequest, request: Request) -> ChatResponse:
        chat_service = _get_service(request)
        try:
            return await run_in_threadpool(chat_service.answer, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.exception("Chat inference failed: %s", exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    _mount_static_ui(app, cfg.static_dir)
    return app
