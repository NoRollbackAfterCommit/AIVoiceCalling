"""Application entry point."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from vaani.api import routes, ws_voice
from vaani.api import settings as settings_api
from vaani.config import Settings, get_settings
from vaani.core.logging import configure_logging, get_logger
from vaani.core.registry import build_services
from vaani.db.repository import CallRepository
from vaani.pipeline.manager import CallManager
from vaani.settings_store import SettingsStore

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # The store layers persisted admin-portal overrides on top of the
    # environment, so what boots is what the operator last saved — not what the
    # container was originally started with.
    store = SettingsStore()
    settings: Settings = store.settings
    app.state.settings_store = store
    app.state.settings = settings

    services = build_services(settings)
    repository = CallRepository(settings.database_url)
    await repository.start()
    services.calls = repository
    await services.start()

    app.state.services = services
    app.state.calls = CallManager(max_concurrent=settings.max_concurrent_calls)

    await _seed_knowledge(services)

    log.info("vaani ready", extra={"port": settings.port, "env": settings.env})
    try:
        yield
    finally:
        log.info("shutting down")
        await app.state.calls.drain(timeout=15)
        await services.close()
        await repository.close()


async def _seed_knowledge(services) -> None:
    """Index anything sitting in ./knowledge at boot.

    Makes a fresh checkout demonstrable — drop a policy PDF in the folder, start
    the server, and the agent can already answer questions about it.
    """
    folder = Path("./knowledge")
    if not folder.is_dir():
        return
    paths = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf", ".docx", ".html"}
    ]
    if not paths:
        return
    with contextlib.suppress(Exception):
        count = await services.retriever.index_paths(paths)
        log.info("seeded knowledge", extra={"files": len(paths), "chunks": count})


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Vaani — AI Voice Agent Platform",
        version="0.1.0",
        description=(
            "Self-hosted, open-source voice agent platform. "
            "Control plane under /api, media plane at /ws/call."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes.router, prefix="/api")
    app.include_router(settings_api.router, prefix="/api")
    app.include_router(ws_voice.router)

    static_dir = Path(__file__).parent / "web" / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(str(static_dir / "index.html"))

        @app.get("/settings", include_in_schema=False)
        async def settings_page() -> FileResponse:
            return FileResponse(str(static_dir / "settings.html"))

        @app.get("/knowledge", include_in_schema=False)
        async def knowledge_page() -> FileResponse:
            return FileResponse(str(static_dir / "knowledge.html"))

    return app


app = create_app()
