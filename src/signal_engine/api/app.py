"""FastAPI application factory.

    uvicorn signal_engine.api.app:app --reload
    python -m signal_engine.api.app
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from signal_engine.api.routes import router, shutdown_engine
from signal_engine.config import get_settings

_ROOT = Path(__file__).resolve().parents[3]
UI_DIST = _ROOT / "ui" / "dist"          # built React app (npm run build)
UI_LEGACY = _ROOT / "ui" / "index.html"  # single-file fallback


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings().paths.ensure()
    yield
    await shutdown_engine()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Signal Engine",
        version="0.1.0",
        description=(
            "An AI-guided relationship discovery engine. The LLM generates hypotheses; a "
            "deterministic statistical engine decides what is true; Elasticsearch is the "
            "persistent evidence memory."
        ),
        lifespan=lifespan,
    )

    # The demo UI is served from this same origin, so CORS is only needed for
    # local development against a separate front-end port.
    # The bundled UI is same-origin, so CORS only matters when the front end is
    # hosted separately (Vercel) against this API. SIGNAL_CORS_ORIGINS is a
    # comma-separated allowlist; the Vite dev server is always permitted.
    extra = [
        o.strip()
        for o in os.environ.get("SIGNAL_CORS_ORIGINS", "").split(",")
        if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", *extra],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(router)

    # The built React bundle is served from this same process, so deployment is
    # one command and one port. Hashed asset filenames get a long cache; the
    # HTML entry point must not be cached or a deploy would serve stale JS.
    if (UI_DIST / "assets").is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=UI_DIST / "assets"),
            name="assets",
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> HTMLResponse:
        built = UI_DIST / "index.html"
        if built.exists():
            return HTMLResponse(
                built.read_text(),
                headers={"Cache-Control": "no-cache, must-revalidate"},
            )
        if UI_LEGACY.exists():
            return HTMLResponse(UI_LEGACY.read_text())
        return HTMLResponse(
            "<h1>Signal Engine</h1>"
            "<p>API is running. Build the UI with <code>cd web &amp;&amp; npm run build</code>, "
            "or see <a href='/docs'>/docs</a>.</p>"
        )

    return app


app = create_app()


def main() -> None:  # pragma: no cover - entry point
    import uvicorn

    # 0.0.0.0 inside a container; the default stays loopback for local runs.
    uvicorn.run(
        app,
        host=os.environ.get("SIGNAL_HOST", "127.0.0.1"),
        port=int(os.environ.get("SIGNAL_PORT", "8000")),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
