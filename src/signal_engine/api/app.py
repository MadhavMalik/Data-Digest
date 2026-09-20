"""FastAPI application factory.

    uvicorn signal_engine.api.app:app --reload
    python -m signal_engine.api.app
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from signal_engine.api.routes import router, shutdown_engine
from signal_engine.config import get_settings

UI_PATH = Path(__file__).resolve().parents[3] / "ui" / "index.html"


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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> HTMLResponse:
        if UI_PATH.exists():
            return HTMLResponse(UI_PATH.read_text())
        return HTMLResponse(
            "<h1>Signal Engine</h1><p>API is running. See <a href='/docs'>/docs</a>.</p>"
        )

    return app


app = create_app()


def main() -> None:  # pragma: no cover - entry point
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
