"""API routes.

Analyses run as background jobs because a full search takes tens of seconds:
POST returns an id immediately, and GET reports progress while it runs.

Nothing here ever returns a credential.  `/health` reports availability
BOOLEANS via `Settings.redacted_dump()`, which is the only path config takes
out of the process.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from signal_engine.config import get_settings
from signal_engine.evidence.base import RetrievalQuery
from signal_engine.evidence.schemas import ExplanationStatus
from signal_engine.ingestion.base import register_local_dataset
from signal_engine.ingestion.tlc import (
    YELLOW_ACCOUNTING_IDENTITIES,
    YELLOW_DATASET_NOTES,
    YELLOW_TAXI_DICTIONARY,
    TLCSource,
    TLCVehicle,
)
from signal_engine.jsonutil import to_jsonable
from signal_engine.profiling.profiler import profile_dataset
from signal_engine.reporting import write_run_report
from signal_engine.search.orchestrator import AnalysisRequest, AnalysisResult, SignalEngine
from signal_engine.telemetry.metrics import get_registry

router = APIRouter()


# ---------------------------------------------------------------------------
# Job registry
# ---------------------------------------------------------------------------


@dataclass
class Job:
    analysis_id: str
    question: str
    dataset_id: str
    status: str = "queued"  # queued | running | completed | failed
    result: AnalysisResult | None = None
    error: str | None = None
    task: asyncio.Task | None = field(default=None, repr=False)

    def summary(self) -> dict:
        metrics = get_registry().get(self.analysis_id)
        payload: dict[str, Any] = {
            "analysis_id": self.analysis_id,
            "question": self.question,
            "dataset_id": self.dataset_id,
            "status": self.status,
            "error": self.error,
        }
        if metrics is not None:
            snapshot = metrics.to_dict()
            payload["progress"] = {
                "elapsed_seconds": snapshot["elapsed_seconds"],
                "statistical_tests": snapshot["counters"].get("statistical_tests", 0),
                "candidates_considered": snapshot["counters"].get("candidates_considered", 0),
                "plots_rendered": snapshot["counters"].get("plots_rendered", 0),
                "evidence_created": snapshot["counters"].get("evidence_created", 0),
                "llm_calls": snapshot["counters"].get("llm_calls", 0),
                "vlm_calls": snapshot["counters"].get("vlm_calls", 0),
            }
        return payload


JOBS: dict[str, Job] = {}
# One engine per process: it owns the evidence-memory connection pool.
_ENGINE: SignalEngine | None = None


def get_engine() -> SignalEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SignalEngine(get_settings())
    return _ENGINE


async def shutdown_engine() -> None:
    global _ENGINE
    if _ENGINE is not None:
        await _ENGINE.aclose()
        _ENGINE = None


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ProfileRequest(BaseModel):
    path: str | None = Field(None, description="Path to a local Parquet/CSV file")
    vehicle: str = "yellow"
    year: int = 2026
    month: int = 1
    dataset_id: str | None = None


class AnalysisCreateRequest(BaseModel):
    question: str = Field(
        "What factors are associated with the amount passengers pay for NYC yellow taxi trips?"
    )
    path: str | None = None
    vehicle: str = "yellow"
    year: int = 2026
    month: int = 1
    max_rounds: int = Field(2, ge=1, le=6)
    max_visualizations: int = Field(6, ge=1, le=20)
    enable_web_grounding: bool = False


# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------


def _resolve_dataset(path: str | None, vehicle: str, year: int, month: int):
    settings = get_settings()
    if path:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = settings.paths.root / resolved
    else:
        source = TLCSource(vehicle=TLCVehicle(vehicle), year=year, month=month)
        resolved = settings.paths.raw / source.local_filename()

    if not resolved.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"dataset not found at {resolved.name}. Run: python scripts/fetch_tlc.py "
                f"--vehicle {vehicle} --year {year} --month {month}"
            ),
        )
    return register_local_dataset(
        resolved,
        dataset_id=f"nyc_tlc_{vehicle}_{year:04d}_{month:02d}",
        description=f"NYC TLC {vehicle} taxi trip records, {year:04d}-{month:02d}.",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/health")
async def health() -> dict:
    """Service status.  Returns availability booleans only, never secrets."""
    settings = get_settings()
    engine = get_engine()
    payload = {
        "status": "ok",
        "config": settings.redacted_dump(),
        "missing_credentials": settings.missing_credentials(),
        "evidence_store": engine.memory.store.name,
        "brave": engine.brave.status,
    }
    store = engine.memory.store
    if hasattr(store, "ping"):
        payload["evidence_store_health"] = await store.ping()
    return payload


@router.post("/datasets/profile")
async def profile_endpoint(request: ProfileRequest) -> JSONResponse:
    settings = get_settings()
    handle = _resolve_dataset(request.path, request.vehicle, request.year, request.month)
    profile = profile_dataset(
        handle,
        dictionary=YELLOW_TAXI_DICTIONARY,
        accounting_identities=YELLOW_ACCOUNTING_IDENTITIES,
        dataset_notes=YELLOW_DATASET_NOTES,
        cache_dir=settings.paths.cache,
    )
    return JSONResponse(to_jsonable({
        "dataset": profile.dataset.to_dict(),
        "columns": {name: card.to_dict() for name, card in profile.columns.items()},
        "numeric_columns": profile.numeric_columns(),
        "categorical_columns": profile.categorical_columns(),
        "datetime_columns": profile.datetime_columns(),
        "prompt_text": profile.to_prompt_text(),
        "prompt_text_chars": len(profile.to_prompt_text()),
        "stats": profile.stats,
    }))


@router.post("/analyses", status_code=202)
async def create_analysis(
    request: AnalysisCreateRequest, background: BackgroundTasks
) -> dict:
    handle = _resolve_dataset(request.path, request.vehicle, request.year, request.month)
    analysis_id = f"an_{uuid.uuid4().hex[:12]}"

    job = Job(analysis_id=analysis_id, question=request.question, dataset_id=handle.dataset_id)
    JOBS[analysis_id] = job

    async def _run() -> None:
        job.status = "running"
        try:
            result = await get_engine().analyze(
                AnalysisRequest(
                    question=request.question,
                    dataset_handle=handle,
                    dictionary=YELLOW_TAXI_DICTIONARY,
                    accounting_identities=YELLOW_ACCOUNTING_IDENTITIES,
                    dataset_notes=YELLOW_DATASET_NOTES,
                    max_rounds=request.max_rounds,
                    max_visualizations=request.max_visualizations,
                    enable_web_grounding=request.enable_web_grounding,
                    analysis_id=analysis_id,
                )
            )
            job.result = result
            job.status = "completed"
            write_run_report(result, get_settings().paths.artifacts)
        except Exception as exc:  # noqa: BLE001 - surfaced through the job record
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"

    job.task = asyncio.create_task(_run())
    return {"analysis_id": analysis_id, "status": "queued", "poll": f"/analyses/{analysis_id}"}


@router.get("/analyses")
async def list_analyses() -> JSONResponse:
    return JSONResponse(to_jsonable({"analyses": [job.summary() for job in JOBS.values()]}))


@router.get("/analyses/{analysis_id}")
async def get_analysis(analysis_id: str, full: bool = Query(False)) -> JSONResponse:
    job = JOBS.get(analysis_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown analysis {analysis_id}")

    payload = job.summary()
    if job.result is not None:
        result = job.result
        payload["result"] = (
            result.to_dict()
            if full
            else {
                "dataset": result.profile.dataset.to_dict(),
                "final_answer": (
                    result.final_answer.model_dump() if result.final_answer else None
                ),
                "stop_reason": (
                    result.state.stop_reason.value if result.state.stop_reason else None
                ),
                "findings": [t.to_dict() for t in result.state.ranked_tests(limit=15)],
                "evidence_count": len(result.evidence),
                "metrics": result.metrics.to_dict() if result.metrics else {},
                "pruning": result.prune_stats.to_dict(),
                "analysis_view": result.view_report,
                "covariance_model": result.covariance_summary,
                "degradations": result.degradations,
                "branches": [b.to_dict() for b in result.state.branches.values()],
                "hypotheses": result.state.hypotheses,
                "notes": result.state.notes,
            }
        )
    return JSONResponse(to_jsonable(payload))


@router.get("/analyses/{analysis_id}/evidence")
async def get_analysis_evidence(analysis_id: str) -> JSONResponse:
    job = JOBS.get(analysis_id)
    if job is None or job.result is None:
        raise HTTPException(status_code=404, detail="analysis not found or not finished")
    return JSONResponse(to_jsonable({
        "analysis_id": analysis_id,
        "evidence": [e.to_dict() for e in job.result.evidence],
    }))


@router.get("/evidence/{evidence_id}")
async def get_evidence(evidence_id: str) -> JSONResponse:
    evidence = await get_engine().memory.store.get(evidence_id)
    if evidence is None:
        raise HTTPException(status_code=404, detail=f"unknown evidence {evidence_id}")
    return JSONResponse(to_jsonable(evidence.to_dict()))


@router.get("/evidence")
async def search_evidence(
    q: str = Query("", description="free-text query"),
    features: str = Query("", description="comma-separated feature names"),
    status: str = Query("", description="comma-separated explanation statuses"),
    dataset_id: str = Query(""),
    limit: int = Query(10, ge=1, le=50),
) -> JSONResponse:
    statuses: list[ExplanationStatus] = []
    for token in (s.strip() for s in status.split(",") if s.strip()):
        try:
            statuses.append(ExplanationStatus(token))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"unknown status {token!r}") from None

    results = await get_engine().memory.search(
        RetrievalQuery(
            text=q,
            feature_names=[f.strip() for f in features.split(",") if f.strip()],
            dataset_id=dataset_id or None,
            statuses=statuses,
            limit=limit,
        )
    )
    return JSONResponse(to_jsonable({
        "query": {"text": q, "features": features, "status": status, "limit": limit},
        "count": len(results),
        "results": [{**r.to_dict(), "evidence": r.evidence.to_dict()} for r in results],
    }))


@router.get("/metrics")
async def metrics_endpoint(analysis_id: str | None = None) -> JSONResponse:
    registry = get_registry()
    if analysis_id:
        metrics = registry.get(analysis_id)
        if metrics is None:
            raise HTTPException(status_code=404, detail=f"no metrics for {analysis_id}")
        return JSONResponse(
            to_jsonable({**metrics.to_dict(), "recent_events": metrics.recent_events(100)})
        )

    engine = get_engine()
    return JSONResponse(to_jsonable({
        **registry.aggregate(),
        "evidence_memory": engine.memory.telemetry(),
        "brave": engine.brave.telemetry(),
    }))


@router.get("/artifacts/{analysis_id}/{filename}")
async def get_artifact(analysis_id: str, filename: str) -> FileResponse:
    """Serve a generated plot.

    Both path components are sanitized and the resolved path is confirmed to
    sit inside the artifacts directory, so `../` cannot escape it.
    """
    if "/" in filename or "\\" in filename or ".." in filename or ".." in analysis_id:
        raise HTTPException(status_code=400, detail="invalid path")

    root = get_settings().paths.artifacts.resolve()
    path = (root / analysis_id / filename).resolve()
    if not str(path).startswith(str(root)) or not path.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(path)
