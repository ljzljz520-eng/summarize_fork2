"""FastAPI HTTP server for the summarizer package.

Exposes all CLI functionality via a REST API. Auto-generated docs at /docs.
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
try:
    from pydantic import BaseModel, Field, model_validator

    def _before_model_validator(func):
        return model_validator(mode="before")(classmethod(func))

except ImportError:
    from pydantic import BaseModel, Field, root_validator

    def _before_model_validator(func):
        return root_validator(pre=True)(func)

from summarizer.config_file import load_config_file, merge_configs, find_config_file
from summarizer.exceptions import SummarizerError, ConfigurationError
from summarizer.jobs import JobKind, JobState
from summarizer.jobs.runner import get_runner
from summarizer.jobs.states import is_terminal
from summarizer.jobs.store import new_job_id
from summarizer.prompts import get_available_prompts
from summarizer.api_utils import (
    build_runtime_config,
    format_output,
    SOURCE_TYPES,
    OUTPUT_FORMATS,
    TRANSCRIPTION_METHODS,
    WHISPER_MODELS,
    DEFAULT_MAX_UPLOAD_MB,
    redact_config_response,
)

# Default wait budget for the legacy synchronous compatibility endpoints.
SYNC_WAIT_TIMEOUT = float(os.environ.get("SUMMARIZER_SYNC_WAIT_TIMEOUT", "3600"))


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────────────────────────────────────

SourceType = Literal[tuple(SOURCE_TYPES)]  # type: ignore[misc]
OutputFormat = Literal[tuple(OUTPUT_FORMATS)]  # type: ignore[misc]
TranscriptionMethod = Literal[tuple(TRANSCRIPTION_METHODS)]  # type: ignore[misc]
WhisperModel = Literal[tuple(WHISPER_MODELS)]  # type: ignore[misc]


def _reject_legacy_audio_speed_field(data: Any) -> Any:
    if isinstance(data, dict) and "audio_speed" in data:
        raise ValueError("audio_speed is no longer supported; use speed instead")
    return data


class SummarizeRequest(BaseModel):
    source: str = Field(..., description="Video URL or file path")
    type: SourceType = Field("YouTube Video", description="Source type")
    provider: Optional[str] = Field(None, description="Provider name from config")
    prompt_type: Optional[str] = Field(None, description="Summary style")
    chunk_size: Optional[int] = Field(
        None, ge=100, le=500_000, description="Characters per chunk"
    )
    parallel_calls: Optional[int] = Field(
        None, ge=1, le=200, description="Concurrent API requests"
    )
    max_tokens: Optional[int] = Field(
        None, ge=1, le=1_000_000, description="Max output tokens per chunk"
    )
    language: Optional[str] = Field(None, description="Caption/transcription language")
    output_language: Optional[str] = Field(None, description="Summary output language")
    force_download: bool = Field(False, description="Skip captions, download audio")
    transcription: Optional[TranscriptionMethod] = Field(
        None, description="Cloud Whisper or Local Whisper"
    )
    whisper_model: Optional[WhisperModel] = Field(None, description="Whisper model size")
    speed: Optional[float] = Field(
        None, gt=0.0, le=10.0, description="Playback speed for audio preprocessing or visual-mode video"
    )
    output_format: OutputFormat = Field("markdown", description="markdown, json, or html")
    visual: bool = Field(False, description="Send video directly to vision model")
    use_proxy: Optional[bool] = Field(None, description="Route through the configured HTTP proxy")
    api_key: Optional[str] = Field(None, description="Override API key")
    base_url: Optional[str] = Field(None, description="Override API base URL")
    model: Optional[str] = Field(None, description="Override model name")
    cobalt_url: Optional[str] = Field(None, description="Cobalt base URL")
    verbose: bool = Field(False, description="Verbose progress output")
    idempotency_key: Optional[str] = Field(
        None, description="Client-generated idempotency key (or send Idempotency-Key header)"
    )

    @_before_model_validator
    def _reject_legacy_audio_speed(cls, data: Any) -> Any:
        return _reject_legacy_audio_speed_field(data)


class SummarizeResponse(BaseModel):
    success: bool
    source: str
    summary: str
    format: str
    model: Optional[str] = None
    prompt_type: Optional[str] = None
    processing_time_seconds: float
    error: Optional[str] = None
    error_type: Optional[str] = None


class BatchRequest(BaseModel):
    sources: List[str] = Field(..., min_length=1, description="List of URLs or file paths")
    type: SourceType = Field("YouTube Video", description="Source type for all items")
    provider: Optional[str] = Field(None, description="Provider name from config")
    prompt_type: Optional[str] = Field(None, description="Summary style")
    chunk_size: Optional[int] = Field(
        None, ge=100, le=500_000, description="Characters per chunk"
    )
    parallel_calls: Optional[int] = Field(
        None, ge=1, le=200, description="Concurrent API requests"
    )
    max_tokens: Optional[int] = Field(
        None, ge=1, le=1_000_000, description="Max output tokens per chunk"
    )
    language: Optional[str] = Field(None, description="Caption/transcription language")
    output_language: Optional[str] = Field(None, description="Summary output language")
    force_download: bool = Field(False, description="Skip captions, download audio")
    transcription: Optional[TranscriptionMethod] = Field(
        None, description="Cloud Whisper or Local Whisper"
    )
    whisper_model: Optional[WhisperModel] = Field(None, description="Whisper model size")
    speed: Optional[float] = Field(
        None, gt=0.0, le=10.0, description="Playback speed for audio preprocessing or visual-mode video"
    )
    output_format: OutputFormat = Field("markdown", description="markdown, json, or html")
    visual: bool = Field(False, description="Send video directly to vision model")
    use_proxy: Optional[bool] = Field(None, description="Route through the configured HTTP proxy")
    api_key: Optional[str] = Field(None, description="Override API key")
    base_url: Optional[str] = Field(None, description="Override API base URL")
    model: Optional[str] = Field(None, description="Override model name")
    cobalt_url: Optional[str] = Field(None, description="Cobalt base URL")
    verbose: bool = Field(False, description="Verbose progress output")
    idempotency_key: Optional[str] = Field(
        None, description="Client-generated idempotency key for the whole batch"
    )

    @_before_model_validator
    def _reject_legacy_audio_speed(cls, data: Any) -> Any:
        return _reject_legacy_audio_speed_field(data)


class JobChildRef(BaseModel):
    job_id: str
    source: str
    dep_index: int


class JobEnqueueResponse(BaseModel):
    job_id: str
    state: str
    kind: str
    idempotency_key: Optional[str] = None
    idempotent_hit: bool = False
    created_at: float
    children: Optional[List[JobChildRef]] = None


class BatchResult(BaseModel):
    source: str
    success: bool
    summary: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    processing_time_seconds: float


class BatchResponse(BaseModel):
    success_count: int
    total_count: int
    results: List[BatchResult]
    overall_processing_time_seconds: float


class ProviderInfo(BaseModel):
    name: str
    base_url: Optional[str] = None
    model: Optional[str] = None
    chunk_size: Optional[int] = None


class ConfigResponse(BaseModel):
    default_provider: Optional[str] = None
    providers: Dict[str, Any]
    defaults: Dict[str, Any]
    config_file_path: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────

SNAKE_OVERRIDES = {
    "provider": "provider",
    "api_key": "api_key",
    "base_url": "base_url",
    "model": "model",
    "prompt_type": "prompt_type",
    "chunk_size": "chunk_size",
    "parallel_calls": "parallel_api_calls",
    "max_tokens": "max_output_tokens",
    "language": "language",
    "output_language": "output_language",
    "transcription": "transcription_method",
    "whisper_model": "whisper_model",
    "speed": "speed",
    "cobalt_url": "cobalt_base_url",
    "use_proxy": "use_proxy",
    "visual": "visual",
}


def _build_overrides(req: SummarizeRequest) -> Dict[str, Any]:
    """Build a CLI-args-style dict from a request for merge_configs."""
    overrides: Dict[str, Any] = {}
    for field, target in SNAKE_OVERRIDES.items():
        value = getattr(req, field)
        if value is not None:
            overrides[target] = value
    return overrides


def _build_runtime_config_from_request(
    req: SummarizeRequest,
    source_override: Optional[str] = None,
    type_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Build runtime config from request, using the same merge path as CLI."""
    file_config = load_config_file()
    overrides = _build_overrides(req)
    merged = merge_configs(file_config, overrides)
    # Ensure per-request overrides win even when merge_configs is mocked in tests.
    merged.update(overrides)

    # Match the guard that exists in the CLI after merging (gives actionable errors
    # for Raycast / API users when they specify a provider name).
    if not merged.get("base_url") or not merged.get("model"):
        prov = overrides.get("provider")
        if prov:
            raise ConfigurationError(
                f"Provider '{prov}' not found in config file (or the provider section is missing base_url/model). "
                f"Check your summarizer.yaml or set base_url + model directly."
            )
        raise ConfigurationError(
            "base_url and model are required. Either use a named 'provider' that exists in summarizer.yaml, "
            "or provide base_url and model explicitly."
        )

    return build_runtime_config(
        merged=merged,
        source=source_override or req.source,
        type_of_source=type_override or req.type,
        verbose=req.verbose,
        force_download=req.force_download,
    )


def _error_response(
    source: str,
    output_format: str,
    elapsed: float,
    exc: Exception,
) -> SummarizeResponse:
    """Build a structured error response."""
    return SummarizeResponse(
        success=False,
        source=source,
        summary="",
        format=output_format,
        error=str(exc),
        error_type=exc.__class__.__name__,
        processing_time_seconds=round(elapsed, 2),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Job-system helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_idempotency_key(req: Any, request: Optional[Request] = None) -> Optional[str]:
    key = getattr(req, "idempotency_key", None)
    if key:
        return key
    if request is not None:
        header_key = request.headers.get("Idempotency-Key")
        if header_key:
            return header_key.strip()
    return None


def _single_request_for(source: str, req: BatchRequest) -> SummarizeRequest:
    """Instantiate the per-item request for a batch entry."""
    return SummarizeRequest(
        source=source,
        type=req.type,
        provider=req.provider,
        prompt_type=req.prompt_type,
        chunk_size=req.chunk_size,
        parallel_calls=req.parallel_calls,
        max_tokens=req.max_tokens,
        language=req.language,
        output_language=req.output_language,
        force_download=req.force_download,
        transcription=req.transcription,
        whisper_model=req.whisper_model,
        speed=req.speed,
        output_format=req.output_format,
        visual=req.visual,
        use_proxy=req.use_proxy,
        api_key=req.api_key,
        base_url=req.base_url,
        model=req.model,
        cobalt_url=req.cobalt_url,
        verbose=req.verbose,
    )


async def _submit_and_wait_single(
    config: Dict[str, Any],
    *,
    display_source: str,
    output_format: str,
    idempotency_key: Optional[str],
    job_id: Optional[str] = None,
) -> SummarizeResponse:
    """Legacy synchronous adapter: enqueue one job and block for completion."""
    runner = get_runner()
    row, _hit = runner.submit_single(
        config,
        source=config.get("source_url_or_path", display_source),
        output_format=output_format,
        idempotency_key=idempotency_key,
        job_id=job_id,
    )
    try:
        final = await runner.wait_for_terminal(row["job_id"], timeout=SYNC_WAIT_TIMEOUT)
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail={"job_id": row["job_id"], "error": str(exc)},
        )

    started = final.get("started_at") or final["created_at"]
    finished = final.get("finished_at") or time.time()
    elapsed = max(0.0, finished - started)

    if final["state"] == JobState.COMPLETED.value and final.get("result_ref"):
        raw = runner.store.read_artifact_text(final["result_ref"])
        formatted = format_output(
            raw,
            display_source,
            output_format,
            {"prompt_type": config.get("prompt_type", ""), "model": config.get("model", "")},
        )
        return SummarizeResponse(
            success=True,
            source=display_source,
            summary=formatted,
            format=output_format,
            model=config.get("model"),
            prompt_type=config.get("prompt_type"),
            processing_time_seconds=round(elapsed, 2),
        )

    if final["state"] == JobState.CANCELLED.value:
        return SummarizeResponse(
            success=False, source=display_source, summary="", format=output_format,
            error="Job cancelled", error_type="JobCancelled",
            processing_time_seconds=round(elapsed, 2),
        )

    return SummarizeResponse(
        success=False,
        source=display_source,
        summary="",
        format=output_format,
        error=final.get("error") or "Job failed",
        error_type=final.get("error_type") or "Unknown",
        processing_time_seconds=round(elapsed, 2),
    )


async def _submit_and_wait_batch(
    req: BatchRequest,
    idempotency_key: Optional[str],
) -> BatchResponse:
    """Legacy synchronous adapter for /summarize/batch."""
    runner = get_runner()
    items = []
    configs = []
    for source in req.sources:
        single_req = _single_request_for(source, req)
        config = _build_runtime_config_from_request(single_req)
        configs.append(config)
        items.append({"source": source, "config": config,
                      "output_format": req.output_format})

    parent, _children, _hit = runner.submit_batch(
        items,
        parent_config=configs[0],
        output_format=req.output_format,
        idempotency_key=idempotency_key,
    )
    try:
        final = await runner.wait_for_terminal(parent["job_id"], timeout=SYNC_WAIT_TIMEOUT)
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail={"job_id": parent["job_id"], "error": str(exc)},
        )

    view = runner.view(final["job_id"])
    aggregate = (view or {}).get("batch")
    if final["state"] != JobState.COMPLETED.value or aggregate is None:
        # Cancellation / unexpected failure: synthesize the legacy shape.
        results = [
            BatchResult(
                source=source,
                success=False,
                error="Batch job cancelled" if final["state"] == JobState.CANCELLED.value
                else (final.get("error") or "Batch job failed"),
                error_type="JobCancelled" if final["state"] == JobState.CANCELLED.value
                else (final.get("error_type") or "Unknown"),
                processing_time_seconds=0.0,
            )
            for source in req.sources
        ]
        return BatchResponse(
            success_count=0,
            total_count=len(req.sources),
            results=results,
            overall_processing_time_seconds=round(
                time.time() - (final.get("started_at") or final["created_at"]), 2
            ),
        )

    return BatchResponse(
        success_count=aggregate["success_count"],
        total_count=aggregate["total_count"],
        results=[BatchResult(**{
            k: v for k, v in item.items() if k in {
                "source", "success", "summary", "error",
                "error_type", "processing_time_seconds",
            }
        }) for item in aggregate["results"]],
        overall_processing_time_seconds=aggregate["overall_processing_time_seconds"],
    )


async def _job_event_stream(
    job_id: str,
    last_event_id: int = 0,
    heartbeat_seconds: float = 15.0,
):
    """SSE generator: durable event-log replay + live bus wakeups."""
    runner = get_runner()
    if runner.get_job(job_id) is None:
        yield f"event: error\ndata: {json.dumps({'error': 'job not found'})}\n\n"
        return

    subscription = runner.bus.subscribe(job_id)
    seq = int(last_event_id or 0)
    try:
        while True:
            events = await run_in_threadpool(runner.events_since, job_id, seq)
            for event in events:
                seq = int(event["seq"])
                payload = {"event": event["event"], **(event.get("payload") or {})}
                yield (
                    f"id: {seq}\n"
                    f"event: {event['event']}\n"
                    f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
                )
            row = runner.get_job(job_id)
            if row is not None and is_terminal(row["state"]):
                return
            try:
                await asyncio.wait_for(subscription._queue.get(), timeout=heartbeat_seconds)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
    finally:
        subscription.close()


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app factory
# ──────────────────────────────────────────────────────────────────────────────

def create_app(allow_origins: Optional[List[str]] = None) -> FastAPI:
    """Create the FastAPI application with configurable CORS origins.

    If ``allow_origins`` is not provided, origins are read from the
    ``SUMMARIZER_CORS_ORIGINS`` environment variable.
    """
    application = FastAPI(
        title="Summarize API",
        description="Transcribe and summarize videos from any source using any OpenAI-compatible LLM.",
        version="0.1.0",
    )

    if allow_origins is None:
        origins_env = os.getenv("SUMMARIZER_CORS_ORIGINS", "")
        if origins_env == "*":
            allow_origins = ["*"]
        else:
            allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()]

    if allow_origins:
        # Credentials cannot be used with wildcard origins.
        allow_credentials = "*" not in allow_origins
        application.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_credentials=allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @application.get("/health")
    async def health() -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok", "service": "summarize"}

    @application.get("/providers")
    async def providers() -> List[ProviderInfo]:
        """List all configured providers from summarizer.yaml."""
        file_config = load_config_file()
        providers_cfg = file_config.get("providers", {})
        result = []
        for name, cfg in providers_cfg.items():
            result.append(ProviderInfo(
                name=name,
                base_url=cfg.get("base_url"),
                model=cfg.get("model"),
                chunk_size=cfg.get("chunk_size"),
            ))
        return result

    @application.get("/prompts")
    async def prompts() -> List[str]:
        """List all available summary prompt types."""
        return get_available_prompts()

    @application.get("/config")
    async def config() -> ConfigResponse:
        """Get the active merged configuration (sensitive keys redacted)."""
        file_config = load_config_file()
        safe_config = redact_config_response(file_config)
        defaults = safe_config.get("defaults", {})
        providers_cfg = safe_config.get("providers", {})
        default_provider = safe_config.get("default_provider")
        config_path = find_config_file()
        return ConfigResponse(
            default_provider=default_provider,
            providers=providers_cfg,
            defaults=defaults,
            config_file_path=config_path.as_posix() if config_path else None,
        )

    # ── Asynchronous job API ──────────────────────────────────────────────

    @application.post("/jobs", response_model=JobEnqueueResponse, status_code=202)
    async def enqueue_job(request: Request, req: SummarizeRequest) -> JobEnqueueResponse:
        """Quickly enqueue a single summarization job and return its job_id."""
        config = _build_runtime_config_from_request(req)
        runner = get_runner()
        row, hit = runner.submit_single(
            config,
            output_format=req.output_format,
            idempotency_key=_resolve_idempotency_key(req, request),
        )
        return JobEnqueueResponse(
            job_id=row["job_id"],
            state=row["state"],
            kind=row["kind"],
            idempotency_key=row.get("idempotency_key"),
            idempotent_hit=hit,
            created_at=row["created_at"],
        )

    @application.post("/jobs/batch", response_model=JobEnqueueResponse, status_code=202)
    async def enqueue_batch(request: Request, req: BatchRequest) -> JobEnqueueResponse:
        """Enqueue a batch: one parent job plus one child job per source."""
        runner = get_runner()
        items = []
        configs = []
        for source in req.sources:
            single_req = _single_request_for(source, req)
            config = _build_runtime_config_from_request(single_req)
            configs.append(config)
            items.append({"source": source, "config": config,
                          "output_format": req.output_format})
        parent, children, hit = runner.submit_batch(
            items,
            parent_config=configs[0],
            output_format=req.output_format,
            idempotency_key=_resolve_idempotency_key(req, request),
        )
        return JobEnqueueResponse(
            job_id=parent["job_id"],
            state=parent["state"],
            kind=parent["kind"],
            idempotency_key=parent.get("idempotency_key"),
            idempotent_hit=hit,
            created_at=parent["created_at"],
            children=[
                JobChildRef(job_id=c["job_id"], source=c["source"], dep_index=c["dep_index"])
                for c in children
            ],
        )

    @application.get("/jobs")
    async def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
        """List recent top-level jobs (newest first)."""
        limit = max(1, min(int(limit), 500))
        return get_runner().list_jobs(limit=limit)

    @application.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> Dict[str, Any]:
        """Get the full state, progress, snapshot pointer and result of a job."""
        view = get_runner().view(job_id)
        if view is None:
            raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
        return view

    @application.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> Dict[str, Any]:
        """Cancel a job. For batches, cancels every live child as well."""
        runner = get_runner()
        if runner.get_job(job_id) is None:
            raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
        return runner.cancel(job_id)

    @application.get("/jobs/{job_id}/events")
    async def job_events(
        job_id: str,
        request: Request,
        last_event_id: Optional[int] = None,
    ) -> StreamingResponse:
        """Server-Sent Events stream for a job (supports Last-Event-ID resume).

        Batch parents also receive ``child`` events for each subtask.
        """
        runner = get_runner()
        if runner.get_job(job_id) is None:
            raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
        resume = last_event_id
        if resume is None:
            header = request.headers.get("Last-Event-ID")
            if header and header.isdigit():
                resume = int(header)
        return StreamingResponse(
            _job_event_stream(job_id, last_event_id=resume or 0),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Legacy synchronous compatibility endpoints ────────────────────────

    @application.post("/summarize", response_model=SummarizeResponse)
    async def summarize(request: Request, req: SummarizeRequest) -> SummarizeResponse:
        """Summarize a video from a URL or file path.

        Compatibility adapter: enqueues a persistent job and waits for its
        completion. Prefer ``POST /jobs`` + ``GET /jobs/{id}/events`` for
        long-running work.
        """
        try:
            config = _build_runtime_config_from_request(req)
            return await _submit_and_wait_single(
                config,
                display_source=req.source,
                output_format=req.output_format,
                idempotency_key=_resolve_idempotency_key(req, request),
            )
        except HTTPException:
            raise
        except SummarizerError as e:
            return _error_response(req.source, req.output_format, 0.0, e)
        except Exception as e:
            return _error_response(req.source, req.output_format, 0.0, e)

    @application.post("/summarize/upload", response_model=SummarizeResponse)
    async def summarize_upload(
        request: Request,
        file: UploadFile = File(..., description="Video or text file to summarize"),
        type: Optional[str] = Form(None, description="Source type (auto-detected if omitted)"),
        provider: Optional[str] = Form(None),
        prompt_type: Optional[str] = Form(None),
        chunk_size: Optional[int] = Form(None),
        parallel_calls: Optional[int] = Form(None),
        max_tokens: Optional[int] = Form(None),
        language: Optional[str] = Form(None),
        output_language: Optional[str] = Form(None),
        force_download: bool = Form(False),
        transcription: Optional[str] = Form(None),
        whisper_model: Optional[str] = Form(None),
        speed: Optional[float] = Form(None),
        output_format: str = Form("markdown"),
        visual: bool = Form(False),
        use_proxy: Optional[bool] = Form(None),
        api_key: Optional[str] = Form(None),
        base_url: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
        cobalt_url: Optional[str] = Form(None),
        verbose: bool = Form(False),
        idempotency_key: Optional[str] = Form(None),
    ) -> SummarizeResponse:
        """Summarize an uploaded file (compatibility adapter, blocks on job).

        Accepts video files (.mp4, .mp3, .wav, .m4a, .webm) or text files
        (.txt, .md, .vtt, .srt, .csv, .log, .rst, .html, .xml, .json).
        Text files bypass audio processing entirely.
        """
        filename = file.filename or "upload"
        ext = Path(filename).suffix.lower()
        text_extensions = {
            ".txt", ".md", ".vtt", ".srt", ".csv",
            ".log", ".rst", ".html", ".xml", ".json",
        }
        detected_type = type or ("TXT" if ext in text_extensions else "Local File")
        tmp_path: Optional[str] = None

        try:
            max_upload_bytes = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
            suffix = ext or ".bin"
            total_read = 0
            stream_chunk_size = 1024 * 1024  # 1 MB chunks

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="wb") as tmp:
                while True:
                    chunk = await file.read(stream_chunk_size)
                    if not chunk:
                        break
                    total_read += len(chunk)
                    if total_read > max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File exceeds maximum upload size of {DEFAULT_MAX_UPLOAD_MB} MB",
                        )
                    tmp.write(chunk)
                tmp_path = tmp.name

            req = SummarizeRequest(
                source=tmp_path,
                type=detected_type,  # type: ignore[arg-type]
                provider=provider,
                prompt_type=prompt_type,
                chunk_size=chunk_size,
                parallel_calls=parallel_calls,
                max_tokens=max_tokens,
                language=language,
                output_language=output_language,
                force_download=force_download,
                transcription=transcription,  # type: ignore[arg-type]
                whisper_model=whisper_model,  # type: ignore[arg-type]
                speed=speed,
                output_format=output_format,  # type: ignore[arg-type]
                visual=visual,
                use_proxy=use_proxy,
                api_key=api_key,
                base_url=base_url,
                model=model,
                cobalt_url=cobalt_url,
                verbose=verbose,
                idempotency_key=idempotency_key
                or request.headers.get("Idempotency-Key"),
            )
            config = _build_runtime_config_from_request(
                req, source_override=tmp_path, type_override=detected_type
            )

            # Move the upload into the job workspace before enqueueing so an
            # independent worker process can read it.
            runner = get_runner()
            job_id = new_job_id()
            with open(tmp_path, "rb") as src:
                workspace_path = runner.persist_upload(job_id, filename, src.read())
            config["source_url_or_path"] = workspace_path

            return await _submit_and_wait_single(
                config,
                display_source=filename,
                output_format=output_format,
                idempotency_key=_resolve_idempotency_key(req, request),
                job_id=job_id,
            )

        except HTTPException:
            raise
        except SummarizerError as e:
            return _error_response(filename, output_format, 0.0, e)
        except Exception as e:
            return _error_response(filename, output_format, 0.0, e)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @application.post("/summarize/batch", response_model=BatchResponse)
    async def summarize_batch(request: Request, req: BatchRequest) -> BatchResponse:
        """Summarize multiple sources in one request (blocks until all finish).

        Compatibility adapter backed by a batch parent + child jobs. Results
        are returned in the same order as the input sources list.
        """
        try:
            return await _submit_and_wait_batch(
                req, idempotency_key=_resolve_idempotency_key(req, request)
            )
        except HTTPException:
            raise
        except SummarizerError as e:
            return BatchResponse(
                success_count=0,
                total_count=len(req.sources),
                results=[
                    BatchResult(
                        source=source, success=False, error=str(e),
                        error_type=e.__class__.__name__, processing_time_seconds=0.0,
                    )
                    for source in req.sources
                ],
                overall_processing_time_seconds=0.0,
            )

    return application


# Default app instance used by uvicorn and production imports.
app = create_app()
