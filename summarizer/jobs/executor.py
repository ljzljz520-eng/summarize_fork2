"""Cancellable, resumable stage pipeline for a leased job.

Stage mapping (state machine names live in :mod:`summarizer.jobs.states`):

* ``acquiring``    -> captions / TXT read / audio (or visual source) download;
* ``transcribing`` -> audio -> transcript text (no-op for text/visual);
* ``summarizing``  -> one model call per content chunk/segment, each result
                      checkpointed immediately so a crashed or retried stage
                      never bills the model twice for finished chunks;
* ``finalizing``   -> timestamped assembly, artifact commit, job completion.

Every stage is idempotent: on resume it first inspects committed artifacts and
``job_chunks`` and only performs the work that is still missing. Cancellation
is cooperative and checked at stage entry and before every model call.

The worker calls :func:`execute_job` (single jobs) or
:func:`execute_batch` (batch parent aggregation). Tests/alternative runtimes
may monkey-patch ``summarizer.jobs.executor.execute_job``.
"""

import asyncio
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from ..api import (
    extract_and_clean_chunks,
    format_summary_with_timestamps,
    process_chunk,
)
from ..api_utils import format_output
from ..config import validate_config
from ..exceptions import (
    APIError,
    ConfigurationError,
    SummarizerError,
    TranscriptError,
)
from ..prompts import load_prompt_template
from ..transcription import acquire_transcript_source, transcribe_audio
from ..transcript_cache import get_cached_transcript, put_cached_transcript
from .states import JobState

logger = logging.getLogger(__name__)


class JobCancelled(SummarizerError):
    """Raised inside the pipeline when the cancel flag was observed."""


class LeaseLostError(Exception):
    """Raised when the worker's lease was taken away; stop without writes."""


# ─────────────────────────────────────────────────────────────────────────────
# Execution context
# ─────────────────────────────────────────────────────────────────────────────

class JobContext:
    """Everything a stage function needs, backed by durable store calls."""

    def __init__(self, store, job: Dict[str, Any], owner: str, bus=None) -> None:
        self.store = store
        self.job_id: str = job["job_id"]
        self.parent_id = job.get("parent_id")
        self.owner = owner
        self.bus = bus
        self.state: str = job["state"]
        self.config: Dict[str, Any] = store.read_config_snapshot(job["snapshot_hash"])
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    # ── liveness ──
    def check_cancelled(self) -> None:
        if self._stop.is_set() or self.store.is_cancel_requested(self.job_id):
            raise JobCancelled(f"job {self.job_id} cancelled")

    def check_alive(self) -> None:
        self.check_cancelled()
        if not self.store.lease_is_valid(self.job_id, self.owner):
            raise LeaseLostError(f"lease lost for {self.job_id}")

    # ── events / progress ──
    def emit(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self.store.append_event(self.job_id, event, payload)
        self._wake()

    def _wake(self) -> None:
        if self.bus is not None:
            self.bus.publish(self.job_id)
            if self.parent_id:
                self.bus.publish(self.parent_id)

    def report_progress(self, done: int, total: int) -> None:
        self.store.set_progress(self.job_id, total, done)
        self.emit("progress", {"stage": self.state, "done": done, "total": total})

    # ── stage boundary (atomic state + artifact refs) ──
    def advance(
        self,
        to_state: str,
        *,
        checkpoint: Optional[Dict[str, Any]] = None,
        artifacts: Optional[List[Dict[str, Any]]] = None,
        progress_total: Optional[int] = None,
        progress_done: Optional[int] = None,
    ) -> None:
        self.store.commit_stage(
            self.job_id,
            to_state,
            owner=self.owner,
            checkpoint=checkpoint,
            artifacts=artifacts,
            progress_total=progress_total,
            progress_done=progress_done,
        )
        self.state = to_state
        self._wake()

    # ── artifacts ──
    def write_file(self, stage: str, name: str, data) -> Dict[str, Any]:
        return self.store.write_stage_file(self.job_id, stage, name, data)

    def artifact(self, stage: str, name: str) -> Optional[Dict[str, Any]]:
        return self.store.get_artifact(self.job_id, stage, name)

    def read_ref(self, ref: str) -> str:
        return self.store.read_artifact_text(ref)

    def checkpoint(self) -> Dict[str, Any]:
        row = self.store.require(self.job_id)
        return row.get("checkpoint") or {}

    # ── chunks ──
    def completed_chunks(self) -> Dict[int, Dict[str, Any]]:
        return self.store.completed_chunk_indices(self.job_id)

    def commit_chunk(self, index: int, summary: str) -> Dict[str, Any]:
        rec = self.write_file(
            JobState.SUMMARIZING.value, f"chunk_{index:05d}.txt", summary
        )
        self.store.upsert_chunk(
            self.job_id,
            index,
            "completed",
            ref=rec["ref"],
            content_hash=rec["content_hash"],
        )
        return rec

    def skip_chunk(self, index: int, status: str, error: Optional[str] = None) -> None:
        self.store.upsert_chunk(self.job_id, index, status, error=error)

    # ── completion ──
    # The worker owns terminal completion (store.complete_with_result) so the
    # executor entry point stays a simple ``(config, ctx) -> raw_summary``
    # seam that alternative runtimes / tests can replace wholesale.


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────

def execute_job(config: Dict[str, Any], ctx: JobContext) -> str:
    """Run the staged pipeline for a leased single job. Returns raw summary."""
    validate_config(config)

    if config.get("visual"):
        return _execute_visual(config, ctx)
    return _execute_transcript(config, ctx)


def execute_batch(ctx: JobContext) -> Dict[str, Any]:
    """Aggregate terminated children into the batch result artifact."""
    ctx.check_cancelled()
    parent = ctx.store.require(ctx.job_id)
    children = ctx.store.children(ctx.job_id)
    pending = [c for c in children if c["state"] not in (
        "completed", "failed", "cancelled"
    )]
    if pending:
        # Claim logic should prevent this; stay defensive.
        raise APIError(f"batch parent claimed with {len(pending)} active children")

    results = []
    success_count = 0
    for child in children:
        started = child.get("started_at") or child["created_at"]
        finished = child.get("finished_at") or started
        elapsed = round(max(0.0, finished - started), 2)
        if child["state"] == "completed" and child.get("result_ref"):
            try:
                raw = ctx.store.read_artifact_text(child["result_ref"])
                child_config = ctx.store.read_config_snapshot(child["snapshot_hash"])
                formatted = format_output(
                    raw,
                    child["source"],
                    child.get("output_format") or "markdown",
                    {
                        "prompt_type": child_config.get("prompt_type", ""),
                        "model": child_config.get("model", ""),
                    },
                )
                success_count += 1
                results.append({
                    "source": child["source"],
                    "success": True,
                    "summary": formatted,
                    "error": None,
                    "error_type": None,
                    "processing_time_seconds": elapsed,
                    "job_id": child["job_id"],
                })
            except Exception as exc:  # corrupted artifact -> surface as item failure
                results.append({
                    "source": child["source"],
                    "success": False,
                    "summary": None,
                    "error": f"failed to read child result: {exc}",
                    "error_type": exc.__class__.__name__,
                    "processing_time_seconds": elapsed,
                    "job_id": child["job_id"],
                })
        elif child["state"] == "cancelled":
            results.append({
                "source": child["source"],
                "success": False,
                "summary": None,
                "error": "Job cancelled",
                "error_type": "JobCancelled",
                "processing_time_seconds": elapsed,
                "job_id": child["job_id"],
            })
        else:
            results.append({
                "source": child["source"],
                "success": False,
                "summary": None,
                "error": child.get("error") or "Job failed",
                "error_type": child.get("error_type") or "Unknown",
                "processing_time_seconds": elapsed,
                "job_id": child["job_id"],
            })

    overall = round(max(0.0, (parent.get("finished_at") or _now()) -
                        (parent.get("started_at") or parent["created_at"])), 2)
    aggregate = {
        "success_count": success_count,
        "total_count": len(children),
        "results": results,
        "overall_processing_time_seconds": overall,
    }
    rec = ctx.write_file(
        JobState.FINALIZING.value,
        "batch_result.json",
        json.dumps(aggregate, ensure_ascii=False, indent=2),
    )
    ctx.store.complete_job(
        ctx.job_id,
        owner=ctx.owner,
        result_ref=rec["ref"],
        artifacts=[rec],
        progress_total=len(children),
        progress_done=len(children),
        payload={"success_count": success_count, "total_count": len(children)},
    )
    ctx.state = JobState.COMPLETED.value
    ctx._wake()
    return aggregate


def _now() -> float:
    import time
    return time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Transcript pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _execute_transcript(config: Dict[str, Any], ctx: JobContext) -> str:
    if ctx.state == JobState.ACQUIRING.value:
        _stage_acquiring(config, ctx)
    ctx.check_alive()

    if ctx.state == JobState.TRANSCRIBING.value:
        _stage_transcribing(config, ctx)
    ctx.check_alive()

    if ctx.state == JobState.SUMMARIZING.value:
        _stage_summarizing(config, ctx)
    ctx.check_alive()

    if ctx.state == JobState.FINALIZING.value:
        raw = _stage_finalize_transcript(config, ctx)
        return raw

    raise APIError(f"unexpected pipeline state for transcript job: {ctx.state}")


def _stage_acquiring(config: Dict[str, Any], ctx: JobContext) -> None:
    ctx.check_cancelled()

    # Transcript cache hit: skip acquisition AND transcription entirely.
    if config.get("cache_transcript", True):
        cached, _key, _source = get_cached_transcript(config)
        if cached is not None:
            rec = ctx.write_file(JobState.TRANSCRIBING.value, "transcript.txt", cached)
            ctx.advance(
                JobState.TRANSCRIBING.value,
                artifacts=[rec],
                checkpoint={"transcript_source": "cache"},
            )
            return

    kind, payload = acquire_transcript_source(config)
    if kind == "text":
        rec = ctx.write_file(JobState.ACQUIRING.value, "source.txt", payload)
        ctx.advance(
            JobState.TRANSCRIBING.value,
            artifacts=[rec],
            checkpoint={"source_kind": "text"},
        )
        return

    # audio: keep the external/temp path in the artifact metadata; the file
    # itself stays out of the content-addressed store (it is often huge).
    audio_path, should_delete = payload
    rec = {
        "stage": JobState.ACQUIRING.value,
        "name": "source_audio",
        "ref": audio_path,
        "content_hash": None,
        "size_bytes": None,
        "meta": {"kind": "audio", "should_delete": bool(should_delete)},
    }
    # The record goes through the normal artifact table in the stage tx.
    ctx.advance(
        JobState.TRANSCRIBING.value,
        artifacts=[rec],
        checkpoint={"source_kind": "audio",
                    "audio_path": audio_path,
                    "should_delete": bool(should_delete)},
    )


def _stage_transcribing(config: Dict[str, Any], ctx: JobContext) -> None:
    ctx.check_cancelled()

    existing = ctx.artifact(JobState.TRANSCRIBING.value, "transcript.txt")
    if existing is not None:
        ctx.advance(JobState.SUMMARIZING.value)
        return

    cp = ctx.checkpoint()
    source_kind = cp.get("source_kind")
    # ``source.txt`` is written during acquiring and registered in the same
    # atomic stage commit, so it lives under the acquiring stage namespace.
    source_rec = ctx.artifact(JobState.ACQUIRING.value, "source.txt")

    if source_kind == "text" or source_rec is not None:
        if source_rec is None:
            raise TranscriptError("acquired text source artifact is missing")
        transcript = ctx.read_ref(source_rec["ref"])
    else:
        audio_meta = ctx.artifact(JobState.ACQUIRING.value, "source_audio")
        if audio_meta is None:
            raise TranscriptError("acquired source artifact is missing")
        audio_path = audio_meta["ref"]
        raw_meta = audio_meta.get("meta")
        if isinstance(raw_meta, str):
            try:
                raw_meta = json.loads(raw_meta)
            except ValueError:
                raw_meta = {}
        should_delete = bool((raw_meta or {}).get("should_delete"))
        if not os.path.exists(audio_path):
            # Temp audio vanished with the process - reacquire internally.
            kind, payload = acquire_transcript_source(config)
            if kind != "audio":
                raise TranscriptError(
                    "source type changed while resuming; expected audio, got text"
                )
            audio_path, should_delete = payload
        ctx.check_cancelled()
        try:
            transcript = transcribe_audio(
                audio_path,
                config.get("transcription_method", "Cloud Whisper"),
                config.get("verbose", False),
                config.get("whisper_model", "tiny"),
                config.get("language", "auto"),
            )
        finally:
            if should_delete and audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

    if not transcript or not transcript.strip():
        raise TranscriptError("transcription produced no content")

    if config.get("cache_transcript", True):
        try:
            put_cached_transcript(config, transcript)
        except Exception:  # caching must never fail the stage
            logger.debug("transcript cache write failed", exc_info=True)

    rec = ctx.write_file(JobState.TRANSCRIBING.value, "transcript.txt", transcript)
    ctx.advance(
        JobState.SUMMARIZING.value,
        artifacts=[rec],
        checkpoint={"transcript_source": "stage"},
    )


def _stage_summarizing(config: Dict[str, Any], ctx: JobContext) -> None:
    ctx.check_cancelled()
    transcript_rec = ctx.artifact(JobState.TRANSCRIBING.value, "transcript.txt")
    if transcript_rec is None:
        raise TranscriptError("transcript artifact missing at summarizing stage")
    transcript = ctx.read_ref(transcript_rec["ref"])

    chunks = extract_and_clean_chunks(
        transcript, config.get("chunk_size", 10000)
    )
    if not chunks:
        raise TranscriptError("Failed to create content chunks")

    template = load_prompt_template(config.get("prompt_type", "Questions and answers"))

    # Only non-empty chunks are billed (matches the legacy pipeline).
    eligible = [
        (idx, ts, text)
        for idx, (ts, text) in enumerate(chunks)
        if text.strip()
    ]
    total = len(eligible)
    if not total:
        raise APIError("No valid content chunks to process")

    done_before = ctx.completed_chunks()
    ctx.report_progress(len(done_before), total)
    pending = [item for item in eligible if item[0] not in done_before]

    if pending:
        _run_async(_process_pending_chunks(config, ctx, template, pending))
        ctx.check_alive()

    completed = ctx.completed_chunks()
    valid = [i for (i, _ts, _text) in eligible if i in completed]
    ctx.report_progress(len(valid), total)
    if not valid:
        raise APIError("No valid summaries generated")

    ctx.advance(
        JobState.FINALIZING.value,
        progress_total=total,
        progress_done=len(valid),
    )


async def _process_pending_chunks(
    config: Dict[str, Any],
    ctx: JobContext,
    template: str,
    pending: List[Tuple[int, str, str]],
) -> None:
    semaphore = asyncio.Semaphore(config.get("parallel_api_calls", 5))
    total = len(pending) + len(ctx.completed_chunks())

    async def one(index: int, timestamp: str, text: str) -> None:
        async with semaphore:
            # Cooperative cancellation: boundary before every paid call.
            ctx.check_cancelled()
            try:
                summary = await process_chunk(text, template, config)
            except JobCancelled:
                raise
            except Exception as exc:
                logger.error("chunk %s failed: %s", index, exc)
                ctx.skip_chunk(index, "failed", error=str(exc))
                return
            ctx.check_cancelled()
            if summary and summary.strip():
                ctx.commit_chunk(index, summary.strip())
            else:
                ctx.skip_chunk(index, "empty")
            ctx.report_progress(len(ctx.completed_chunks()), total)

    tasks = [
        asyncio.ensure_future(one(index, ts, text))
        for index, ts, text in pending
    ]
    await asyncio.gather(*tasks)


def _stage_finalize_transcript(config: Dict[str, Any], ctx: JobContext) -> str:
    ctx.check_cancelled()
    transcript_rec = ctx.artifact(JobState.TRANSCRIBING.value, "transcript.txt")
    transcript = ctx.read_ref(transcript_rec["ref"])
    chunks = extract_and_clean_chunks(
        transcript, config.get("chunk_size", 10000)
    )
    completed = ctx.completed_chunks()
    summaries: List[Tuple[str, str]] = []
    for idx, (timestamp, _text) in enumerate(chunks):
        row = completed.get(idx)
        if row is None or not row.get("ref"):
            continue
        summaries.append((timestamp, ctx.read_ref(row["ref"])))

    if not summaries:
        raise APIError("No valid summaries to finalize")

    return format_summary_with_timestamps(summaries, config)


def _run_async(factory):
    """Run an async coroutine in a fresh loop (the worker is thread-based)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(factory)
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


# ─────────────────────────────────────────────────────────────────────────────
# Visual pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _execute_visual(config: Dict[str, Any], ctx: JobContext) -> str:
    if ctx.state == JobState.ACQUIRING.value:
        _stage_visual_acquire(config, ctx)
    ctx.check_alive()
    if ctx.state == JobState.TRANSCRIBING.value:
        ctx.advance(JobState.SUMMARIZING.value)
    ctx.check_alive()
    if ctx.state == JobState.SUMMARIZING.value:
        _stage_visual_summarize(config, ctx)
    ctx.check_alive()
    if ctx.state == JobState.FINALIZING.value:
        return _stage_visual_finalize(config, ctx)
    raise APIError(f"unexpected pipeline state for visual job: {ctx.state}")


def _stage_visual_acquire(config: Dict[str, Any], ctx: JobContext) -> None:
    from ..visual import (  # imported lazily like core.main does
        get_visual_profile,
        resolve_visual_url,
        resolve_video_source,
        normalize_video,
        validate_video_limits,
        build_visual_segments,
        split_video_segments,
    )

    ctx.check_cancelled()
    profile = get_visual_profile(config)
    visual_url = resolve_visual_url(config, profile)
    if visual_url:
        plan = {"mode": "url", "url": visual_url}
        rec = ctx.write_file(
            JobState.ACQUIRING.value, "visual_plan.json", json.dumps(plan)
        )
        ctx.advance(
            JobState.TRANSCRIBING.value,
            artifacts=[rec],
            checkpoint={"visual_mode": "url"},
        )
        return

    original_path, should_delete = resolve_video_source(config)
    normalized_path = normalize_video(original_path, profile, config)
    segments = build_visual_segments(normalized_path, profile, config)
    if len(segments) > 1 and not profile.get("supports_chunking"):
        validate_video_limits(normalized_path, profile, config)
    segment_paths = split_video_segments(normalized_path, segments, config)
    for segment in segment_paths:
        validate_video_limits(segment["path"], profile, config)

    plan = {
        "mode": "segments",
        "segments": segment_paths,
        "original_path": original_path,
        "should_delete_original": bool(should_delete),
        "normalized_path": normalized_path,
    }
    rec = ctx.write_file(
        JobState.ACQUIRING.value, "visual_plan.json", json.dumps(plan, default=str)
    )
    ctx.advance(
        JobState.TRANSCRIBING.value,
        artifacts=[rec],
        checkpoint={"visual_mode": "segments",
                    "cleanup": {
                        "original_path": original_path,
                        "should_delete_original": bool(should_delete),
                        "normalized_path": normalized_path,
                    }},
    )


def _visual_plan(ctx: JobContext) -> Dict[str, Any]:
    rec = ctx.artifact(JobState.ACQUIRING.value, "visual_plan.json")
    if rec is None:
        raise ConfigurationError("visual plan artifact missing")
    return json.loads(ctx.read_ref(rec["ref"]))


def _stage_visual_summarize(config: Dict[str, Any], ctx: JobContext) -> None:
    from ..visual import get_visual_profile
    from ..visual_api import process_video

    ctx.check_cancelled()
    profile = get_visual_profile(config)
    plan = _visual_plan(ctx)
    done_before = ctx.completed_chunks()

    if plan["mode"] == "url":
        total = 1
        if 0 not in done_before:
            ctx.check_cancelled()
            summary = _run_async(process_video(config, plan["url"], profile))
            ctx.check_cancelled()
            if summary and summary.strip():
                ctx.commit_chunk(0, summary.strip())
            else:
                ctx.skip_chunk(0, "empty")
        ctx.report_progress(len(ctx.completed_chunks()), total)
        if not ctx.completed_chunks():
            raise APIError("No valid summary generated")
        ctx.advance(JobState.FINALIZING.value, progress_total=1, progress_done=1)
        return

    segments = plan["segments"]
    total = len(segments)
    pending = [s for s in segments if int(s["index"]) - 1 not in done_before]
    for segment in pending:
        ctx.check_cancelled()
        index = int(segment["index"]) - 1
        segment_config = dict(config)
        segment_config.update({
            "visual_segment_start": segment.get("timestamp"),
            "visual_segment_end": segment.get("end_timestamp"),
            "visual_segment_index": segment.get("index"),
            "visual_segment_total": segment.get("total"),
        })
        summary = _run_async(process_video(segment_config, segment["path"], profile))
        ctx.check_cancelled()
        if summary and summary.strip():
            ctx.commit_chunk(index, summary.strip())
        else:
            ctx.skip_chunk(index, "empty")
        ctx.report_progress(len(ctx.completed_chunks()), total)

    if not ctx.completed_chunks():
        raise APIError("No valid summaries generated")
    ctx.advance(
        JobState.FINALIZING.value,
        progress_total=total,
        progress_done=len(ctx.completed_chunks()),
    )


def _stage_visual_finalize(config: Dict[str, Any], ctx: JobContext) -> str:
    ctx.check_cancelled()
    plan = _visual_plan(ctx)
    completed = ctx.completed_chunks()

    if plan["mode"] == "url":
        summaries = [("", ctx.read_ref(completed[0]["ref"]))]
    else:
        segments = plan["segments"]
        summaries = []
        for segment in segments:
            idx = int(segment["index"]) - 1
            row = completed.get(idx)
            if row and row.get("ref"):
                summaries.append((segment.get("timestamp", ""), ctx.read_ref(row["ref"])))

    final_summary = format_summary_with_timestamps(summaries, config)
    _visual_cleanup(plan)
    return final_summary


def _visual_cleanup(plan: Dict[str, Any]) -> None:
    """Best-effort removal of temp visual files after durable completion."""
    if plan.get("mode") != "segments":
        return
    for segment in plan.get("segments", []):
        path = segment.get("path")
        if segment.get("should_delete") and path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    cleanup = plan  # original/normalized paths live in the plan payload
    original = cleanup.get("original_path")
    if cleanup.get("should_delete_original") and original and os.path.exists(original):
        try:
            os.remove(original)
        except OSError:
            pass
    normalized = cleanup.get("normalized_path")
    if normalized and normalized != original and os.path.exists(normalized):
        try:
            os.remove(normalized)
        except OSError:
            pass
