"""High-level facade used by the HTTP layer.

Owns the process-wide :class:`JobStore`, :class:`EventBus` and (by default)
an embedded daemon :class:`Worker` so that a single ``uvicorn`` process works
zero-config. Set ``SUMMARIZER_EMBEDDED_WORKER=0`` when running one or more
standalone workers (``python -m summarizer.jobs.worker``).
"""

import asyncio
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from .bus import EventBus
from .states import TERMINAL_STATES, JobKind, is_terminal
from .store import JobStore, new_job_id
from .worker import Worker

logger = logging.getLogger(__name__)


def _embedded_enabled() -> bool:
    return os.environ.get("SUMMARIZER_EMBEDDED_WORKER", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


class JobRunner:
    def __init__(
        self,
        store: Optional[JobStore] = None,
        *,
        start_embedded_worker: Optional[bool] = None,
        worker_concurrency: Optional[int] = None,
        lease_seconds: Optional[float] = None,
    ) -> None:
        self.store = store or JobStore()
        self.bus = EventBus()
        if start_embedded_worker is None:
            start_embedded_worker = _embedded_enabled()
        self.worker: Optional[Worker] = None
        if start_embedded_worker:
            self.worker = Worker(
                self.store,
                bus=self.bus,
                concurrency=worker_concurrency
                or int(os.environ.get("SUMMARIZER_WORKER_CONCURRENCY", "2")),
                lease_seconds=lease_seconds
                or float(os.environ.get("SUMMARIZER_JOB_LEASE_SECONDS", "30")),
            )
            self.worker.start()

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

    # ── submission ────────────────────────────────────────────────────────

    def submit_single(
        self,
        config: Dict[str, Any],
        *,
        source: Optional[str] = None,
        output_format: str = "markdown",
        idempotency_key: Optional[str] = None,
        job_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        dep_index: Optional[int] = None,
        max_attempts: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        row, hit = self.store.create_job(
            config=config,
            kind=JobKind.SINGLE.value,
            source=source if source is not None else config.get("source_url_or_path", ""),
            output_format=output_format,
            job_id=job_id,
            parent_id=parent_id,
            dep_index=dep_index,
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
        )
        self.bus.publish(row["job_id"])
        if parent_id:
            self.bus.publish(parent_id)
        return row, hit

    def submit_batch(
        self,
        items: List[Dict[str, Any]],
        *,
        parent_config: Dict[str, Any],
        output_format: str = "markdown",
        idempotency_key: Optional[str] = None,
        max_attempts: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], bool]:
        """Create a batch parent plus one child per item, in order.

        ``items`` is a list of ``{"source", "config"}`` dicts. Children have
        an implicit dependency: the parent's ``finalizing`` stage can only be
        claimed once every child is terminal (enforced by the claim query).
        """
        parent_id = new_job_id()
        parent, parent_hit = self.store.create_job(
            config=parent_config,
            kind=JobKind.BATCH.value,
            source=f"batch:{len(items)}",
            output_format=output_format,
            job_id=parent_id,
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
        )
        if parent_hit:
            children = self.store.children(parent["job_id"])
            return parent, children, True

        children: List[Dict[str, Any]] = []
        for index, item in enumerate(items):
            child_key = (
                f"{idempotency_key}:child:{index}" if idempotency_key else None
            )
            child, _hit = self.submit_single(
                item["config"],
                source=item["source"],
                output_format=item.get("output_format", output_format),
                job_id=new_job_id(),
                parent_id=parent_id,
                dep_index=index,
                idempotency_key=child_key,
                max_attempts=max_attempts,
            )
            children.append(child)
        return parent, children, False

    def persist_upload(self, job_id: str, filename: str, data: bytes) -> str:
        return self.store.save_input_file(job_id, filename, data)

    # ── queries ───────────────────────────────────────────────────────────

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.store.get(job_id)

    def view(self, job_id: str, *, include_results: bool = True) -> Optional[Dict[str, Any]]:
        row = self.store.get(job_id)
        if row is None:
            return None
        return self._build_view(row, include_results=include_results)

    def _build_view(
        self, row: Dict[str, Any], *, include_results: bool = True
    ) -> Dict[str, Any]:
        view: Dict[str, Any] = {
            "job_id": row["job_id"],
            "parent_id": row.get("parent_id"),
            "kind": row["kind"],
            "dep_index": row.get("dep_index"),
            "state": row["state"],
            "stage": row["state"] if row["state"] not in TERMINAL_STATES else None,
            "cancel_requested": bool(row["cancel_requested"]),
            "source": row.get("source"),
            "output_format": row.get("output_format"),
            "attempt": row["attempt"],
            "max_attempts": row["max_attempts"],
            "progress": {
                "total": row["progress_total"],
                "done": row["progress_done"],
            },
            "error": row.get("error"),
            "error_type": row.get("error_type"),
            "next_attempt_at": row.get("next_attempt_at"),
            "lease_owner": row.get("lease_owner"),
            "lease_expires_at": row.get("lease_expires_at"),
            "input_snapshot": {
                "version": row["snapshot_version"],
                "hash": row["snapshot_hash"],
            },
            "created_at": row["created_at"],
            "started_at": row.get("started_at"),
            "updated_at": row.get("updated_at"),
            "finished_at": row.get("finished_at"),
            "idempotency_key": row.get("idempotency_key"),
        }

        if include_results and row["state"] == "completed" and row.get("result_ref"):
            try:
                raw = self.store.read_artifact_text(row["result_ref"])
                if row["kind"] == JobKind.BATCH.value:
                    view["batch"] = json.loads(raw)
                else:
                    view["summary"] = raw
            except Exception as exc:  # pragma: no cover - defensive
                view["result_error"] = str(exc)

        if row["kind"] == JobKind.BATCH.value:
            view["children"] = [
                self._build_view(child, include_results=False)
                for child in self.store.children(row["job_id"])
            ]
        return view

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [self._build_view(r, include_results=False) for r in self.store.list_jobs(limit=limit)]

    def events_since(self, job_id: str, after_seq: int = 0) -> List[Dict[str, Any]]:
        return self.store.events_since(job_id, after_seq)

    def latest_seq(self, job_id: str) -> int:
        return self.store.latest_seq(job_id)

    # ── cancellation ──────────────────────────────────────────────────────

    def cancel(self, job_id: str) -> Dict[str, Any]:
        result = self.store.request_cancel_tree(job_id)
        for affected in result["affected"]:
            self.bus.publish(affected["job_id"])
        self.bus.publish(job_id)
        return self.view(job_id)

    # ── waiting ───────────────────────────────────────────────────────────

    async def wait_for_terminal(
        self,
        job_id: str,
        *,
        timeout: float = 3600.0,
        poll_interval: float = 1.0,
    ) -> Dict[str, Any]:
        """Poll + bus-wait until the job is terminal or timeout."""
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        subscription = self.bus.subscribe(job_id)
        try:
            while True:
                row = await loop.run_in_executor(None, self.store.get, job_id)
                if row is not None and is_terminal(row["state"]):
                    return row
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")
                await subscription.wait(min(poll_interval, remaining))
        finally:
            subscription.close()


# ── process-wide singleton ─────────────────────────────────────────────────

_runner: Optional[JobRunner] = None
_runner_lock = threading.Lock()


def get_runner() -> JobRunner:
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                _runner = JobRunner()
    return _runner


def reset_runner(runner: Optional[JobRunner] = None) -> JobRunner:
    """Replace the singleton (tests / custom bootstrap)."""
    global _runner
    with _runner_lock:
        if _runner is not None:
            _runner.shutdown()
        _runner = runner or JobRunner()
        return _runner
