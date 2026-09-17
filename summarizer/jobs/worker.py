"""Independent lease-claiming worker.

A worker polls the durable queue, claims due jobs with an expiring lease,
runs the cancellable stage pipeline, heartbeats the lease while work is in
flight and performs crash recovery for leases that expired.

Run modes:

* embedded (default inside the FastAPI process) - daemon thread started
  lazily by :class:`~summarizer.jobs.runner.JobRunner`;
* standalone process - ``python -m summarizer.jobs.worker`` (CLI below).

Multiple workers (embedded + standalone, or several replicas) are safe:
SQLite claims are conditional updates, so a job is only ever leased to one
owner at a time.
"""

import argparse
import logging
import os
import random
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ..exceptions import (
    APIError,
    APIKeyError,
    ConfigurationError,
    SourceNotFoundError,
    UnsupportedSourceError,
    VideoValidationError,
)
from . import executor as executor_mod
from .bus import EventBus
from .executor import JobContext, JobCancelled, LeaseLostError
from .states import JobKind, JobState
from .store import JobStore

logger = logging.getLogger(__name__)


# Errors that never deserve an automatic stage retry.
_TERMINAL_ERRORS = (
    ConfigurationError,
    APIKeyError,
    SourceNotFoundError,
    UnsupportedSourceError,
    VideoValidationError,
)


def default_owner() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _backoff_seconds(attempt: int) -> float:
    return min(60.0, 2.0 ** max(0, attempt - 1)) + random.uniform(0.0, 0.5)


class Worker:
    """Polls the store and executes jobs with a bounded thread pool."""

    def __init__(
        self,
        store: JobStore,
        bus: Optional[EventBus] = None,
        *,
        owner: Optional[str] = None,
        lease_seconds: float = 30.0,
        concurrency: int = 2,
        poll_interval: float = 0.5,
        idle_exit: Optional[float] = None,
    ) -> None:
        self.store = store
        self.bus = bus or EventBus()
        self.owner = owner or default_owner()
        self.lease_seconds = float(lease_seconds)
        self.concurrency = max(1, int(concurrency))
        self.poll_interval = float(poll_interval)
        # When set, stop after the queue has been empty for this many seconds.
        self.idle_exit = idle_exit

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pool: Optional[ThreadPoolExecutor] = None
        self._inflight = 0
        self._inflight_lock = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run, name=f"summarizer-worker-{self.owner}", daemon=True
        )
        self._thread.start()

    def stop(self, wait: bool = True, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None and wait:
            self._thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run(self) -> None:
        logger.info(
            "worker %s starting (concurrency=%s lease=%ss)",
            self.owner, self.concurrency, self.lease_seconds,
        )
        self._pool = ThreadPoolExecutor(
            max_workers=self.concurrency,
            thread_name_prefix=f"job-{self.owner}",
        )
        last_work = time.time()
        try:
            while not self._stop.is_set():
                try:
                    self.store.reconcile()
                except Exception:
                    logger.exception("lease reconciliation sweep failed")

                with self._inflight_lock:
                    free_slots = self.concurrency - self._inflight

                progressed = False
                if free_slots > 0:
                    job = None
                    try:
                        job = self.store.claim_one(
                            self.owner, lease_seconds=self.lease_seconds
                        )
                    except Exception:
                        logger.exception("claim query failed")
                    if job is not None:
                        progressed = True
                        last_work = time.time()
                        with self._inflight_lock:
                            self._inflight += 1
                        self._pool.submit(self._run_job_safely, job["job_id"])
                    else:
                        if self.idle_exit is not None and not self._has_inflight():
                            if time.time() - last_work > self.idle_exit:
                                break

                if not progressed:
                    self._stop.wait(self.poll_interval)
        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
            logger.info("worker %s stopped", self.owner)

    def _has_inflight(self) -> bool:
        with self._inflight_lock:
            return self._inflight > 0

    # ── job execution ─────────────────────────────────────────────────────

    def _run_job_safely(self, job_id: str) -> None:
        try:
            self._run_job(job_id)
        except Exception:
            logger.exception("worker crashed while handling %s", job_id)
        finally:
            with self._inflight_lock:
                self._inflight -= 1

    def _run_job(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return
        if job["cancel_requested"]:
            self.store.mark_cancelled(job_id, owner=self.owner,
                                      reason="cancel requested before execution")
            self._publish(job_id)
            return
        if job["state"] not in (
            JobState.ACQUIRING.value,
            JobState.TRANSCRIBING.value,
            JobState.SUMMARIZING.value,
            JobState.FINALIZING.value,
        ):
            return  # someone else handled it concurrently

        config = self.store.read_config_snapshot(job["snapshot_hash"])
        ctx = JobContext(self.store, job, self.owner, bus=self.bus)
        heartbeat = self._start_heartbeat(job_id, ctx)
        try:
            heartbeat.start()
            if job["kind"] == JobKind.BATCH.value:
                if job["state"] == JobState.FINALIZING.value:
                    executor_mod.execute_batch(ctx)
                else:
                    # Batch parents start directly in finalizing by definition;
                    # anything else is a store inconsistency -> retry later.
                    self.store.requeue_for_retry(
                        job_id,
                        "batch parent claimed before finalizing",
                        "InvalidBatchState",
                        owner=self.owner,
                        retry_after=1.0,
                    )
            else:
                raw_summary = executor_mod.execute_job(config, ctx)
                if not isinstance(raw_summary, str) or not raw_summary.strip():
                    raise APIError("executor produced an empty summary")
                # Atomic final boundary: result artifact + completed state.
                self.store.complete_with_result(
                    job_id, raw_summary, owner=self.owner
                )
        except JobCancelled:
            logger.info("job %s acknowledged cancellation", job_id)
            try:
                self.store.mark_cancelled(job_id, owner=self.owner)
            except Exception:
                logger.exception("failed to mark %s cancelled", job_id)
        except LeaseLostError:
            logger.warning("job %s lease lost; another worker will resume", job_id)
            ctx.request_stop()
        except _TERMINAL_ERRORS as exc:
            logger.info("job %s failed permanently: %s", job_id, exc)
            self.store.fail_job(
                job_id, str(exc), exc.__class__.__name__, owner=self.owner
            )
        except Exception as exc:
            attempt = int(job["attempt"]) + 1
            retry_after = _backoff_seconds(attempt)
            logger.warning(
                "job %s stage %s failed (attempt %s): %s; retry in %.1fs",
                job_id, job.get("resume_stage"), attempt, exc, retry_after,
            )
            try:
                self.store.requeue_for_retry(
                    job_id,
                    str(exc),
                    exc.__class__.__name__,
                    owner=self.owner,
                    retry_after=retry_after,
                )
            except Exception:
                logger.exception("failed to requeue %s", job_id)
        finally:
            heartbeat.stop()
            self._publish(job_id)

    def _start_heartbeat(self, job_id: str, ctx: JobContext) -> "_Heartbeat":
        return _Heartbeat(
            self.store, job_id, self.owner,
            interval=max(2.0, self.lease_seconds / 3),
            on_lost=ctx.request_stop,
        )

    def _publish(self, job_id: str) -> None:
        if self.bus is None:
            return
        self.bus.publish(job_id)
        row = self.store.get(job_id)
        if row and row.get("parent_id"):
            self.bus.publish(row["parent_id"])


class _Heartbeat:
    """Extends the lease on a fixed cadence; flags the context on lease loss."""

    def __init__(self, store: JobStore, job_id: str, owner: str,
                 interval: float, on_lost) -> None:
        self.store = store
        self.job_id = job_id
        self.owner = owner
        self.interval = interval
        self.on_lost = on_lost
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"heartbeat-{self.job_id[:8]}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                ok = self.store.heartbeat(self.job_id, self.owner)
            except Exception:
                logger.exception("heartbeat update failed for %s", self.job_id)
                continue
            if not ok:
                logger.warning("lease lost for %s; signalling executor stop", self.job_id)
                self.on_lost()
                return


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI: python -m summarizer.jobs.worker
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m summarizer.jobs.worker",
        description="Independent lease-claiming worker for the summarizer job queue.",
    )
    parser.add_argument(
        "--jobs-dir",
        default=os.environ.get("SUMMARIZER_JOBS_DIR"),
        help="Persistent jobs directory (default: $SUMMARIZER_JOBS_DIR or ~/.summarizer/jobs)",
    )
    parser.add_argument("--concurrency", type=int,
                        default=int(os.environ.get("SUMMARIZER_WORKER_CONCURRENCY", "2")))
    parser.add_argument("--lease-seconds", type=float,
                        default=float(os.environ.get("SUMMARIZER_JOB_LEASE_SECONDS", "30")))
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument(
        "--idle-exit",
        type=float,
        default=None,
        help="Exit after the queue has stayed empty for N seconds (useful for one-shot runs).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    store = JobStore(base_dir=args.jobs_dir) if args.jobs_dir else JobStore()
    worker = Worker(
        store,
        concurrency=args.concurrency,
        lease_seconds=args.lease_seconds,
        poll_interval=args.poll_interval,
        idle_exit=args.idle_exit,
    )

    import signal

    def _shutdown(_signum, _frame):
        worker.stop(wait=False)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _shutdown)
        except (ValueError, OSError):
            pass

    worker.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
