"""In-process event bus for instant SSE wakeups.

The SQLite ``job_events`` table is the durable source of truth and also the
cross-process channel (SSE readers poll it). This bus only removes latency
when the worker happens to run inside the API process: publishers call
:meth:`EventBus.publish` from worker threads, subscribers are asyncio queues
belonging to SSE response loops.
"""

import asyncio
import threading
from typing import Dict, Set


class EventBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}

    def subscribe(self, job_id: str) -> "Subscription":
        queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        with self._lock:
            self._subscribers.setdefault(job_id, set()).add(queue)
        return Subscription(self, job_id, queue)

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id)
            if subs:
                subs.discard(queue)
                if not subs:
                    self._subscribers.pop(job_id, None)

    def publish(self, job_id: str, payload=None) -> None:
        """Thread-safe; safe to call when no event loop is attached."""
        with self._lock:
            subs = list(self._subscribers.get(job_id, ()))
        for queue in subs:
            loop = getattr(queue, "_loop", None)
            try:
                if loop is not None and loop.is_running():
                    loop.call_soon_threadsafe(self._offer, queue, payload)
                else:
                    self._offer(queue, payload)
            except RuntimeError:
                self._offer(queue, payload)

    @staticmethod
    def _offer(queue: asyncio.Queue, payload) -> None:
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            # SSE loop also polls the DB, so dropping a wakeup is harmless.
            pass


class Subscription:
    def __init__(self, bus: EventBus, job_id: str, queue: asyncio.Queue) -> None:
        self._bus = bus
        self.job_id = job_id
        self._queue = queue
        self._loop = asyncio.get_event_loop()
        # Remember the owning loop on the queue for thread-safe publishing.
        self._queue._loop = self._loop  # type: ignore[attr-defined]

    async def wait(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for a wakeup. Returns True if woken."""
        try:
            await asyncio.wait_for(self._queue.get(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def close(self) -> None:
        self._bus.unsubscribe(self.job_id, self._queue)
