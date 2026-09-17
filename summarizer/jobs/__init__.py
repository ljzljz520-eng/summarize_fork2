"""Persistent job system for the summarizer.

Public surface::

    JobState / JobKind          - persistent state machine definitions
    JobStore                    - SQLite-backed persistence (jobs, events,
                                  artifacts, chunk checkpoints, idempotency,
                                  leases, versioned input snapshots)
    JobRunner                   - submit / query / cancel / wait facade
    Worker                      - independent lease-claiming worker loop
"""

from .states import JobState, JobKind, STAGES, TERMINAL_STATES, ACTIVE_STATES
from .store import JobStore
from .runner import JobRunner, get_runner, reset_runner
from .worker import Worker

__all__ = [
    "JobState",
    "JobKind",
    "STAGES",
    "TERMINAL_STATES",
    "ACTIVE_STATES",
    "JobStore",
    "JobRunner",
    "get_runner",
    "reset_runner",
    "Worker",
]
