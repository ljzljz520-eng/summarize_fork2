"""Persistent task state machine.

States::

    queued       - admitted to the queue, waiting for a worker lease
    acquiring    - worker stage 1: download / open / fetch the source
    transcribing - worker stage 2: captions/audio -> transcript text
    summarizing  - worker stage 3: model calls per content chunk
    finalizing   - worker stage 4: assemble + persist the final artifact
    completed    - terminal: success, result artifact committed
    failed       - terminal: retries exhausted, error recorded
    cancelled    - terminal: cancellation acknowledged

``queued`` and the four stage names are the "active" lifecycle states a job
can occupy while the system is making progress (or trying to). Only the store
(never user API code) may move a leased job back to ``queued`` for retry or
crash recovery; the stage pointer and committed artifacts then let the next
lease resume instead of redoing paid work.
"""

from enum import Enum


class JobState(str, Enum):
    QUEUED = "queued"
    ACQUIRING = "acquiring"
    TRANSCRIBING = "transcribing"
    SUMMARIZING = "summarizing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.value


class JobKind(str, Enum):
    SINGLE = "single"
    BATCH = "batch"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.value


# Ordered worker stages; the state machine uses these state names directly.
STAGES = (
    JobState.ACQUIRING.value,
    JobState.TRANSCRIBING.value,
    JobState.SUMMARIZING.value,
    JobState.FINALIZING.value,
)

ACTIVE_STATES = frozenset(
    [
        JobState.QUEUED.value,
        JobState.ACQUIRING.value,
        JobState.TRANSCRIBING.value,
        JobState.SUMMARIZING.value,
        JobState.FINALIZING.value,
    ]
)

RUNNING_STATES = frozenset(
    [
        JobState.ACQUIRING.value,
        JobState.TRANSCRIBING.value,
        JobState.SUMMARIZING.value,
        JobState.FINALIZING.value,
    ]
)

TERMINAL_STATES = frozenset(
    [
        JobState.COMPLETED.value,
        JobState.FAILED.value,
        JobState.CANCELLED.value,
    ]
)


def is_terminal(state: str) -> bool:
    return state in TERMINAL_STATES


def stage_index(stage: str) -> int:
    """Return the ordinal of a stage state (queued == -1)."""
    if stage == JobState.QUEUED.value:
        return -1
    return STAGES.index(stage)


def can_transition(src: str, dst: str) -> bool:
    """Validate an external/forward state transition.

    Allowed edges:

    * queued      -> any stage (claim jumps straight to the checkpoint stage);
    * stage       -> same/later stage (idempotent re-commit, forward progress);
    * any active  -> a terminal state;
    * queued      -> queued (store-internal scheduling metadata update).

    The store is allowed to move a leased job back to ``queued`` for retry /
    crash recovery regardless of this table (it owns retry semantics).
    """
    if src == dst:
        return True
    if src not in ACTIVE_STATES:
        return False
    if dst in TERMINAL_STATES:
        return True
    if src == JobState.QUEUED.value and dst in STAGES:
        return True
    if src in STAGES and dst in STAGES:
        return stage_index(dst) >= stage_index(src)
    return False
