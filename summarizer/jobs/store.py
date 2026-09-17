"""SQLite-backed persistence for the persistent job system.

The store owns every durable fact about a job:

* ``jobs``             - lifecycle state, lease, retry bookkeeping, progress,
                         versioned input-snapshot pointer, result pointer;
* ``job_events``       - append-only per-job event log (SSE replay source);
* ``job_artifacts``    - content-addressed file references committed at stage
                         boundaries, atomically with the state transition;
* ``job_chunks``       - per-model-chunk checkpoint so completed (already
                         billed) chunks are never sent to the model again;
* ``idempotency_keys`` - submit dedupe for single and batch requests.

Layout on disk::

    {base_dir}/jobs.db
    {base_dir}/snapshots/{sha256}.json       # versioned input snapshots
    {base_dir}/artifacts/{job_id}/{stage}/{name}

SQLite runs in WAL mode so an independent worker process and the FastAPI
process can read/write concurrently.
"""

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .states import JobKind, JobState, STAGES, TERMINAL_STATES, can_transition, is_terminal

SCHEMA_VERSION = 1
SNAPSHOT_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id                TEXT PRIMARY KEY,
    parent_id             TEXT,
    kind                  TEXT NOT NULL,
    dep_index             INTEGER,
    idempotency_key       TEXT,
    state                 TEXT NOT NULL,
    resume_stage          TEXT NOT NULL,
    cancel_requested      INTEGER NOT NULL DEFAULT 0,
    attempt               INTEGER NOT NULL DEFAULT 0,
    max_attempts          INTEGER NOT NULL DEFAULT 3,
    source                TEXT,
    output_format         TEXT,
    snapshot_hash         TEXT NOT NULL,
    snapshot_version      INTEGER NOT NULL DEFAULT 1,
    result_ref            TEXT,
    checkpoint            TEXT,
    progress_total        INTEGER NOT NULL DEFAULT 0,
    progress_done         INTEGER NOT NULL DEFAULT 0,
    error                 TEXT,
    error_type            TEXT,
    next_attempt_at       REAL,
    lease_owner           TEXT,
    lease_expires_at      REAL,
    created_at            REAL NOT NULL,
    updated_at            REAL NOT NULL,
    started_at            REAL,
    finished_at           REAL,
    FOREIGN KEY (parent_id) REFERENCES jobs(job_id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_state_time ON jobs(state, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_parent ON jobs(parent_id);
CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs(state, lease_expires_at);

CREATE TABLE IF NOT EXISTS job_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL,
    origin_id   TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    event       TEXT NOT NULL,
    payload     TEXT,
    created_at  REAL NOT NULL,
    UNIQUE(job_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_events_job_seq ON job_events(job_id, seq);

CREATE TABLE IF NOT EXISTS job_artifacts (
    job_id      TEXT NOT NULL,
    stage       TEXT NOT NULL,
    name        TEXT NOT NULL,
    ref         TEXT NOT NULL,
    content_hash TEXT,
    size_bytes  INTEGER,
    meta        TEXT,
    created_at  REAL NOT NULL,
    PRIMARY KEY (job_id, stage, name)
);

CREATE TABLE IF NOT EXISTS job_chunks (
    job_id      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    status      TEXT NOT NULL,
    ref         TEXT,
    content_hash TEXT,
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    updated_at  REAL NOT NULL,
    PRIMARY KEY (job_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    idem_key    TEXT PRIMARY KEY,
    job_id      TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""


class JobNotFoundError(Exception):
    pass


class InvalidTransitionError(Exception):
    pass


def new_job_id() -> str:
    return uuid.uuid4().hex


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


class JobStore:
    """All persistence operations. One instance per process is enough;
    individual SQLite connections are short-lived and thread-confined."""

    def __init__(
        self,
        base_dir: Optional[os.PathLike] = None,
        db_path: Optional[os.PathLike] = None,
    ) -> None:
        if base_dir is None:
            env_dir = os.environ.get("SUMMARIZER_JOBS_DIR", "").strip()
            base_dir = Path(env_dir) if env_dir else Path.home() / ".summarizer" / "jobs"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.base_dir, 0o700)
        except OSError:
            pass
        (self.base_dir / "snapshots").mkdir(exist_ok=True)
        (self.base_dir / "artifacts").mkdir(exist_ok=True)

        self.db_path = Path(db_path) if db_path else self.base_dir / "jobs.db"
        self._init_lock = threading.Lock()
        self._init_db()

    # ── low level ─────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            isolation_level=None,  # explicit BEGIN/COMMIT
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._init_lock:
            conn = self._connect()
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.executescript(_DDL)
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            finally:
                conn.close()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        if d.get("checkpoint"):
            try:
                d["checkpoint"] = json.loads(d["checkpoint"])
            except (TypeError, ValueError):
                pass
        return d

    # ── snapshots & artifact files ────────────────────────────────────────

    def write_config_snapshot(self, config: Dict[str, Any]) -> Tuple[str, int, str]:
        """Persist a versioned input snapshot, content-addressed.

        Returns ``(snapshot_hash, version, ref)``. Snapshot files are written
        with 0600 because runtime configs may carry API keys.
        """
        envelope = {
            "version": SNAPSHOT_VERSION,
            "kind": "summarizer-runtime-config",
            "created_at": time.time(),
            "config": config,
        }
        raw = json.dumps(envelope, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.sha256(_canonical_json(envelope["config"]).encode("utf-8")).hexdigest()
        ref = f"snapshots/{digest}.json"
        path = self.base_dir / ref
        if not path.exists():
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(raw)
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, path)
        return digest, SNAPSHOT_VERSION, ref

    def read_config_snapshot(self, snapshot_hash: str) -> Dict[str, Any]:
        path = self.base_dir / "snapshots" / f"{snapshot_hash}.json"
        with open(path, "r", encoding="utf-8") as fh:
            envelope = json.load(fh)
        return envelope["config"]

    def _safe_artifact_path(self, ref: str) -> Path:
        path = (self.base_dir / ref).resolve()
        root = self.base_dir.resolve()
        if root not in path.parents and path != root:
            raise ValueError(f"artifact ref escapes jobs dir: {ref}")
        return path

    def save_input_file(self, job_id: str, filename: str, data: bytes) -> str:
        """Persist an uploaded input file into the job's workspace."""
        safe_name = Path(filename).name or "upload.bin"
        rel = Path("artifacts") / job_id / "input" / safe_name
        path = self.base_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return str(path)

    def save_artifact_file(
        self,
        job_id: str,
        stage: str,
        name: str,
        data,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write an artifact file and record its reference (own transaction).

        Use :meth:`commit_stage` with ``artifacts=[...]`` instead when the
        reference must be committed atomically together with a state
        transition.
        """
        rec = self._write_artifact_file(job_id, stage, name, data)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._insert_artifact(conn, job_id, rec, meta)
            conn.commit()
        finally:
            conn.close()
        return rec

    def _write_artifact_file(
        self, job_id: str, stage: str, name: str, data
    ) -> Dict[str, Any]:
        if isinstance(data, str):
            payload = data.encode("utf-8")
        else:
            payload = bytes(data)
        rel = Path("artifacts") / job_id / stage / name
        path = self.base_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            fh.write(payload)
        os.replace(tmp, path)
        digest = hashlib.sha256(payload).hexdigest()
        return {
            "stage": stage,
            "name": name,
            "ref": rel.as_posix(),
            "content_hash": digest,
            "size_bytes": len(payload),
        }

    @staticmethod
    def _insert_artifact(
        conn: sqlite3.Connection,
        job_id: str,
        rec: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO job_artifacts
                (job_id, stage, name, ref, content_hash, size_bytes, meta, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id, stage, name) DO UPDATE SET
                ref=excluded.ref,
                content_hash=excluded.content_hash,
                size_bytes=excluded.size_bytes,
                meta=excluded.meta,
                created_at=excluded.created_at
            """,
            (
                job_id,
                rec["stage"],
                rec["name"],
                rec["ref"],
                rec.get("content_hash"),
                rec.get("size_bytes"),
                json.dumps(meta, default=str) if meta is not None else None,
                time.time(),
            ),
        )

    def write_stage_file(
        self, job_id: str, stage: str, name: str, data
    ) -> Dict[str, Any]:
        """Write a stage file WITHOUT recording it in job_artifacts.

        Used for high-frequency checkpoints (model chunks) that are tracked
        in ``job_chunks`` instead.
        """
        return self._write_artifact_file(job_id, stage, name, data)

    def complete_with_result(
        self,
        job_id: str,
        data,
        *,
        owner: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist the final result artifact and complete the job together.

        The artifact reference is inserted in the same transaction as the
        ``completed`` transition, so a completed job always has a readable
        result and vice versa. A second call for an already terminal job is a
        no-op (the existing result file is never overwritten).
        """
        current = self.get(job_id)
        if current is not None and current["state"] in TERMINAL_STATES:
            return current
        rec = self._write_artifact_file(
            job_id, JobState.FINALIZING.value, "result.txt", data
        )
        return self.complete_job(
            job_id,
            owner=owner,
            result_ref=rec["ref"],
            artifacts=[rec],
            payload=payload,
        )

    def read_artifact(self, ref: str) -> bytes:
        return self._safe_artifact_path(ref).read_bytes()

    def read_artifact_text(self, ref: str) -> str:
        return self.read_artifact(ref).decode("utf-8")

    def get_artifact(self, job_id: str, stage: str, name: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM job_artifacts WHERE job_id=? AND stage=? AND name=?",
                (job_id, stage, name),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    # ── creation / lookup ─────────────────────────────────────────────────

    def create_job(
        self,
        *,
        config: Dict[str, Any],
        kind: str,
        source: str,
        output_format: str,
        job_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        dep_index: Optional[int] = None,
        idempotency_key: Optional[str] = None,
        max_attempts: Optional[int] = None,
        now: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        """Insert a job (and its idempotency mapping) in one transaction.

        Returns ``(job_row, idempotency_hit)``. When ``idempotency_key`` was
        already registered, no row is inserted and the existing job is
        returned instead.
        """
        now = time.time() if now is None else now
        job_id = job_id or new_job_id()
        digest, version, _ = self.write_config_snapshot(config)
        resume_stage = (
            JobState.FINALIZING.value if kind == JobKind.BATCH.value else STAGES[0]
        )
        if max_attempts is None:
            max_attempts = int(os.environ.get("SUMMARIZER_JOB_MAX_ATTEMPTS", "3"))

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            if idempotency_key:
                existing = conn.execute(
                    "SELECT job_id FROM idempotency_keys WHERE idem_key=?",
                    (idempotency_key,),
                ).fetchone()
                if existing:
                    conn.commit()
                    row = self.get(existing["job_id"])
                    return row, True

            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, parent_id, kind, dep_index, idempotency_key,
                    state, resume_stage, cancel_requested, attempt, max_attempts,
                    source, output_format, snapshot_hash, snapshot_version,
                    checkpoint, progress_total, progress_done,
                    created_at, updated_at
                ) VALUES (?,?,?,?,?, ?,?,?,?,?, ?,?,?,?, ?,?,?, ?,?)
                """,
                (
                    job_id, parent_id, kind, dep_index, idempotency_key,
                    JobState.QUEUED.value, resume_stage, 0, 0, max_attempts,
                    source, output_format, digest, version,
                    None, 0, 0, now, now,
                ),
            )
            if idempotency_key:
                conn.execute(
                    "INSERT INTO idempotency_keys (idem_key, job_id, created_at) VALUES (?,?,?)",
                    (idempotency_key, job_id, now),
                )
            self._insert_event(
                conn,
                job_id,
                job_id,
                "queued",
                {
                    "kind": kind,
                    "source": source,
                    "parent_id": parent_id,
                    "dep_index": dep_index,
                    "idempotency_key": idempotency_key,
                },
                now,
            )
            if parent_id:
                self._insert_event(
                    conn,
                    parent_id,
                    job_id,
                    "child",
                    {"event": "queued", "child_job_id": job_id, "dep_index": dep_index},
                    now,
                )
            conn.commit()
            return self.get(job_id), False
        finally:
            conn.close()

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return self._row_to_dict(row) if row else None
        finally:
            conn.close()

    def require(self, job_id: str) -> Dict[str, Any]:
        row = self.get(job_id)
        if row is None:
            raise JobNotFoundError(job_id)
        return row

    def children(self, parent_id: str) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE parent_id=? ORDER BY dep_index ASC, created_at ASC",
                (parent_id,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def list_jobs(self, limit: int = 50, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            if kind:
                rows = conn.execute(
                    "SELECT * FROM jobs WHERE kind=? AND parent_id IS NULL "
                    "ORDER BY created_at DESC LIMIT ?",
                    (kind, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM jobs WHERE parent_id IS NULL "
                    "ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        finally:
            conn.close()

    # ── events ────────────────────────────────────────────────────────────

    @staticmethod
    def _insert_event(
        conn: sqlite3.Connection,
        job_id: str,
        origin_id: str,
        event: str,
        payload: Optional[Dict[str, Any]],
        now: float,
    ) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM job_events WHERE job_id=?",
            (job_id,),
        ).fetchone()
        seq = int(row["next_seq"])
        conn.execute(
            """
            INSERT INTO job_events (job_id, origin_id, seq, event, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job_id, origin_id, seq, event,
             json.dumps(payload, default=str, ensure_ascii=False) if payload is not None else None,
             now),
        )
        return seq

    def append_event(
        self,
        job_id: str,
        event: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        origin_id: Optional[str] = None,
    ) -> int:
        """Append an event; child events fan out to the batch parent stream."""
        now = time.time()
        origin_id = origin_id or job_id
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            seq = self._insert_event(conn, job_id, origin_id, event, payload, now)
            parent_id = None
            if origin_id != job_id:
                pass  # already a fan-out
            else:
                row = conn.execute(
                    "SELECT parent_id FROM jobs WHERE job_id=?", (job_id,)
                ).fetchone()
                if row and row["parent_id"]:
                    parent_id = row["parent_id"]
            if parent_id:
                wrapped = dict(payload or {})
                wrapped.update({"event": event, "child_job_id": job_id})
                self._insert_event(conn, parent_id, job_id, "child", wrapped, now)
            conn.commit()
            return seq
        finally:
            conn.close()

    def events_since(self, job_id: str, after_seq: int = 0) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT seq, event, payload, created_at, origin_id FROM job_events "
                "WHERE job_id=? AND seq>? ORDER BY seq ASC",
                (job_id, after_seq),
            ).fetchall()
            out = []
            for r in rows:
                item = {
                    "seq": r["seq"],
                    "event": r["event"],
                    "created_at": r["created_at"],
                    "origin_id": r["origin_id"],
                }
                if r["payload"]:
                    try:
                        item["payload"] = json.loads(r["payload"])
                    except ValueError:
                        item["payload"] = {"raw": r["payload"]}
                else:
                    item["payload"] = {}
                out.append(item)
            return out
        finally:
            conn.close()

    def latest_seq(self, job_id: str) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq),0) AS s FROM job_events WHERE job_id=?",
                (job_id,),
            ).fetchone()
            return int(row["s"])
        finally:
            conn.close()

    # ── leasing ───────────────────────────────────────────────────────────

    def claim_one(
        self,
        owner: str,
        lease_seconds: float = 30.0,
        now: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Atomically claim one due job.

        Singles are eligible when queued; batch parents become eligible only
        once every child reached a terminal state. Cancellation requests and
        retry backoffs are honored.
        """
        now = time.time() if now is None else now
        expires = now + lease_seconds
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT job_id, resume_stage FROM jobs AS j
                WHERE j.state = ?
                  AND j.cancel_requested = 0
                  AND (j.next_attempt_at IS NULL OR j.next_attempt_at <= ?)
                  AND (
                    j.kind = ?
                    OR (
                        j.kind = ?
                        AND NOT EXISTS (
                            SELECT 1 FROM jobs c
                            WHERE c.parent_id = j.job_id
                              AND c.state NOT IN ('completed', 'failed', 'cancelled')
                        )
                    )
                  )
                ORDER BY j.created_at ASC
                LIMIT 1
                """,
                (
                    JobState.QUEUED.value,
                    now,
                    JobKind.SINGLE.value,
                    JobKind.BATCH.value,
                ),
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            job_id, stage = row["job_id"], row["resume_stage"]
            conn.execute(
                """
                UPDATE jobs
                SET state=?, lease_owner=?, lease_expires_at=?,
                    started_at=COALESCE(started_at, ?), updated_at=?
                WHERE job_id=?
                """,
                (stage, owner, expires, now, now, job_id),
            )
            self._insert_event(
                conn, job_id, job_id, "stage_changed",
                {"to": stage, "lease_owner": owner, "attempt_context": "claimed"}, now,
            )
            conn.commit()
            return self.get(job_id)
        finally:
            conn.close()

    def heartbeat(
        self,
        job_id: str,
        owner: str,
        lease_seconds: float = 30.0,
        now: Optional[float] = None,
    ) -> bool:
        """Extend a lease. Returns False if the lease was lost (e.g. a
        recovery sweep after a long stall) — the executor must stop."""
        now = time.time() if now is None else now
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT lease_owner FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None or row["lease_owner"] != owner:
                conn.commit()
                return False
            conn.execute(
                "UPDATE jobs SET lease_expires_at=?, updated_at=? WHERE job_id=?",
                (now + lease_seconds, now, job_id),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def lease_is_valid(self, job_id: str, owner: str, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT lease_owner, lease_expires_at, state FROM jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            return (
                row is not None
                and row["lease_owner"] == owner
                and row["lease_expires_at"] is not None
                and row["lease_expires_at"] > now
                and row["state"] not in TERMINAL_STATES
            )
        finally:
            conn.close()

    def is_cancel_requested(self, job_id: str) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT cancel_requested FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            return bool(row and row["cancel_requested"])
        finally:
            conn.close()

    # ── state transitions ─────────────────────────────────────────────────

    def commit_stage(
        self,
        job_id: str,
        to_state: str,
        *,
        owner: Optional[str] = None,
        checkpoint: Optional[Dict[str, Any]] = None,
        progress_total: Optional[int] = None,
        progress_done: Optional[int] = None,
        artifacts: Optional[Sequence[Dict[str, Any]]] = None,
        event_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Atomically move to a later stage and commit artifact references.

        Successful stage progress clears retry bookkeeping (attempts reset at
        every durable boundary).
        """
        now = time.time()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFoundError(job_id)
            if owner is not None and row["lease_owner"] != owner:
                raise InvalidTransitionError(
                    f"lease lost for {job_id}: owned by {row['lease_owner']!r}, "
                    f"update by {owner!r}"
                )
            current = row["state"]
            if is_terminal(current):
                raise InvalidTransitionError(f"{job_id} already terminal ({current})")
            if not can_transition(current, to_state):
                raise InvalidTransitionError(f"{job_id}: {current} -> {to_state}")

            new_checkpoint = dict(row["checkpoint"] and json.loads(row["checkpoint"]) or {})
            if checkpoint:
                new_checkpoint.update(checkpoint)

            sets = ["state=?", "resume_stage=?", "checkpoint=?", "updated_at=?",
                    "attempt=0", "next_attempt_at=NULL", "error=NULL", "error_type=NULL"]
            params: List[Any] = [to_state, to_state,
                                 json.dumps(new_checkpoint, default=str, ensure_ascii=False), now]
            if progress_total is not None:
                sets.append("progress_total=?")
                params.append(int(progress_total))
            if progress_done is not None:
                sets.append("progress_done=?")
                params.append(int(progress_done))
            params.append(job_id)
            conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id=?", params)

            for rec in artifacts or []:
                self._insert_artifact(conn, job_id, rec, rec.get("meta"))

            payload = {"from": current, "to": to_state}
            if event_payload:
                payload.update(event_payload)
            self._insert_event(conn, job_id, job_id, "stage_changed", payload, now)
            self._fan_parent_stage(conn, job_id, payload, now)
            conn.commit()
            return self.require(job_id)
        finally:
            conn.close()

    def complete_job(
        self,
        job_id: str,
        *,
        owner: Optional[str] = None,
        result_ref: Optional[str] = None,
        progress_total: Optional[int] = None,
        progress_done: Optional[int] = None,
        artifacts: Optional[Sequence[Dict[str, Any]]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFoundError(job_id)
            # Terminal rows are immutable: the idempotent no-op takes
            # precedence over lease ownership (the lease is released when the
            # terminal transition commits).
            if row["state"] in TERMINAL_STATES:
                conn.commit()
                return self._row_to_dict(row)
            if owner is not None and row["lease_owner"] != owner:
                raise InvalidTransitionError(f"lease lost for {job_id}")
            total = progress_total if progress_total is not None else row["progress_total"]
            done = progress_done if progress_done is not None else max(
                row["progress_done"], total or 0
            )
            conn.execute(
                """
                UPDATE jobs SET state=?, result_ref=COALESCE(?, result_ref),
                    progress_total=?, progress_done=?,
                    lease_owner=NULL, lease_expires_at=NULL,
                    cancel_requested=0, finished_at=?, updated_at=?
                WHERE job_id=?
                """,
                (JobState.COMPLETED.value, result_ref, total, done, now, now, job_id),
            )
            for rec in artifacts or []:
                self._insert_artifact(conn, job_id, rec, rec.get("meta"))
            ev_payload = {"result_ref": result_ref}
            if payload:
                ev_payload.update(payload)
            self._insert_event(conn, job_id, job_id, "completed", ev_payload, now)
            self._fan_parent_terminal(conn, job_id, "completed", ev_payload, now)
            if row["parent_id"]:
                self._refresh_parent_progress(conn, row["parent_id"], now)
            conn.commit()
            return self.require(job_id)
        finally:
            conn.close()

    def fail_job(
        self,
        job_id: str,
        error: str,
        error_type: str,
        *,
        owner: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFoundError(job_id)
            if row["state"] in TERMINAL_STATES:
                conn.commit()
                return self._row_to_dict(row)
            if owner is not None and row["lease_owner"] != owner:
                raise InvalidTransitionError(f"lease lost for {job_id}")
            conn.execute(
                """
                UPDATE jobs SET state=?, error=?, error_type=?,
                    lease_owner=NULL, lease_expires_at=NULL,
                    next_attempt_at=NULL, finished_at=?, updated_at=?
                WHERE job_id=?
                """,
                (JobState.FAILED.value, error, error_type, now, now, job_id),
            )
            ev_payload = {"error": error, "error_type": error_type}
            if payload:
                ev_payload.update(payload)
            self._insert_event(conn, job_id, job_id, "failed", ev_payload, now)
            self._fan_parent_terminal(conn, job_id, "failed", ev_payload, now)
            if row["parent_id"]:
                self._refresh_parent_progress(conn, row["parent_id"], now)
            conn.commit()
            return self.require(job_id)
        finally:
            conn.close()

    def mark_cancelled(
        self,
        job_id: str,
        *,
        owner: Optional[str] = None,
        reason: str = "cancelled",
    ) -> Dict[str, Any]:
        now = time.time()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFoundError(job_id)
            if owner is not None and row["lease_owner"] not in (owner, None):
                raise InvalidTransitionError(f"lease lost for {job_id}")
            if row["state"] in TERMINAL_STATES:
                conn.commit()
                return self._row_to_dict(row)
            conn.execute(
                """
                UPDATE jobs SET state=?, cancel_requested=0,
                    lease_owner=NULL, lease_expires_at=NULL,
                    next_attempt_at=NULL, finished_at=?, updated_at=?
                WHERE job_id=?
                """,
                (JobState.CANCELLED.value, now, now, job_id),
            )
            payload = {"reason": reason}
            self._insert_event(conn, job_id, job_id, "cancelled", payload, now)
            self._fan_parent_terminal(conn, job_id, "cancelled", payload, now)
            if row["parent_id"]:
                self._refresh_parent_progress(conn, row["parent_id"], now)
            conn.commit()
            return self.require(job_id)
        finally:
            conn.close()

    def requeue_for_retry(
        self,
        job_id: str,
        error: str,
        error_type: str,
        *,
        owner: Optional[str] = None,
        retry_after: float = 1.0,
        now: Optional[float] = None,
        reason: str = "error",
    ) -> Dict[str, Any]:
        """Give the lease back to the queue with backoff, or fail permanently.

        The attempt budget resets at every successful :meth:`commit_stage`,
        so ``max_attempts`` effectively bounds consecutive failures of one
        stage.
        """
        now = time.time() if now is None else now
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFoundError(job_id)
            if row["state"] in TERMINAL_STATES:
                conn.commit()
                return self._row_to_dict(row)
            if owner is not None and row["lease_owner"] != owner:
                raise InvalidTransitionError(f"lease lost for {job_id}")

            attempt = int(row["attempt"]) + 1
            resume_stage = row["resume_stage"]
            max_attempts = int(row["max_attempts"])
            if attempt >= max_attempts:
                conn.execute(
                    """
                    UPDATE jobs SET state=?, attempt=?, error=?, error_type=?,
                        lease_owner=NULL, lease_expires_at=NULL,
                        next_attempt_at=NULL, finished_at=?, updated_at=?
                    WHERE job_id=?
                    """,
                    (JobState.FAILED.value, attempt, error, error_type, now, now, job_id),
                )
                payload = {"error": error, "error_type": error_type,
                           "attempt": attempt, "reason": reason}
                self._insert_event(conn, job_id, job_id, "failed", payload, now)
                self._fan_parent_terminal(conn, job_id, "failed", payload, now)
                if row["parent_id"]:
                    self._refresh_parent_progress(conn, row["parent_id"], now)
                conn.commit()
                return self.require(job_id)

            conn.execute(
                """
                UPDATE jobs SET state=?, attempt=?, error=?, error_type=?,
                    lease_owner=NULL, lease_expires_at=NULL,
                    next_attempt_at=?, updated_at=?
                WHERE job_id=?
                """,
                (JobState.QUEUED.value, attempt, error, error_type,
                 now + max(0.0, retry_after), now, job_id),
            )
            payload = {"error": error, "error_type": error_type,
                       "attempt": attempt, "retry_after": retry_after,
                       "resume_stage": resume_stage, "reason": reason}
            self._insert_event(conn, job_id, job_id, "retry_scheduled", payload, now)
            self._fan_parent_stage(conn, job_id, payload, now, event="retry_scheduled")
            conn.commit()
            return self.require(job_id)
        finally:
            conn.close()

    # ── cancellation ──────────────────────────────────────────────────────

    def request_cancel_tree(self, root_id: str) -> Dict[str, Any]:
        """Request cancellation of a job (and, for batches, all live children).

        Unleased queued jobs terminate immediately as ``cancelled``; leased
        jobs get ``cancel_requested=1`` and are terminated by their executor
        at the next cancellation boundary (or by the lease-recovery sweep).
        """
        now = time.time()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            root = conn.execute("SELECT * FROM jobs WHERE job_id=?", (root_id,)).fetchone()
            if root is None:
                raise JobNotFoundError(root_id)

            target_ids = [root_id]
            if root["kind"] == JobKind.BATCH.value:
                kids = conn.execute(
                    "SELECT job_id FROM jobs WHERE parent_id=?", (root_id,)
                ).fetchall()
                target_ids.extend(k["job_id"] for k in kids)

            affected: List[Dict[str, Any]] = []
            for jid in target_ids:
                row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (jid,)).fetchone()
                if row["state"] in TERMINAL_STATES:
                    affected.append({"job_id": jid, "state": row["state"]})
                    continue
                if row["state"] == JobState.QUEUED.value:
                    conn.execute(
                        """
                        UPDATE jobs SET state=?, cancel_requested=0,
                            lease_owner=NULL, lease_expires_at=NULL,
                            next_attempt_at=NULL, finished_at=?, updated_at=?
                        WHERE job_id=?
                        """,
                        (JobState.CANCELLED.value, now, now, jid),
                    )
                    self._insert_event(
                        conn, jid, jid, "cancelled",
                        {"reason": "cancelled while queued"}, now,
                    )
                    self._fan_parent_terminal(
                        conn, jid, "cancelled", {"reason": "cancelled while queued"}, now
                    )
                    if row["parent_id"]:
                        self._refresh_parent_progress(conn, row["parent_id"], now)
                    affected.append({"job_id": jid, "state": JobState.CANCELLED.value})
                else:
                    conn.execute(
                        "UPDATE jobs SET cancel_requested=1, updated_at=? WHERE job_id=?",
                        (now, jid),
                    )
                    self._insert_event(
                        conn, jid, jid, "cancel_requested", {"lease_owner": row["lease_owner"]}, now,
                    )
                    self._fan_parent_stage(
                        conn, jid, {"child_job_id": jid, "event": "cancel_requested"}, now,
                        event="child",
                    )
                    affected.append({"job_id": jid, "state": row["state"],
                                     "cancel_requested": True})

            conn.commit()
            root_after = self.get(root_id)
            return {"root": root_after, "affected": affected}
        finally:
            conn.close()

    # ── chunk checkpoints ─────────────────────────────────────────────────

    def upsert_chunk(
        self,
        job_id: str,
        chunk_index: int,
        status: str,
        *,
        ref: Optional[str] = None,
        content_hash: Optional[str] = None,
        error: Optional[str] = None,
        attempts: Optional[int] = None,
        now: Optional[float] = None,
    ) -> None:
        now = time.time() if now is None else now
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            if attempts is None:
                row = conn.execute(
                    "SELECT attempts FROM job_chunks WHERE job_id=? AND chunk_index=?",
                    (job_id, chunk_index),
                ).fetchone()
                attempts = int(row["attempts"]) + 1 if row else 1
            conn.execute(
                """
                INSERT INTO job_chunks (job_id, chunk_index, status, ref, content_hash,
                                        attempts, error, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id, chunk_index) DO UPDATE SET
                    status=excluded.status,
                    ref=COALESCE(excluded.ref, job_chunks.ref),
                    content_hash=COALESCE(excluded.content_hash, job_chunks.content_hash),
                    attempts=excluded.attempts,
                    error=excluded.error,
                    updated_at=excluded.updated_at
                """,
                (job_id, chunk_index, status, ref, content_hash, attempts, error, now),
            )
            conn.commit()
        finally:
            conn.close()

    def get_chunks(self, job_id: str) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM job_chunks WHERE job_id=? ORDER BY chunk_index ASC",
                (job_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def completed_chunk_indices(self, job_id: str) -> Dict[int, Dict[str, Any]]:
        return {
            int(r["chunk_index"]): dict(r)
            for r in self.get_chunks(job_id)
            if r["status"] == "completed"
        }

    def set_progress(self, job_id: str, total: int, done: int) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE jobs SET progress_total=?, progress_done=?, updated_at=? WHERE job_id=?",
                (int(total), int(done), time.time(), job_id),
            )
            conn.commit()
        finally:
            conn.close()

    # ── recovery ──────────────────────────────────────────────────────────

    def reconcile(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Sweep dead work. Safe to run from every worker loop.

        * leased jobs whose lease expired -> requeue (attempt+1, short backoff)
          or fail when the attempt budget is exhausted;
        * queued jobs with a pending cancel flag -> cancelled;
        * cancel-requested batch parents whose children all terminated ->
          cancelled.
        """
        now = time.time() if now is None else now
        actions: List[Dict[str, Any]] = []
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            stale = conn.execute(
                """
                SELECT * FROM jobs
                WHERE state IN ('acquiring','transcribing','summarizing','finalizing')
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at < ?
                """,
                (now,),
            ).fetchall()
            for row in stale:
                if row["cancel_requested"]:
                    self._do_cancel_locked(conn, row, "worker lease expired after cancel", now)
                    actions.append({"job_id": row["job_id"], "action": "cancelled"})
                    continue
                attempt = int(row["attempt"]) + 1
                retry_after = min(60.0, 2.0 ** max(0, attempt - 1))
                if attempt >= int(row["max_attempts"]):
                    conn.execute(
                        """
                        UPDATE jobs SET state='failed', attempt=?,
                            error='worker lease expired; stage never completed',
                            error_type='LeaseExpired',
                            lease_owner=NULL, lease_expires_at=NULL,
                            next_attempt_at=NULL, finished_at=?, updated_at=?
                        WHERE job_id=?
                        """,
                        (attempt, now, now, row["job_id"]),
                    )
                    payload = {"error_type": "LeaseExpired", "attempt": attempt,
                               "reason": "lease_expired"}
                    self._insert_event(conn, row["job_id"], row["job_id"],
                                       "failed", payload, now)
                    self._fan_parent_terminal(
                        conn, row["job_id"], "failed", payload, now
                    )
                    if row["parent_id"]:
                        self._refresh_parent_progress(conn, row["parent_id"], now)
                    actions.append({"job_id": row["job_id"], "action": "failed"})
                else:
                    conn.execute(
                        """
                        UPDATE jobs SET state='queued', attempt=?,
                            error='worker lease expired; resuming from checkpoint',
                            error_type='LeaseExpired',
                            lease_owner=NULL, lease_expires_at=NULL,
                            next_attempt_at=?, updated_at=?
                        WHERE job_id=?
                        """,
                        (attempt, now + retry_after, now, row["job_id"]),
                    )
                    payload = {"attempt": attempt, "retry_after": retry_after,
                               "resume_stage": row["resume_stage"],
                               "reason": "lease_expired"}
                    self._insert_event(
                        conn, row["job_id"], row["job_id"], "retry_scheduled", payload, now
                    )
                    self._fan_parent_stage(
                        conn, row["job_id"], payload, now, event="retry_scheduled"
                    )
                    actions.append({"job_id": row["job_id"], "action": "requeued",
                                    "resume_stage": row["resume_stage"]})

            # queued singletons that carry a cancel flag (cancel raced with claim)
            flagged = conn.execute(
                "SELECT * FROM jobs WHERE state='queued' AND cancel_requested=1",
            ).fetchall()
            for row in flagged:
                if row["kind"] == JobKind.BATCH.value:
                    pending = conn.execute(
                        "SELECT COUNT(1) AS n FROM jobs WHERE parent_id=? "
                        "AND state NOT IN ('completed','failed','cancelled')",
                        (row["job_id"],),
                    ).fetchone()["n"]
                    if pending:
                        continue
                self._do_cancel_locked(conn, row, "cancel requested", now)
                actions.append({"job_id": row["job_id"], "action": "cancelled"})

            conn.commit()
            return actions
        finally:
            conn.close()

    @staticmethod
    def _do_cancel_locked(
        conn: sqlite3.Connection, row: sqlite3.Row, reason: str, now: float
    ) -> None:
        conn.execute(
            """
            UPDATE jobs SET state='cancelled', cancel_requested=0,
                lease_owner=NULL, lease_expires_at=NULL,
                next_attempt_at=NULL, finished_at=?, updated_at=?
            WHERE job_id=?
            """,
            (now, now, row["job_id"]),
        )
        JobStore._insert_event(
            conn, row["job_id"], row["job_id"], "cancelled", {"reason": reason}, now
        )
        JobStore._fan_parent_terminal(
            conn, row["job_id"], "cancelled", {"reason": reason}, now
        )
        if row["parent_id"]:
            JobStore._refresh_parent_progress(conn, row["parent_id"], now)

    # ── batch helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _refresh_parent_progress(
        conn: sqlite3.Connection, parent_id: str, now: float
    ) -> None:
        total = conn.execute(
            "SELECT COUNT(1) AS n FROM jobs WHERE parent_id=?", (parent_id,)
        ).fetchone()["n"]
        done = conn.execute(
            "SELECT COUNT(1) AS n FROM jobs WHERE parent_id=? "
            "AND state IN ('completed','failed','cancelled')",
            (parent_id,),
        ).fetchone()["n"]
        conn.execute(
            "UPDATE jobs SET progress_total=?, progress_done=?, updated_at=? WHERE job_id=?",
            (total, done, now, parent_id),
        )

    @staticmethod
    def _fan_parent_terminal(
        conn: sqlite3.Connection,
        child_id: str,
        event: str,
        payload: Dict[str, Any],
        now: float,
    ) -> None:
        row = conn.execute("SELECT parent_id FROM jobs WHERE job_id=?", (child_id,)).fetchone()
        if not row or not row["parent_id"]:
            return
        wrapped = dict(payload or {})
        wrapped.update({"event": event, "child_job_id": child_id})
        JobStore._insert_event(conn, row["parent_id"], child_id, "child", wrapped, now)

    @staticmethod
    def _fan_parent_stage(
        conn: sqlite3.Connection,
        child_id: str,
        payload: Dict[str, Any],
        now: float,
        event: str = "child",
    ) -> None:
        row = conn.execute("SELECT parent_id FROM jobs WHERE job_id=?", (child_id,)).fetchone()
        if not row or not row["parent_id"]:
            return
        wrapped = dict(payload or {})
        wrapped.setdefault("event", "stage_changed")
        wrapped["child_job_id"] = child_id
        JobStore._insert_event(conn, row["parent_id"], child_id, event, wrapped, now)
