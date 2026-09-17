"""Tests for the persistent job state machine, store, executor and worker.

Covers:
* the eight-state machine transition table;
* JobStore: snapshots, idempotency, leases, atomic stage commits, retries,
  cancellation trees, crash recovery (reconcile) and chunk checkpoints;
* the executor pipeline: stage resume and "completed model chunks are never
  billed twice";
* batch parent dependency + aggregation;
* the JobRunner facade and the FastAPI /jobs + SSE surface.
"""

import asyncio
import json
import time
from unittest.mock import patch

import pytest

from summarizer.exceptions import APIError
from summarizer.jobs import JobKind, JobState
from summarizer.jobs.bus import EventBus
from summarizer.jobs.executor import JobContext, JobCancelled, execute_batch, execute_job
from summarizer.jobs.runner import JobRunner
from summarizer.jobs.states import (
    ACTIVE_STATES,
    STAGES,
    TERMINAL_STATES,
    can_transition,
    is_terminal,
    stage_index,
)
from summarizer.jobs.store import (
    InvalidTransitionError,
    JobStore,
)


OWNER = "test-worker-1"


def make_config(**overrides):
    cfg = {
        "base_url": "https://api.example.com/v1",
        "model": "test-model",
        "source_url_or_path": "https://example.com/watch?v=abc",
        "chunk_size": 500,
        "parallel_api_calls": 1,
        "cache_transcript": False,
        "prompt_type": "Questions and answers",
    }
    cfg.update(overrides)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

class TestStateMachine:
    def test_eight_states_and_sets(self):
        values = {s.value for s in JobState}
        assert values == {
            "queued", "acquiring", "transcribing", "summarizing",
            "finalizing", "completed", "failed", "cancelled",
        }
        assert len(TERMINAL_STATES) == 3
        assert len(ACTIVE_STATES) == 5
        assert STAGES == (
            "acquiring", "transcribing", "summarizing", "finalizing"
        )
        assert stage_index("queued") == -1
        assert stage_index("finalizing") == 3

    def test_valid_forward_edges(self):
        assert can_transition("queued", "acquiring")
        # a claim may jump straight to a checkpointed later stage
        assert can_transition("queued", "summarizing")
        assert can_transition("acquiring", "transcribing")
        assert can_transition("transcribing", "finalizing")
        assert can_transition("summarizing", "summarizing")  # idempotent commit
        for state in ACTIVE_STATES:
            for terminal in TERMINAL_STATES:
                assert can_transition(state, terminal)

    def test_invalid_edges(self):
        assert not can_transition("summarizing", "acquiring")  # no going back
        assert not can_transition("finalizing", "queued")
        for terminal in TERMINAL_STATES:
            assert not can_transition(terminal, "acquiring")
            assert not is_terminal("queued")
        assert is_terminal("completed")


# ─────────────────────────────────────────────────────────────────────────────
# Store
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs")


@pytest.fixture
def runner(store):
    r = JobRunner(store=store, start_embedded_worker=False)
    yield r
    r.shutdown()


def create_single(store, key=None, config=None, **kw):
    return store.create_job(
        config=config or make_config(),
        kind=JobKind.SINGLE.value,
        source="src-1",
        output_format="markdown",
        idempotency_key=key,
        **kw
    )


class TestJobStoreSubmission:
    def test_create_job_snapshots_input(self, store):
        row, hit = create_single(store)
        assert hit is False
        assert row["state"] == JobState.QUEUED.value
        assert row["snapshot_version"] == 1
        assert len(row["snapshot_hash"]) == 64  # sha256 hex
        snap = store.read_config_snapshot(row["snapshot_hash"])
        assert snap["model"] == "test-model"

    def test_snapshot_is_content_addressed_and_versioned(self, store):
        cfg_a = make_config()
        cfg_b = make_config(model="other-model")
        row_a, _ = create_single(store, config=cfg_a)
        row_b, _ = create_single(store, config=dict(cfg_a))
        row_c, _ = create_single(store, config=cfg_b)
        # identical inputs share one snapshot blob
        assert row_a["snapshot_hash"] == row_b["snapshot_hash"]
        assert row_a["snapshot_hash"] != row_c["snapshot_hash"]

    def test_idempotency_key_returns_existing_job(self, store):
        row1, hit1 = create_single(store, key="key-1")
        row2, hit2 = create_single(store, key="key-1", config=make_config(model="x"))
        assert hit1 is False
        assert hit2 is True
        assert row1["job_id"] == row2["job_id"]
        # the original snapshot is kept; the duplicate request is ignored
        assert store.read_config_snapshot(row2["snapshot_hash"])["model"] == "test-model"

    def test_queued_event_is_emitted(self, store):
        row, _ = create_single(store)
        events = store.events_since(row["job_id"])
        assert events[0]["event"] == "queued"
        assert events[0]["seq"] == 1


class TestLeasing:
    def test_claim_atomic_and_ordered(self, store):
        row1, _ = create_single(store)
        row2, _ = create_single(store)
        claimed = store.claim_one(OWNER, lease_seconds=30)
        assert claimed["job_id"] == row1["job_id"]
        assert claimed["state"] == JobState.ACQUIRING.value
        assert claimed["lease_owner"] == OWNER
        assert claimed["lease_expires_at"] > time.time()
        # second claim gets the other job, not the already leased one
        second = store.claim_one("other")
        assert second["job_id"] == row2["job_id"]
        assert store.claim_one("third") is None

    def test_claim_respects_backoff(self, store):
        row, _ = create_single(store)
        claimed = store.claim_one(OWNER)
        store.requeue_for_retry(
            row["job_id"], "boom", "APIError", owner=OWNER, retry_after=30
        )
        assert store.claim_one(OWNER, now=time.time() + 10) is None
        future = store.claim_one(OWNER, now=time.time() + 60)
        assert future is not None and future["job_id"] == row["job_id"]

    def test_claim_skips_cancel_requested(self, store):
        row, _ = create_single(store)
        conn = store._connect()
        try:
            conn.execute(
                "UPDATE jobs SET cancel_requested=1 WHERE job_id=?", (row["job_id"],)
            )
            conn.commit()
        finally:
            conn.close()
        assert store.claim_one(OWNER) is None

    def test_heartbeat_and_lease_validity(self, store):
        row, _ = create_single(store)
        store.claim_one(OWNER, lease_seconds=30)
        assert store.lease_is_valid(row["job_id"], OWNER)
        assert not store.lease_is_valid(row["job_id"], "impostor")
        assert store.heartbeat(row["job_id"], OWNER, lease_seconds=30) is True
        # another owner cannot extend the lease
        assert store.heartbeat(row["job_id"], "impostor") is False


class TestStageCommit:
    def test_commit_stage_atomic_artifacts_and_attempt_reset(self, store):
        row, _ = create_single(store)
        store.claim_one(OWNER)
        store.requeue_for_retry(
            row["job_id"], "early", "E", owner=OWNER, retry_after=5
        )
        store.claim_one(OWNER, now=time.time() + 10)
        assert store.require(row["job_id"])["attempt"] == 1

        rec = store.write_stage_file(
            row["job_id"], JobState.ACQUIRING.value, "source.txt", "hello"
        )
        updated = store.commit_stage(
            row["job_id"],
            JobState.TRANSCRIBING.value,
            owner=OWNER,
            checkpoint={"source_kind": "text"},
            artifacts=[rec],
        )
        assert updated["state"] == JobState.TRANSCRIBING.value
        assert updated["resume_stage"] == JobState.TRANSCRIBING.value
        assert updated["attempt"] == 0  # successful boundary resets retries
        assert updated["checkpoint"] == {"source_kind": "text"}
        got = store.get_artifact(
            row["job_id"], JobState.ACQUIRING.value, "source.txt"
        )
        assert got is not None
        assert store.read_artifact_text(got["ref"]) == "hello"

    def test_commit_stage_rejects_backward_transition(self, store):
        row, _ = create_single(store)
        store.claim_one(OWNER)
        store.commit_stage(row["job_id"], JobState.SUMMARIZING.value, owner=OWNER)
        with pytest.raises(InvalidTransitionError):
            store.commit_stage(
                row["job_id"], JobState.ACQUIRING.value, owner=OWNER
            )

    def test_commit_stage_rejects_wrong_owner(self, store):
        row, _ = create_single(store)
        store.claim_one(OWNER)
        with pytest.raises(InvalidTransitionError):
            store.commit_stage(
                row["job_id"], JobState.TRANSCRIBING.value, owner="someone-else"
            )

    def test_complete_with_result_links_artifact(self, store):
        row, _ = create_single(store)
        store.claim_one(OWNER)
        final = store.complete_with_result(row["job_id"], "FINAL", owner=OWNER)
        assert final["state"] == JobState.COMPLETED.value
        assert final["lease_owner"] is None
        assert final["finished_at"] is not None
        assert store.read_artifact_text(final["result_ref"]) == "FINAL"
        # terminal writes are idempotent no-ops
        again = store.complete_with_result(row["job_id"], "OTHER", owner=OWNER)
        assert store.read_artifact_text(again["result_ref"]) == "FINAL"


class TestRetry:
    def test_backoff_then_exhaustion(self, store):
        row, _ = create_single(store, max_attempts=2)
        store.claim_one(OWNER)
        once = store.requeue_for_retry(
            row["job_id"], "transient", "APIError", owner=OWNER, retry_after=5
        )
        assert once["state"] == JobState.QUEUED.value
        assert once["attempt"] == 1
        assert once["next_attempt_at"] is not None

        store.claim_one(OWNER, now=time.time() + 10)
        dead = store.requeue_for_retry(
            row["job_id"], "still failing", "APIError", owner=OWNER
        )
        assert dead["state"] == JobState.FAILED.value
        assert dead["error"] == "still failing"
        assert dead["error_type"] == "APIError"
        assert dead["attempt"] == 2


class TestCancellation:
    def test_cancel_queued_single_is_immediate(self, runner):
        row, _ = runner.submit_single(make_config(), source="s")
        view = runner.cancel(row["job_id"])
        assert view["state"] == JobState.CANCELLED.value
        assert store_of(runner).claim_one(OWNER) is None

    def test_cancel_leased_job_sets_flag(self, store):
        row, _ = create_single(store)
        store.claim_one(OWNER)
        result = store.request_cancel_tree(row["job_id"])
        assert result["root"]["state"] == JobState.ACQUIRING.value
        assert store.is_cancel_requested(row["job_id"])
        # executor observes the flag at the next boundary
        ctx = JobContext(store, store.require(row["job_id"]), OWNER)
        with pytest.raises(JobCancelled):
            ctx.check_cancelled()
        final = store.mark_cancelled(row["job_id"], owner=OWNER)
        assert final["state"] == JobState.CANCELLED.value

    def test_cancel_batch_cancels_children(self, runner):
        items = [
            {"source": "a", "config": make_config(source_url_or_path="a")},
            {"source": "b", "config": make_config(source_url_or_path="b")},
        ]
        parent, children, _ = runner.submit_batch(items, parent_config=make_config())
        runner.cancel(parent["job_id"])
        assert runner.get_job(parent["job_id"])["state"] == JobState.CANCELLED.value
        for child in children:
            assert runner.get_job(child["job_id"])["state"] == JobState.CANCELLED.value


def store_of(runner):
    return runner.store


class TestReconcile:
    def test_expired_lease_resumes_at_checkpoint(self, store):
        row, _ = create_single(store, max_attempts=3)
        store.claim_one(OWNER, lease_seconds=30)
        store.commit_stage(
            row["job_id"], JobState.SUMMARIZING.value, owner=OWNER
        )
        # worker dies: lease goes stale
        conn = store._connect()
        try:
            conn.execute(
                "UPDATE jobs SET lease_expires_at=? WHERE job_id=?",
                (time.time() - 5, row["job_id"]),
            )
            conn.commit()
        finally:
            conn.close()

        actions = store.reconcile()
        assert actions == [{
            "job_id": row["job_id"],
            "action": "requeued",
            "resume_stage": JobState.SUMMARIZING.value,
        }]
        resumed = store.require(row["job_id"])
        assert resumed["state"] == JobState.QUEUED.value
        assert resumed["lease_owner"] is None
        assert resumed["attempt"] == 1
        # next claim jumps straight back to the checkpointed stage
        claimed = store.claim_one(OWNER, now=time.time() + 10)
        assert claimed["state"] == JobState.SUMMARIZING.value

    def test_expired_lease_exhausts_attempt_budget(self, store):
        row, _ = create_single(store, max_attempts=1)
        store.claim_one(OWNER)
        conn = store._connect()
        try:
            conn.execute(
                "UPDATE jobs SET lease_expires_at=? WHERE job_id=?",
                (time.time() - 5, row["job_id"]),
            )
            conn.commit()
        finally:
            conn.close()
        actions = store.reconcile()
        assert actions[0]["action"] == "failed"
        assert store.require(row["job_id"])["state"] == JobState.FAILED.value


class TestChunkCheckpoints:
    def test_chunk_rows_track_completion(self, store):
        row, _ = create_single(store)
        store.upsert_chunk(row["job_id"], 0, "completed", ref="artifacts/0.txt")
        store.upsert_chunk(row["job_id"], 1, "failed", error="rate limit")
        store.upsert_chunk(row["job_id"], 2, "empty")

        done = store.completed_chunk_indices(row["job_id"])
        assert set(done) == {0}
        # a failed/empty chunk is still eligible for a retry on resume
        all_rows = {r["chunk_index"]: r for r in store.get_chunks(row["job_id"])}
        assert all_rows[1]["attempts"] == 1
        store.upsert_chunk(row["job_id"], 1, "completed", ref="artifacts/1.txt")
        assert set(store.completed_chunk_indices(row["job_id"])) == {0, 1}


class TestBatchDependency:
    def test_parent_blocked_until_children_terminal(self, runner):
        s = runner.store
        items = [
            {"source": "a", "config": make_config(source_url_or_path="a")},
            {"source": "b", "config": make_config(source_url_or_path="b")},
        ]
        parent, children, _ = runner.submit_batch(items, parent_config=make_config())

        # children are claimed first; the parent is never eligible meanwhile
        first = s.claim_one(OWNER)
        assert first["job_id"] in {c["job_id"] for c in children}
        s.complete_with_result(first["job_id"], "summary A", owner=OWNER)

        second = s.claim_one(OWNER)
        assert second["job_id"] != parent["job_id"]
        s.fail_job(second["job_id"], "bad source", "SourceNotFoundError", owner=OWNER)

        batch_claim = s.claim_one(OWNER)
        assert batch_claim["job_id"] == parent["job_id"]
        assert batch_claim["state"] == JobState.FINALIZING.value

        ctx = JobContext(s, batch_claim, OWNER, bus=runner.bus)
        aggregate = execute_batch(ctx)
        assert aggregate["success_count"] == 1
        assert aggregate["total_count"] == 2
        done = s.require(parent["job_id"])
        assert done["state"] == JobState.COMPLETED.value
        payload = json.loads(s.read_artifact_text(done["result_ref"]))
        statuses = {(r["source"], r["success"]) for r in payload["results"]}
        assert statuses == {("a", True), ("b", False)}


class TestEventLog:
    def test_parent_receives_child_events(self, runner):
        items = [{"source": "a", "config": make_config(source_url_or_path="a")}]
        parent, children, _ = runner.submit_batch(items, parent_config=make_config())
        runner.store.append_event(children[0]["job_id"], "progress", {"done": 1})
        stream = runner.events_since(parent["job_id"])
        kinds = [e["event"] for e in stream]
        assert "child" in kinds
        child_events = [e for e in stream if e["event"] == "child"]
        assert child_events[-1]["payload"]["child_job_id"] == children[0]["job_id"]


# ─────────────────────────────────────────────────────────────────────────────
# Executor: resume semantics and per-chunk billing
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_TEXTS = {
    0: "alpha " * 100,
    1: "bravo " * 100,
    2: "charlie " * 100,
}


def _patched_pipeline(process_chunk_impl):
    """Patch all external boundaries of the executor pipeline."""
    chunks = [(f"00:{i:02d}:00", text) for i, text in CHUNK_TEXTS.items()]
    return [
        patch(
            "summarizer.jobs.executor.acquire_transcript_source",
            return_value=("text", "WORD " * 200),
        ),
        patch(
            "summarizer.jobs.executor.extract_and_clean_chunks",
            return_value=chunks,
        ),
        patch(
            "summarizer.jobs.executor.load_prompt_template",
            return_value="TEMPLATE",
        ),
        patch(
            "summarizer.jobs.executor.format_summary_with_timestamps",
            side_effect=lambda summaries, cfg: "\n".join(
                text for _ts, text in summaries
            ),
        ),
        patch(
            "summarizer.jobs.executor.process_chunk",
            side_effect=process_chunk_impl,
        ),
    ]


class TestExecutorPipeline:
    def test_text_source_happy_path(self, runner):
        s = runner.store
        row, _ = runner.submit_single(make_config(), source="src")
        claimed = s.claim_one(OWNER, lease_seconds=60)

        calls = []

        async def fake_process(text, template, cfg):
            calls.append(text)
            return f"SUM[{text.split()[0]}]"

        patches = _patched_pipeline(fake_process)
        for p in patches:
            p.start()
        try:
            raw = execute_job(s.read_config_snapshot(claimed["snapshot_hash"]),
                              JobContext(s, claimed, OWNER, bus=runner.bus))
            s.complete_with_result(row["job_id"], raw, owner=OWNER)
        finally:
            for p in patches:
                p.stop()

        final = s.require(row["job_id"])
        assert final["state"] == JobState.COMPLETED.value
        assert len(calls) == 3
        assert set(calls) == set(CHUNK_TEXTS.values())
        # every finished chunk has a durable artifact + checkpoint row
        done = s.completed_chunk_indices(row["job_id"])
        assert set(done) == {0, 1, 2}
        for rec in done.values():
            assert s.read_artifact_text(rec["ref"]).startswith("SUM[")

    def test_completed_chunks_are_not_rebilled_after_crash(self, runner):
        s = runner.store
        row, _ = runner.submit_single(
            make_config(), source="src", max_attempts=3
        )

        call_counts = {0: 0, 1: 0, 2: 0}
        text_to_index = {text: i for i, text in CHUNK_TEXTS.items()}

        async def fake_process(text, template, cfg):
            index = text_to_index[text]
            call_counts[index] += 1
            # chunk 1 fails during the first (crashed) attempt
            if index == 1 and call_counts[index] == 1:
                raise APIError("simulated model rate limit")
            return f"SUM-{index}"

        async def expire_lease_during_last_chunk(text, template, cfg):
            if text_to_index[text] == 2:
                # simulate worker process death while the stage is running
                conn = s._connect()
                try:
                    conn.execute(
                        "UPDATE jobs SET lease_expires_at=? WHERE job_id=?",
                        (time.time() - 60, row["job_id"]),
                    )
                    conn.commit()
                finally:
                    conn.close()
            return await fake_process(text, template, cfg)

        patches = _patched_pipeline(expire_lease_during_last_chunk)

        # ── first lease: process chunks 0,1,2 then crash at the stage gate ──
        claimed = s.claim_one(OWNER, lease_seconds=60)
        ctx = JobContext(s, claimed, OWNER, bus=runner.bus)
        for p in patches:
            p.start()
        with pytest.raises(Exception):
            execute_job(s.read_config_snapshot(claimed["snapshot_hash"]), ctx)

        # recovery sweep: expired lease -> resume from summarizing checkpoint
        actions = s.reconcile()
        assert actions[0]["resume_stage"] == JobState.SUMMARIZING.value
        for p in patches:
            p.stop()

        # ── second lease: only the unfinished chunk may be billed again ────
        claimed2 = s.claim_one("worker-2", lease_seconds=60, now=time.time() + 10)
        ctx2 = JobContext(s, claimed2, "worker-2", bus=runner.bus)
        for p in patches:
            p.start()
        try:
            raw = execute_job(
                s.read_config_snapshot(claimed2["snapshot_hash"]), ctx2
            )
            s.complete_with_result(row["job_id"], raw, owner="worker-2")
        finally:
            for p in patches:
                p.stop()

        assert s.require(row["job_id"])["state"] == JobState.COMPLETED.value
        # completed chunks (0, 2) ran exactly once; the failed chunk 1 retried
        assert call_counts == {0: 1, 1: 2, 2: 1}
        assert set(s.completed_chunk_indices(row["job_id"])) == {0, 1, 2}
        result = s.read_artifact_text(s.require(row["job_id"])["result_ref"])
        assert {"SUM-0", "SUM-1", "SUM-2"} <= set(result.splitlines())


# ─────────────────────────────────────────────────────────────────────────────
# Runner facade
# ─────────────────────────────────────────────────────────────────────────────

class TestRunnerFacade:
    def test_wait_for_terminal_and_view(self, runner):
        row, _ = runner.submit_single(make_config(), source="src")

        def complete_soon():
            time.sleep(0.05)
            claimed = runner.store.claim_one("late-worker", lease_seconds=30)
            assert claimed["job_id"] == row["job_id"]
            runner.store.complete_with_result(
                row["job_id"], "done", owner="late-worker"
            )

        import threading
        t = threading.Thread(target=complete_soon)
        t.start()
        final = asyncio.run(
            runner.wait_for_terminal(row["job_id"], timeout=5)
        )
        t.join()
        assert final["state"] == JobState.COMPLETED.value

        view = runner.view(row["job_id"])
        assert view["summary"] == "done"
        assert view["input_snapshot"]["version"] == 1
        assert len(view["input_snapshot"]["hash"]) == 64

    def test_wait_timeout(self, runner):
        row, _ = runner.submit_single(make_config(), source="src")
        with pytest.raises(TimeoutError):
            asyncio.run(
                runner.wait_for_terminal(row["job_id"], timeout=0.05,
                                         poll_interval=0.02)
            )

    def test_batch_idempotency_replays_parent(self, runner):
        items = [{"source": "a", "config": make_config(source_url_or_path="a")}]
        parent1, children1, hit1 = runner.submit_batch(
            items, parent_config=make_config(), idempotency_key="batch-1"
        )
        parent2, children2, hit2 = runner.submit_batch(
            items, parent_config=make_config(), idempotency_key="batch-1"
        )
        assert hit1 is False and hit2 is True
        assert parent1["job_id"] == parent2["job_id"]
        assert children1[0]["job_id"] == children2[0]["job_id"]
        assert children1[0]["idempotency_key"] == "batch-1:child:0"


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI surface (/jobs + SSE)
# ─────────────────────────────────────────────────────────────────────────────

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from summarizer.server import create_app  # noqa: E402
from summarizer.jobs.runner import reset_runner  # noqa: E402


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("SUMMARIZER_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("SUMMARIZER_JOB_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("SUMMARIZER_SYNC_WAIT_TIMEOUT", "30")
    runner = reset_runner()  # embedded worker enabled
    client = TestClient(create_app())
    yield client
    client.close()
    runner.shutdown()


class TestJobsAPI:
    PAYLOAD = {"source": "/tmp/x", "base_url": "https://api.example/v1", "model": "m"}

    def test_enqueue_get_cancel_lifecycle(self, api):
        # deterministic queue/cancel semantics: no worker claims in this test
        from summarizer.jobs.runner import get_runner
        get_runner().shutdown()

        r = api.post("/jobs", json=self.PAYLOAD)
        assert r.status_code == 202
        body = r.json()
        job_id = body["job_id"]
        assert body["state"] == "queued"
        assert body["idempotent_hit"] is False

        r2 = api.post(
            "/jobs",
            json=self.PAYLOAD,
            headers={"Idempotency-Key": "fixed-key"},
        )
        r3 = api.post(
            "/jobs",
            json=dict(self.PAYLOAD, source="/tmp/other"),
            headers={"Idempotency-Key": "fixed-key"},
        )
        assert r3.json()["idempotent_hit"] is True
        assert r2.json()["job_id"] == r3.json()["job_id"]

        listed = api.get("/jobs").json()
        assert any(j["job_id"] == job_id for j in listed)
        assert api.get(f"/jobs/{job_id}").json()["job_id"] == job_id
        assert api.get("/jobs/nope").status_code == 404

        # a fresh queued job can be cancelled immediately
        rc = api.post(
            "/jobs",
            json=dict(self.PAYLOAD, source="/tmp/cancel-me"),
            headers={"Idempotency-Key": "cancel-key"},
        )
        cancel_id = rc.json()["job_id"]
        cancelled = api.post(f"/jobs/{cancel_id}/cancel").json()
        assert cancelled["state"] == "cancelled"

    def test_end_to_end_completion_and_sse(self, api):
        with patch(
            "summarizer.jobs.executor.execute_job",
            return_value="API SUMMARY",
        ):
            enq = api.post(
                "/jobs",
                json=dict(self.PAYLOAD, source="https://example.com/v"),
            )
            job_id = enq.json()["job_id"]

            # poll the view until the embedded worker completes the job
            deadline = time.time() + 10
            view = None
            while time.time() < deadline:
                view = api.get(f"/jobs/{job_id}").json()
                if view["state"] in ("completed", "failed"):
                    break
                time.sleep(0.05)
            assert view["state"] == "completed"
            assert view["summary"] == "API SUMMARY"

            # SSE: replay the durable event log from the beginning; the stream
            # closes on its own after the terminal event
            with api.stream("GET", f"/jobs/{job_id}/events") as resp:
                assert resp.status_code == 200
                assert resp.headers["content-type"].startswith("text/event-stream")
                text = b"".join(resp.iter_bytes())
            assert b"event: queued" in text
            assert b"event: completed" in text
            assert job_id.encode() in text

    def test_sse_resume_after_last_event_id(self, api):
        with patch(
            "summarizer.jobs.executor.execute_job",
            return_value="X",
        ):
            job_id = api.post(
                "/jobs", json=dict(self.PAYLOAD, source="s")
            ).json()["job_id"]
            deadline = time.time() + 10
            while api.get(f"/jobs/{job_id}").json()["state"] != "completed":
                time.sleep(0.05)
                assert time.time() < deadline

        with api.stream(
            "GET", f"/jobs/{job_id}/events?last_event_id=1"
        ) as resp:
            text = b"".join(resp.iter_bytes())
        # seq 1 (queued) must be skipped, later events retained
        assert b"id: 1\n" not in text
        assert b"event: completed" in text
