"""B2 final lifecycle corrections.

1. A startup that fails after its process exists returns the retained job id when cleanup is unconfirmed,
   and no id otherwise. Its owner can inspect and stop it; other sessions cannot.
2. A runtime is CLOSED only after its worker thread has actually stopped. A timed-out shutdown keeps the
   worker owned and refuses work, and it never permits a second worker.

Blocking is real: a stuck shutdown blocks the worker's event loop, and the test releases it explicitly.
"""

import asyncio
import json
import re
import threading
import time

import pytest

from src import jobs, session
from src.job_runtime import WorkerRuntime
from tests.test_jobs import _run_session, _tool_message
from tests.test_jobs_audit import (  # noqa: F401  (audit_env is the fixture; the rest are helpers)
    PY, _alive, _gone, _json, _new_proc, _start, _status, _sync, _table, audit_env,
)
from src.tools import shell


def _worker_threads():
    return [t for t in threading.enumerate() if t.name == "agent-loop-jobs" and t.is_alive()]


def _count_runtimes(monkeypatch):
    created = []
    real_factory = jobs._GATE._factory

    def counted():
        created.append(1)
        return real_factory()

    monkeypatch.setattr(jobs._GATE, "_factory", counted)
    return created


def _deny_signals(monkeypatch):
    state = {"deny": True}
    real = shell._signal_group
    monkeypatch.setattr(shell, "_signal_group", lambda pgid, sig: False if state["deny"] else real(pgid, sig))
    return state


def _break_readers(monkeypatch):
    def broken(pipe):
        raise OSError("reader could not be attached")

    monkeypatch.setattr(shell, "_open_reader", broken)


# ---------------------------------------------------------------------------
# 1. Startup failure: the response contract
# ---------------------------------------------------------------------------

@pytest.mark.real_process
def test_failure_before_subprocess_creation_returns_an_error_without_an_id(audit_env, monkeypatch):
    def refuse_to_spawn(*args, **kwargs):
        raise OSError("no such executable")

    monkeypatch.setattr(jobs.subprocess, "Popen", refuse_to_spawn)
    result = _sync("start_job", {"command": f"{PY} sleeper.py"})

    assert result["isError"] is True
    assert result["errorCategory"] == "validation"
    assert "job_" not in result["content"]
    assert _table() == {}


@pytest.mark.real_process
def test_failure_after_creation_with_confirmed_cleanup_returns_an_error_without_an_id(audit_env, monkeypatch):
    _break_readers(monkeypatch)
    result = _sync("start_job", {"command": f"{PY} sleeper.py"})

    assert result["isError"] is True
    assert result["errorCategory"] == "validation", "a confirmed cleanup is a definitive failure"
    assert "job_" not in result["content"]
    assert "Nothing is left running" in result["content"]
    assert all(p.poll() is not None for p in audit_env.created)
    assert _table() == {}


@pytest.mark.real_process
def test_failure_after_creation_with_unconfirmed_cleanup_returns_the_retained_id(audit_env, monkeypatch):
    signals = _deny_signals(monkeypatch)
    _break_readers(monkeypatch)
    result = _sync("start_job", {"command": f"{PY} ignore_term.py"})

    assert result["isError"] is True
    assert result["errorCategory"] == "unknown_outcome"
    assert result["isRetryable"] is False
    payload = json.loads(result["content"])
    job_id = payload["job_id"]
    assert re.fullmatch(r"job_[0-9a-f]{32}", job_id)
    assert payload["state"] == "unknown"
    assert _alive(_new_proc(audit_env).pid), "the process is unconfirmed, so it stays tracked and alive"

    status = _json(_sync("status_job", {"job_id": job_id}))
    assert status["state"] == "unknown"
    assert status["group"] == "unconfirmed"

    signals["deny"] = False


@pytest.mark.real_process
def test_owner_can_inspect_and_terminate_an_unknown_startup_job(audit_env, monkeypatch):
    signals = _deny_signals(monkeypatch)
    _break_readers(monkeypatch)
    result = _sync("start_job", {"command": f"{PY} ignore_term.py"})
    job_id = json.loads(result["content"])["job_id"]
    proc = _new_proc(audit_env)

    signals["deny"] = False  # the stop can now be confirmed
    stopped = _json(_sync("terminate_job", {"job_id": job_id}))

    assert stopped["state"] == "terminated", "a verified stop on request is reported as terminated"
    assert "confirmed stopped on retry" in stopped["reason"]
    assert _gone(proc.pid)
    assert jobs._inspect().manager.retained == []


@pytest.mark.real_process
def test_other_sessions_cannot_reach_a_retained_startup_job(audit_env, monkeypatch):
    signals = _deny_signals(monkeypatch)
    _break_readers(monkeypatch)
    result = _sync("start_job", {"command": f"{PY} ignore_term.py"}, owner="sess-A")
    job_id = json.loads(result["content"])["job_id"]

    for tool, args in (("status_job", {"job_id": job_id}),
                       ("collect_job", {"job_id": job_id}),
                       ("terminate_job", {"job_id": job_id})):
        denied = _sync(tool, args, owner="sess-B")
        assert denied["isError"] is True
        assert "belongs to this session" in denied["content"]
        assert job_id not in denied["content"]
    assert _alive(_new_proc(audit_env).pid), "a foreign session must not stop the job"

    signals["deny"] = False
    _sync("terminate_job", {"job_id": job_id}, owner="sess-A")


@pytest.mark.real_process
def test_journal_and_tool_envelope_preserve_unknown_and_the_id(audit_env, monkeypatch, tmp_sessions):
    signals = _deny_signals(monkeypatch)
    _break_readers(monkeypatch)
    post = _run_session(monkeypatch, [
        _start_reply("call_unknown"),
        _final_reply(),
    ], session_id="unknown-startup")

    message = json.loads(_raw_tool_message(post.payloads[1], "call_unknown"))
    assert message["errorCategory"] == "unknown_outcome"
    assert message["isRetryable"] is False
    payload = json.loads(message["content"])
    job_id = payload["job_id"]
    assert payload["state"] == "unknown"

    record = session.load_session("unknown-startup")["execution"][0]
    assert record["status"] == "unknown", "the Phase A journal must record UNKNOWN, not completed or failed"
    assert job_id in record["outcome"]

    after_run = _json(_sync("status_job", {"job_id": job_id}, owner="unknown-startup"))
    assert after_run["state"] == "unknown", "ending the session must not fabricate a confirmed termination"

    signals["deny"] = False
    _sync("terminate_job", {"job_id": job_id}, owner="unknown-startup")


def _start_reply(call_id):
    from tests.conftest import make_streamed_response
    return make_streamed_response(finish_reason="tool_calls", tool_calls=[
        {"id": call_id, "name": "start_job", "arguments": json.dumps({"command": f"{PY} ignore_term.py"})}])


def _final_reply():
    from tests.conftest import make_streamed_response
    return make_streamed_response(finish_reason="stop", content="done")


def _raw_tool_message(payload, call_id):
    return next(m["content"] for m in payload["messages"]
                if m.get("role") == "tool" and m.get("tool_call_id") == call_id)


# ---------------------------------------------------------------------------
# 2. Shutdown: CLOSED only after the worker has stopped
# ---------------------------------------------------------------------------

def test_worker_stop_reports_false_while_the_thread_is_still_running():
    worker = WorkerRuntime("test-stop-false")
    release = threading.Event()

    async def blocker():
        release.wait(20)  # blocks the worker loop, as a stuck inspection would

    worker.submit(blocker())
    time.sleep(0.05)
    assert worker.stop(timeout=0.2) is False
    assert worker.thread.is_alive()
    release.set()
    assert worker.stop(timeout=10) is True
    assert not worker.thread.is_alive()


@pytest.fixture
def stuck_shutdown(audit_env, monkeypatch):
    """A runtime whose manager shutdown blocks the worker until the test releases it."""
    monkeypatch.setattr(jobs, "_STOP_SECONDS", 0.2)
    release = threading.Event()
    real_shutdown = jobs._Manager.shutdown

    async def stuck(self, owner, reason):
        await asyncio.to_thread(release.wait, 30)  # a stuck inspection holding the worker
        return await real_shutdown(self, owner, reason)

    monkeypatch.setattr(jobs._Manager, "shutdown", stuck)
    _sync("status_job", {"job_id": "job_" + "0" * 32})  # starts the runtime
    yield release
    release.set()
    jobs.shutdown_runtime(timeout=20)


def test_timed_out_shutdown_keeps_the_worker_owned_and_the_runtime_closing(stuck_shutdown):
    release = stuck_shutdown
    started = time.monotonic()
    unresolved = jobs.shutdown_runtime(timeout=0.5)

    assert time.monotonic() - started < 10, "a timed-out shutdown is bounded"
    assert any(entry["job_id"] is None for entry in unresolved), "the incomplete shutdown is reported"
    assert jobs._GATE.state() == "closing", "CLOSED must not be published while the worker runs"
    assert jobs._inspect().worker_alive(), "the original worker is still owned, not forgotten"
    release.set()


def test_reopen_before_the_worker_exits_is_refused(stuck_shutdown, audit_env):
    jobs.shutdown_runtime(timeout=0.5)
    with pytest.raises(RuntimeError):
        jobs.reopen_runtime()
    refused = _sync("start_job", {"command": f"{PY} sleeper.py"})
    assert refused["isError"] is True and "shutting down" in refused["content"]


def test_repeated_shutdown_after_a_timeout_is_bounded_and_creates_no_worker(stuck_shutdown, monkeypatch):
    created = _count_runtimes(monkeypatch)
    for _ in range(3):
        started = time.monotonic()
        assert any(e["job_id"] is None for e in jobs.shutdown_runtime(timeout=0.3))
        assert time.monotonic() - started < 10
    assert created == [], "no replacement runtime while the old worker is alive"
    assert len(_worker_threads()) == 1, "exactly one worker thread exists"


def test_concurrent_shutdown_and_acquisition_during_an_incomplete_attempt(stuck_shutdown, monkeypatch):
    created = _count_runtimes(monkeypatch)
    jobs.shutdown_runtime(timeout=0.3)  # first attempt, incomplete
    results, errors = [], []

    def acquire_repeatedly():
        try:
            for _ in range(5):
                results.append(_sync("start_job", {"command": f"{PY} sleeper.py"}))
        except Exception as e:  # any exception is a race bug
            errors.append(e)

    def concurrent_shutdown():
        try:
            results.append(jobs.shutdown_runtime(timeout=0.3))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=acquire_repeatedly), threading.Thread(target=concurrent_shutdown)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    refusals = [r for r in results if isinstance(r, dict)]
    assert refusals and all(r["isError"] for r in refusals)
    assert created == [], "acquisition during an incomplete shutdown must not create a runtime"


def test_later_successful_shutdown_transitions_to_closed(stuck_shutdown):
    release = stuck_shutdown
    runtime = jobs._inspect()
    jobs.shutdown_runtime(timeout=0.3)
    assert jobs._GATE.state() == "closing"

    release.set()
    unresolved = jobs.shutdown_runtime(timeout=20)

    assert jobs._GATE.state() == "closed"
    assert not runtime.worker_alive()
    assert not any(e["job_id"] is None for e in unresolved)


def test_reopen_is_permitted_only_after_the_worker_has_stopped(stuck_shutdown, monkeypatch):
    release = stuck_shutdown
    created = _count_runtimes(monkeypatch)
    jobs.shutdown_runtime(timeout=0.3)
    with pytest.raises(RuntimeError):
        jobs.reopen_runtime()

    release.set()
    jobs.shutdown_runtime(timeout=20)
    jobs.reopen_runtime()
    _sync("status_job", {"job_id": "job_" + "0" * 32})

    assert created == [1], "a new runtime starts only after the explicit reopen"
    assert len(_worker_threads()) == 1
