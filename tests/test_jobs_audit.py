"""B2 audit corrections: shutdown races, startup ownership, task failures, blocking inspection, cursor integrity.

Real-process tests run harmless local child interpreters in a temporary directory. Waits are bounded.
Teardown reopens the runtime, stops it, and kills every process the test created.
"""

import asyncio
import gc
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import warnings
from types import SimpleNamespace

import pytest

from src import config, envelope, jobs
from src import job_output
from src.tools import shell
from tests.test_jobs import _alive, _gone

PY = sys.executable

HELPERS = {
    "sleeper.py": "import time\ntime.sleep(60)\n",
    "ignore_term.py": ("import os, signal, time\n"
                       "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                       "open('pid', 'w').write(str(os.getpid()))\n"
                       "time.sleep(60)\n"),
    "leader_spawns.py": ("import subprocess, sys\n"
                         "p = subprocess.Popen([sys.executable, 'grandchild.py'], stdin=subprocess.DEVNULL,\n"
                         "                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                         "open('gc', 'w').write(str(p.pid))\n"
                         "print('leader done', flush=True)\n"),
    "grandchild.py": "import time\ntime.sleep(60)\n",
    "burst.py": ("import sys, time\n"
                 "sys.stdout.write('a' * 100 + '\\n'); sys.stdout.flush()\n"
                 "time.sleep(1.0)\n"
                 "sys.stdout.write('b' * 200000); sys.stdout.flush()\n"
                 "time.sleep(60)\n"),
}


@pytest.fixture
def audit_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "ALLOWED_TOOLS", {PY, "echo"})
    monkeypatch.setattr(config, "ALLOW_ALL", False)
    monkeypatch.setattr(config, "JOB_GRACE_SECONDS", 0.3)
    monkeypatch.setattr(config, "JOB_KILL_CONFIRM_SECONDS", 0.5)
    for name, body in HELPERS.items():
        (tmp_path / name).write_text(body)
    created = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        created.append(proc)
        return proc

    # The job module sees a Popen that records every process it creates; ps still uses the real subprocess
    monkeypatch.setattr(jobs, "subprocess", SimpleNamespace(
        Popen=recording_popen, DEVNULL=subprocess.DEVNULL, PIPE=subprocess.PIPE))
    yield SimpleNamespace(path=tmp_path, created=created)
    jobs._reset_for_tests()
    for proc in created:
        if proc.poll() is None:
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            proc.wait()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync(tool, args, owner="sess-A"):
    jobs_token = jobs.bind_owner(owner)
    try:
        return jobs.call_tool_sync(tool, args)
    finally:
        jobs.unbind_owner(jobs_token)


def _json(result):
    assert not result.get("isError"), result
    return json.loads(result["content"])


def _start(command, owner="sess-A", **extra):
    return _json(_sync("start_job", {"command": command, **extra}, owner))["job_id"]


def _status(job_id, owner="sess-A"):
    return _json(_sync("status_job", {"job_id": job_id}, owner))


def _wait_state(job_id, states, seconds=25.0, owner="sess-A"):
    end = time.monotonic() + seconds
    status = _status(job_id, owner)
    while status["state"] not in states and time.monotonic() < end:
        time.sleep(0.05)
        status = _status(job_id, owner)
    assert status["state"] in states, f"state stayed {status['state']}: {status['reason']}"
    return status


def _table():
    runtime = jobs._inspect()
    return runtime.manager.jobs if runtime is not None else {}


def _pid_file(path, name="pid", seconds=10.0):
    end = time.monotonic() + seconds
    while not (path / name).exists() and time.monotonic() < end:
        time.sleep(0.02)
    return int((path / name).read_text())


def _new_proc(audit_env):
    """The most recently created process by the job module (its leader)."""
    return audit_env.created[-1]


# ---------------------------------------------------------------------------
# 1. Runtime shutdown race
# ---------------------------------------------------------------------------

@pytest.mark.real_process
def test_concurrent_acquisition_and_shutdown_never_creates_a_replacement_runtime(audit_env, monkeypatch):
    created = []
    real_factory = jobs._GATE._factory

    def counted():
        created.append(1)
        return real_factory()

    monkeypatch.setattr(jobs._GATE, "_factory", counted)
    results, errors = [], []

    def hammer(index):
        try:
            token = jobs.bind_owner(f"sess-{index}")
            try:
                for _ in range(6):
                    results.append(jobs.call_tool_sync("start_job", {"command": f"{PY} sleeper.py"}))
                    time.sleep(0.01)
            finally:
                jobs.unbind_owner(token)
        except Exception as e:  # any exception here is a race bug
            errors.append(e)

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    unresolved = jobs.shutdown_runtime(timeout=30)
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert all(isinstance(r, dict) and "isError" in r for r in results)
    assert unresolved == []
    assert len(created) == 1, "shutdown must never be followed by a replacement runtime"
    assert all(job.state == "terminated" for job in _table().values() if job.state != "failed")
    for job in _table().values():
        if job.proc is not None:
            assert _gone(job.proc.pid)


@pytest.mark.real_process
def test_submission_during_shutdown_is_refused_until_shutdown_completes(audit_env, monkeypatch):
    release = threading.Event()
    real_shutdown = jobs._Manager.shutdown

    async def paused_shutdown(self, owner, reason):
        await asyncio.to_thread(release.wait, 15)
        return await real_shutdown(self, owner, reason)

    monkeypatch.setattr(jobs._Manager, "shutdown", paused_shutdown)
    _start(f"{PY} sleeper.py")
    closer = threading.Thread(target=jobs.shutdown_runtime, kwargs={"timeout": 30})
    closer.start()

    refused = None
    end = time.monotonic() + 10
    while time.monotonic() < end:
        result = _sync("start_job", {"command": f"{PY} sleeper.py"})
        if result.get("isError") and "shutting down" in result["content"]:
            refused = result
            break
        time.sleep(0.02)
    assert refused is not None, "a job must be refused while shutdown is underway"

    release.set()
    closer.join(timeout=30)
    after = _sync("start_job", {"command": f"{PY} sleeper.py"})
    assert after["isError"] is True and "shut down" in after["content"]


@pytest.mark.real_process
def test_repeated_shutdown_is_idempotent_and_performs_one_shutdown(audit_env, monkeypatch):
    calls = []
    real_shutdown = jobs._Manager.shutdown

    async def counted(self, owner, reason):
        calls.append(owner)
        return await real_shutdown(self, owner, reason)

    monkeypatch.setattr(jobs._Manager, "shutdown", counted)
    _start(f"{PY} sleeper.py")
    outcomes = []
    threads = [threading.Thread(target=lambda: outcomes.append(jobs.shutdown_runtime(timeout=30))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    again = jobs.shutdown_runtime(timeout=30)

    assert len(outcomes) == 4 and all(o == [] for o in outcomes)
    assert again == []
    assert calls == [None], "exactly one caller performs the runtime shutdown"


@pytest.mark.real_process
def test_shutdown_racing_an_explicit_termination_leaves_the_job_terminal(audit_env):
    job_id = _start(f"{PY} ignore_term.py")
    pid = _pid_file(audit_env.path)
    outcomes = []

    def terminate():
        token = jobs.bind_owner("sess-A")
        try:
            outcomes.append(jobs.call_tool_sync("terminate_job", {"job_id": job_id}))
        finally:
            jobs.unbind_owner(token)

    t1 = threading.Thread(target=terminate)
    t2 = threading.Thread(target=lambda: jobs.shutdown_runtime(timeout=30))
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    assert len(outcomes) == 1 and isinstance(outcomes[0], dict)
    assert _gone(pid)
    assert _table()[job_id].state == "terminated"


def test_no_runtime_threads_remain_after_shutdown(audit_env):
    _start(f"{PY} sleeper.py")
    jobs.shutdown_runtime(timeout=30)
    leftover = [t.name for t in threading.enumerate() if t.name.startswith("agent-loop-jobs") and t.is_alive()]
    assert leftover == [], f"orphaned runtime threads: {leftover}"


# ---------------------------------------------------------------------------
# 2. Process ownership after reader failure
# ---------------------------------------------------------------------------

@pytest.mark.real_process
def test_stdout_reader_failure_leaves_no_live_process_and_no_partial_job(audit_env, monkeypatch):
    def broken(pipe):
        raise OSError("stdout reader could not be attached")

    monkeypatch.setattr(shell, "_open_reader", broken)
    result = _sync("start_job", {"command": f"{PY} sleeper.py"})

    assert result["isError"] is True
    assert "could not be started" in result["content"] or "stopped" in result["content"]
    proc = _new_proc(audit_env)
    assert proc.poll() is not None, "the started process must be stopped"
    assert _table() == {}, "a partially initialized job must never be exposed"


@pytest.mark.real_process
def test_stderr_reader_failure_closes_the_stdout_transport_too(audit_env, monkeypatch):
    real_open = shell._open_reader
    transports = []
    attempts = {"n": 0}

    async def second_fails(pipe):
        attempts["n"] += 1
        if attempts["n"] == 2:
            raise OSError("stderr reader could not be attached")
        reader, transport = await real_open(pipe)
        transports.append(transport)
        return reader, transport

    monkeypatch.setattr(shell, "_open_reader", second_fails)
    result = _sync("start_job", {"command": f"{PY} sleeper.py"})

    assert result["isError"] is True
    assert len(transports) == 1 and transports[0].is_closing(), "the already-attached transport must be closed"
    assert _new_proc(audit_env).poll() is not None
    assert _table() == {}


@pytest.mark.real_process
def test_unconfirmed_startup_cleanup_keeps_the_handle_and_shutdown_finalizes_it(audit_env, monkeypatch):
    state = {"deny": True}
    real_signal = shell._signal_group
    monkeypatch.setattr(shell, "_signal_group", lambda pgid, sig: False if state["deny"] else real_signal(pgid, sig))

    def broken(pipe):
        raise OSError("reader could not be attached")

    monkeypatch.setattr(shell, "_open_reader", broken)
    result = _sync("start_job", {"command": f"{PY} ignore_term.py"})

    assert result["isError"] is True
    assert result["errorCategory"] == "unknown_outcome", "an unconfirmed stop must not read as a clean failure"
    proc = _new_proc(audit_env)
    assert proc.poll() is None, "the unconfirmed process is still alive and must remain tracked"
    runtime = jobs._inspect()
    assert len(runtime.manager.retained) == 1
    assert [j.state for j in _table().values()] == ["unknown"]

    state["deny"] = False
    unresolved = jobs.shutdown_runtime(timeout=30)

    assert unresolved == []
    assert _gone(proc.pid)
    assert runtime.manager.retained == []


@pytest.mark.real_process
def test_registry_insertion_failure_leaves_no_unmanaged_live_process(audit_env, monkeypatch):
    def refuse_registration(self, job):
        raise RuntimeError("registry insertion failed")

    monkeypatch.setattr(jobs._Manager, "_register", refuse_registration)
    result = _sync("start_job", {"command": f"{PY} sleeper.py"})

    assert result["isError"] is True
    proc = _new_proc(audit_env)
    assert proc.poll() is not None, "no live process may exist outside the registry"
    assert _table() == {}


# ---------------------------------------------------------------------------
# 3. Worker task exception handling
# ---------------------------------------------------------------------------

def _no_unretrieved_exception_logs(caplog):
    gc.collect()
    return [r for r in caplog.records if "exception was never retrieved" in r.getMessage()] == []


@pytest.mark.real_process
def test_pump_failure_stops_the_job_and_is_recorded(audit_env, monkeypatch, caplog):
    caplog.set_level(logging.ERROR)

    async def broken_pump(reader, stream):
        raise RuntimeError("pump failed unexpectedly")

    monkeypatch.setattr(jobs, "_pump", broken_pump)
    job_id = _start(f"{PY} sleeper.py")
    status = _wait_state(job_id, {"terminated", "unknown"}, seconds=25)

    assert status["state"] == "terminated", status["reason"]
    assert "pump failed" in status["reason"]
    assert any(d["task"] == "pump" for d in jobs._inspect().manager.diagnostics)
    assert _no_unretrieved_exception_logs(caplog)


@pytest.mark.real_process
def test_watchdog_failure_stops_the_job_instead_of_leaving_it_unmanaged(audit_env, monkeypatch, caplog):
    caplog.set_level(logging.ERROR)

    async def broken_watchdog(self, job):
        raise RuntimeError("watchdog failed unexpectedly")

    monkeypatch.setattr(jobs._Manager, "_watchdog", broken_watchdog)
    job_id = _start(f"{PY} sleeper.py", lifetime_seconds=600)
    status = _wait_state(job_id, {"terminated", "unknown"}, seconds=25)

    assert status["state"] == "terminated", status["reason"]
    assert "watchdog failed" in status["reason"]
    assert any(d["task"] == "watchdog" for d in jobs._inspect().manager.diagnostics)
    assert _no_unretrieved_exception_logs(caplog)


@pytest.mark.real_process
def test_supervisor_failure_stops_the_job_instead_of_leaving_it_unmanaged(audit_env, monkeypatch, caplog):
    caplog.set_level(logging.ERROR)

    async def broken_supervisor(self, job):
        raise RuntimeError("supervisor failed unexpectedly")

    monkeypatch.setattr(jobs._Manager, "_supervise", broken_supervisor)
    job_id = _start(f"{PY} sleeper.py")
    status = _wait_state(job_id, {"terminated", "unknown"}, seconds=25)

    assert status["state"] == "terminated", status["reason"]
    assert any(d["task"] == "supervisor" for d in jobs._inspect().manager.diagnostics)
    assert _no_unretrieved_exception_logs(caplog)


@pytest.mark.real_process
def test_janitor_failure_is_recorded_and_the_janitor_keeps_running(audit_env, monkeypatch, caplog):
    caplog.set_level(logging.ERROR)
    calls = {"n": 0}
    real = jobs._Manager._retry_retained

    async def fails_once(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("janitor pass failed")
        return await real(self)

    monkeypatch.setattr(jobs._Manager, "_retry_retained", fails_once)
    _sync("status_job", {"job_id": "job_" + "0" * 32})  # starts the runtime and its janitor
    end = time.monotonic() + 6
    while calls["n"] < 3 and time.monotonic() < end:
        time.sleep(0.05)

    assert any(d["task"] == "janitor" for d in jobs._inspect().manager.diagnostics)
    assert calls["n"] >= 3, "the janitor must survive a failing pass and keep running"
    assert _no_unretrieved_exception_logs(caplog)


@pytest.mark.real_process
def test_worker_shutdown_with_a_failing_task_completes_and_reports(audit_env, monkeypatch, caplog):
    caplog.set_level(logging.ERROR)

    async def broken_pump(reader, stream):
        raise RuntimeError("pump failed during shutdown")

    monkeypatch.setattr(jobs, "_pump", broken_pump)
    _start(f"{PY} sleeper.py")
    unresolved = jobs.shutdown_runtime(timeout=30)

    assert unresolved == []
    assert _no_unretrieved_exception_logs(caplog)
    assert any(d["task"] == "pump" for d in jobs._inspect().manager.diagnostics)


# ---------------------------------------------------------------------------
# 4. Blocking membership inspection
# ---------------------------------------------------------------------------

def _slow_membership(monkeypatch, seconds):
    real = shell._group_members

    def slow(pgid):
        time.sleep(seconds)
        return real(pgid)

    monkeypatch.setattr(shell, "_group_members", slow)


@pytest.mark.real_process
def test_status_stays_responsive_while_the_janitor_inspects_slowly(audit_env, monkeypatch):
    # A job whose leader exits while a same-group grandchild lives: its leader stays retained, so the
    # janitor inspects the group every second. Each inspection is slow.
    state = {"deny": True}
    real_signal = shell._signal_group
    monkeypatch.setattr(shell, "_signal_group", lambda pgid, sig: False if state["deny"] else real_signal(pgid, sig))
    _slow_membership(monkeypatch, 1.0)
    job_id = _start(f"{PY} leader_spawns.py")
    _wait_state(job_id, {"completed"}, seconds=25)

    durations = []
    end = time.monotonic() + 3.0
    while time.monotonic() < end:
        started = time.monotonic()
        _status(job_id)
        durations.append(time.monotonic() - started)
        time.sleep(0.1)
    assert max(durations) < 0.6, f"a status call was blocked for {max(durations):.2f}s by inspection"

    state["deny"] = False
    grandchild = int((audit_env.path / "gc").read_text())
    jobs.shutdown_runtime(timeout=30)
    assert _gone(grandchild)


@pytest.mark.real_process
def test_watchdog_deadline_is_observed_under_slow_inspection(audit_env, monkeypatch):
    _slow_membership(monkeypatch, 1.0)
    started = time.monotonic()
    job_id = _start(f"{PY} sleeper.py", lifetime_seconds=1)
    time.sleep(1.8)
    status = _status(job_id)  # responsive even while the watchdog's stop is waiting on inspection
    assert status["state"] in ("running", "terminated", "unknown")
    _wait_state(job_id, {"terminated", "unknown"}, seconds=40)
    assert time.monotonic() - started < 40
    assert "lifetime limit reached" in _status(job_id)["reason"]


@pytest.mark.real_process
def test_concurrent_status_and_termination_both_complete(audit_env, monkeypatch):
    _slow_membership(monkeypatch, 0.5)
    job_id = _start(f"{PY} ignore_term.py")
    _pid_file(audit_env.path)
    seen = []

    def poll():
        for _ in range(10):
            seen.append(_sync("status_job", {"job_id": job_id}))
            time.sleep(0.05)

    poller = threading.Thread(target=poll)
    poller.start()
    stopped = _sync("terminate_job", {"job_id": job_id})
    poller.join(timeout=30)

    assert _json(stopped)["state"] in ("terminated", "unknown")
    assert len(seen) == 10 and all("isError" in r for r in seen)


@pytest.mark.real_process
def test_membership_inspection_failure_is_unknown_not_terminated(audit_env, monkeypatch):
    real_members = shell._group_members
    monkeypatch.setattr(shell, "_group_members", lambda pgid: None)
    job_id = _start(f"{PY} ignore_term.py")
    _pid_file(audit_env.path)
    status = _json(_sync("terminate_job", {"job_id": job_id}))

    assert status["state"] == "unknown", "a failed membership check must never prove the group is empty"
    assert "could not be confirmed" in status["reason"]
    proc = _new_proc(audit_env)

    monkeypatch.setattr(shell, "_group_members", real_members)  # inspection works again
    assert jobs.shutdown_runtime(timeout=30) == []
    assert _gone(proc.pid)


@pytest.mark.real_process
def test_runtime_shutdown_while_inspection_is_pending_completes_within_bounds(audit_env, monkeypatch):
    _slow_membership(monkeypatch, 2.0)
    _start(f"{PY} ignore_term.py")
    _pid_file(audit_env.path)
    proc = _new_proc(audit_env)
    started = time.monotonic()
    unresolved = jobs.shutdown_runtime(timeout=60)

    assert time.monotonic() - started < 60
    assert unresolved == []
    assert _gone(proc.pid)


# ---------------------------------------------------------------------------
# 5. Job-bound, authenticated cursors
# ---------------------------------------------------------------------------

@pytest.mark.real_process
def test_cursor_from_another_job_is_rejected_without_revealing_it(audit_env):
    first = _start(f"{PY} sleeper.py")
    second = _start(f"{PY} sleeper.py", owner="sess-A")
    page = _json(_sync("collect_job", {"job_id": first}))
    foreign = _sync("collect_job", {"job_id": second, "cursor": page["cursor"]})

    assert foreign["isError"] is True
    assert "cursor is not valid for this job" in foreign["content"]
    assert first not in foreign["content"] and second not in foreign["content"]
    jobs.shutdown_runtime(timeout=30)


@pytest.mark.real_process
def test_stale_cursor_after_ring_buffer_truncation_reports_drops(audit_env, monkeypatch):
    monkeypatch.setattr(config, "JOB_OUTPUT_BUFFER_BYTES", 4096)
    job_id = _start(f"{PY} burst.py")
    early = None
    end = time.monotonic() + 15
    while early is None and time.monotonic() < end:
        page = _json(_sync("collect_job", {"job_id": job_id}))
        if "a" * 50 in page["stdout"]:
            early = page["cursor"]
        time.sleep(0.05)
    assert early is not None
    while _status(job_id)["stdout_bytes_written"] < 200_000 and time.monotonic() < end:
        time.sleep(0.05)

    stale = _json(_sync("collect_job", {"job_id": job_id, "cursor": early}))
    assert stale["stdout_dropped_bytes"] > 0 and stale["truncated"] is True
    assert len(stale["stdout"].encode()) <= 16 << 10
    _sync("terminate_job", {"job_id": job_id})


@pytest.mark.parametrize("token", [
    "not-a-cursor",
    "!!!!",
    "A" * 300,
    "",
    "YWJj",  # base64 with no authentication
])
def test_corrupt_cursors_are_rejected_with_one_answer(audit_env, token):
    result = _sync("collect_job", {"job_id": "job_" + "0" * 32, "cursor": token})
    assert result["isError"] is True
    assert "cursor is not valid for this job" in result["content"] or "No job" in result["content"]


@pytest.mark.real_process
def test_tampered_cursor_is_rejected(audit_env):
    job_id = _start(f"{PY} sleeper.py")
    cursor = _json(_sync("collect_job", {"job_id": job_id}))["cursor"]
    flipped = cursor[:-2] + ("A" if cursor[-2] != "A" else "B") + cursor[-1]
    result = _sync("collect_job", {"job_id": job_id, "cursor": flipped})
    assert result["isError"] is True and "cursor is not valid for this job" in result["content"]
    jobs.shutdown_runtime(timeout=30)


def test_cursor_codec_binds_offsets_to_one_job():
    job_a, job_b = "job_" + "a" * 32, "job_" + "b" * 32
    token = job_output.encode_cursor(job_a, 10, 20)
    assert job_output.decode_cursor(token, job_a) == (10, 20)
    assert job_output.decode_cursor(token, job_b) is None
    assert job_output.decode_cursor(token + "x", job_a) is None


@pytest.mark.real_process
def test_collection_after_normal_completion_returns_the_remaining_output(audit_env):
    job_id = _start(f"{PY} -c \"print('finished output')\"")
    _wait_state(job_id, {"completed"})
    page = _json(_sync("collect_job", {"job_id": job_id}))
    assert "finished output" in page["stdout"]
    again = _json(_sync("collect_job", {"job_id": job_id, "cursor": page["cursor"]}))
    assert again["stdout"] == "" and again["more"] is False
