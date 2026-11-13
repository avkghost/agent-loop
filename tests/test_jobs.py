"""B2 managed background jobs: lifecycle, ownership, bounded output, watchdog, shutdown, and cleanup.

Real-process tests run harmless child interpreters inside a temporary directory. Every wait is bounded,
and teardown stops the runtime and kills anything left behind.
"""

import asyncio
import copy
import gc
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import warnings

import pytest

from src import config, jobs, loop, session
from src.tools import shell
from src.tools.mcp_client import MCPClientManager
from tests.conftest import make_streamed_response

PY = sys.executable

HELPERS = {
    "sleeper.py": "import time\ntime.sleep(60)\n",
    "ignore_term.py": ("import os, signal, time\n"
                       "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                       "open('pid', 'w').write(str(os.getpid()))\n"
                       "time.sleep(60)\n"),
    "chatty.py": ("import sys\n"
                  "for i in range(3000):\n"
                  "    sys.stdout.write(('line %05d ' % i) + 'x' * 1000 + '\\n')\n"
                  "sys.stdout.flush()\n"),
    "printer.py": ("import sys, time\n"
                   "print('first', flush=True)\n"
                   "time.sleep(0.6)\n"
                   "print('second', flush=True)\n"
                   "time.sleep(60)\n"),
    "grandchild.py": "import time\ntime.sleep(60)\n",
    "leader_spawns.py": ("import subprocess, sys\n"
                         "p = subprocess.Popen([sys.executable, 'grandchild.py'], stdin=subprocess.DEVNULL,\n"
                         "                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                         "open('gc', 'w').write(str(p.pid))\n"
                         "print('leader done', flush=True)\n"),
}


@pytest.fixture
def jobs_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "ALLOWED_TOOLS", {PY, "echo"})
    monkeypatch.setattr(config, "ALLOW_ALL", False)
    monkeypatch.setattr(config, "JOB_GRACE_SECONDS", 0.3)
    monkeypatch.setattr(config, "JOB_KILL_CONFIRM_SECONDS", 0.5)
    for name, body in HELPERS.items():
        (tmp_path / name).write_text(body)
    yield tmp_path
    jobs._reset_for_tests()
    for pid_file in tmp_path.glob("*"):
        if pid_file.name in ("pid", "gc"):
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except (ValueError, OSError):
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _gone(pid: int, seconds: float = 5.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


def _sync(tool, args, owner="sess-A"):
    token = jobs.bind_owner(owner)
    try:
        return jobs.call_tool_sync(tool, args)
    finally:
        jobs.unbind_owner(token)


def _async_call(tool, args, owner="sess-A"):
    async def go():
        token = jobs.bind_owner(owner)
        try:
            return await jobs.call_tool(tool, args)
        finally:
            jobs.unbind_owner(token)
    return go()


def _json(result):
    assert not result.get("isError"), result
    return json.loads(result["content"])


def _start(command, owner="sess-A", **extra):
    return _json(_sync("start_job", {"command": command, **extra}, owner))["job_id"]


def _status(job_id, owner="sess-A"):
    return _json(_sync("status_job", {"job_id": job_id}, owner))


def _wait_state(job_id, states, seconds=20.0, owner="sess-A"):
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


def _record(job_id):
    return _table()[job_id]


def _pid(tmp_path, name="pid"):
    deadline = time.monotonic() + 10
    while not (tmp_path / name).exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    return int((tmp_path / name).read_text())


# ---------------------------------------------------------------------------
# Contract: input validation, approval, and command rules (no processes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool,args", [
    ("start_job", {}),
    ("start_job", {"command": "   "}),
    ("start_job", {"command": "echo hi", "unexpected": 1}),
    ("start_job", {"command": "echo hi", "lifetime_seconds": 0}),
    ("start_job", {"command": "echo hi", "lifetime_seconds": True}),
    ("start_job", {"command": "echo hi", "lifetime_seconds": 99999}),
    ("status_job", {"job_id": "not-a-job"}),
    ("status_job", {}),
    ("collect_job", {"job_id": "job_" + "a" * 32, "cursor": "banana"}),
    ("terminate_job", {"job_id": 7}),
])
def test_invalid_job_inputs_are_rejected_before_anything_starts(jobs_env, tool, args):
    result = _sync(tool, args)
    assert result["isError"] is True
    assert _table() == {}


def test_background_and_detaching_commands_are_rejected_before_any_process_starts(jobs_env, monkeypatch):
    spawned = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: spawned.append(a) or (_ for _ in ()).throw(AssertionError))
    for command in ("sleep 100 &", "nohup sleep 100", "setsid sleep 100"):
        result = _sync("start_job", {"command": command})
        assert result["isError"] is True
    assert spawned == []


def test_shell_operators_require_approval_and_start_nothing(jobs_env):
    result = _sync("start_job", {"command": f"{PY} sleeper.py | cat"})
    assert result.get("requiresPermission") is True
    assert result.get("kind") == "shell_command"
    assert _table() == {}


def test_jobs_are_refused_outside_an_agent_run(jobs_env):
    result = jobs.call_tool_sync("start_job", {"command": f"{PY} sleeper.py"})
    assert result["isError"] is True
    assert "agent run" in result["content"]


def test_missing_ps_refuses_before_any_process_starts(jobs_env, monkeypatch):
    real_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None if name == "ps" else real_which(name, *a, **k))
    result = _sync("start_job", {"command": f"{PY} -c \"open('started','w').close()\""})
    assert result["isError"] is True
    assert "'ps'" in result["content"]
    assert not (jobs_env / "started").exists()


def test_job_tools_are_advertised_with_strict_schemas():
    from src.tools import get_local_tools

    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in get_local_tools()
               if t["function"]["name"] in jobs.TOOL_NAMES}
    assert set(schemas) == jobs.TOOL_NAMES
    for schema in schemas.values():
        assert schema["additionalProperties"] is False
    assert schemas["start_job"]["required"] == ["command"]
    assert schemas["collect_job"]["required"] == ["job_id"]


# ---------------------------------------------------------------------------
# Starting, running, and ownership
# ---------------------------------------------------------------------------

@pytest.mark.real_process
def test_job_id_is_opaque_and_the_job_survives_across_separate_calls(jobs_env):
    job_id = _start(f"{PY} sleeper.py")
    assert re.fullmatch(r"job_[0-9a-f]{32}", job_id), "the id must be opaque, not a PID"
    assert _status(job_id)["state"] == "running"
    time.sleep(0.5)
    assert _status(job_id)["state"] == "running", "a job must stay alive between calls"
    _sync("terminate_job", {"job_id": job_id})


@pytest.mark.real_process
def test_foreground_command_runs_while_a_job_stays_active(jobs_env):
    job_id = _start(f"{PY} sleeper.py")
    foreground = asyncio.run(shell.execute_command_async(f"{PY} -c \"print('foreground ok')\"", 20))
    assert "foreground ok" in foreground["content"]
    assert _status(job_id)["state"] == "running"
    _sync("terminate_job", {"job_id": job_id})


@pytest.mark.real_process
def test_jobs_are_owned_by_one_session(jobs_env):
    job_id = _start(f"{PY} sleeper.py", owner="sess-A")
    for tool, args in (("status_job", {"job_id": job_id}),
                       ("collect_job", {"job_id": job_id}),
                       ("terminate_job", {"job_id": job_id})):
        result = _sync(tool, args, owner="sess-B")
        assert result["isError"] is True
        assert "belongs to this session" in result["content"]
    assert _status(job_id, owner="sess-A")["state"] == "running", "a foreign session must not stop the job"
    _sync("terminate_job", {"job_id": job_id}, owner="sess-A")


@pytest.mark.real_process
def test_concurrent_jobs_are_capped_and_capacity_is_recovered(jobs_env, monkeypatch):
    monkeypatch.setattr(config, "JOB_MAX_CONCURRENT", 2)
    first = _start(f"{PY} sleeper.py")
    second = _start(f"{PY} sleeper.py")
    refused = _sync("start_job", {"command": f"{PY} sleeper.py"})
    assert refused["isError"] is True and "already running" in refused["content"]

    _sync("terminate_job", {"job_id": first})
    third = _start(f"{PY} sleeper.py")
    assert _status(third)["state"] == "running"
    _sync("terminate_job", {"job_id": second})
    _sync("terminate_job", {"job_id": third})


# ---------------------------------------------------------------------------
# Output: incremental, bounded, and honest about drops
# ---------------------------------------------------------------------------

@pytest.mark.real_process
def test_collect_is_incremental_and_never_repeats_output(jobs_env):
    job_id = _start(f"{PY} printer.py")
    cursor, seen = None, ""
    end = time.monotonic() + 15
    while "second" not in seen and time.monotonic() < end:
        args = {"job_id": job_id} if cursor is None else {"job_id": job_id, "cursor": cursor}
        page = _json(_sync("collect_job", args))
        seen += page["stdout"]
        cursor = page["cursor"]
        time.sleep(0.05)
    assert seen.count("first") == 1 and seen.count("second") == 1, "each byte is returned once"
    assert page["truncated"] is False
    assert _status(job_id)["state"] == "running"
    _sync("terminate_job", {"job_id": job_id})


@pytest.mark.real_process
def test_high_volume_output_is_bounded_truncated_reported_and_never_deadlocks(jobs_env, monkeypatch):
    monkeypatch.setattr(config, "JOB_OUTPUT_BUFFER_BYTES", 64 << 10)
    job_id = _start(f"{PY} chatty.py")
    status = _wait_state(job_id, {"completed"}, seconds=30)
    assert status["stdout_bytes_written"] > 3_000_000, "the child must have finished writing: no backpressure stall"

    record = _record(job_id)
    assert record.stdout.size <= 64 << 10, "the in-memory buffer must stay within its limit"

    payload = _json(_sync("collect_job", {"job_id": job_id}))
    assert payload["truncated"] is True and payload["stdout_dropped_bytes"] > 0
    assert len(payload["stdout"].encode()) <= shell.MODEL_STREAM_BYTES, "the model still sees the bounded window"
    assert payload["more"] is True


# ---------------------------------------------------------------------------
# Lifecycle: exit, failure, termination, timeout, and identity
# ---------------------------------------------------------------------------

@pytest.mark.real_process
def test_normal_completion_is_completed_with_exit_zero(jobs_env):
    job_id = _start(f"{PY} -c \"print('done')\"")
    status = _wait_state(job_id, {"completed"})
    assert status["exit_status"] == 0 and status["group"] == "empty"
    assert "done" in _json(_sync("collect_job", {"job_id": job_id}))["stdout"]


@pytest.mark.real_process
def test_nonzero_exit_is_failed_and_reports_its_code(jobs_env):
    job_id = _start(f"{PY} -c \"import sys; sys.exit(3)\"")
    status = _wait_state(job_id, {"failed"})
    assert status["exit_status"] == 3 and "exited with code 3" in status["reason"]


@pytest.mark.real_process
def test_explicit_termination_confirms_the_stop_and_escalates_past_sigterm(jobs_env):
    job_id = _start(f"{PY} ignore_term.py")
    pid = _pid(jobs_env)
    status = _json(_sync("terminate_job", {"job_id": job_id}))
    assert status["state"] == "terminated"
    assert "confirmed stopped" in status["reason"]
    assert _gone(pid), "a SIGTERM-ignoring job must be SIGKILLed and reaped"


@pytest.mark.real_process
def test_lifetime_watchdog_terminates_without_any_tool_call(jobs_env):
    job_id = _start(f"{PY} sleeper.py", lifetime_seconds=1)
    time.sleep(2.5)  # the caller makes no tool calls: the worker must enforce the limit on its own
    status = _status(job_id)
    assert status["state"] == "terminated"
    assert "lifetime limit reached" in status["reason"]


@pytest.mark.real_process
def test_terminating_an_already_finished_job_is_a_no_op(jobs_env):
    job_id = _start(f"{PY} -c \"print('quick')\"")
    _wait_state(job_id, {"completed"})
    status = _json(_sync("terminate_job", {"job_id": job_id}))
    assert status["state"] == "completed", "a finished job must not be reclassified as terminated"


@pytest.mark.real_process
def test_leader_exit_with_same_group_descendant_stops_the_descendant(jobs_env):
    job_id = _start(f"{PY} leader_spawns.py")
    status = _wait_state(job_id, {"completed"})
    grandchild = int((jobs_env / "gc").read_text())
    assert status["group"] == "empty"
    assert _gone(grandchild), "a same-group descendant must not outlive its job"


@pytest.mark.real_process
def test_lost_process_identity_is_unknown_and_nothing_is_signalled(jobs_env, monkeypatch):
    signals = []
    real = shell._signal_group
    monkeypatch.setattr(shell, "_signal_group", lambda pgid, sig: signals.append((pgid, sig)) or real(pgid, sig))
    previous = signal.signal(signal.SIGCHLD, signal.SIG_IGN)  # the kernel reaps our children itself
    try:
        job_id = _start(f"{PY} -c \"import sys; sys.exit(3)\"")
        status = _wait_state(job_id, {"unknown"})
    finally:
        signal.signal(signal.SIGCHLD, previous)
    assert "identity was lost" in status["reason"]
    assert status["exit_status"] is None, "no exit status may be inferred from a lost identity"
    assert signals == [], "a group whose leader was reaped elsewhere must never be signalled"


@pytest.mark.real_process
def test_repeated_cancellation_does_not_interrupt_a_termination(jobs_env):
    job_id = _start(f"{PY} ignore_term.py")
    pid = _pid(jobs_env)

    async def scenario():
        task = asyncio.ensure_future(_async_call("terminate_job", {"job_id": job_id}))
        await asyncio.sleep(0.1)
        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    status = _wait_state(job_id, {"terminated", "unknown"}, seconds=20)
    assert status["state"] == "terminated", status["reason"]
    assert _gone(pid), "the caller's cancellation must not leave the job's leader running"


@pytest.mark.real_process
def test_unconfirmed_termination_is_unknown_not_terminated_and_later_resolves(jobs_env, monkeypatch):
    state = {"deny": True}
    real = shell._signal_group
    monkeypatch.setattr(shell, "_signal_group", lambda pgid, sig: False if state["deny"] else real(pgid, sig))
    monkeypatch.setattr(config, "JOB_MAX_RETAINED", 1)

    job_id = _start(f"{PY} ignore_term.py")
    pid = _pid(jobs_env)
    status = _json(_sync("terminate_job", {"job_id": job_id}))
    assert status["state"] == "unknown", "an unconfirmed stop must not be reported as terminated"
    assert "could not be confirmed" in status["reason"]
    assert _alive(pid)

    refused = _sync("start_job", {"command": f"{PY} sleeper.py"})
    assert refused["isError"] is True, "the retention cap must refuse new jobs, not discard the live one"

    state["deny"] = False
    resolved = _wait_state(job_id, {"terminated"}, seconds=20)
    assert "verified empty later" in resolved["reason"]
    assert _gone(pid)


# ---------------------------------------------------------------------------
# Runtime shutdown, cleanup, and no restart
# ---------------------------------------------------------------------------

@pytest.mark.real_process
def test_runtime_shutdown_terminates_all_jobs(jobs_env):
    first = _start(f"{PY} ignore_term.py")
    pid_one = _pid(jobs_env)
    second = _start(f"{PY} sleeper.py", owner="sess-B")
    pid_two = _record(second).proc.pid

    unresolved = jobs.shutdown_runtime(timeout=20)

    assert unresolved == []
    assert _gone(pid_one) and _gone(pid_two)
    assert _record(first).state == "terminated"
    assert _record(second).state == "terminated"


@pytest.mark.real_process
def test_jobs_do_not_survive_into_a_fresh_runtime_and_are_never_restarted(jobs_env):
    job_id = _start(f"{PY} sleeper.py")
    jobs.shutdown_runtime(timeout=20)
    refused = _sync("status_job", {"job_id": job_id})
    assert refused["isError"] is True, "a closed runtime refuses work until it is explicitly reopened"

    jobs.reopen_runtime()
    result = _sync("status_job", {"job_id": job_id})
    assert result["isError"] is True, "a job id from an earlier runtime must not resolve"
    assert _table() == {}, "nothing is restored or restarted in a new runtime"


@pytest.mark.real_process
def test_job_resources_are_released_with_no_resource_warnings(jobs_env, monkeypatch):
    closes = []
    real_open = shell._open_reader

    async def recording_open(pipe):
        reader, transport = await real_open(pipe)
        real_close = transport.close

        def close():
            closes.append(True)
            real_close()

        transport.close = close
        return reader, transport

    monkeypatch.setattr(shell, "_open_reader", recording_open)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        job_id = _start(f"{PY} -c \"print('x')\"")
        _wait_state(job_id, {"completed"})
        jobs.shutdown_runtime(timeout=20)
        gc.collect()
    assert len(closes) == 2, "stdout and stderr transports must each be closed explicitly"
    assert not [w for w in caught if issubclass(w.category, ResourceWarning)]


# ---------------------------------------------------------------------------
# Agent-loop integration: cross-turn, blocked model calls, session end, journal
# ---------------------------------------------------------------------------

class _BlockingPost:
    """requests.post stand-in. Each reply is a stream or a callable that blocks, like a real model call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.payloads = []

    def __call__(self, *args, **kwargs):
        self.payloads.append(copy.deepcopy(kwargs["json"]))
        if not self.replies:
            raise AssertionError("unexpected extra model request")
        item = self.replies.pop(0)
        return item() if callable(item) else item


def _run_session(monkeypatch, replies, session_id):
    post = _BlockingPost(replies)
    monkeypatch.setattr(loop.requests, "post", post)
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)
    event_loop = asyncio.new_event_loop()
    try:
        event_loop.run_until_complete(loop.agent_loop_async(
            "goal", MCPClientManager(), event_loop, session_id=session_id))
    finally:
        event_loop.close()
    return post


def _tool_message(payload, call_id):
    """The tool's content as the model sees it: the serialized envelope's inner text."""
    raw = next(m["content"] for m in payload["messages"]
               if m.get("role") == "tool" and m.get("tool_call_id") == call_id)
    return json.loads(raw)["content"]


@pytest.mark.real_process
def test_job_survives_model_turns_and_its_watchdog_fires_while_a_model_call_blocks(jobs_env, monkeypatch, tmp_sessions):
    def blocked_status_reply():
        time.sleep(2.5)  # a blocking model call: the agent's event loop cannot run anything else
        job_id = next(iter(_table()))
        return make_streamed_response(finish_reason="tool_calls", tool_calls=[
            {"id": "call_status", "name": "status_job", "arguments": json.dumps({"job_id": job_id})}])

    post = _run_session(monkeypatch, [
        make_streamed_response(finish_reason="tool_calls", tool_calls=[
            {"id": "call_start", "name": "start_job",
             "arguments": json.dumps({"command": f"{PY} sleeper.py", "lifetime_seconds": 1})}]),
        blocked_status_reply,
        make_streamed_response(finish_reason="stop", content="done"),
    ], session_id="jobs-blocked")

    status = json.loads(_tool_message(post.payloads[2], "call_status"))
    assert status["state"] == "terminated", "the watchdog must not depend on the agent making progress"
    assert "lifetime limit reached" in status["reason"]


@pytest.mark.real_process
def test_running_command_and_job_coexist_across_turns_and_the_job_ends_with_its_session(jobs_env, monkeypatch, tmp_sessions):
    post = _run_session(monkeypatch, [
        make_streamed_response(finish_reason="tool_calls", tool_calls=[
            {"id": "call_start", "name": "start_job", "arguments": json.dumps({"command": f"{PY} sleeper.py"})}]),
        make_streamed_response(finish_reason="tool_calls", tool_calls=[
            {"id": "call_fg", "name": "run_command", "arguments": json.dumps({"command": f"{PY} -c \"print('fg')\""})}]),
        make_streamed_response(finish_reason="stop", content="done"),
    ], session_id="jobs-session")

    job_id = json.loads(_tool_message(post.payloads[1], "call_start"))["job_id"]
    assert "fg" in _tool_message(post.payloads[2], "call_fg")

    record = _record(job_id)
    assert record.state == "terminated"
    assert record.reason == "owning session ended; process group confirmed stopped"
    assert record.owner == "jobs-session"


@pytest.mark.real_process
def test_job_start_is_journaled_with_its_identifier(jobs_env, monkeypatch, tmp_sessions):
    _run_session(monkeypatch, [
        make_streamed_response(finish_reason="tool_calls", tool_calls=[
            {"id": "call_journal", "name": "start_job", "arguments": json.dumps({"command": f"{PY} sleeper.py"})}]),
        make_streamed_response(finish_reason="stop", content="done"),
    ], session_id="jobs-journal")

    saved = session.load_session("jobs-journal")
    record = saved["execution"][0]
    assert record["status"] == "completed"
    assert re.search(r"job_[0-9a-f]{32}", record["outcome"])


@pytest.mark.real_process
def test_completed_job_metadata_is_retained_for_a_bounded_time_and_count(jobs_env, monkeypatch):
    monkeypatch.setattr(config, "JOB_RETENTION_SECONDS", 0.5)
    monkeypatch.setattr(config, "JOB_RETENTION_MAX", 2)
    finished = [_start(f"{PY} -c \"print({i})\"") for i in range(3)]
    for job_id in finished:
        _wait_state(job_id, {"completed"})

    time.sleep(1.5)  # the janitor applies retention without any tool call
    survivors = [job_id for job_id in finished if _table().get(job_id) is not None]
    assert len(survivors) <= 2, "finished-job metadata is bounded by count"
    assert _sync("status_job", {"job_id": finished[0]})["isError"] is True, "and by age"


@pytest.mark.real_process
def test_finished_job_metadata_is_bounded_by_count_alone(jobs_env, monkeypatch):
    monkeypatch.setattr(config, "JOB_RETENTION_SECONDS", 3600)
    monkeypatch.setattr(config, "JOB_RETENTION_MAX", 2)
    finished = [_start(f"{PY} -c \"print({i})\"") for i in range(4)]
    for job_id in finished:
        _wait_state(job_id, {"completed"})
    by_finish = sorted(finished, key=lambda job_id: _record(job_id).finished)  # retention orders by finish time
    _start(f"{PY} sleeper.py")  # starting a job runs housekeeping
    kept = {job_id for job_id in finished if job_id in _table()}
    assert kept == set(by_finish[-2:]), "the two jobs that finished last are kept"
