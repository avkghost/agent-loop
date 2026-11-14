"""B3.3 shutdown safety: a busy session refuses runtime shutdown without destroying shared resources.

Scripted model, real managed jobs (bounded local process), fake MCP connections. No LLM, no network.
"""

import asyncio
import json
import os

import pytest

from src import jobs, loop, runs, session
from src.runtime import (AgentRuntime, RuntimeShutdownRefused, SessionBusy, SessionClosed, SessionError)
from src import config
from tests import scripted_provider
from tests.conftest import make_streamed_response
from tests.test_session_ownership import JOB_COMMAND, _alive, _job_id_from_start, _pid_of, _stop, _tool_turn
from tests.test_session_ownership import _ScriptedPost


class FakeMCP:
    """Bounded stand-in for an MCP client. Records connect and close, and can fail close a set number of times."""

    instances = []

    def __init__(self, close_failures=0):
        self.connected = False
        self.closes = 0
        self.close_failures = close_failures
        self.tools_cache = {}
        FakeMCP.instances.append(self)

    async def connect_all(self, config_data):
        self.connected = True

    def get_openai_tools(self):
        return []

    async def close_all(self):
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("injected MCP close failure")
        self.closes += 1
        self.connected = False


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)
    FakeMCP.instances = []


@pytest.fixture
def script(monkeypatch):
    def install(turns):
        post = _ScriptedPost(turns)
        monkeypatch.setattr(scripted_provider, "POST", post)
        return post
    return install


TASKS = []  # goals started by a test; a failing test must not leave one running into the next test


@pytest.fixture
def runtime_factory(monkeypatch):
    made = []

    def make(mcp_close_failures=0):
        rt = AgentRuntime(mcp_client_factory=lambda: FakeMCP(close_failures=mcp_close_failures))
        made.append(rt)
        return rt

    yield make
    for task in TASKS:
        if not task.done():
            task.cancel()
            try:
                task.get_loop().run_until_complete(task)
            except BaseException:
                pass
    TASKS.clear()
    for rt in made:
        try:
            rt.shutdown()
        except RuntimeShutdownRefused:
            pass


def _busy_goal(rt, sess, goal):
    """Start a goal that is suspended on a job call, so the session is busy while the loop is idle."""
    task = rt.event_loop.create_task(sess.run_goal(goal))
    TASKS.append(task)
    for _ in range(100):
        rt.event_loop.run_until_complete(asyncio.sleep(0))
        if sess.busy:
            break
    assert sess.busy and not task.done()
    return task


def _finish(rt, task):
    return rt.event_loop.run_until_complete(task)


def _start_job_script(script, call_id="j1"):
    return script([_tool_turn("start_job", {"command": JOB_COMMAND, "lifetime_seconds": 120}, call_id),
                   _stop("started")])


# --- reproduction and refusal -------------------------------------------------------------------------

def test_shutdown_while_a_session_is_busy_is_refused_and_destroys_nothing(runtime_factory, script, tmp_sessions):
    rt = runtime_factory()
    sess = rt.open_session("busy-one")
    _start_job_script(script)
    task = _busy_goal(rt, sess, "start a job then wait")

    with pytest.raises(RuntimeShutdownRefused) as refused:
        rt.shutdown()
    assert refused.value.busy == ["busy-one"]

    assert rt.state == "open"
    assert not rt.event_loop.is_closed()
    rt.event_loop.run_until_complete(asyncio.sleep(0))  # the loop is still operational
    assert jobs.runtime_state() == jobs.RUNNING
    assert not sess.closed
    assert FakeMCP.instances and FakeMCP.instances[0].closes == 0

    outcome = _finish(rt, task)
    assert outcome.status == runs.RunStatus.COMPLETED_UNVERIFIED.value
    job_pid = _pid_of(_job_id_from_start("busy-one", "j1"))
    assert _alive(job_pid), "the job must still be running after a refused shutdown"

    assert rt.shutdown() == []
    assert rt.state == "closed"
    assert not _alive(job_pid)


def test_refusal_leaves_idle_sessions_usable(runtime_factory, script, tmp_sessions):
    rt = runtime_factory()
    idle = rt.open_session("idle-two")
    busy = rt.open_session("busy-two")
    _start_job_script(script)
    task = _busy_goal(rt, busy, "busy goal")

    with pytest.raises(RuntimeShutdownRefused) as refused:
        rt.shutdown()
    assert refused.value.busy == ["busy-two"]
    assert not idle.closed and not idle.busy

    script([_stop("idle still works")])
    outcome = rt.event_loop.run_until_complete(idle.run_goal("idle goal"))
    assert outcome.status == runs.RunStatus.COMPLETED_UNVERIFIED.value
    _finish(rt, task)


def test_new_goal_is_refused_while_shutdown_is_committed(runtime_factory, script, tmp_sessions):
    """The race: a goal scheduled before shutdown must not start while shutdown closes sessions. Session
    close runs the loop (MCP close), so a scheduled goal would otherwise get a turn in the middle."""
    rt = runtime_factory()
    sess = rt.open_session("racing")
    post = script([_stop("should never run")])
    starter = rt.event_loop.create_task(sess.run_goal("late goal"))
    TASKS.append(starter)
    assert rt.shutdown() == []
    # The scheduled goal ran during session close, was refused, and never reached the model
    assert starter.done()
    assert isinstance(starter.exception(), SessionClosed)
    assert len(post.turns) == 1, "the refused goal must not have made a model request"
    assert sess.closed


# --- retry, partial cleanup, and state ---------------------------------------------------------------

def test_active_run_can_be_cancelled_and_then_shutdown_succeeds(runtime_factory, script, tmp_sessions):
    rt = runtime_factory()
    sess = rt.open_session("cancelled")
    _start_job_script(script)
    task = _busy_goal(rt, sess, "to be cancelled")
    with pytest.raises(RuntimeShutdownRefused):
        rt.shutdown()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        _finish(rt, task)
    assert not sess.busy
    assert [r["status"] for r in session.load_session("cancelled")["runs"]] == ["interrupted"]

    assert rt.shutdown() == []
    assert rt.state == "closed"


def test_concurrent_shutdown_attempt_is_refused_while_one_is_in_progress(runtime_factory, script, tmp_sessions):
    rt = runtime_factory()
    rt.open_session("only")
    seen = {}

    async def reentrant():
        try:
            rt.shutdown()
        except RuntimeShutdownRefused as e:
            seen["error"] = str(e)

    real_close = rt.sessions()[0].close

    def close_and_reenter():
        rt.event_loop.run_until_complete(reentrant())
        return real_close()

    rt.sessions()[0].close = close_and_reenter
    assert rt.shutdown() == []
    assert "already in progress" in seen["error"]


def test_mcp_close_failure_keeps_runtime_recoverable_then_retry_completes(runtime_factory, script, tmp_sessions):
    rt = runtime_factory(mcp_close_failures=1)
    sess = rt.open_session("mcp-retry")
    _start_job_script(script)
    _finish(rt, _busy_goal_done(rt, sess, "start job"))
    job_pid = _pid_of(_job_id_from_start("mcp-retry", "j1"))

    problems = rt.shutdown()
    assert problems and "MCP close" in problems[0]["reason"]
    assert rt.state == "cleanup_pending"
    assert not rt.event_loop.is_closed(), "the loop stays owned until MCP cleanup is confirmed"
    assert jobs.runtime_state() == jobs.RUNNING, "the job worker stays owned for the retry"
    assert not _alive(job_pid), "the job stop was confirmed before the MCP failure, and is not repeated"

    assert rt.shutdown() == []
    assert rt.state == "closed"
    assert rt.event_loop.is_closed()
    assert FakeMCP.instances[0].closes == 1, "MCP must be closed exactly once"


def test_job_cleanup_timeout_then_retry_does_not_double_stop(runtime_factory, script, tmp_sessions, monkeypatch):
    rt = runtime_factory()
    sess = rt.open_session("slow-job")
    _start_job_script(script)
    _finish(rt, _busy_goal_done(rt, sess, "start job"))
    job_pid = _pid_of(_job_id_from_start("slow-job", "j1"))

    calls = {"n": 0}
    real_shutdown = jobs._Manager.shutdown

    async def slow_once(self, owner, reason):
        calls["n"] += 1
        if calls["n"] == 1:
            await asyncio.sleep(0.5)
        return await real_shutdown(self, owner, reason)

    monkeypatch.setattr(jobs._Manager, "shutdown", slow_once)
    monkeypatch.setattr(config, "JOB_SHUTDOWN_TIMEOUT_SECONDS", 0.1)
    problems = rt.shutdown()
    assert problems and "did not finish within" in problems[0]["reason"]
    assert rt.state == "cleanup_pending"
    assert _alive(job_pid)

    monkeypatch.undo()
    assert rt.shutdown() == []
    assert rt.state == "closed"
    assert not _alive(job_pid)


def test_repeated_successful_shutdown_is_idempotent(runtime_factory, script, tmp_sessions):
    rt = runtime_factory()
    sess = rt.open_session("repeat")
    script([_stop("done")])
    _finish(rt, _busy_goal_done(rt, sess, "one goal"))
    assert rt.shutdown() == []
    assert rt.shutdown() == []
    assert rt.state == "closed"
    assert rt.closed and not rt.accepting
    assert rt.event_loop.is_closed(), "a final runtime releases its loop"
    # No job was ever started, so the job runtime was never created; if it exists it must be stopped
    assert jobs.runtime_state() in (None, jobs.CLOSED), "a final runtime has stopped its job worker"
    assert FakeMCP.instances[0].closes == 1


def test_unconfirmed_job_stop_after_worker_stopped_is_reported_as_final_unconfirmed(runtime_factory, script,
                                                                                  tmp_sessions, monkeypatch):
    rt = runtime_factory()
    sess = rt.open_session("unconfirmed")
    _start_job_script(script)
    _finish(rt, _busy_goal_done(rt, sess, "start job"))

    real_shutdown_runtime = jobs.shutdown_runtime
    entry = {"job_id": "job_" + "0" * 32, "state": "unknown", "leader_pid": 1, "reason": "injected"}
    monkeypatch.setattr(jobs, "shutdown_runtime", lambda *a, **k: real_shutdown_runtime(*a, **k) + [entry])
    problems = rt.shutdown()
    assert problems and problems[-1]["job_id"] == entry["job_id"]
    assert rt.state == "closed_unconfirmed"
    assert rt.closed and rt.event_loop.is_closed()
    assert rt.shutdown() == problems, "a final runtime reports the same diagnostics and does not retry"


def test_closed_runtime_refuses_new_sessions_and_goals(runtime_factory, script, tmp_sessions):
    rt = runtime_factory()
    sess = rt.open_session("final")
    rt.shutdown()
    with pytest.raises(SessionClosed):
        rt.open_session("late")
    with pytest.raises(SessionClosed):
        asyncio.run(sess.run_goal("after close"))


def test_rejected_shutdown_does_not_touch_another_runtime_session(runtime_factory, script, tmp_sessions):
    rt = runtime_factory()
    sess = rt.open_session("only-here")
    script([_stop("ok")])
    _finish(rt, _busy_goal_done(rt, sess, "goal"))
    rt.shutdown()
    rt2 = runtime_factory()
    other = rt2.open_session("other")
    _start_job_script(script)
    task = _busy_goal(rt2, other, "other goal")
    assert rt.shutdown() == []  # already closed: a no-op that must not touch rt2
    assert jobs.runtime_state() == jobs.RUNNING
    _finish(rt2, task)
    assert rt2.shutdown() == []


def _busy_goal_done(rt, sess, goal):
    """A finished goal (no job): the session is idle again afterwards. Returns a task to finish."""
    return rt.event_loop.create_task(sess.run_goal(goal))


def test_ctrl_c_during_a_repl_goal_records_the_interrupt_and_exits_cleanly(tmp_sessions, script, monkeypatch):
    from src import cli

    monkeypatch.setattr(cli, "load_mcp_config", lambda: {})
    inputs = iter(["interrupted goal", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    def ctrl_c():
        raise KeyboardInterrupt()

    script([ctrl_c, _stop("never reached")])
    cli._main(["repl"])  # must return; the scripted interrupt is handled inside the REPL

    files = list(tmp_sessions.glob("*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert [r["status"] for r in saved["runs"]] == ["interrupted"]
    assert jobs.runtime_state() in (None, jobs.CLOSED)
