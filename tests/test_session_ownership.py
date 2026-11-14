"""B3.3 runtime, session and goal ownership. Scripted model, real managed jobs, no MCP server, no network."""

import asyncio
import json
import os
import re
import subprocess

import pytest

from src import cli, config, jobs, loop, runs, runtime, session
from src.authorization import SessionAuthorization
from src.runtime import AgentRuntime, SessionBusy, SessionClosed, SessionError, SessionStateUnknown
from tests import scripted_provider
from tests.conftest import make_streamed_response

# Bounded, deterministic background process: prints a marker, then waits for the lifetime to end.
JOB_COMMAND = 'python3 -u -c "import time; print(\'ready\', flush=True); time.sleep(60)"'
JOB_PATTERN = re.compile(r"job_[0-9a-f]{32}")


def _tool_turn(name, arguments, call_id):
    return make_streamed_response(finish_reason="tool_calls", tool_calls=[
        {"id": call_id, "name": name, "arguments": json.dumps(arguments)}])


def _stop(text):
    return make_streamed_response(finish_reason="stop", content=text)


class _ScriptedPost:
    """Replays a list of turns. An entry that is an exception class is raised instead of returned."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.payloads = []

    def __call__(self, *args, **kwargs):
        self.payloads.append(json.loads(json.dumps(kwargs["json"])))
        if not self.turns:
            raise AssertionError("unexpected extra model request")
        turn = self.turns.pop(0)
        if isinstance(turn, type) and issubclass(turn, BaseException):
            raise turn()
        if callable(turn):
            return turn()
        return turn


@pytest.fixture
def script(monkeypatch):
    def install(turns):
        post = _ScriptedPost(turns)
        monkeypatch.setattr(scripted_provider, "POST", post)
        return post
    return install


@pytest.fixture
def runtimes():
    made = []

    def make(**kwargs):
        rt = AgentRuntime(**kwargs)
        made.append(rt)
        return rt

    yield make
    for rt in made:
        rt.shutdown()


@pytest.fixture(autouse=True)
def _quiet_project(monkeypatch):
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)


def _run(rt, coroutine):
    return rt.event_loop.run_until_complete(coroutine)


def _saved(sid):
    return session.load_session(sid)


def _tool_content(messages, call_id):
    for message in messages:
        if message.get("role") == "tool" and message.get("tool_call_id") == call_id:
            return message["content"]
    raise AssertionError(f"no tool result for {call_id}")


def _job_id_from_start(sid, call_id):
    return JOB_PATTERN.search(_tool_content(_saved(sid)["messages"], call_id)).group(0)


def _pid_of(job_id):
    return jobs._inspect().manager.jobs[job_id].proc.pid


def _alive(pid):
    """True while the process exists and is not a zombie."""
    try:
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    except OSError:
        return False
    return bool(state) and not state.startswith("Z")


# --- lifecycle, ownership and the acceptance scenario ------------------------------------------------

def test_acceptance_two_goals_share_one_session_and_job(runtimes, script, tmp_sessions):
    rt = runtimes()
    sess = rt.open_session("acceptance")
    script([
        _tool_turn("start_job", {"command": JOB_COMMAND, "lifetime_seconds": 120}, "c1"),
        _stop("job started"),
    ])
    first = _run(rt, sess.run_goal("start a background job"))
    assert first.status == runs.RunStatus.COMPLETED_UNVERIFIED.value

    job_id = _job_id_from_start("acceptance", "c1")
    pid = _pid_of(job_id)
    assert _alive(pid), "the job must survive the end of run 1"

    # Goal 2 uses the same live session and the same job
    post = script([
        _tool_turn("collect_job", {"job_id": job_id}, "c3"),
        _tool_turn("terminate_job", {"job_id": job_id}, "c4"),
        _stop("collected and terminated"),
    ])
    second = _run(rt, sess.run_goal("collect and stop the job"))
    assert second.status == runs.RunStatus.COMPLETED_UNVERIFIED.value

    assert "ready" in _tool_content(_saved("acceptance")["messages"], "c3")
    assert not _alive(pid)

    saved = _saved("acceptance")
    assert saved["session_id"] == "acceptance"
    assert [r["run_id"] for r in saved["runs"]] == [first.run_id, second.run_id]
    assert all(r["status"] == "completed_unverified" for r in saved["runs"])
    assert saved["goal"] == "start a background job"
    assert "Next goal: collect and stop the job" in json.dumps(post.payloads[0]["messages"])
    session.validate_session_data(saved)

    assert sess.close() == []
    assert sess.close() == []  # idempotent
    assert rt.shutdown() == []


def test_goal_two_appends_and_keeps_goal_one_context(runtimes, script, tmp_sessions):
    rt = runtimes()
    sess = rt.open_session("context")
    script([_stop("first answer")])
    _run(rt, sess.run_goal("alpha"))
    post = script([_stop("second answer")])
    _run(rt, sess.run_goal("beta"))

    saved = _saved("context")
    assert saved["objective"] == "Objective: alpha"
    assert saved["goal"] == "alpha"
    user_messages = [m["content"] for m in saved["messages"] if m["role"] == "user"]
    assert user_messages == ["Objective: alpha", "Next goal: beta"]
    assert saved["messages"][-1]["content"] == "second answer"
    assert len(post.payloads[0]["messages"]) > 2  # the second request carries the first run's history
    assert len(sess.run_records) == 2


def test_open_session_refuses_existing_saved_state(runtimes, script, tmp_sessions):
    rt = runtimes()
    sess = rt.open_session("taken")
    script([_stop("done")])
    _run(rt, sess.run_goal("first"))
    sess.close()
    with pytest.raises(SessionError, match="already has saved state"):
        rt.open_session("taken")


def test_a_second_live_runtime_is_refused(runtimes):
    rt = runtimes()
    with pytest.raises(RuntimeError, match="another AgentRuntime is live"):
        AgentRuntime()
    rt.shutdown()
    AgentRuntime().shutdown()  # allowed again after shutdown


# --- session close, cross-session isolation and job ownership ----------------------------------------

def test_cross_session_job_access_is_refused_as_not_found(runtimes, script, tmp_sessions):
    rt = runtimes()
    owner_session = rt.open_session("owner")
    other = rt.open_session("intruder")
    script([_tool_turn("start_job", {"command": JOB_COMMAND, "lifetime_seconds": 120}, "s1"), _stop("started")])
    _run(rt, owner_session.run_goal("start"))
    job_id = _job_id_from_start("owner", "s1")

    script([_tool_turn("status_job", {"job_id": job_id}, "x1"),
            _tool_turn("terminate_job", {"job_id": job_id}, "x2"), _stop("tried")])
    _run(rt, other.run_goal("try the foreign job"))

    for call_id in ("x1", "x2"):
        content = _tool_content(_saved("intruder")["messages"], call_id)
        assert "No job with that id belongs to this session." in content
        assert "ready" not in content
    assert _alive(_pid_of(job_id)), "a foreign session must not stop the job"

    owner_session.close()
    assert not _alive(_pid_of(job_id))


def test_closing_one_session_leaves_another_sessions_jobs_running(runtimes, script, tmp_sessions):
    rt = runtimes()
    a = rt.open_session("session-a")
    b = rt.open_session("session-b")
    script([_tool_turn("start_job", {"command": JOB_COMMAND, "lifetime_seconds": 120}, "a1"), _stop("a")])
    _run(rt, a.run_goal("start in A"))
    script([_tool_turn("start_job", {"command": JOB_COMMAND, "lifetime_seconds": 120}, "b1"), _stop("b")])
    _run(rt, b.run_goal("start in B"))
    job_a = _pid_of(_job_id_from_start("session-a", "a1"))
    job_b = _pid_of(_job_id_from_start("session-b", "b1"))

    assert a.close() == []
    assert not _alive(job_a)
    assert _alive(job_b)
    rt.shutdown()
    assert not _alive(job_b)


def test_closed_session_refuses_new_goals(runtimes, script, tmp_sessions):
    rt = runtimes()
    sess = rt.open_session("closing")
    script([_stop("done")])
    _run(rt, sess.run_goal("only goal"))
    assert sess.close() == []
    with pytest.raises(SessionClosed):
        _run(rt, sess.run_goal("after close"))


def test_concurrent_goal_in_same_session_is_refused(runtimes, script, tmp_sessions):
    rt = runtimes()
    sess = rt.open_session("busy")
    # The first goal suspends on a job call, so a second goal can arrive while the first is running
    script([_tool_turn("start_job", {"command": JOB_COMMAND, "lifetime_seconds": 120}, "b1"), _stop("started")])

    async def second_goal():
        while not sess.busy:
            await asyncio.sleep(0)
        with pytest.raises(SessionBusy):
            await sess.run_goal("overlapping goal")
        return "refused"

    async def both():
        return await asyncio.gather(sess.run_goal("first goal"), second_goal())

    first, verdict = _run(rt, both())
    assert verdict == "refused"
    assert first.status == runs.RunStatus.COMPLETED_UNVERIFIED.value
    assert [r["goal"] for r in _saved("busy")["runs"]] == ["first goal"]
    sess.close()


def test_run_cancellation_interrupts_the_run_and_keeps_session_usable(runtimes, script, tmp_sessions, monkeypatch):
    rt = runtimes()
    sess = rt.open_session("cancel")
    sess.authorization.grant("command", "true")  # a grant made before the cancellation

    script([asyncio.CancelledError, _stop("after cancel")])
    with pytest.raises(asyncio.CancelledError):
        _run(rt, sess.run_goal("cancelled goal"))

    assert sess.closed is False
    assert [r["status"] for r in _saved("cancel")["runs"]] == ["interrupted"]
    assert "true" in sess.authorization.snapshot()["command"], "cancellation must not reset grants"

    _run(rt, sess.run_goal("next goal"))
    saved = _saved("cancel")
    assert [r["status"] for r in saved["runs"]] == ["interrupted", "completed_unverified"]
    assert any(m.get("content") == "Objective: cancelled goal" for m in saved["messages"])
    sess.close()


def test_failed_persistence_refuses_the_next_goal(runtimes, script, tmp_sessions, monkeypatch):
    rt = runtimes()
    sess = rt.open_session("unsaved")
    real_save = session.save_session

    def fail_terminal_save(*args, **kwargs):
        if any(r.get("status") != runs.RUNNING for r in kwargs.get("runs", [])):
            raise OSError("injected disk failure")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(session, "save_session", fail_terminal_save)
    script([_stop("answer")])
    _run(rt, sess.run_goal("unconfirmed goal"))

    with pytest.raises(SessionStateUnknown):
        _run(rt, sess.run_goal("refused goal"))
    assert [r["status"] for r in _saved("unsaved")["runs"]] == ["running"]


def test_runtime_shutdown_during_an_active_job_stops_it_and_closes_sessions(runtimes, script, tmp_sessions):
    rt = runtimes()
    sess = rt.open_session("shutdown")
    script([_tool_turn("start_job", {"command": JOB_COMMAND, "lifetime_seconds": 120}, "d1"), _stop("started")])
    _run(rt, sess.run_goal("start"))
    pid = _pid_of(_job_id_from_start("shutdown", "d1"))

    assert rt.shutdown() == []
    assert sess.closed
    assert not _alive(pid)
    assert rt.shutdown() == []  # idempotent
    with pytest.raises(SessionClosed):
        rt.open_session("late")


def test_job_cleanup_timeout_is_reported_not_silent(runtimes, script, tmp_sessions, monkeypatch, capsys):
    rt = runtimes()
    sess = rt.open_session("slow-close")
    script([_stop("no job")])
    _run(rt, sess.run_goal("create the job runtime"))
    jobs._GATE.acquire()  # the job runtime exists, so close must wait for its worker

    async def slow_shutdown(self, owner, reason):
        await asyncio.sleep(0.5)
        return []

    monkeypatch.setattr(jobs._Manager, "shutdown", slow_shutdown)
    monkeypatch.setattr(config, "JOB_SHUTDOWN_TIMEOUT_SECONDS", 0.1)
    problems = sess.close()
    assert problems and "did not finish within" in problems[0]["reason"]
    assert "did not finish within" in capsys.readouterr().out

    monkeypatch.undo()
    rt.shutdown()


def test_grants_are_session_local_and_survive_runs_of_one_session(runtimes, script, tmp_sessions, monkeypatch):
    rt = runtimes()
    a = rt.open_session("grants-a")
    b = rt.open_session("grants-b")
    answers = iter(["2"])  # session A: approve "true" for this session
    monkeypatch.setattr(loop, "_get_single_key_input", lambda prompt, valid_keys: next(answers))

    script([_tool_turn("run_command", {"command": "true"}, "g1"), _stop("ran")])
    _run(rt, a.run_goal("run true first time"))
    script([_tool_turn("run_command", {"command": "true"}, "g2"), _stop("ran again")])
    _run(rt, a.run_goal("run true second time, no prompt"))  # no key input left: a prompt would fail
    assert "requiresPermission" not in _tool_content(_saved("grants-a")["messages"], "g2")

    denied = iter(["3"])
    monkeypatch.setattr(loop, "_get_single_key_input", lambda prompt, valid_keys: next(denied))
    script([_tool_turn("run_command", {"command": "true"}, "g3"), _stop("tried")])
    _run(rt, b.run_goal("session B runs true"))
    assert "Command/tool denied" in _tool_content(_saved("grants-b")["messages"], "g3")
    assert "true" not in b.authorization.snapshot()["command"]
    assert "true" in a.authorization.snapshot()["command"]


def test_shell_operator_approval_is_exact_and_one_time_when_asked_once(runtimes, script, tmp_sessions, monkeypatch):
    rt = runtimes()
    sess = rt.open_session("exact-shell")
    monkeypatch.setattr(loop, "_get_single_key_input", lambda prompt, valid_keys: "1")  # this use only
    script([_tool_turn("run_command", {"command": "echo hi | cat"}, "e1"), _stop("ok")])
    _run(rt, sess.run_goal("pipe once"))
    assert "hi" in _tool_content(_saved("exact-shell")["messages"], "e1")
    assert "echo hi | cat" not in sess.authorization.snapshot()["shell"]

    monkeypatch.setattr(loop, "_get_single_key_input", lambda prompt, valid_keys: "3")
    script([_tool_turn("run_command", {"command": "echo hi | cat"}, "e2"), _stop("again")])
    _run(rt, sess.run_goal("pipe again"))
    assert "denied" in _tool_content(_saved("exact-shell")["messages"], "e2").lower()
    sess.close()


def test_from_config_snapshot_does_not_share_grants(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ALLOW_ALL", False)
    auth = SessionAuthorization.from_config()
    auth.grant("command", "uname")
    assert SessionAuthorization.from_config().allows("command", "uname") is False
    assert auth.allows("command", "uname") is True


def test_legacy_session_resumes_in_a_live_session(runtimes, script, tmp_sessions):
    session.save_session("legacy", "old goal", [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Objective: old goal"},
        {"role": "assistant", "content": "earlier answer"},
    ])
    data = _saved("legacy")
    data["format"] = 4
    data.update({"objective": "Objective: old goal", "summary": None, "execution": []})
    for key in ("runs", "evidence", "legacy_history"):
        data.pop(key, None)
    (tmp_sessions / "legacy.json").write_text(json.dumps(data))

    rt = runtimes()
    sess = rt.resume_session("legacy")
    post = script([_stop("continued")])
    outcome = _run(rt, sess.run_goal("new goal"))
    assert outcome.status == runs.RunStatus.COMPLETED_UNVERIFIED.value
    assert post.payloads[0]["messages"][-1]["content"] == "Next goal: new goal"
    sess.close()


def test_dispatched_call_left_running_is_not_redispatched_after_resume(runtimes, script, tmp_sessions, monkeypatch):
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Objective: old goal"},
        {"role": "assistant", "content": "calling", "tool_calls": [
            {"id": "u1", "type": "function", "function": {"name": "run_command", "arguments": '{"command": "true"}'}}]},
    ]
    session.save_session("unknown", "old goal", messages, objective="Objective: old goal", execution=[
        {"tool_call_id": "u1", "tool_name": "run_command", "arguments": {"command": "true"}, "status": "running",
         "outcome": None, "error_category": None}])
    dispatched = []

    async def recording_dispatch(tool_name, tool_input, *args, **kwargs):
        dispatched.append(tool_name)
        return {}

    monkeypatch.setattr(loop.dispatch, "dispatch_tool_async", recording_dispatch)

    rt = runtimes()
    sess = rt.resume_session("unknown")
    post = script([_stop("continued after unknown")])
    _run(rt, sess.run_goal("continue"))
    assert dispatched == []
    tool = [m for m in post.payloads[0]["messages"] if m.get("role") == "tool"]
    assert tool and "not re-executed" in tool[0]["content"]
    sess.close()


# --- CLI: one stable session across REPL goals ------------------------------------------------------

def test_repl_uses_one_session_for_every_goal_and_cleans_up_on_exit(tmp_sessions, script, monkeypatch):
    monkeypatch.setattr(cli, "load_mcp_config", lambda: {})
    inputs = iter(["start a job", "collect and stop it", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    def the_job():
        return next(iter(jobs._inspect().manager.jobs))

    script([
        _tool_turn("start_job", {"command": JOB_COMMAND, "lifetime_seconds": 120}, "r1"), _stop("started"),
        lambda: _tool_turn("terminate_job", {"job_id": the_job()}, "r2"), _stop("stopped"),
    ])
    cli._main(["repl"])

    files = list(tmp_sessions.glob("*.json"))
    assert len(files) == 1, [f.name for f in files]
    saved = json.loads(files[0].read_text())
    assert len(saved["runs"]) == 2
    assert saved["goal"] == "start a job"
    assert [m["content"] for m in saved["messages"] if m["role"] == "user"] == [
        "Objective: start a job", "Next goal: collect and stop it"]


def test_saved_format_five_session_resumes_with_history_runs_and_fresh_grants(runtimes, script, tmp_sessions):
    rt = runtimes()
    sess = rt.open_session("five")
    script([_stop("first answer")])
    first = _run(rt, sess.run_goal("first goal"))
    sess.close()
    rt.shutdown()

    rt2 = runtimes()
    resumed = rt2.resume_session("five")
    post = script([_stop("second answer")])
    second = _run(rt2, resumed.run_goal("second goal"))

    assert [r["run_id"] for r in _saved("five")["runs"]] == [first.run_id, second.run_id]
    assert "first answer" in json.dumps(post.payloads[0]["messages"])
    assert post.payloads[0]["messages"][-1]["content"] == "Next goal: second goal"
    assert resumed.authorization is not sess.authorization  # grants are not restored from disk
    resumed.close()
