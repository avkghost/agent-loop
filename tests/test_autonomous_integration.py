"""B3.7 integration gaps: a full autonomous run reloaded by a fresh runtime, provider failure after a successful
tool effect, and attempt numbering across sequential goals. Scripted provider, real file and shell tools."""

import asyncio
import json

import requests

from src import acceptance, jobs, loop, runs, session
from src.runtime import AgentRuntime
from tests import scripted_provider
from tests.conftest import make_streamed_response

CHECK = 'python3 -c "from calc import add; assert add(2, 3) == 5"'


def _turn(name, arguments, call_id):
    return make_streamed_response(finish_reason="tool_calls", tool_calls=[
        {"id": call_id, "name": name, "arguments": json.dumps(arguments)}])


def _stop(text):
    return make_streamed_response(finish_reason="stop", content=text)


def _criterion(kind, criterion_id, **kwargs):
    expected = kwargs.pop("expected", True)
    return acceptance.AcceptanceCriterion(criterion_id=criterion_id, run_id="", criterion_type=kind,
                                          expected=expected, origin="user", parameters=kwargs)


def _script(monkeypatch, turns):
    """Replay turns. A callable turn is evaluated at request time; an exception class is raised."""
    queue = list(turns)
    seen = []

    def post(*args, **kwargs):
        seen.append(json.loads(json.dumps(kwargs["json"])))
        if not queue:
            raise AssertionError("unexpected model request")
        turn = queue.pop(0)
        if isinstance(turn, type) and issubclass(turn, BaseException):
            raise turn()
        return turn() if callable(turn) else turn

    monkeypatch.setattr(scripted_provider, "POST", post)
    return seen


def _setup(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)


def test_autonomous_run_is_verified_reloaded_by_a_fresh_runtime_and_leaves_no_jobs(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    seen = _script(monkeypatch, [
        _turn("list_directory", {"path": "."}, "inspect"),                        # 1. inspect the workspace
        _turn("write_file", {"path": "calc.py", "content": "def add(a, b):\n    return a - b\n"}, "defect"),
        _turn("run_command", {"command": CHECK}, "check"),                        # 2. a real check fails
        _turn("write_file", {"path": "calc.py", "content": "def add(a, b):\n    return a + b  # fixed\n"}, "fix"),
        _stop("calc.py now adds correctly"),
    ])
    criteria = [_criterion("file_content", "source", path="calc.py", contains="return a + b"),
                _criterion("command_exit", "behaviour", command=CHECK, expected=0)]

    rt = AgentRuntime()
    try:
        sess = rt.open_session("autonomous")
        outcome = rt.event_loop.run_until_complete(sess.run_goal("make add(2, 3) return 5", criteria=criteria))
        sess.close()
    finally:
        rt.shutdown()

    assert outcome.run_status is runs.RunStatus.COMPLETED_VERIFIED
    assert len(seen) == 5
    # The model saw the failing check as a tool result before it repaired the defect
    assert any("exit code 1" in str(m.get("content", "")) for m in seen[3]["messages"] if m.get("role") == "tool")

    # A fresh runtime loads and validates the saved session
    fresh = AgentRuntime()
    try:
        reloaded = fresh.resume_session("autonomous")
        saved = session.load_session("autonomous")
        session.validate_session_data(saved)
        assert saved["format"] == 6
        assert [r["status"] for r in saved["runs"]] == [runs.RunStatus.COMPLETED_VERIFIED.value]
        results = {(r["criterion_id"], r["attempt"], r["status"]) for r in saved["verification_results"]}
        assert results == {("source", 1, "passed"), ("behaviour", 1, "passed")}
        tools = [r["tool_name"] for r in saved["execution"]]
        assert tools == ["list_directory", "write_file", "run_command", "write_file", "verification_command"], tools
        assert all(r["status"] == "completed" for r in saved["execution"])
        reloaded.close()
    finally:
        fresh.shutdown()
    assert jobs.runtime_state() in (None, jobs.CLOSED), "no managed job may remain after the session closes"


def test_provider_failure_after_a_successful_write_is_not_completion(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    _script(monkeypatch, [
        _turn("write_file", {"path": "note.txt", "content": "done"}, "w1"),
        requests.exceptions.ConnectionError,                                     # the next request fails
    ])
    criteria = [_criterion("file_content", "note", path="note.txt", contains="done")]

    rt = AgentRuntime()
    try:
        sess = rt.open_session("provider-after-write")
        outcome = rt.event_loop.run_until_complete(sess.run_goal("write a note", criteria=criteria))
        sess.close()
    finally:
        rt.shutdown()

    assert (tmp_path / "note.txt").read_text() == "done", "the earlier successful write really happened"
    assert outcome.run_status is runs.RunStatus.PROVIDER_FAILED
    assert outcome.run_status is not runs.RunStatus.COMPLETED_VERIFIED
    saved = session.load_session("provider-after-write")
    session.validate_session_data(saved)
    write = next(r for r in saved["execution"] if r["tool_name"] == "write_file")
    assert write["status"] == "completed", "the journal must keep the real outcome of the completed write"
    assert saved["verification_results"] == [], "no verification may run after a provider failure"


def test_attempt_numbering_restarts_for_a_new_goal_and_earlier_history_is_intact(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")
    _script(monkeypatch, [_stop("first"), _stop("second"), _stop("third")])

    rt = AgentRuntime()
    try:
        sess = rt.open_session("numbering")
        first = rt.event_loop.run_until_complete(sess.run_goal(
            "goal one", criteria=[_criterion("file_content", "a", path="a.txt", contains="alpha")], verification_cycles=2))
        second = rt.event_loop.run_until_complete(sess.run_goal(
            "goal two", criteria=[_criterion("file_content", "b", path="b.txt", contains="gamma")], verification_cycles=2))
        sess.close()
    finally:
        rt.shutdown()

    assert first.run_status is runs.RunStatus.COMPLETED_VERIFIED
    assert second.run_status is runs.RunStatus.NEEDS_ATTENTION
    saved = session.load_session("numbering")
    session.validate_session_data(saved)
    by_run = {}
    for item in saved["verification_results"]:
        by_run.setdefault(item["run_id"], []).append((item["criterion_id"], item["attempt"], item["status"]))
    assert by_run[first.run_id] == [("a", 1, "passed")], "run one's history is preserved exactly"
    assert sorted(by_run[second.run_id]) == [("b", 1, "failed"), ("b", 2, "failed")], "numbering restarts for run two"
    assert [r["status"] for r in saved["runs"]] == ["completed_verified", "needs_attention"]
