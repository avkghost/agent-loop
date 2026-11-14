"""B3.5 end-to-end and negative acceptance cases. Scripted model; real bounded local tools; no network."""

import asyncio
import json

import pytest

from src import acceptance, loop, runs, session, verification
from src.runtime import AgentRuntime
from src.tools.mcp_client import MCPClientManager
from tests import scripted_provider
from tests.conftest import make_streamed_response

CHECK = 'python3 -c "from calc import add; assert add(2, 3) == 5"'


def _criterion(kind, criterion_id, origin="user", **kwargs):
    expected = kwargs.pop("expected", True)
    return acceptance.AcceptanceCriterion(
        criterion_id=criterion_id, run_id="", criterion_type=kind, expected=expected,
        origin=origin, parameters=kwargs)


def _turn(name, arguments, call_id):
    return make_streamed_response(finish_reason="tool_calls", tool_calls=[
        {"id": call_id, "name": name, "arguments": json.dumps(arguments)}])


def _stop(text):
    return make_streamed_response(finish_reason="stop", content=text)


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)


def _script(monkeypatch, replies):
    scripted = list(replies)
    seen = []

    def post(*args, **kwargs):
        seen.append(json.loads(json.dumps(kwargs["json"])))
        if not scripted:
            raise AssertionError("unexpected model request")
        return scripted.pop(0)

    monkeypatch.setattr(scripted_provider, "POST", post)
    return seen


def _run(criteria, sid="accept", cycles=2, dispatch=None, monkeypatch=None):
    if dispatch is not None:
        monkeypatch.setattr(loop.dispatch, "dispatch_tool_async", dispatch)

    async def go():
        return await loop.agent_loop_async("build a calc module", MCPClientManager(),
                                           asyncio.get_running_loop(), session_id=sid,
                                           criteria=criteria, verification_cycles=cycles)
    return asyncio.run(go())


def _saved(sid="accept"):
    return session.load_session(sid)


# --- mandatory end-to-end scenario --------------------------------------------------------------

def test_end_to_end_defect_check_fix_declare_then_runtime_verifies(monkeypatch, tmp_path):
    seen = _script(monkeypatch, [
        _turn("write_file", {"path": "calc.py", "content": "def add(a, b):\n    return a - b\n"}, "w1"),
        _turn("run_command", {"command": CHECK}, "r1"),        # the model's own check, which fails
        # A different length: an equal-size rewrite within one second can leave a stale bytecode cache entry
        _turn("write_file", {"path": "calc.py", "content": "def add(a, b):\n    return a + b  # fixed\n"}, "w2"),
        _stop("calc.py is fixed and the check passes"),
    ])
    criteria = [
        _criterion("file_content", "calc_source", path="calc.py", contains="return a + b"),
        _criterion("command_exit", "calc_behaviour", command=CHECK, expected=0),
    ]
    outcome = _run(criteria)
    assert outcome.run_status is runs.RunStatus.COMPLETED_VERIFIED
    assert len(seen) == 4

    model_saw = seen[2]["messages"]  # the request after the model's own check ran
    assert any("AssertionError" in str(m.get("content", "")) or "exit code 1" in str(m.get("content", ""))
               for m in model_saw if m.get("role") == "tool"), "the model's own check should have failed"

    saved = _saved()
    assert saved["format"] == 6
    assert {r["criterion_id"]: r["status"] for r in saved["verification_results"]} == {
        "calc_source": "passed", "calc_behaviour": "passed"}
    session.validate_session_data(saved)
    assert [r["status"] for r in saved["runs"]] == ["completed_verified"]


# --- negative cases ------------------------------------------------------------------------------

def test_model_claims_success_with_required_file_missing(monkeypatch, tmp_path):
    _script(monkeypatch, [_stop("done"), _stop("still done")])
    outcome = _run([_criterion("file_exists", "f1", path="calc.py", type="file")], cycles=2)
    assert outcome.run_status is runs.RunStatus.NEEDS_ATTENTION
    assert "2 of 2 attempts used" in outcome.reason
    assert _saved()["verification_results"][-1]["status"] == "failed"


def test_failed_check_feedback_names_the_criterion_and_the_repair_is_reverified(monkeypatch, tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    seen = _script(monkeypatch, [
        _stop("I think it is done"),
        _turn("write_file", {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"}, "w1"),
        _stop("fixed"),
    ])
    outcome = _run([_criterion("file_content", "calc_source", path="calc.py", contains="return a + b")], cycles=2)
    assert outcome.run_status is runs.RunStatus.COMPLETED_VERIFIED
    feedback = [m["content"] for m in seen[1]["messages"] if m.get("role") == "user" and "calc_source" in m["content"]]
    assert feedback, "failed criteria must be named in the feedback"


def test_timeout_during_verification_is_not_a_pass(monkeypatch, tmp_path):
    _script(monkeypatch, [_stop("done"), _stop("done again")])
    slow = 'python3 -c "import time; time.sleep(5)"'
    criteria = [_criterion("command_exit", "slow", command=slow, expected=0, timeout_seconds=1)]
    outcome = _run(criteria, cycles=1)
    assert outcome.run_status is runs.RunStatus.NEEDS_ATTENTION
    assert _saved()["verification_results"][0]["status"] == "inconclusive"


def test_unauthorized_verification_command_is_denied_and_not_run(monkeypatch, tmp_path):
    marker = tmp_path / "ran.txt"
    _script(monkeypatch, [_stop("done"), _stop("again")])
    # uname is not on the CLI policy and no grant exists, so the verification command must be refused
    criteria = [_criterion("command_exit", "denied", command="uname -s", expected=0)]
    outcome = _run(criteria, cycles=1)
    assert outcome.run_status is runs.RunStatus.NEEDS_ATTENTION
    saved = _saved()
    assert saved["verification_results"][0]["status"] == "inconclusive"
    record = next(item for item in saved["execution"] if item["tool_name"] == "verification_command")
    assert record["status"] == "failed"
    assert record["error_category"] == "permission"
    assert not marker.exists()


def test_verification_dispatch_exception_is_unknown_and_never_passes(monkeypatch, tmp_path):
    _script(monkeypatch, [_stop("done"), _stop("again")])

    async def raising(tool_name, tool_input, *args, **kwargs):
        if tool_name == "run_command":
            raise RuntimeError("injected dispatch failure")
        return {"content": "", "isError": False}

    criteria = [_criterion("command_exit", "boom", command="python3 -c \"print(1)\"", expected=0)]
    outcome = _run(criteria, cycles=1, dispatch=raising, monkeypatch=monkeypatch)
    assert outcome.run_status is runs.RunStatus.NEEDS_ATTENTION
    saved = _saved()
    assert saved["verification_results"][0]["status"] == "inconclusive"
    record = next(item for item in saved["execution"] if item["tool_name"] == "verification_command")
    assert record["status"] == "unknown"


def test_signal_terminated_verification_command_is_not_a_pass(monkeypatch, tmp_path):
    _script(monkeypatch, [_stop("done"), _stop("again")])

    async def signalled(tool_name, tool_input, *args, **kwargs):
        return {"content": "[terminated by signal 9]", "isError": False, "exitCode": -9,
                "processGroupEmpty": True}

    criteria = [_criterion("command_exit", "sig", command="python3 -c \"print(1)\"", expected=0)]
    outcome = _run(criteria, cycles=1, dispatch=signalled, monkeypatch=monkeypatch)
    assert outcome.run_status is runs.RunStatus.NEEDS_ATTENTION
    assert _saved()["verification_results"][0]["status"] == "inconclusive"


def test_missing_exit_status_or_unconfirmed_group_is_not_a_pass(monkeypatch, tmp_path):
    _script(monkeypatch, [_stop("done"), _stop("again")])

    async def unconfirmed(tool_name, tool_input, *args, **kwargs):
        return {"content": "", "isError": False, "exitCode": 0, "processGroupEmpty": False}

    criteria = [_criterion("command_exit", "group", command="python3 -c \"print(1)\"", expected=0)]
    outcome = _run(criteria, cycles=1, dispatch=unconfirmed, monkeypatch=monkeypatch)
    assert outcome.run_status is runs.RunStatus.NEEDS_ATTENTION
    assert _saved()["verification_results"][0]["status"] == "inconclusive"


def test_required_criterion_cannot_be_silenced_by_a_model_proposal(monkeypatch, tmp_path):
    _script(monkeypatch, [_stop("done"), _stop("again")])
    criteria = [
        _criterion("file_exists", "required", path="calc.py", type="file"),
        _criterion("file_exists", "proposal", origin="model", path="README.md", type="file"),
    ]
    outcome = _run(criteria, cycles=1)
    assert outcome.run_status is runs.RunStatus.NEEDS_ATTENTION
    results = {r["criterion_id"]: r["status"] for r in _saved()["verification_results"]}
    assert results == {"required": "failed"}, "only trusted criteria are verified, and the required one failed"


def test_model_proposed_passing_criterion_cannot_create_verified_success(monkeypatch, tmp_path):
    (tmp_path / "README.md").write_text("hi")
    _script(monkeypatch, [_stop("done")])
    outcome = _run([_criterion("file_exists", "proposal", origin="model", path="README.md", type="file")])
    assert outcome.run_status is runs.RunStatus.COMPLETED_UNVERIFIED


def test_unknown_outcome_from_this_run_withholds_verified_completion(monkeypatch, tmp_path):
    (tmp_path / "calc.py").write_text("x = 1\n")
    _script(monkeypatch, [
        _turn("write_file", {"path": "calc.py", "content": "x = 1\n"}, "w1"),  # outcome is unknown
        _stop("done"),
    ])

    async def unknown_write(tool_name, tool_input, *args, **kwargs):
        if tool_name == "write_file":
            return {"content": "ambiguous", "isError": True, "errorCategory": "unknown_outcome",
                    "isRetryable": False}
        return {"content": "", "isError": False, "exitCode": 0, "processGroupEmpty": True}

    criteria = [_criterion("file_content", "present", path="calc.py", contains="x = 1")]
    outcome = _run(criteria, cycles=1, dispatch=unknown_write, monkeypatch=monkeypatch)
    assert outcome.run_status is runs.RunStatus.NEEDS_ATTENTION
    assert "unknown" in outcome.reason
    saved = _saved()
    assert any(r["tool_name"] == "write_file" and r["status"] == "unknown" for r in saved["execution"])


def test_cancellation_during_verification_records_interrupted(monkeypatch, tmp_path):
    _script(monkeypatch, [_stop("done")])

    async def cancelled(tool_name, tool_input, *args, **kwargs):
        if tool_name == "run_command":
            raise asyncio.CancelledError()
        return {"content": "", "isError": False}

    criteria = [_criterion("command_exit", "c", command="python3 -c \"print(1)\"", expected=0)]
    with pytest.raises(asyncio.CancelledError):
        _run(criteria, cycles=1, dispatch=cancelled, monkeypatch=monkeypatch)
    saved = _saved()
    assert [r["status"] for r in saved["runs"]] == ["interrupted"]
    assert not any(r["status"] == "completed_verified" for r in saved["runs"])


def test_persisted_evidence_from_another_run_is_rejected(monkeypatch, tmp_path):
    _script(monkeypatch, [_stop("done")])
    outcome = _run([_criterion("command_exit", "present", command="python3 -c \"print(1)\"", expected=0)])
    assert outcome.run_status is runs.RunStatus.COMPLETED_VERIFIED
    data = _saved()
    assert data["evidence"], "a command verification must leave an evidence record"
    data["evidence"][0]["run_id"] = "run_" + "0" * 32
    with pytest.raises(session.SessionValidationError):
        session.validate_session_data(data)


def test_verified_run_without_passed_criteria_is_rejected_on_load(monkeypatch, tmp_path):
    (tmp_path / "calc.py").write_text("x = 1\n")
    _script(monkeypatch, [_stop("done")])
    _run([_criterion("file_content", "present", path="calc.py", contains="x = 1")])
    data = _saved()
    data["verification_results"] = []
    with pytest.raises(session.SessionValidationError):
        session.validate_session_data(data)


def test_format_five_cannot_carry_completed_verified(monkeypatch, tmp_path):
    (tmp_path / "calc.py").write_text("x = 1\n")
    _script(monkeypatch, [_stop("done")])
    _run([_criterion("file_content", "present", path="calc.py", contains="x = 1")])
    data = _saved()
    data["format"] = 5
    data.pop("criteria", None)
    data.pop("verification_results", None)
    with pytest.raises(session.SessionValidationError):
        session.validate_session_data(data)


def test_sequential_goals_keep_separate_criteria(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")
    _script(monkeypatch, [_stop("first"), _stop("second")])
    rt = AgentRuntime()
    try:
        sess = rt.open_session("two-goals")
        first = rt.event_loop.run_until_complete(sess.run_goal(
            "goal one", criteria=[_criterion("file_content", "g1", path="a.txt", contains="alpha")]))
        second = rt.event_loop.run_until_complete(sess.run_goal("goal two"))
        assert first.run_status is runs.RunStatus.COMPLETED_VERIFIED
        assert second.run_status is runs.RunStatus.COMPLETED_UNVERIFIED
        saved = _saved("two-goals")
        assert {c["run_id"] for c in saved["criteria"]} == {first.run_id}
        assert [r["criterion_id"] for r in saved["verification_results"]] == ["g1"]
        sess.close()
    finally:
        rt.shutdown()


def test_existing_unknown_from_an_earlier_run_is_not_reclassified(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("alpha")
    _script(monkeypatch, [_stop("first")])
    rt = AgentRuntime()
    try:
        sess = rt.open_session("prior-unknown")
        rt.event_loop.run_until_complete(sess.run_goal("first goal"))
        sess.close()
    finally:
        rt.shutdown()
    data = _saved("prior-unknown")
    data["execution"].append({"tool_call_id": "old_call", "tool_name": "write_file",
                              "arguments": {"path": "a.txt"}, "status": "unknown",
                              "outcome": "unknown outcome from an earlier run",
                              "error_category": "unknown_outcome"})
    session.save_session("prior-unknown", data["goal"], data["messages"], objective=data["objective"],
                         summary=data["summary"], execution=data["execution"], runs=data["runs"],
                         evidence=data["evidence"], legacy_history=data["legacy_history"])
    assert any(r["tool_call_id"] == "old_call" and r["status"] == "unknown" for r in _saved("prior-unknown")["execution"])


# --- regressions found in review ---------------------------------------------------------------------

def test_cli_resume_keeps_prior_run_records(monkeypatch, tmp_path):
    """CLI resume used to replace the resume kwargs and drop earlier RunOutcomes. They must survive."""
    from src import cli
    _script(monkeypatch, [_stop("first")])
    rt = AgentRuntime()
    try:
        sess = rt.open_session("cli-resume")
        first = rt.event_loop.run_until_complete(sess.run_goal("first goal"))
        sess.close()
    finally:
        rt.shutdown()
    _script(monkeypatch, [_stop("second")])
    monkeypatch.setattr(cli, "load_mcp_config", lambda: {})
    cli._main(["resume", "cli-resume"])
    saved = _saved("cli-resume")
    assert [r["run_id"] for r in saved["runs"]][0] == first.run_id, "earlier RunOutcomes must not be lost"
    assert len(saved["runs"]) == 2


def test_resumed_verified_session_keeps_criteria_and_results_and_new_runs_get_their_own(monkeypatch, tmp_path):
    (tmp_path / "calc.py").write_text("x = 1\n")
    _script(monkeypatch, [_stop("done")])
    first = _run([_criterion("file_content", "present", path="calc.py", contains="x = 1")], sid="carry")
    assert first.run_status is runs.RunStatus.COMPLETED_VERIFIED
    before = _saved("carry")

    (tmp_path / "calc.py").write_text("x = 2 # changed\n")
    _script(monkeypatch, [_stop("second")])

    async def resumed():
        return await loop.agent_loop_async(
            "second goal", MCPClientManager(), asyncio.get_running_loop(), session_id="carry",
            resume_messages=before["messages"], resume_objective=before["objective"],
            resume_summary=before["summary"], resume_execution=before["execution"],
            resume_runs=before["runs"], resume_evidence=before["evidence"],
            resume_criteria=before["criteria"], resume_verification_results=before["verification_results"],
            criteria=[_criterion("file_content", "later", path="calc.py", contains="x = 2")])
    second = asyncio.run(resumed())
    assert second.run_status is runs.RunStatus.COMPLETED_VERIFIED
    after = _saved("carry")
    session.validate_session_data(after)
    assert after["format"] == 6
    assert {c["criterion_id"] for c in after["criteria"]} == {"present", "later"}
    old = [r for r in after["verification_results"] if r["criterion_id"] == "present"]
    assert old == [r for r in before["verification_results"] if r["criterion_id"] == "present"], \
        "earlier verification observations must not be rewritten"


def test_too_many_criteria_are_recorded_as_not_checked_never_passed():
    import asyncio as _asyncio
    criteria = [acceptance.AcceptanceCriterion(f"c{i}", "run_x", "file_exists", True, "user",
                                               parameters={"path": "."}) for i in range(3)]
    engine = verification.VerificationEngine(max_checks=1)
    results = _asyncio.run(engine.verify(criteria, "run_x", attempt=1))
    statuses = {r.criterion_id: r.status for r in results}
    assert statuses["c1"] == "inconclusive" and statuses["c2"] == "inconclusive"
    assert "not checked" in next(r.observation for r in results if r.criterion_id == "c2")


def test_job_state_is_read_from_a_real_status_envelope():
    from src import envelope
    assert verification._job_state(envelope.success('{"state": "running", "job_id": "x"}')) == "running"
    assert verification._job_state(envelope.error_validation("no such job", fixable=False)) is None
    assert verification._job_state({"state": "completed"}) == "completed"
    assert verification._job_state(envelope.success("not json")) is None


def test_results_are_unique_per_criterion_and_attempt():
    criterion = acceptance.AcceptanceCriterion("c", "run_" + "a" * 32, "file_exists", True, "user",
                                               parameters={"path": "."})
    first = verification.VerificationResult("c", criterion.run_id, 1, 2, "failed", "x", "2026-01-01T00:00:00")
    second = verification.VerificationResult("c", criterion.run_id, 2, 2, "passed", "y", "2026-01-01T00:00:01")
    assert [r.attempt for r in verification.validate_results([first.to_dict(), second.to_dict()], [criterion])] == [1, 2]
    with pytest.raises(ValueError):
        verification.validate_results([first.to_dict(), first.to_dict()], [criterion])


def test_evidence_pointing_at_a_foreign_journal_call_is_rejected(monkeypatch, tmp_path):
    _script(monkeypatch, [_stop("done")])
    outcome = _run([_criterion("command_exit", "present", command="python3 -c \"print(1)\"", expected=0)])
    assert outcome.run_status is runs.RunStatus.COMPLETED_VERIFIED
    data = _saved()
    data["evidence"][0]["journal_call_id"] = "call_not_in_this_journal"
    with pytest.raises(session.SessionValidationError):
        session.validate_session_data(data)


def test_only_the_verified_save_failing_is_not_reported_as_verified(monkeypatch, tmp_path):
    (tmp_path / "result.txt").write_text("fixed")
    real_save = session.save_session

    def fail_only_verified(*args, **kwargs):
        if any(r.get("status") == runs.RunStatus.COMPLETED_VERIFIED.value for r in kwargs.get("runs", [])):
            raise OSError("simulated disk failure on the verified save")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(session, "save_session", fail_only_verified)
    _script(monkeypatch, [_stop("done")])
    outcome = _run([_criterion("file_content", "result", path="result.txt", contains="fixed")])
    assert outcome.run_status is runs.RunStatus.NEEDS_ATTENTION
    assert "not saved" in outcome.reason
    # The durable file never claims verified success
    assert all(r["status"] != "completed_verified" for r in (session.load_session("accept") or {}).get("runs", []))
