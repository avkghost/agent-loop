"""B3.5 verification attempt integrity. Scripted model, real bounded local tools, no network."""

import asyncio
import copy
import json

import pytest

from src import acceptance, loop, runs, session, verification
from src.runtime import AgentRuntime
from src.tools.mcp_client import MCPClientManager
from tests import scripted_provider
from tests.conftest import make_streamed_response
from tests.test_acceptance_negative import _criterion, _env, _run, _saved, _script, _stop, _turn  # noqa: F401
from tests.test_session_ownership import JOB_COMMAND, _alive, _job_id_from_start, _pid_of, _tool_turn


def _verified_document(monkeypatch, tmp_path, sid="history", cycles=2):
    """A saved format-6 session whose run verified in its first cycle."""
    (tmp_path / "result.txt").write_text("fixed")
    _script(monkeypatch, [_stop("done")])
    outcome = _run([_criterion("file_content", "result", path="result.txt", contains="fixed")],
                   sid=sid, cycles=cycles)
    assert outcome.run_status is runs.RunStatus.COMPLETED_VERIFIED
    return copy.deepcopy(_saved(sid))


def _rejects(document):
    with pytest.raises(session.SessionValidationError):
        session.validate_session_data(document)


def _result(criterion_id, run_id, attempt, budget, status="passed"):
    return {"criterion_id": criterion_id, "run_id": run_id, "attempt": attempt, "budget": budget,
            "status": status, "observation": "observed", "checked_at": "2026-01-01T00:00:00"}


# 1-3, 6-7: attempt numbers ----------------------------------------------------------------------------

def test_lone_attempt_999_marked_verified_is_rejected(monkeypatch, tmp_path):
    document = _verified_document(monkeypatch, tmp_path)
    document["verification_results"][0]["attempt"] = 999
    _rejects(document)


def test_lone_attempt_2_without_attempt_1_is_rejected(monkeypatch, tmp_path):
    document = _verified_document(monkeypatch, tmp_path)
    document["verification_results"][0]["attempt"] = 2
    _rejects(document)


def test_valid_single_attempt_passing_sequence_is_accepted(monkeypatch, tmp_path):
    document = _verified_document(monkeypatch, tmp_path)
    assert [r["attempt"] for r in document["verification_results"]] == [1]
    session.validate_session_data(document)


@pytest.mark.parametrize("bad", [True, False, "1", 0, -1, 1.0, None])
def test_invalid_attempt_types_and_values_are_rejected(monkeypatch, tmp_path, bad):
    document = _verified_document(monkeypatch, tmp_path)
    document["verification_results"][0]["attempt"] = bad
    _rejects(document)


def test_attempts_above_the_configured_budget_are_rejected(monkeypatch, tmp_path):
    document = _verified_document(monkeypatch, tmp_path, cycles=2)
    document["verification_results"][0]["attempt"] = 3
    _rejects(document)


def test_budget_above_the_maximum_is_rejected(monkeypatch, tmp_path):
    document = _verified_document(monkeypatch, tmp_path)
    document["verification_results"][0]["budget"] = verification.MAX_VERIFICATION_CYCLES + 1
    _rejects(document)


# 4-5, 9: multi-cycle history --------------------------------------------------------------------------

def test_failed_attempt_1_then_passed_attempt_2_is_accepted_and_history_is_kept(monkeypatch, tmp_path):
    (tmp_path / "calc.py").write_text("x = 1\n")
    _script(monkeypatch, [
        _stop("first"),                                             # cycle 1 fails: content not yet written
        _turn("write_file", {"path": "calc.py", "content": "x = 1  # fixed\n"}, "w1"),
        _stop("second"),                                            # cycle 2 passes
    ])
    outcome = _run([_criterion("file_content", "present", path="calc.py", contains="# fixed")], sid="two-cycles", cycles=2)
    assert outcome.run_status is runs.RunStatus.COMPLETED_VERIFIED
    saved = _saved("two-cycles")
    session.validate_session_data(saved)
    history = [(r["attempt"], r["status"]) for r in saved["verification_results"]]
    assert history == [(1, "failed"), (2, "passed")], "earlier failure must be preserved"


def test_valid_multi_criterion_history_is_accepted(monkeypatch, tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")
    _script(monkeypatch, [_stop("first"), _stop("second")])
    criteria = [_criterion("file_content", "a", path="a.txt", contains="alpha"),
                _criterion("file_content", "b", path="b.txt", contains="gamma")]
    outcome = _run(criteria, sid="multi", cycles=2)
    assert outcome.run_status is runs.RunStatus.NEEDS_ATTENTION
    saved = _saved("multi")
    session.validate_session_data(saved)
    assert {(r["criterion_id"], r["attempt"]) for r in saved["verification_results"]} == {
        ("a", 1), ("b", 1), ("a", 2), ("b", 2)}


# 8: partial and cancelled histories -------------------------------------------------------------------

def test_cancelled_second_cycle_leaves_recoverable_history(monkeypatch, tmp_path):
    modes = iter(["fail", "cancel"])  # cycle 1 fails, cycle 2 is cancelled during its check
    state = {"mode": None}
    real = loop.dispatch.dispatch_tool_async

    async def scripted_verification(tool_name, tool_input, *args, **kwargs):
        if tool_name == "run_command" and state["mode"] is not None:
            if state["mode"] == "cancel":
                raise asyncio.CancelledError()
            exit_code = 1 if state["mode"] == "fail" else 0
            return {"content": "", "isError": False, "exitCode": exit_code, "processGroupEmpty": True}
        return await real(tool_name, tool_input, *args, **kwargs)

    monkeypatch.setattr(loop.dispatch, "dispatch_tool_async", scripted_verification)
    _script(monkeypatch, [_stop("first"), _stop("second")])
    rt = AgentRuntime()
    try:
        sess = rt.open_session("cancel-cycle")
        criteria = [_criterion("command_exit", "check", command="python3 -c \"print(1)\"", expected=0)]
        state["mode"] = next(modes)
        original_verify = loop.verification.VerificationEngine.verify

        async def mode_per_cycle(self, *args, **kwargs):
            if kwargs.get("attempt", 1) > 1:
                state["mode"] = next(modes, "cancel")
            return await original_verify(self, *args, **kwargs)

        monkeypatch.setattr(loop.verification.VerificationEngine, "verify", mode_per_cycle)
        with pytest.raises(asyncio.CancelledError):
            rt.event_loop.run_until_complete(sess.run_goal("cancel during cycle two", criteria=criteria,
                                                           verification_cycles=2))
        saved = _saved("cancel-cycle")
        session.validate_session_data(saved)
        assert [(r["attempt"], r["status"]) for r in saved["verification_results"]] == [(1, "failed")]
        assert [r["status"] for r in saved["runs"]] == ["interrupted"]

        # The session remains usable: a new goal with its own criteria verifies in a new run
        _script(monkeypatch, [_stop("recovered")])
        state["mode"] = "ok"
        second = rt.event_loop.run_until_complete(sess.run_goal(
            "recover", criteria=[_criterion("command_exit", "recheck", command="python3 -c \"print(1)\"", expected=0)]))
        assert second.run_status is runs.RunStatus.COMPLETED_VERIFIED
        session.validate_session_data(_saved("cancel-cycle"))
        sess.close()
    finally:
        rt.shutdown()


# 10: forged entries ------------------------------------------------------------------------------------

def test_forged_high_number_pass_cannot_override_a_failed_latest_attempt(monkeypatch, tmp_path):
    (tmp_path / "calc.py").write_text("x = 1\n")
    _script(monkeypatch, [_stop("first"), _stop("second")])
    outcome = _run([_criterion("file_content", "present", path="calc.py", contains="never")], sid="forge", cycles=2)
    assert outcome.run_status is runs.RunStatus.NEEDS_ATTENTION
    document = copy.deepcopy(_saved("forge"))
    run_id = document["runs"][-1]["run_id"]
    document["verification_results"].append(_result("present", run_id, 999, 2))
    _rejects(document)


def test_forged_duplicate_pass_for_a_failed_attempt_is_rejected(monkeypatch, tmp_path):
    (tmp_path / "calc.py").write_text("x = 1\n")
    _script(monkeypatch, [_stop("first"), _stop("second")])
    _run([_criterion("file_content", "present", path="calc.py", contains="never")], sid="forge-dup", cycles=2)
    document = copy.deepcopy(_saved("forge-dup"))
    failed = next(r for r in document["verification_results"] if r["attempt"] == 2)
    document["verification_results"].append(dict(failed, status="passed"))
    _rejects(document)


def test_verified_run_with_a_missing_cycle_result_is_rejected(monkeypatch, tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")
    _script(monkeypatch, [_stop("done")])
    _run([_criterion("file_content", "a", path="a.txt", contains="alpha"),
          _criterion("file_content", "b", path="b.txt", contains="beta")], sid="partial-verified", cycles=2)
    document = copy.deepcopy(_saved("partial-verified"))
    document["verification_results"] = [r for r in document["verification_results"] if r["criterion_id"] != "b"]
    _rejects(document)


# 11-12: save and load -----------------------------------------------------------------------------------

def test_invalid_save_preserves_the_previously_valid_file(monkeypatch, tmp_path):
    document = _verified_document(monkeypatch, tmp_path, sid="kept")
    path = tmp_path / "kept.json"
    before = path.read_bytes()
    messages = document["messages"]
    bad_results = [dict(document["verification_results"][0], attempt=999)]
    with pytest.raises(session.SessionValidationError):
        session.save_session("kept", document["goal"], messages, objective=document["objective"],
                             summary=document["summary"], execution=document["execution"],
                             runs=document["runs"], evidence=document["evidence"],
                             legacy_history=document["legacy_history"],
                             criteria=document["criteria"], verification_results=bad_results)
    assert path.read_bytes() == before
    assert not (tmp_path / "kept.json.tmp").exists()


def test_save_and_load_preserve_attempt_history_and_run_outcomes(monkeypatch, tmp_path):
    (tmp_path / "calc.py").write_text("x = 1\n")
    _script(monkeypatch, [
        _stop("first"),
        _turn("write_file", {"path": "calc.py", "content": "x = 1  # fixed\n"}, "w1"),
        _stop("second"),
    ])
    outcome = _run([_criterion("file_content", "present", path="calc.py", contains="x = 1")], sid="roundtrip", cycles=2)
    loaded = _saved("roundtrip")
    session.validate_session_data(loaded)
    session.save_session("roundtrip", loaded["goal"], loaded["messages"], objective=loaded["objective"],
                         summary=loaded["summary"], execution=loaded["execution"], runs=loaded["runs"],
                         evidence=loaded["evidence"], legacy_history=loaded["legacy_history"],
                         criteria=loaded["criteria"], verification_results=loaded["verification_results"])
    again = _saved("roundtrip")
    assert again["verification_results"] == loaded["verification_results"]
    assert again["runs"] == loaded["runs"]
    assert again["runs"][-1]["run_id"] == outcome.run_id and again["runs"][-1]["status"] == "completed_verified"


# Managed-job coverage (secondary) ----------------------------------------------------------------------

def test_job_state_criterion_observes_a_real_owned_job_and_isolation_holds(monkeypatch, tmp_path):
    from src import jobs
    from src.authorization import SessionAuthorization
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)
    rt = AgentRuntime()
    try:
        owner = rt.open_session("job-owner")
        _script(monkeypatch, [_tool_turn("start_job", {"command": JOB_COMMAND, "lifetime_seconds": 60}, "j1"),
                              _stop("started")])
        rt.event_loop.run_until_complete(owner.run_goal("start a job"))
        job_id = _job_id_from_start("job-owner", "j1")
        pid = _pid_of(job_id)
        assert _alive(pid)

        # The owning session verifies the real observed state
        _script(monkeypatch, [_stop("checked")])
        verified = rt.event_loop.run_until_complete(owner.run_goal(
            "verify the job", criteria=[_criterion("job_state", "job", job_id=job_id, expected="running")]))
        assert verified.run_status is runs.RunStatus.COMPLETED_VERIFIED

        # Another session cannot observe it: the answer is "not found", which is inconclusive, never a pass
        intruder = rt.open_session("job-intruder")
        _script(monkeypatch, [_stop("checked"), _stop("again")])
        denied = rt.event_loop.run_until_complete(intruder.run_goal(
            "verify someone else's job", criteria=[_criterion("job_state", "job", job_id=job_id, expected="running")],
            verification_cycles=1))
        assert denied.run_status is runs.RunStatus.NEEDS_ATTENTION
        assert _saved("job-intruder")["verification_results"][0]["status"] == "inconclusive"

        owner.close()
        assert not _alive(pid), "session close must stop the owned job"
        intruder.close()
    finally:
        rt.shutdown()


def test_verification_budget_outside_the_supported_range_is_refused_before_any_model_request(monkeypatch, tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    post = _script(monkeypatch, [_stop("should never be requested")])
    outcome = _run([_criterion("file_content", "a", path="a.txt", contains="alpha")], sid="budget-range",
                   cycles=verification.MAX_VERIFICATION_CYCLES + 1)
    assert outcome.run_status is runs.RunStatus.NEEDS_ATTENTION
    assert "verification cycles" in outcome.reason
    assert post == [], "no model request may be made for a refused budget"
