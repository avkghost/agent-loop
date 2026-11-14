"""B3.5 deterministic acceptance criteria and verification-gate tests."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from src import acceptance, config, loop, runs, session, verification
from src import provider
from src.tools.mcp_client import MCPClientManager
from tests import scripted_provider
from tests.conftest import make_streamed_response


def _criterion(kind, **kwargs):
    return acceptance.AcceptanceCriterion(
        criterion_id=kwargs.pop("criterion_id", "c1"), run_id=kwargs.pop("run_id", ""),
        criterion_type=kind, expected=kwargs.pop("expected", True),
        origin=kwargs.pop("origin", "user"), parameters=kwargs)


def _run(monkeypatch, tmp_path, replies, criteria, *, verification_cycles=2, dispatch=None):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)
    scripted = list(replies)

    def post(*args, **kwargs):
        if not scripted:
            raise AssertionError("unexpected model request")
        return scripted.pop(0)

    monkeypatch.setattr(scripted_provider, "POST", post)
    if dispatch is None:
        async def dispatch(*args, **kwargs):
            # What the foreground shell reports for a command that exited 0 with its group confirmed empty
            return {"content": "", "isError": False, "exitCode": 0, "processGroupEmpty": True}
    monkeypatch.setattr(loop.dispatch, "dispatch_tool_async", dispatch)
    async def run():
        return await loop.agent_loop_async(
            "build and verify", MCPClientManager(), asyncio.get_running_loop(), session_id="verify",
            criteria=criteria, verification_cycles=verification_cycles)
    return asyncio.run(run())


def test_file_criterion_gate_verifies_and_round_trips(monkeypatch, tmp_path):
    (tmp_path / "result.txt").write_text("fixed")
    criterion = _criterion("file_content", path="result.txt", contains="fixed")
    outcome = _run(monkeypatch, tmp_path, [make_streamed_response(content="done")], [criterion])
    assert outcome.run_status is runs.RunStatus.COMPLETED_VERIFIED
    saved = session.load_session("verify")
    assert saved["format"] == 6
    assert saved["criteria"][0]["criterion_id"] == "c1"
    assert saved["verification_results"][0]["status"] == "passed"
    session.validate_session_data(saved)


def test_command_criterion_uses_journal_and_evidence(monkeypatch, tmp_path):
    criterion = _criterion("command_exit", command="true", expected=0)
    outcome = _run(monkeypatch, tmp_path, [make_streamed_response(content="done")], [criterion])
    assert outcome.run_status is runs.RunStatus.COMPLETED_VERIFIED
    saved = session.load_session("verify")
    record = next(item for item in saved["execution"] if item["tool_name"] == "verification_command")
    assert record["status"] == "completed"
    assert any(item["journal_call_id"] == record["tool_call_id"] for item in saved["evidence"])


def test_scripted_create_then_fix_then_verify(monkeypatch, tmp_path):
    def write_reply(content, call_id):
        return make_streamed_response(finish_reason="tool_calls", tool_calls=[
            {"id": call_id, "name": "write_file", "arguments": json.dumps(
                {"path": "service.py", "content": content})}])

    async def write_tool(tool_name, tool_input, *args, **kwargs):
        if tool_name == "write_file":
            (tmp_path / tool_input["path"]).write_text(tool_input["content"])
            return {"content": "written", "isError": False}
        return {"content": "", "isError": False}

    outcome = _run(monkeypatch, tmp_path,
                   [write_reply("broken", "w1"), write_reply("fixed", "w2"), make_streamed_response(content="ready")],
                   [_criterion("file_content", path="service.py", contains="fixed")], dispatch=write_tool)
    assert outcome.run_status is runs.RunStatus.COMPLETED_VERIFIED


def test_verification_persistence_failure_is_not_reported_as_verified(monkeypatch, tmp_path):
    (tmp_path / "result.txt").write_text("fixed")
    real_save = session.save_session
    calls = {"count": 0}

    def fail_after_initial(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] >= 2:
            raise OSError("simulated disk failure")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(session, "save_session", fail_after_initial)
    outcome = _run(monkeypatch, tmp_path, [make_streamed_response(content="done")],
                   [_criterion("file_content", path="result.txt", contains="fixed")])
    assert outcome.run_status is runs.RunStatus.NEEDS_ATTENTION
    assert outcome.run_status is not runs.RunStatus.COMPLETED_VERIFIED


def test_missing_required_file_cannot_be_claimed_verified(monkeypatch, tmp_path):
    criterion = _criterion("file_exists", path="missing.py", type="file")
    outcome = _run(monkeypatch, tmp_path,
                   [make_streamed_response(content="done"), make_streamed_response(content="still done")],
                   [criterion], verification_cycles=2)
    assert outcome.run_status is runs.RunStatus.NEEDS_ATTENTION
    assert "criteria" in outcome.reason
    assert all(item["status"] == "failed" for item in session.load_session("verify")["verification_results"])


def test_model_proposed_criterion_is_not_trusted(monkeypatch, tmp_path):
    criterion = _criterion("file_exists", path="missing.py", origin="model")
    outcome = _run(monkeypatch, tmp_path, [make_streamed_response(content="claimed")], [criterion])
    assert outcome.run_status is runs.RunStatus.COMPLETED_UNVERIFIED
    assert "no verification" in outcome.reason


def test_unknown_or_denied_command_never_passes(monkeypatch, tmp_path):
    criterion = _criterion("command_exit", command="echo verified", expected=0)

    async def denied(*args, **kwargs):
        return {"isError": True, "errorCategory": "permission", "isRetryable": False, "content": "denied"}

    outcome = _run(monkeypatch, tmp_path,
                   [make_streamed_response(content="claimed"), make_streamed_response(content="claimed again")],
                   [criterion], verification_cycles=2, dispatch=denied)
    assert outcome.run_status is runs.RunStatus.NEEDS_ATTENTION
    assert session.load_session("verify")["verification_results"][0]["status"] == "inconclusive"


def test_criterion_and_result_ownership_is_strict():
    criterion = _criterion("file_exists", criterion_id="criterion-a", run_id="run_a")
    result = verification.VerificationResult("criterion-a", "run_b", 1, 2, "passed", "observed", "2026-01-01T00:00:00")
    with pytest.raises(ValueError, match="not owned"):
        verification.validate_results([result.to_dict()], [criterion])


def test_format5_cannot_smuggle_verified_status_without_criteria():
    recorder = runs.RunRecorder(goal="g")
    recorder.end_verified("criteria passed")
    candidate = session.build_candidate(
        "sid", "g", [{"role": "system", "content": "s"}],
        runs=[recorder.finish().to_dict()], evidence=[])
    with pytest.raises(session.SessionValidationError, match="format 6"):
        session.validate_session_data(candidate)


def test_engine_rejects_escape_and_bounds_file_reads(tmp_path):
    import os
    old = os.getcwd()
    os.chdir(tmp_path)
    criterion = _criterion("file_content", path="../outside", contains="x", run_id="run_x")
    try:
        result = asyncio.run(verification.VerificationEngine().verify([criterion], "run_x"))[0]
        assert result.status == verification.VerificationStatus.FAILED.value
    finally:
        os.chdir(old)


def test_job_criterion_uses_observable_owned_state():
    criterion = _criterion("job_state", job_id="job_1", expected="completed", run_id="run_x")

    async def checker(job_id):
        assert job_id == "job_1"
        return {"state": "completed"}

    result = asyncio.run(verification.VerificationEngine().verify([criterion], "run_x", job_checker=checker))[0]
    assert result.status == verification.VerificationStatus.PASSED.value

