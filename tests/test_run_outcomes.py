"""B3.2: explicit run outcomes, evidence records, format 5 persistence, and recovery.

Deterministic and local: scripted model replies, scripted tools, temporary session directories.
No live model, MCP server, network service, or destructive command.
"""

import asyncio
import json
import os

import pytest

from src import config, dispatch, envelope, execution, loop, runs, session
from src.session import SessionValidationError, validate_session_data
from src.tools.mcp_client import MCPClientManager
from tests import scripted_provider
from tests.conftest import make_streamed_response

S = runs.RunStatus


def _drive(monkeypatch, replies, *, dispatch_fn=None, session_id=None, **overrides):
    """Run one goal through the real loop. Returns (outcome, post, dispatched)."""
    post = scripted_provider.POST
    scripted = list(replies)

    def scripted_post(url, json=None, timeout=None, stream=False):
        if not scripted:
            raise AssertionError("unexpected extra model request")
        item = scripted.pop(0)
        if callable(item) and not hasattr(item, "iter_lines"):
            return item(json)
        return item

    dispatched = []

    async def fake_dispatch(tool_name, tool_input, *args, **kwargs):
        dispatched.append((tool_name, tool_input))
        return dispatch_fn(tool_name, tool_input) if dispatch_fn else envelope.success("tool output")

    monkeypatch.setattr(scripted_provider, "POST", scripted_post)
    monkeypatch.setattr(dispatch, "dispatch_tool_async", fake_dispatch)
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)
    for name, value in overrides.items():
        monkeypatch.setattr(config, name, value)
    async def run():
        return await loop.agent_loop_async("goal", MCPClientManager(), asyncio.get_running_loop(),
                                           session_id=session_id)

    return asyncio.run(run()), post, dispatched


def _call_reply(call_id, name="list_directory", arguments=None):
    return make_streamed_response(finish_reason="tool_calls", tool_calls=[
        {"id": call_id, "name": name, "arguments": json.dumps(arguments or {"path": "."})}])


def _stop_reply(content="done"):
    return make_streamed_response(finish_reason="stop", content=content)


# ---------------------------------------------------------------------------
# Run status: every exit path produces one explicit outcome
# ---------------------------------------------------------------------------

def test_model_stop_is_completed_unverified_never_verified(monkeypatch):
    outcome, _, _ = _drive(monkeypatch, [_stop_reply("all finished")])
    assert outcome.run_status is S.COMPLETED_UNVERIFIED
    assert outcome.final_message == "all finished"
    assert "no verification" in outcome.reason
    assert outcome.iteration_count == 1 and outcome.finished_at is not None


def test_completed_verified_requires_the_explicit_verification_gate():
    recorder = runs.RunRecorder(goal="g")
    with pytest.raises(runs.RunRecordError, match="reserved"):
        recorder.end(S.COMPLETED_VERIFIED, "forged")
    recorder.end_verified("criteria passed")
    assert recorder.finish().run_status is S.COMPLETED_VERIFIED
    assert runs.RunOutcome.from_dict(dict(_outcome_dict(), status="completed_verified"))


def test_provider_timeout_exhaustion_is_provider_failed(monkeypatch):
    import requests

    def timeout(url, json=None, timeout=None, stream=False):
        raise requests.exceptions.Timeout("slow")

    outcome, _, dispatched = _drive(monkeypatch, [timeout] * 3, LM_STUDIO_RETRY_BASE=0)
    assert outcome.run_status is S.PROVIDER_FAILED and "timed out" in outcome.reason
    assert dispatched == []


def test_incomplete_stream_is_provider_failed_not_completed(monkeypatch):
    partial = make_streamed_response(finish_reason="stop", content="cut")
    partial._lines = [line for line in partial._lines if "finish_reason\":\"stop\"" not in line
                      and "[DONE]" not in line]
    outcome, _, dispatched = _drive(monkeypatch, [partial])
    assert outcome.run_status is S.PROVIDER_FAILED and "incomplete" in outcome.reason
    assert dispatched == []


def test_malformed_stream_is_provider_failed(monkeypatch):
    class Broken:
        def raise_for_status(self):
            return None

        def iter_lines(self):
            return iter([b"data: {not json", b"data: [DONE]"])

        def close(self):
            return None

    outcome, _, dispatched = _drive(monkeypatch, [Broken()])
    assert outcome.run_status is S.PROVIDER_FAILED and "malformed" in outcome.reason
    assert dispatched == []


def test_length_limited_response_is_limit_reached(monkeypatch):
    outcome, _, dispatched = _drive(monkeypatch, [make_streamed_response(finish_reason="length", content="x")])
    assert outcome.run_status is S.LIMIT_REACHED and "length" in outcome.reason
    assert dispatched == []


def test_iteration_budget_is_limit_reached(monkeypatch):
    outcome, _, _ = _drive(monkeypatch, [_call_reply("c1"), _call_reply("c2")], MAX_ITERATIONS=2)
    assert outcome.run_status is S.LIMIT_REACHED and outcome.iteration_count == 2
    assert "iteration limit" in outcome.reason


def test_recoverable_tool_failure_continues_the_run(monkeypatch):
    outcome, _, dispatched = _drive(
        monkeypatch,
        [_call_reply("c1"), _stop_reply("recovered")],
        dispatch_fn=lambda name, args: envelope.error_validation("missing path", fixable=True),
    )
    assert len(dispatched) == 1
    assert outcome.run_status is S.COMPLETED_UNVERIFIED, "a recoverable tool error must not end the run"


def test_user_permission_denial_is_blocked_and_stops_the_run(monkeypatch, tmp_sessions):
    monkeypatch.setattr(loop, "_get_single_key_input", lambda prompt, valid_keys: "3")
    monkeypatch.setattr(config, "ALLOW_ALL", False)
    monkeypatch.setattr(config, "ALLOWED_TOOLS", set())

    def gate(name, args):
        # What the real dispatcher answers for a command outside the allowlist
        return {"requiresPermission": True, "kind": "command", "identifier": name, "arguments": []}

    outcome, _, dispatched = _drive(monkeypatch, [
        make_streamed_response(finish_reason="tool_calls", tool_calls=[
            {"id": "c_deny", "name": "grep", "arguments": json.dumps({"pattern": "x"})},
            {"id": "c_after", "name": "list_directory", "arguments": json.dumps({"path": "."})}]),
        _stop_reply("never reached"),
    ], dispatch_fn=gate, session_id="denied")
    assert outcome.run_status is S.BLOCKED
    assert "denied permission" in outcome.reason
    assert [name for name, _ in dispatched] == ["grep"], "the calls after a denial are never dispatched"
    journal = session.load_session("denied")["execution"]
    after = next(r for r in journal if r["tool_call_id"] == "c_after")
    assert after["status"] == "failed" and after["error_category"] == "permission"


def test_ambiguous_external_operation_makes_a_model_stop_unknown(monkeypatch):
    outcome, _, _ = _drive(
        monkeypatch,
        [_call_reply("c_ambiguous", name="write_file", arguments={"path": "x", "content": "y"}),
         _stop_reply("it is written")],
        dispatch_fn=lambda name, args: envelope.error_unknown_outcome("connection lost mid-write"),
    )
    assert outcome.run_status is S.UNKNOWN, "a model stop cannot clear an unknown external effect"
    assert "unknown outcome" in outcome.reason


def test_cancellation_records_interrupted_and_propagates(monkeypatch, tmp_sessions):
    class Cancelled:
        def raise_for_status(self):
            return None

        def iter_lines(self):
            raise asyncio.CancelledError()

        def close(self):
            return None

    with pytest.raises(asyncio.CancelledError):
        _drive(monkeypatch, [Cancelled()], session_id="cancelled-run")
    saved = session.load_session("cancelled-run")
    assert saved["runs"][0]["status"] == "interrupted"
    assert "cancelled" in saved["runs"][0]["reason"]


def test_unexpected_error_is_unknown_and_not_swallowed(monkeypatch, tmp_sessions):
    def boom(tool_name, args):
        raise RuntimeError("tool bug")

    with pytest.raises(RuntimeError, match="tool bug"):
        _drive(monkeypatch, [_call_reply("c_boom")], dispatch_fn=boom, session_id="bug-run")
    saved = session.load_session("bug-run")
    assert saved["runs"][0]["status"] == "unknown"
    assert "RuntimeError" in saved["runs"][0]["reason"]


def test_runtime_interrupt_is_recorded_as_interrupted(monkeypatch, tmp_sessions):
    def interrupt(tool_name, args):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        _drive(monkeypatch, [_call_reply("c_int")], dispatch_fn=interrupt, session_id="int-run")
    assert session.load_session("int-run")["runs"][0]["status"] == "interrupted"


def test_unsaveable_session_is_needs_attention(monkeypatch, tmp_sessions):
    def refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(session, "save_session", refuse)
    outcome, _, dispatched = _drive(monkeypatch, [_stop_reply()], session_id="no-disk")
    assert outcome.run_status is S.NEEDS_ATTENTION and "durability" in outcome.reason
    assert dispatched == []


def test_every_run_has_a_distinct_identifier_separate_from_the_session(monkeypatch, tmp_sessions):
    first, _, _ = _drive(monkeypatch, [_stop_reply()], session_id="session-one")
    second, _, _ = _drive(monkeypatch, [_stop_reply()], session_id="session-two")
    assert first.run_id != second.run_id and first.run_id.startswith("run_")
    assert [r["run_id"] for r in session.load_session("session-one")["runs"]] == [first.run_id]
    assert [r["run_id"] for r in session.load_session("session-two")["runs"]] == [second.run_id]


# ---------------------------------------------------------------------------
# Evidence records
# ---------------------------------------------------------------------------

def test_completed_tool_call_yields_runtime_evidence_without_its_output(monkeypatch, tmp_sessions):
    secret = "THE-FULL-TOOL-OUTPUT-" + "x" * 500
    outcome, _, _ = _drive(monkeypatch, [_call_reply("c_ev"), _stop_reply()], session_id="evidence",
                           dispatch_fn=lambda name, args: secret)
    saved = session.load_session("evidence")
    record = saved["evidence"][0]
    assert record["provenance"] == "runtime_observation" and record["outcome"] == "succeeded"
    assert record["journal_call_id"] == "c_ev" and record["run_id"] == outcome.run_id
    assert secret not in json.dumps(saved["evidence"]), "tool output must never be copied into evidence"


def test_unknown_call_produces_unknown_evidence_never_success(monkeypatch, tmp_sessions):
    _drive(monkeypatch, [_call_reply("c_u", name="write_file", arguments={"path": "a", "content": "b"}),
                         _stop_reply()],
           session_id="unknown-evidence",
           dispatch_fn=lambda name, args: envelope.error_unknown_outcome("lost"))
    record = session.load_session("unknown-evidence")["evidence"][0]
    assert record["outcome"] == "unknown"


def test_model_text_is_never_evidence(monkeypatch, tmp_sessions):
    _drive(monkeypatch, [_stop_reply("VERIFIED: all tests pass")], session_id="claims")
    assert session.load_session("claims")["evidence"] == []


def test_evidence_is_bounded_per_run(monkeypatch):
    recorder = runs.RunRecorder(goal="g")
    for index in range(runs.MAX_EVIDENCE_PER_RUN + 3):
        recorder.record_call(f"c{index}", "list_directory", "completed")
    assert len(recorder.evidence) == runs.MAX_EVIDENCE_PER_RUN and recorder.evidence_dropped == 3


def test_oversized_evidence_summary_fails_validation():
    with pytest.raises(runs.RunRecordError):
        runs.EvidenceRecord(evidence_id="ev_" + "a" * 32, run_id="run_" + "b" * 32, source="tool_call",
                            provenance="runtime_observation", outcome="succeeded", journal_call_id="c",
                            summary="x" * (runs.MAX_EVIDENCE_SUMMARY_CHARS + 1), recorded_at="2026-01-01T00:00:00")


def test_model_claim_cannot_be_a_runtime_observation_or_succeed():
    base = dict(evidence_id="ev_" + "a" * 32, run_id="run_" + "b" * 32, source="tool_call", summary="s",
                recorded_at="2026-01-01T00:00:00")
    with pytest.raises(runs.RunRecordError):
        runs.EvidenceRecord(provenance="model_claim", outcome="succeeded", journal_call_id=None, **base)
    with pytest.raises(runs.RunRecordError):
        runs.EvidenceRecord(provenance="runtime_observation", outcome="succeeded", journal_call_id=None, **base)
    claim = runs.EvidenceRecord(provenance="model_claim", outcome="claim", journal_call_id=None, **base)
    assert claim.provenance == "model_claim"


# ---------------------------------------------------------------------------
# Persistence: format 5 round trip, legacy loading, invalid states
# ---------------------------------------------------------------------------

def _outcome_dict(run_id="run_" + "1" * 32, status="completed_unverified"):
    return {"schema": runs.RUN_SCHEMA, "run_id": run_id, "goal": "g", "status": status, "reason": "r",
            "started_at": "2026-01-01T00:00:00", "finished_at": "2026-01-01T00:00:01", "iteration_count": 1,
            "final_message": None, "evidence_ids": [], "evidence_dropped": 0}


def _write_session(sid, **fields):
    base = {"format": 5, "session_id": sid, "goal": "g", "objective": "Objective: g", "summary": None,
            "messages": [{"role": "system", "content": "sys"}], "execution": [], "runs": [],
            "evidence": [], "legacy_history": False, "saved_at": "2026-01-01T00:00:00"}
    base.update(fields)
    session.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    (session.SESSIONS_DIR / f"{sid}.json").write_text(json.dumps(base))
    return base


def test_format_five_round_trip_preserves_runs_and_evidence(tmp_sessions):
    run = runs.RunRecorder(goal="g")
    run.record_call("c1", "list_directory", "completed")
    run.end(S.COMPLETED_UNVERIFIED, "stopped", "done")
    outcome = run.finish()
    session.save_session("rt", "g", [{"role": "system", "content": "sys"}],
                         execution=[{"tool_call_id": "c1", "tool_name": "list_directory", "arguments": {},
                                     "status": "completed", "outcome": "ok", "error_category": None}],
                         runs=[outcome.to_dict()], evidence=[e.to_dict() for e in run.evidence])
    loaded = session.load_session("rt")
    assert loaded["format"] == 5
    assert runs.RunOutcome.from_dict(loaded["runs"][0]) == outcome
    assert session.resume_history(loaded)[0][0]["status"] == "completed_unverified"


def test_legacy_format_four_session_loads_with_unrecorded_history(tmp_sessions):
    _write_session("old", format=4, runs=None, evidence=None, legacy_history=None)
    data = json.loads((session.SESSIONS_DIR / "old.json").read_text())
    for key in ("runs", "evidence", "legacy_history"):
        data.pop(key)
    (session.SESSIONS_DIR / "old.json").write_text(json.dumps(data))
    assert validate_session_data(data) == "structured"
    prior, evidence, legacy = session.resume_history(data)
    assert prior == [] and evidence == [] and legacy is True, "no outcome may be invented for a legacy session"


def test_invalid_persisted_status_is_rejected(tmp_sessions):
    bad = _write_session("bad-status", runs=[_outcome_dict(status="succeeded_probably")])
    with pytest.raises(SessionValidationError, match="invalid run history"):
        validate_session_data(bad)


def test_corrupt_or_missing_run_identifiers_are_rejected(tmp_sessions):
    with pytest.raises(runs.RunRecordError):
        runs.RunOutcome.from_dict(dict(_outcome_dict(), run_id="run-42"))
    missing = _outcome_dict()
    del missing["run_id"]
    with pytest.raises(runs.RunRecordError, match="missing"):
        runs.RunOutcome.from_dict(missing)
    with pytest.raises(SessionValidationError):
        validate_session_data(_write_session("dup", runs=[_outcome_dict(), _outcome_dict()]))


def test_running_run_cannot_claim_a_finish_time_and_terminal_run_needs_one():
    running_with_finish = dict(_outcome_dict(status="running"))
    with pytest.raises(runs.RunRecordError, match="running"):
        runs.RunOutcome.from_dict(running_with_finish)
    terminal_without = dict(_outcome_dict(), finished_at=None)
    with pytest.raises(runs.RunRecordError, match="finished_at"):
        runs.RunOutcome.from_dict(terminal_without)


def test_evidence_from_another_run_is_rejected(tmp_sessions):
    run_a, run_b = "run_" + "a" * 32, "run_" + "b" * 32
    evidence_id = "ev_" + "e" * 32
    owner = dict(_outcome_dict(run_id=run_a), evidence_ids=[evidence_id])
    other = _outcome_dict(run_id=run_b)
    record = {"schema": runs.EVIDENCE_SCHEMA, "evidence_id": evidence_id, "run_id": run_b, "criterion_id": None,
              "source": "tool_call", "provenance": "runtime_observation", "outcome": "succeeded",
              "journal_call_id": "c1", "summary": "s", "recorded_at": "2026-01-01T00:00:00"}
    journal = [{"tool_call_id": "c1", "tool_name": "t", "arguments": {}, "status": "completed",
                "outcome": "ok", "error_category": None}]
    with pytest.raises(SessionValidationError, match="invalid run history"):
        validate_session_data(_write_session("cross", runs=[owner, other], evidence=[record], execution=journal,
                                             messages=[{"role": "system", "content": "sys"}]))


def test_evidence_cannot_claim_an_outcome_its_journal_record_does_not_have(tmp_sessions):
    run = "run_" + "c" * 32
    evidence_id = "ev_" + "f" * 32
    owner = dict(_outcome_dict(run_id=run), evidence_ids=[evidence_id])
    record = {"schema": runs.EVIDENCE_SCHEMA, "evidence_id": evidence_id, "run_id": run, "criterion_id": None,
              "source": "tool_call", "provenance": "runtime_observation", "outcome": "succeeded",
              "journal_call_id": "c2", "summary": "s", "recorded_at": "2026-01-01T00:00:00"}
    journal = [{"tool_call_id": "c2", "tool_name": "t", "arguments": {}, "status": "unknown",
                "outcome": "lost", "error_category": "unknown_outcome"}]
    with pytest.raises(SessionValidationError, match="claims succeeded"):
        validate_session_data(_write_session("fake-success", runs=[owner], evidence=[record], execution=journal))


def test_evidence_citing_a_missing_journal_record_is_rejected(tmp_sessions):
    run = "run_" + "d" * 32
    evidence_id = "ev_" + "0" * 32
    owner = dict(_outcome_dict(run_id=run), evidence_ids=[evidence_id])
    record = {"schema": runs.EVIDENCE_SCHEMA, "evidence_id": evidence_id, "run_id": run, "criterion_id": None,
              "source": "tool_call", "provenance": "runtime_observation", "outcome": "failed",
              "journal_call_id": "nope", "summary": "s", "recorded_at": "2026-01-01T00:00:00"}
    with pytest.raises(SessionValidationError, match="does not exist"):
        validate_session_data(_write_session("missing-ref", runs=[owner], evidence=[record]))


def test_running_run_is_recovered_as_interrupted_not_completed(tmp_sessions):
    _write_session("crashed", runs=[dict(_outcome_dict(status="running"), finished_at=None)])
    prior, _, _ = session.resume_history(json.loads((session.SESSIONS_DIR / "crashed.json").read_text()))
    assert prior[0]["status"] == "interrupted"
    assert "not re-executed" in prior[0]["reason"]


def test_atomic_write_failure_keeps_the_previous_session(tmp_sessions, monkeypatch):
    session.save_session("atomic", "g", [{"role": "system", "content": "sys"}])
    before = (session.SESSIONS_DIR / "atomic.json").read_text()

    def fail_replace(src, dst):
        raise OSError("replace refused")

    monkeypatch.setattr(session.os, "replace", fail_replace)
    with pytest.raises(OSError):
        session.save_session("atomic", "g", [{"role": "system", "content": "new"}])
    assert (session.SESSIONS_DIR / "atomic.json").read_text() == before
    assert not list(session.SESSIONS_DIR.glob("atomic.json.tmp"))


def test_run_recorder_keeps_the_first_terminal_status():
    recorder = runs.RunRecorder(goal="g")
    recorder.end(S.BLOCKED, "first")
    recorder.end(S.COMPLETED_UNVERIFIED, "second")
    assert recorder.finish().status == "blocked"
