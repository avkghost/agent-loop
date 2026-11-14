"""B3.2 persistence integrity: one validator for save and load, atomic replacement, and aggregate bounds.

Every invalid candidate must be rejected before anything touches the disk, so the previously valid session
file is left byte-for-byte unchanged. Every successful save must load.
"""

import asyncio
import json

import pytest

from src import loop, runs, session
from src.session import SessionValidationError, validate_session_data
from src.tools.mcp_client import MCPClientManager
from tests import scripted_provider
from tests.conftest import make_streamed_response

SYS = {"role": "system", "content": "sys"}
DECLARED = {"id": "call_ok", "type": "function", "function": {"name": "list_directory", "arguments": "{}"}}


def _messages(*extra):
    return [SYS, {"role": "assistant", "content": "", "tool_calls": [DECLARED]},
            {"role": "tool", "tool_call_id": "call_ok", "content": "listing"}, *extra]


def _journal(status="completed", outcome="listing", error_category=None):
    return [{"tool_call_id": "call_ok", "tool_name": "list_directory", "arguments": {}, "status": status,
             "outcome": outcome, "error_category": error_category}]


def _seed(sid="integrity"):
    """Save a valid session and return its exact bytes: the state every rejection must preserve."""
    session.save_session(sid, "goal", _messages(), execution=_journal())
    return (session.SESSIONS_DIR / f"{sid}.json").read_bytes()


def _path(sid="integrity"):
    return session.SESSIONS_DIR / f"{sid}.json"


def _run_record(run_id="run_" + "1" * 32, evidence=()):
    return {"schema": runs.RUN_SCHEMA, "run_id": run_id, "goal": "g", "status": "completed_unverified",
            "reason": "r", "started_at": "2026-01-01T00:00:00", "finished_at": "2026-01-01T00:00:01",
            "iteration_count": 1, "final_message": None, "evidence_ids": list(evidence), "evidence_dropped": 0}


def _evidence_record(evidence_id, run_id, call_id="call_ok", outcome="succeeded"):
    return {"schema": runs.EVIDENCE_SCHEMA, "evidence_id": evidence_id, "run_id": run_id, "criterion_id": None,
            "source": "tool_call", "provenance": "runtime_observation", "outcome": outcome,
            "journal_call_id": call_id, "summary": "s", "recorded_at": "2026-01-01T00:00:00"}


def _assert_rejected_and_unchanged(sid, before, action):
    with pytest.raises(SessionValidationError):
        action()
    assert _path(sid).read_bytes() == before, "the previously valid file must not change"
    assert not list(session.SESSIONS_DIR.glob(f"{sid}.json.tmp")), "no temporary file may be left behind"
    validate_session_data(session.load_session(sid))  # the original still loads


# 1-2. Message and journal consistency, rejected before save

def test_declared_tool_call_without_a_journal_record_is_rejected_before_save(tmp_sessions):
    before = _seed()
    ghost = {"id": "call_ghost", "type": "function", "function": {"name": "run_command", "arguments": "{}"}}
    bad = _messages({"role": "assistant", "content": "", "tool_calls": [ghost]})
    _assert_rejected_and_unchanged("integrity", before,
                                   lambda: session.save_session("integrity", "goal", bad, execution=_journal()))


def test_tool_result_without_a_declaration_is_rejected_before_save(tmp_sessions):
    before = _seed()
    orphan = _messages({"role": "tool", "tool_call_id": "call_orphan", "content": "x"})
    _assert_rejected_and_unchanged("integrity", before,
                                   lambda: session.save_session("integrity", "goal", orphan,
                                                                execution=_journal()))


@pytest.mark.parametrize("journal,why", [
    (_journal(status="failed", outcome="listing", error_category="validation"), "success-looking result on a failed call"),
    (_journal(status="completed", outcome="other"), "tool result conflicts with the journal outcome"),
    (_journal(status="running", outcome=None), "active call already has a tool result"),
    (_journal(status="completed", outcome='{"content": "x", "isError": true}'), "error result on a completed call"),
])
def test_journal_states_inconsistent_with_messages_are_rejected(tmp_sessions, journal, why):
    before = _seed()
    candidate = _messages()
    if "error result" in why:
        candidate[2] = {"role": "tool", "tool_call_id": "call_ok", "content": journal[0]["outcome"]}
    with pytest.raises(SessionValidationError):
        session.save_session("integrity", "goal", candidate, execution=journal)
    assert _path().read_bytes() == before, why


# 3. A successful save loads

def test_a_successful_save_is_accepted_by_the_loader(tmp_sessions):
    _seed()
    loaded = session.load_session("integrity")
    assert loaded["format"] == 5
    assert validate_session_data(loaded) == "structured"
    assert loaded["messages"] == _messages()


# 4-6. Run and evidence ownership, evidence references, corrupt identifiers

def test_evidence_listed_by_a_run_that_does_not_own_it_is_rejected_before_save(tmp_sessions):
    before = _seed()
    run_a, run_b = "run_" + "a" * 32, "run_" + "b" * 32
    ev = "ev_" + "e" * 32
    owner = _run_record(run_a, evidence=[ev])
    other = _run_record(run_b)
    _assert_rejected_and_unchanged("integrity", before, lambda: session.save_session(
        "integrity", "goal", _messages(), execution=_journal(), runs=[owner, other],
        evidence=[_evidence_record(ev, run_b)]))


def test_evidence_citing_a_missing_journal_record_is_rejected_before_save(tmp_sessions):
    before = _seed()
    run_id, ev = "run_" + "c" * 32, "ev_" + "0" * 32
    _assert_rejected_and_unchanged("integrity", before, lambda: session.save_session(
        "integrity", "goal", _messages(), execution=_journal(), runs=[_run_record(run_id, [ev])],
        evidence=[_evidence_record(ev, run_id, call_id="call_missing")]))


def test_evidence_outcome_contradicting_its_journal_record_is_rejected_before_save(tmp_sessions):
    before = _seed()
    run_id, ev = "run_" + "d" * 32, "ev_" + "1" * 32
    _assert_rejected_and_unchanged("integrity", before, lambda: session.save_session(
        "integrity", "goal", _messages(), execution=_journal(status="unknown", outcome="lost",
                                                              error_category="unknown_outcome"),
        runs=[_run_record(run_id, [ev])], evidence=[_evidence_record(ev, run_id, outcome="succeeded")]))


@pytest.mark.parametrize("run_id", ["run-42", "run_XYZ", "", 7, None])
def test_corrupt_run_identifiers_are_rejected_before_save(tmp_sessions, run_id):
    before = _seed()
    bad = dict(_run_record(), run_id=run_id)
    _assert_rejected_and_unchanged("integrity", before, lambda: session.save_session(
        "integrity", "goal", _messages(), execution=_journal(), runs=[bad]))


# 7. Legacy compatibility and history preservation

def test_format_four_session_loads_and_its_history_survives_a_format_five_save(tmp_sessions):
    legacy = {"format": 4, "session_id": "old", "goal": "goal", "objective": "Objective: goal",
              "summary": "earlier work", "messages": _messages(), "execution": _journal(),
              "saved_at": "2026-01-01T00:00:00"}
    (session.SESSIONS_DIR / "old.json").write_text(json.dumps(legacy))
    loaded = session.load_session("old")
    assert validate_session_data(loaded) == "structured"
    prior, evidence, legacy_history = session.resume_history(loaded)
    assert prior == [] and evidence == [] and legacy_history is True

    session.save_session("old", "goal", loaded["messages"], objective=loaded["objective"],
                         summary=loaded["summary"], execution=loaded["execution"], legacy_history=True)
    reloaded = session.load_session("old")
    assert reloaded["format"] == 5 and reloaded["legacy_history"] is True
    assert reloaded["messages"] == _messages()
    assert reloaded["objective"] == "Objective: goal" and reloaded["summary"] == "earlier work"
    assert reloaded["execution"] == _journal()
    assert reloaded["runs"] == [], "no outcome may be invented for legacy history"


# 8. Atomic replacement

def test_failed_replace_keeps_the_previous_valid_file(tmp_sessions, monkeypatch):
    before = _seed()

    def refuse(src, dst):
        raise OSError("replace refused")

    monkeypatch.setattr(session.os, "replace", refuse)
    with pytest.raises(OSError):
        session.save_session("integrity", "goal", _messages("extra") if False else _messages(),
                             execution=_journal())
    assert _path().read_bytes() == before
    assert not list(session.SESSIONS_DIR.glob("integrity.json.tmp"))


# 9. Aggregate bounds: explicit errors on save and on load, exact limits accepted

def test_run_count_above_the_session_bound_is_rejected_on_save_and_on_load(tmp_sessions):
    before = _seed()
    too_many = [dict(_run_record(f"run_{i:032x}")) for i in range(runs.MAX_RUNS_PER_SESSION + 1)]
    with pytest.raises(SessionValidationError, match="at most 500 runs"):
        session.save_session("integrity", "goal", _messages(), execution=_journal(), runs=too_many)
    assert _path().read_bytes() == before
    raw = json.loads(before)
    raw.update(runs=too_many, evidence=[])
    with pytest.raises(SessionValidationError, match="at most 500 runs"):
        validate_session_data(raw)


def test_evidence_count_at_the_session_bound_is_accepted_and_one_more_is_rejected(tmp_sessions):
    per_run = runs.MAX_EVIDENCE_PER_RUN
    owners, evidence = [], []
    for run_index in range(runs.MAX_EVIDENCE_PER_SESSION // per_run):
        run_id = f"run_{run_index:032x}"
        ids = [f"ev_{run_index:08x}{k:024x}" for k in range(per_run)]
        owners.append(dict(_run_record(run_id), evidence_ids=ids))
        evidence += [_evidence_record(ev, run_id) for ev in ids]
    session.save_session("bound", "goal", [SYS], execution=_journal(), runs=owners, evidence=evidence)
    assert len(session.load_session("bound")["evidence"]) == runs.MAX_EVIDENCE_PER_SESSION

    extra_id = "ev_" + "f" * 32
    owners[0]["evidence_ids"] = owners[0]["evidence_ids"] + [extra_id]
    with pytest.raises(SessionValidationError, match="evidence"):
        session.save_session("bound", "goal", [SYS], execution=_journal(), runs=owners,
                             evidence=evidence + [_evidence_record(extra_id, owners[0]["run_id"])])


# 10. Diagnostics name the real format and invariant

def test_format_five_diagnostics_name_format_five_not_format_four(tmp_sessions):
    ghost = {"id": "call_ghost", "type": "function", "function": {"name": "x", "arguments": "{}"}}
    with pytest.raises(SessionValidationError) as exc:
        session.save_session("integrity", "goal", _messages({"role": "assistant", "content": "",
                                                              "tool_calls": [ghost]}), execution=_journal())
    message = str(exc.value)
    assert "session format 5" in message and "call_ghost" in message
    assert "format 4" not in message


# 11. The runtime's own save path: a rejected state stops the run and keeps the previous file

def test_runtime_save_of_an_inconsistent_resume_stops_the_run_and_keeps_the_file(tmp_sessions, monkeypatch, capsys):
    before = _seed("runtime-integrity")
    ghost = {"id": "call_ghost", "type": "function", "function": {"name": "run_command", "arguments": "{}"}}
    resume_messages = [SYS, {"role": "user", "content": "Objective: goal"},
                       {"role": "assistant", "content": "", "tool_calls": [ghost]}]
    monkeypatch.setattr(scripted_provider, "POST", lambda *a, **k: make_streamed_response(
        finish_reason="stop", content="never sent"))
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)

    async def run():
        return await loop.agent_loop_async("goal", MCPClientManager(), asyncio.get_running_loop(),
                                           session_id="runtime-integrity", resume_messages=resume_messages,
                                           resume_execution=[])

    outcome = asyncio.run(run())
    assert outcome.run_status is runs.RunStatus.NEEDS_ATTENTION
    assert "was not saved" in capsys.readouterr().out
    assert _path("runtime-integrity").read_bytes() == before
    validate_session_data(session.load_session("runtime-integrity"))
