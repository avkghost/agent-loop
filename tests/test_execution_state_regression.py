"""Deterministic lifecycle and recovery tests for durable tool-call state."""

import copy
import json

import pytest

from src import compaction, dispatch, envelope, execution, session
from tests.conftest import make_streamed_response, protocol_violations


def _call(call_id="call_1"):
    return {
        "id": call_id,
        "name": "read_file",
        "arguments": json.dumps({"path": "README.md"}),
    }


def test_lifecycle_transitions_are_explicit_and_serializable():
    journal = execution.ExecutionJournal()
    journal.register("call_1", "read_file", '{"path":"README.md"}')
    journal.transition("call_1", execution.ToolCallStatus.RUNNING)
    journal.transition("call_1", execution.ToolCallStatus.COMPLETED, "done")

    restored = execution.ExecutionJournal.from_dicts(journal.to_dicts())
    record = restored.get("call_1")
    assert record.status is execution.ToolCallStatus.COMPLETED
    assert record.outcome == "done"


def test_invalid_transition_and_duplicate_id_are_rejected():
    journal = execution.ExecutionJournal()
    journal.register("call_1", "read_file", {})
    with pytest.raises(execution.ExecutionStateError):
        journal.transition("call_1", execution.ToolCallStatus.COMPLETED)
    with pytest.raises(execution.ExecutionStateError):
        journal.register("call_1", "read_file", {})


def test_session_format_four_round_trip_preserves_execution(tmp_sessions):
    record = execution.ToolExecutionRecord(
        "call_1", "read_file", '{}', execution.ToolCallStatus.UNKNOWN, "unknown", "unknown_outcome"
    )
    session.save_session(
        "sid", "goal", [{"role": "system", "content": "SYS"}],
        objective="Objective: goal", execution=[record.to_dict()],
    )
    loaded = session.load_session("sid")
    assert loaded["format"] == session.SESSION_FORMAT == 4
    assert execution.ExecutionJournal.from_dicts(loaded["execution"]).get("call_1").status is execution.ToolCallStatus.UNKNOWN


def test_pending_recovery_becomes_unknown_without_dispatch():
    journal = execution.ExecutionJournal()
    journal.register("call_1", "run_command", {"command": "echo unsafe"})
    recovered = journal.recover_unresolved("interrupted")

    assert recovered[0].status is execution.ToolCallStatus.UNKNOWN
    assert recovered[0].outcome == "interrupted"


def test_pending_is_persisted_before_dispatch_and_terminal_result(tmp_path, monkeypatch, run_loop, streamed_response):
    snapshots = []

    def save(*args, **kwargs):
        snapshots.append(copy.deepcopy(kwargs["execution"]))

    monkeypatch.setattr(session, "save_session", save)
    run_loop(
        [
            streamed_response(finish_reason="tool_calls", tool_calls=[_call()]),
            streamed_response(finish_reason="stop", content="done"),
        ],
        dispatch_fn=lambda *a, **k: "read result",
        session_id="sid",
    )

    statuses = [[record["status"] for record in snapshot] for snapshot in snapshots]
    assert any("pending" in status_list for status_list in statuses)
    assert any("running" in status_list for status_list in statuses)
    assert statuses[-1] == ["completed"]


def test_persistence_failure_before_dispatch_prevents_tool_execution(monkeypatch, run_loop, streamed_response):
    dispatched = []

    def fail(*args, **kwargs):
        raise OSError("session store unavailable")

    monkeypatch.setattr(session, "save_session", fail)
    run_loop(
        [streamed_response(finish_reason="tool_calls", tool_calls=[_call()])],
        dispatch_fn=lambda *a, **k: dispatched.append(a) or "should not run",
        session_id="sid",
    )

    assert dispatched == []


def test_ambiguous_tool_failure_is_persisted_as_unknown(monkeypatch, run_loop, streamed_response):
    snapshots = []
    monkeypatch.setattr(session, "save_session", lambda *a, **k: snapshots.append(copy.deepcopy(k["execution"])))

    def fail(*args, **kwargs):
        raise dispatch.ToolExecutionError("may have run")

    run_loop(
        [
            streamed_response(finish_reason="tool_calls", tool_calls=[_call()]),
            streamed_response(finish_reason="stop", content="done"),
        ],
        dispatch_fn=fail,
        session_id="sid",
    )

    assert snapshots[-1][0]["status"] == "unknown"
    assert snapshots[-1][0]["error_category"] == "unknown_outcome"


def test_compaction_does_not_change_execution_records(monkeypatch):
    state = compaction.CompactionState("Objective: goal", "prior")
    journal = execution.ExecutionJournal()
    journal.register("call_1", "read_file", '{}')
    journal.transition("call_1", execution.ToolCallStatus.RUNNING)
    journal.transition("call_1", execution.ToolCallStatus.COMPLETED, "done")
    before = copy.deepcopy(journal.to_dicts())

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "new summary"}}]}

    monkeypatch.setattr(compaction.requests, "post", lambda *a, **k: Response())
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Objective: goal"},
        {"role": "assistant", "content": "old", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "u"},
    ]
    import asyncio
    event_loop = asyncio.new_event_loop()
    try:
        compacted = event_loop.run_until_complete(compaction.compact_messages(messages, event_loop, state=state))
    finally:
        event_loop.close()

    assert journal.to_dicts() == before
    assert not protocol_violations(compacted)


def test_resume_restores_completed_result_without_redispatch(run_loop, streamed_response):
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Objective: goal"},
        {"role": "assistant", "content": "calling", "tool_calls": [{
            "id": "call_done", "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }]},
    ]
    execution_records = [{
        "tool_call_id": "call_done", "tool_name": "read_file", "arguments": "{}",
        "status": "completed", "outcome": "already done", "error_category": None,
    }]
    dispatched = []
    post, _ = run_loop(
        [streamed_response(finish_reason="stop", content="continued")],
        dispatch_fn=lambda *a, **k: dispatched.append(a),
        resume_messages=messages,
        resume_objective="Objective: goal",
        resume_execution=execution_records,
    )

    assert dispatched == []
    restored = [m for m in post.payloads[0]["messages"] if m.get("role") == "tool"]
    assert restored[0]["tool_call_id"] == "call_done"
    assert restored[0]["content"] == "already done"
    assert not protocol_violations(post.payloads[0]["messages"])


def test_resume_running_call_becomes_unknown_and_is_not_redispatched(run_loop, streamed_response):
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Objective: goal"},
        {"role": "assistant", "content": "calling", "tool_calls": [{
            "id": "call_running", "type": "function",
            "function": {"name": "run_command", "arguments": "{\"command\":\"unsafe\"}"},
        }]},
    ]
    execution_records = [{
        "tool_call_id": "call_running", "tool_name": "run_command",
        "arguments": "{\"command\":\"unsafe\"}", "status": "running",
        "outcome": None, "error_category": None,
    }]
    dispatched = []
    post, _ = run_loop(
        [streamed_response(finish_reason="stop", content="continued")],
        dispatch_fn=lambda *a, **k: dispatched.append(a),
        resume_messages=messages,
        resume_objective="Objective: goal",
        resume_execution=execution_records,
    )

    assert dispatched == []
    tool_messages = [m for m in post.payloads[0]["messages"] if m.get("role") == "tool"]
    assert "Interrupted" in tool_messages[0]["content"]
    assert not protocol_violations(post.payloads[0]["messages"])
