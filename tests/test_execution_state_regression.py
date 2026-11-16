"""Deterministic lifecycle and recovery tests for durable tool-call state."""

from tests import scripted_provider  # noqa: E402
import copy
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from src import compaction, dispatch, envelope, execution, session
from tests.conftest import make_streamed_response, protocol_violations


REPO = Path(__file__).resolve().parents[1]
SUBPROCESS_TIMEOUT_SECONDS = 30


CRASH_AFTER_EXTERNAL_COMMIT = r'''
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path

from src import loop, provider, session


session_dir, effect_db, session_id = sys.argv[1:]
session.SESSIONS_DIR = Path(session_dir)
loop.load_project_guidelines = lambda: None


class OneToolProvider:
    async def complete(self, messages, tools, options, on_text=None):
        return provider.ProviderTurn(
            outcome=provider.TurnOutcome.TOOL_CALLS,
            tool_calls=[{
                "id": "call_external_once",
                "type": "function",
                "function": {"name": "run_command", "arguments": "{}"},
            }],
            finish_reason=provider.FinishReason.TOOL_CALLS,
            usage=provider.TokenUsage.reported(1, 1),
        )

    async def aclose(self):
        return None


async def external_action(tool_name, tool_input, *args, **kwargs):
    connection = sqlite3.connect(effect_db)
    connection.execute("CREATE TABLE IF NOT EXISTS effects (key TEXT PRIMARY KEY)")
    connection.execute("INSERT INTO effects(key) VALUES (?)", ("crash-boundary-effect",))
    connection.commit()
    connection.close()
    # This is the deterministic boundary: the external commit is durable, but the
    # loop has not yet transitioned the journal record to COMPLETED or persisted it.
    os._exit(73)


async def main():
    loop.dispatch.dispatch_tool_async = external_action
    event_loop = asyncio.get_running_loop()
    await loop._agent_loop_run(
        "crash recovery goal",
        type("Mcp", (), {"get_openai_tools": lambda self: []})(),
        event_loop,
        system_prompt_override="crash recovery test",
        session_id=session_id,
        provider=OneToolProvider(),
    )


asyncio.run(main())
'''


RECOVER_AFTER_CRASH = r'''
import asyncio
import json
import sqlite3
import sys
from pathlib import Path

from src import loop, provider, session


session_dir, effect_db, session_id, report_path = sys.argv[1:]
session.SESSIONS_DIR = Path(session_dir)
loop.load_project_guidelines = lambda: None
dispatch_count = {"value": 0}


class StopProvider:
    async def complete(self, messages, tools, options, on_text=None):
        return provider.ProviderTurn(
            outcome=provider.TurnOutcome.COMPLETED,
            content="recovered without redispatch",
            finish_reason=provider.FinishReason.STOP,
            usage=provider.TokenUsage.reported(1, 1),
        )

    async def aclose(self):
        return None


async def forbidden_dispatch(*args, **kwargs):
    dispatch_count["value"] += 1
    raise AssertionError("recovery redispatched the unresolved external action")


async def main():
    data = session.load_session(session_id)
    assert data is not None
    runs, evidence, legacy_history = session.resume_history(data)
    loop.dispatch.dispatch_tool_async = forbidden_dispatch
    event_loop = asyncio.get_running_loop()
    outcome = await loop._agent_loop_run(
        "crash recovery goal",
        type("Mcp", (), {"get_openai_tools": lambda self: []})(),
        event_loop,
        system_prompt_override="crash recovery test",
        session_id=session_id,
        resume_messages=data["messages"],
        resume_objective=data["objective"],
        resume_summary=data["summary"],
        resume_execution=data["execution"],
        resume_runs=runs,
        resume_evidence=evidence,
        legacy_history=legacy_history,
        provider=StopProvider(),
    )
    with sqlite3.connect(effect_db) as connection:
        effect_count = connection.execute(
            "SELECT COUNT(*) FROM effects WHERE key = ?", ("crash-boundary-effect",)
        ).fetchone()[0]
    saved = session.load_session(session_id)
    Path(report_path).write_text(json.dumps({
        "outcome": outcome.to_dict(),
        "dispatch_count": dispatch_count["value"],
        "effect_count": effect_count,
        "execution": saved["execution"],
        "messages": saved["messages"],
    }))


asyncio.run(main())
'''


def test_crash_after_external_commit_recovers_unknown_without_redispatch(tmp_path):
    """A durable side effect may precede the terminal journal write; recovery must not replay it."""
    session_dir = tmp_path / "sessions"
    effect_db = tmp_path / "effects.sqlite"
    report_path = tmp_path / "recovery.json"
    session_id = "crash-boundary"

    try:
        crashed = subprocess.run(
            [
                sys.executable,
                "-c",
                CRASH_AFTER_EXTERNAL_COMMIT,
                str(session_dir),
                str(effect_db),
                session_id,
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"process A timed out after {SUBPROCESS_TIMEOUT_SECONDS}s: {exc}")
    assert crashed.returncode == 73, crashed.stderr

    interrupted = json.loads((session_dir / f"{session_id}.json").read_text())
    assert interrupted["execution"][0]["tool_call_id"] == "call_external_once"
    assert interrupted["execution"][0]["status"] == "running"
    with sqlite3.connect(effect_db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM effects").fetchone()[0] == 1

    try:
        recovered = subprocess.run(
            [
                sys.executable,
                "-c",
                RECOVER_AFTER_CRASH,
                str(session_dir),
                str(effect_db),
                session_id,
                str(report_path),
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"process B timed out after {SUBPROCESS_TIMEOUT_SECONDS}s: {exc}")
    assert recovered.returncode == 0, recovered.stderr
    report = json.loads(report_path.read_text())

    assert report["effect_count"] == 1
    assert report["dispatch_count"] == 0
    assert report["outcome"]["status"] != "completed_verified"
    execution_record = report["execution"][0]
    assert execution_record["tool_call_id"] == "call_external_once"
    assert execution_record["status"] == "unknown"
    assert "was not re-executed" in execution_record["outcome"]
    assert any(
        message.get("role") == "tool"
        and message.get("tool_call_id") == "call_external_once"
        and "was not re-executed" in message.get("content", "")
        for message in report["messages"]
    )


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
    assert loaded["format"] == session.SESSION_FORMAT == 5
    assert execution.ExecutionJournal.from_dicts(loaded["execution"]).get("call_1").status is execution.ToolCallStatus.UNKNOWN


def test_pending_recovery_becomes_unknown_without_dispatch():
    journal = execution.ExecutionJournal()
    journal.register("call_1", "run_command", {"command": "echo unsafe"})
    recovered = journal.recover_unresolved("interrupted")

    assert recovered[0].status is execution.ToolCallStatus.UNKNOWN
    assert recovered[0].outcome == "interrupted"


def test_new_execution_record_is_owned_by_the_current_run(run_loop, streamed_response, tmp_sessions):
    run_loop(
        [
            streamed_response(finish_reason="tool_calls", tool_calls=[_call()]),
            streamed_response(finish_reason="stop", content="done"),
        ],
        dispatch_fn=lambda *a, **k: "read result",
        session_id="owned",
    )

    saved = session.load_session("owned")
    assert saved["execution"][0]["origin_run_id"] == saved["runs"][-1]["run_id"]


def test_legacy_execution_record_without_origin_run_id_remains_compatible():
    journal = execution.ExecutionJournal.from_dicts([{
        "tool_call_id": "call_legacy", "tool_name": "read_file", "arguments": {},
        "status": "pending", "outcome": None, "error_category": None,
    }])

    assert journal.get("call_legacy").origin_run_id is None
    assert "origin_run_id" not in journal.to_dicts()[0]


def test_invalid_origin_run_id_is_rejected():
    with pytest.raises(execution.ExecutionStateError, match="origin_run_id"):
        execution.ExecutionJournal.from_dicts([{
            "tool_call_id": "call_bad", "tool_name": "read_file", "arguments": {},
            "status": "pending", "outcome": None, "error_category": None,
            "origin_run_id": "run-not-valid",
        }])


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

    monkeypatch.setattr(scripted_provider, "POST", lambda *a, **k: Response())
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
        compacted = event_loop.run_until_complete(compaction.compact_messages(messages, state=state))
    finally:
        event_loop.close()

    assert journal.to_dicts() == before
    assert not protocol_violations(compacted)


def test_failed_compaction_keeps_unresolved_call_unknown_on_resume(
    monkeypatch, run_loop, streamed_response, tmp_sessions
):
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Objective: recover safely"},
        {"role": "assistant", "content": "calling", "tool_calls": [{
            "id": "call_unresolved", "type": "function",
            "function": {"name": "run_command", "arguments": "{\"command\":\"unsafe\"}"},
        }]},
        {"role": "assistant", "content": "older context"},
        {"role": "user", "content": "more context"},
        {"role": "assistant", "content": "latest context"},
    ]
    execution_records = [{
        "tool_call_id": "call_unresolved", "tool_name": "run_command",
        "arguments": {"command": "unsafe"}, "status": "running",
        "outcome": None, "error_category": None,
    }]

    original_compact = compaction.compact_messages

    class FailingSummaryProvider:
        async def complete(self, *args, **kwargs):
            raise RuntimeError("summary service unavailable")

    async def fail_summary(messages, provider=None, state=None):
        return await original_compact(messages, FailingSummaryProvider(), state=state)

    monkeypatch.setattr(compaction, "compact_messages", fail_summary)
    post, dispatched = run_loop(
        [streamed_response(finish_reason="stop", content="continued safely")],
        dispatch_fn=lambda *args, **kwargs: pytest.fail("an unresolved call was redispatched"),
        session_id="compaction-recovery",
        resume_messages=messages,
        resume_objective="Objective: recover safely",
        resume_execution=execution_records,
        MAX_TOKENS=200,
        COMPACT_THRESHOLD=0.5,
    )

    saved = session.load_session("compaction-recovery")
    record = saved["execution"][0]
    assert dispatched == []
    assert record["status"] == "unknown"
    assert record["error_category"] == "unknown_outcome"
    assert any(
        message.get("role") == "tool"
        and message.get("tool_call_id") == "call_unresolved"
        and "not re-executed" in message.get("content", "")
        for message in saved["messages"]
    )
    assert any(
        message.get("role") == "tool"
        and message.get("tool_call_id") == "call_unresolved"
        and "not re-executed" in message.get("content", "")
        for message in post.payloads[0]["messages"]
    )


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
