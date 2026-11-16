from tests import scripted_provider  # noqa: E402
import json
import asyncio
from types import SimpleNamespace

import pytest

from src import cli, compaction, loop, runs, session
from tests.conftest import make_streamed_response


def test_session_round_trip_preserves_message_fields(tmp_sessions):
    messages = [
        {"role": "system", "content": "system"},
        {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [{"id": "call_1", "function": {"name": "x", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "done"},
    ]

    session.save_session("sid", "goal", messages, execution=[
        {"tool_call_id": "call_1", "tool_name": "x", "arguments": {}, "status": "completed",
         "outcome": "done", "error_category": None}])
    loaded = session.load_session("sid")

    assert loaded["session_id"] == "sid"
    assert loaded["goal"] == "goal"
    assert loaded["messages"] == messages


def test_missing_session_returns_none(tmp_sessions):
    assert session.load_session("missing") is None


def test_malformed_session_returns_none(tmp_sessions):
    (tmp_sessions / "bad.json").write_text("{not json")
    with pytest.raises(session.SessionLoadError) as exc:
        session.load_session("bad")
    assert exc.value.category == "invalid_json"


def test_cli_inspect_shows_unknown_execution_without_resuming(tmp_sessions, capsys):
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "calling", "tool_calls": [
            {"id": "call_unknown", "type": "function",
             "function": {"name": "write_file", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "call_unknown", "content": session.INTERRUPTED_RESULT},
    ]
    session.save_session("inspect-me", "goal", messages, execution=[{
        "tool_call_id": "call_unknown", "tool_name": "write_file", "arguments": {},
        "status": "unknown", "outcome": session.INTERRUPTED_RESULT,
        "error_category": "unknown_outcome",
    }])

    cli.main(["inspect", "inspect-me"])

    output = capsys.readouterr().out
    assert "status=unknown category=unknown_outcome" in output
    assert session.INTERRUPTED_RESULT in output


def test_cli_inspect_resolves_run_id_to_session(monkeypatch, tmp_sessions, capsys):
    run_id = "run_" + "a" * 32
    session.save_session("inspect-run", "goal", [{"role": "system", "content": "sys"}],
                         runs=[{"schema": runs.RUN_SCHEMA, "run_id": run_id, "goal": "goal",
                                "status": "completed_unverified", "reason": "done",
                                "started_at": "2026-01-01T00:00:00", "finished_at": "2026-01-01T00:00:01",
                                "iteration_count": 1, "final_message": "done", "evidence_ids": [],
                                "evidence_dropped": 0}])

    cli.main(["inspect", run_id])

    output = capsys.readouterr().out
    assert f"Resolved run_id {run_id} to session_id inspect-run" in output
    assert "Session: inspect-run" in output


def test_interrupted_write_preserves_existing_session(tmp_sessions, monkeypatch):
    original = [{"role": "system", "content": "sys"}, {"role": "user", "content": "original"}]
    replacement = [{"role": "system", "content": "sys"}, {"role": "user", "content": "replacement"}]
    session.save_session("sid", "old goal", original)

    original_replace = session.os.replace

    def interrupted_replace(*args, **kwargs):
        raise OSError("injected replacement interruption")

    monkeypatch.setattr(session.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="replacement interruption"):
        session.save_session("sid", "new goal", replacement)

    loaded = session.load_session("sid")
    assert loaded["goal"] == "old goal"
    assert loaded["messages"] == original
    assert not (tmp_sessions / "sid.json.tmp").exists()

    monkeypatch.setattr(session.os, "replace", original_replace)
    session.save_session("sid", "new goal", replacement)
    loaded = session.load_session("sid")
    assert loaded["goal"] == "new goal"
    assert loaded["messages"] == replacement


def test_completed_tool_result_is_persisted_before_next_request(monkeypatch, streamed_response, tmp_sessions):
    class SimulatedInterruption(BaseException):
        pass

    responses = iter([
        streamed_response(finish_reason="tool_calls", tool_calls=[{
            "id": "call_persisted",
            "name": "read_file",
            "arguments": '{"path": "README.md"}',
        }]),
    ])
    requests = {"count": 0}

    def post(*args, **kwargs):
        requests["count"] += 1
        if requests["count"] == 1:
            return next(responses)
        raise SimulatedInterruption("before next response")

    monkeypatch.setattr(scripted_provider, "POST", post)
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)
    monkeypatch.setattr(loop.dispatch, "dispatch_tool", lambda *args, **kwargs: "persisted result")

    event_loop = asyncio.new_event_loop()
    try:
        with pytest.raises(SimulatedInterruption):
            event_loop.run_until_complete(loop.agent_loop_async(
                "goal", SimpleNamespace(get_openai_tools=lambda: []), event_loop, session_id="sid"
            ))
    finally:
        event_loop.close()

    persisted = session.load_session("sid")["messages"]
    result = [message for message in persisted if message.get("role") == "tool"]
    assert result == [{
        "role": "tool",
        "tool_call_id": "call_persisted",
        "content": "persisted result",
    }]

    resumed_dispatches = []
    monkeypatch.setattr(scripted_provider, "POST", lambda *args, **kwargs: make_streamed_response(
        finish_reason="stop", content="resumed"
    ))
    monkeypatch.setattr(loop.dispatch, "dispatch_tool", lambda *args, **kwargs: resumed_dispatches.append(args))
    event_loop = asyncio.new_event_loop()
    try:
        event_loop.run_until_complete(loop.agent_loop_async(
            "goal", SimpleNamespace(get_openai_tools=lambda: []), event_loop,
            session_id="sid", resume_messages=persisted
        ))
    finally:
        event_loop.close()
    assert resumed_dispatches == []


def test_runtime_persists_when_session_id_is_provided(monkeypatch, run_loop, streamed_response):
    saved = []
    monkeypatch.setattr(session, "save_session", lambda *args, **kwargs: saved.append(args))

    run_loop(
        [streamed_response(finish_reason="stop", content="done")],
        session_id="sid",
    )

    assert saved
    assert all(entry[0] == "sid" for entry in saved)


def test_resume_answers_unresolved_tool_call_without_redispatch(
    run_loop, streamed_response, tmp_sessions
):
    resume_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "goal"},
        {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}
            ],
        },
    ]

    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("an unresolved call must not be re-dispatched during resume")

    post, dispatched = run_loop(
        [streamed_response(finish_reason="stop", content="continued")],
        dispatch_fn=unexpected_dispatch,
        session_id="sid",
        resume_messages=resume_messages,
        resume_execution=[{"tool_call_id": "call_1", "tool_name": "x", "arguments": {}, "status": "running",
                           "outcome": None, "error_category": None}],
    )

    assert dispatched == []
    tool_messages = [m for m in post.payloads[0]["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert "not re-executed" in tool_messages[0]["content"]


def test_cli_resume_reconstructs_and_continues_session(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_session", lambda session_id: {
        "session_id": session_id,
        "goal": "goal",
        "messages": [],
    })

    resumed = []
    monkeypatch.setattr(cli.loop, "agent_loop", lambda *args, **kwargs: resumed.append((args, kwargs)))

    cli.main(["resume", "sid"])

    assert resumed == [
        (("goal",), {"session_id": "sid", "resume_messages": [], "criteria": [], "verification_cycles": 2,
                     "resume_runs": [], "resume_evidence": [], "legacy_history": True})
    ]
    assert "Resuming session: goal" in capsys.readouterr().out


def test_multiple_compactions_round_trip_without_nested_summaries(tmp_sessions, monkeypatch):
    class Response:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": self.text}, "finish_reason": "stop"}]}

    summaries = iter(["first summary", "second summary"])
    monkeypatch.setattr(scripted_provider, "POST", lambda *a, **k: Response(next(summaries)))
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Objective: keep me"},
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]
    state = compaction.CompactionState(messages[1]["content"])
    event_loop = asyncio.new_event_loop()
    try:
        messages = event_loop.run_until_complete(compaction.compact_messages(messages, state=state))
        messages += [{"role": "user", "content": "u4"}, {"role": "assistant", "content": "a4"}]
        messages = event_loop.run_until_complete(compaction.compact_messages(messages, state=state))
    finally:
        event_loop.close()

    session.save_session("compacted", "keep me", messages)
    loaded = session.load_session("compacted")
    assert loaded["messages"] == messages
    assert loaded["messages"][1]["content"].startswith("Objective: keep me")
    assert loaded["messages"][1]["content"].count("[Context Summary]") == 1
