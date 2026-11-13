"""Regression coverage for Phase A protocol, persistence, and tool-boundary corrections."""

import pytest

from src import cli, session
from src.tools import files as file_tools


def _format_four_with_tool_calls(tool_calls):
    return {
        "format": 4,
        "session_id": "sid",
        "goal": "goal",
        "objective": "Objective: goal",
        "summary": None,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
        ],
        "execution": [{
            "tool_call_id": "call_1", "tool_name": "read_file", "arguments": {},
            "status": "pending", "outcome": None, "error_category": None,
        }],
        "saved_at": "now",
    }


@pytest.mark.parametrize("tool_calls", [
    {},
    "bad",
    [None],
    [{"id": "call_1"}],
    [{"id": "call_1", "function": {"name": "read_file", "arguments": {}}}],
], ids=["mapping", "string", "null-entry", "missing-function", "non-string-arguments"])
def test_format_four_rejects_malformed_assistant_tool_calls(tool_calls):
    with pytest.raises(session.SessionValidationError):
        session.validate_session_data(_format_four_with_tool_calls(tool_calls))


def test_format_four_accepts_valid_openai_tool_call_structure():
    data = _format_four_with_tool_calls([{
        "id": "call_1",
        "type": "function",
        "function": {"name": "read_file", "arguments": "{}"},
    }])
    assert session.validate_session_data(data) == "structured"


def test_session_exists_failure_is_categorized(monkeypatch, tmp_sessions):
    def fail_exists(self):
        raise OSError("metadata I/O failure")

    monkeypatch.setattr(session.Path, "exists", fail_exists)
    with pytest.raises(session.SessionLoadError) as exc:
        session.load_session("sid")
    assert exc.value.category == "read_error"


@pytest.mark.parametrize("calls", [
    [{"id": None, "name": "read_file", "arguments": "{}"}],
    [
        {"id": "same", "name": "read_file", "arguments": "{}"},
        {"id": "same", "name": "read_file", "arguments": "{}"},
    ],
])
def test_malformed_tool_id_batch_is_rejected_before_dispatch(run_loop, streamed_response, calls):
    post, dispatched = run_loop([
        streamed_response(finish_reason="tool_calls", tool_calls=calls),
    ])

    assert dispatched == []
    assert len(post.payloads) == 1
    assert not any(message.get("tool_calls") for message in post.payloads[0]["messages"])


def test_failed_write_before_replacement_leaves_the_original_file_intact(tmp_path, monkeypatch):
    """The new content is written to a temporary file first; a failure there never touches the target."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "out.txt"
    target.write_text("original content")

    def fail_sync(fd):
        raise OSError("disk full while syncing")

    monkeypatch.setattr(file_tools.os, "fsync", fail_sync)
    result = file_tools.write_file_safe("out.txt", "complete content")

    assert result["isError"] is True
    assert result["errorCategory"] == "validation"
    assert result["isRetryable"] is False
    assert target.read_text() == "original content"
    assert list(tmp_path.glob(".out.txt.*.tmp")) == [], "temporary file must be removed"


def test_replacement_failure_with_temp_file_still_present_is_definitive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "out.txt"
    target.write_text("original content")

    def refuse(src, dst):
        raise OSError("replace refused")

    monkeypatch.setattr(file_tools.os, "replace", refuse)
    result = file_tools.write_file_safe("out.txt", "complete content")

    assert result["errorCategory"] == "validation"
    assert result["isRetryable"] is False
    assert target.read_text() == "original content"


def test_replacement_that_consumed_the_temp_file_then_failed_is_unknown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "out.txt"
    target.write_text("original content")
    real_replace = file_tools.os.replace

    def replace_then_raise(src, dst):
        real_replace(src, dst)
        raise OSError("reported failure after replacing")

    monkeypatch.setattr(file_tools.os, "replace", replace_then_raise)
    result = file_tools.write_file_safe("out.txt", "complete content")

    assert result["errorCategory"] == "unknown_outcome"
    assert result["isRetryable"] is False
    assert target.read_text() == "complete content"


def test_format_four_rejects_conflicting_retained_tool_result():
    data = {
        "format": 4,
        "session_id": "sid",
        "goal": "goal",
        "objective": "Objective: goal",
        "summary": None,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "different"},
        ],
        "execution": [{
            "tool_call_id": "call_1", "tool_name": "read_file", "arguments": {},
            "status": "completed", "outcome": "authoritative", "error_category": None,
        }],
        "saved_at": "now",
    }

    with pytest.raises(session.SessionValidationError, match="conflicts"):
        session.validate_session_data(data)


@pytest.mark.parametrize("data, category", [
    ({"format": 99}, "unsupported"),
    ({"format": 4, "messages": []}, "schema"),
])
def test_cli_reports_invalid_session_without_dispatch(monkeypatch, capsys, data, category):
    data.setdefault("session_id", "sid")
    data.setdefault("goal", "goal")
    monkeypatch.setattr(cli, "load_session", lambda _: data)
    resumed = []
    monkeypatch.setattr(cli.loop, "agent_loop", lambda *args, **kwargs: resumed.append(args))

    cli.main(["resume", "sid"])

    assert resumed == []
    assert "Cannot resume session" in capsys.readouterr().out


def test_cli_reports_invalid_json_session(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_session", lambda _: (_ for _ in ()).throw(
        session.SessionLoadError("invalid_json", "invalid JSON")))
    cli.main(["resume", "sid"])
    output = capsys.readouterr().out
    assert "invalid_json" in output
    assert "Cannot load session" in output


def test_running_state_is_persisted_before_dispatch(monkeypatch, run_loop, streamed_response):
    snapshots = []
    monkeypatch.setattr(session, "save_session", lambda *a, **k: snapshots.append(k["execution"]))
    run_loop([
        streamed_response(finish_reason="tool_calls", tool_calls=[{
            "id": "call_running", "name": "read_file", "arguments": "{}",
        }]),
        streamed_response(finish_reason="stop", content="done"),
    ], dispatch_fn=lambda *a, **k: "ok", session_id="sid")

    assert any(records and records[0]["status"] == "running" for records in snapshots)


def test_recovery_marks_unresolved_calls_unknown_without_dispatch(
    monkeypatch, run_loop, streamed_response
):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Objective: goal"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_pending", "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }]},
    ]
    records = [{
        "tool_call_id": "call_pending", "tool_name": "read_file", "arguments": {},
        "status": "running", "outcome": None, "error_category": None,
    }]
    dispatched = []
    snapshots = []
    monkeypatch.setattr(session, "save_session", lambda *a, **k: snapshots.append(k["execution"]))
    run_loop([
        streamed_response(finish_reason="stop", content="continued"),
    ], dispatch_fn=lambda *a, **k: dispatched.append(a), resume_messages=messages,
             resume_objective="Objective: goal", resume_execution=records, session_id="sid")
    assert dispatched == []
    assert snapshots[0][0]["status"] == "unknown"
    assert snapshots[0][0]["error_category"] == "unknown_outcome"


def test_terminal_persistence_failure_stops_before_next_provider_request(monkeypatch, run_loop, streamed_response):
    saves = {"count": 0}

    def save(*args, **kwargs):
        saves["count"] += 1
        if kwargs["execution"] and kwargs["execution"][0]["status"] == "completed":
            raise OSError("terminal save failed")

    monkeypatch.setattr(session, "save_session", save)
    post, _ = run_loop([
        streamed_response(finish_reason="tool_calls", tool_calls=[{
            "id": "call_terminal", "name": "read_file", "arguments": "{}",
        }]),
        streamed_response(finish_reason="stop", content="must not be requested"),
    ], dispatch_fn=lambda *a, **k: "done", session_id="sid")

    assert len(post.payloads) == 1
