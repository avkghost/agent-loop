"""Phase A final corrections: MCP through the real runtime, file safety, session consistency, M1.

External boundaries are mocked: the LLM (requests.post) and the MCP transport session. The
MCP client, dispatcher, loop, journal, and session code all run for real.
"""

import asyncio
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import config, dispatch, envelope, execution, loop, session
from src.session import SessionValidationError, validate_session_data
from src.tools import files
from src.tools.mcp_client import MCPClientManager
from tests.conftest import make_streamed_response, protocol_violations


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.payloads = []

    def __call__(self, *args, **kwargs):
        self.payloads.append(copy.deepcopy(kwargs["json"]))
        if not self.replies:
            raise AssertionError("unexpected extra model request")
        return self.replies.pop(0)


class FakeTransport:
    """The MCP session: the lowest external boundary. `behaviour` decides what happens on each call."""

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.submitted = 0

    async def call_tool(self, name, arguments):
        self.submitted += 1
        return await self.behaviour(name, arguments)


def run_agent(monkeypatch, replies, *, mcp_client=None, session_id=None, **kwargs):
    llm = ScriptedLLM(replies)
    monkeypatch.setattr(loop.requests, "post", llm)
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)
    client = mcp_client or MCPClientManager()
    event_loop = asyncio.new_event_loop()
    try:
        event_loop.run_until_complete(loop.agent_loop_async(
            "final goal", client, event_loop, session_id=session_id, **kwargs))
    finally:
        event_loop.close()
    return llm


def mcp_run(monkeypatch, tmp_sessions, behaviour, tool="mcp_remote_write", session_id="mcp-run"):
    transport = FakeTransport(behaviour)
    client = MCPClientManager()
    client.sessions = {"remote": transport}
    monkeypatch.setattr(config, "ALLOWED_MCP_TOOLS", {tool})
    llm = run_agent(monkeypatch, [
        make_streamed_response(finish_reason="tool_calls", tool_calls=[
            {"id": "call_mcp", "name": tool, "arguments": json.dumps({"path": "x"})}]),
        make_streamed_response(finish_reason="stop", content="done"),
    ], mcp_client=client, session_id=session_id)
    saved = session.load_session(session_id)
    record = saved["execution"][0] if saved and saved.get("execution") else None
    tool_msg = next((m for m in llm.payloads[1]["messages"] if m.get("role") == "tool"), None) if len(llm.payloads) > 1 else None
    return transport, tool_msg, record, llm


def envelope_of(tool_msg):
    return json.loads(tool_msg["content"])


# ---------------------------------------------------------------------------
# H1: MCP reaches the transport through the real agent runtime
# ---------------------------------------------------------------------------

def test_h1_mcp_call_reaches_transport_through_agent_loop_and_is_journaled(monkeypatch, tmp_sessions):
    async def ok(name, arguments):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=f"wrote {arguments['path']}")], isError=False)

    transport, tool_msg, record, _ = mcp_run(monkeypatch, tmp_sessions, ok)

    assert transport.submitted == 1, "the transport must be reached from inside the running agent"
    assert envelope_of(tool_msg) == {"content": "wrote x", "isError": False}
    assert record["status"] == "completed" and record["outcome"] == tool_msg["content"]


def test_h1_sync_dispatch_inside_running_loop_is_not_nested(monkeypatch):
    """dispatch_tool (synchronous) must not be used for MCP inside a running loop."""
    seen = {}

    async def scenario():
        client = MCPClientManager()
        client.sessions = {"remote": FakeTransport(lambda n, a: asyncio.sleep(0, result=SimpleNamespace(content=[], isError=False)))}
        monkeypatch.setattr(config, "ALLOWED_MCP_TOOLS", {"mcp_remote_write"})
        seen["result"] = await dispatch.dispatch_tool_async("mcp_remote_write", {}, client, asyncio.get_running_loop())

    asyncio.run(scenario())
    assert seen["result"]["isError"] is False


# ---------------------------------------------------------------------------
# H2: malformed MCP responses are never successes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("result", [
    None,
    SimpleNamespace(isError=False),                                 # missing content
    SimpleNamespace(content="just a string", isError=False),         # content not a list
    SimpleNamespace(content=[object()], isError=False),             # item with no text and no type
    SimpleNamespace(content=[], isError="no"),                      # isError not a bool
    {"content": "raw dict is not a result"},                        # dict without isError handling
], ids=["none", "no-content", "content-str", "bad-item", "bad-iserror", "dict-missing-iserror"])
def test_h2_malformed_responses_are_unknown_not_success(monkeypatch, tmp_sessions, result):
    async def malformed(name, arguments):
        return result

    _, tool_msg, record, _ = mcp_run(monkeypatch, tmp_sessions, malformed)
    body = envelope_of(tool_msg)

    assert body["isError"] is True, body
    assert body["errorCategory"] == "unknown_outcome"
    assert body["isRetryable"] is False
    assert "malformed" in body["content"]
    assert record["status"] == "unknown"


def test_h2_valid_empty_content_is_a_success(monkeypatch, tmp_sessions):
    async def empty(name, arguments):
        return SimpleNamespace(content=[], isError=False)

    _, tool_msg, record, _ = mcp_run(monkeypatch, tmp_sessions, empty)

    assert envelope_of(tool_msg)["isError"] is False
    assert envelope_of(tool_msg)["content"] == ""
    assert record["status"] == "completed"


def test_h2_non_text_content_is_represented_not_dropped(monkeypatch, tmp_sessions):
    async def image(name, arguments):
        return SimpleNamespace(content=[SimpleNamespace(type="image")], isError=False)

    _, tool_msg, record, _ = mcp_run(monkeypatch, tmp_sessions, image)
    assert envelope_of(tool_msg)["content"] == "[image content omitted]"
    assert record["status"] == "completed"


# ---------------------------------------------------------------------------
# H3 and M-c: transport failures and remote errors are unknown, never retryable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,behaviour", [
    ("connection refused", lambda n, a: (_ for _ in ()).throw(ConnectionRefusedError("refused"))),
    ("connection lost after send", lambda n, a: (_ for _ in ()).throw(ConnectionResetError("reset"))),
])
def test_h3_transport_failures_are_unknown_and_not_retryable(monkeypatch, tmp_sessions, label, behaviour):
    async def failing(name, arguments):
        return behaviour(name, arguments)  # raises synchronously inside the coroutine

    transport, tool_msg, record, llm = mcp_run(monkeypatch, tmp_sessions, failing)
    body = envelope_of(tool_msg)

    assert body["errorCategory"] == "unknown_outcome", label
    assert body["isRetryable"] is False
    assert record["status"] == "unknown"
    assert transport.submitted == 1, "no automatic redispatch after an ambiguous failure"


def test_h3_timeout_after_submission_is_unknown_and_not_retryable(monkeypatch, tmp_sessions):
    async def slow(name, arguments):
        raise asyncio.TimeoutError()

    _, tool_msg, record, _ = mcp_run(monkeypatch, tmp_sessions, slow)
    body = envelope_of(tool_msg)
    assert body["errorCategory"] == "unknown_outcome" and body["isRetryable"] is False
    assert "timed out" in body["content"]
    assert record["status"] == "unknown"


def test_mc_explicit_remote_error_is_unknown_with_server_message(monkeypatch, tmp_sessions):
    async def remote_error(name, arguments):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="quota exceeded")], isError=True)

    _, tool_msg, record, _ = mcp_run(monkeypatch, tmp_sessions, remote_error)
    body = envelope_of(tool_msg)

    assert body["content"] == "quota exceeded"
    assert body["errorCategory"] == "unknown_outcome" and body["isRetryable"] is False
    assert record["status"] == "unknown", "a remote error does not prove that no side effect occurred"


def test_h3_invalid_tool_name_is_a_definitive_validation_failure_before_transport(monkeypatch, tmp_sessions):
    async def never(name, arguments):
        raise AssertionError("must not reach the transport")

    transport, tool_msg, record, _ = mcp_run(monkeypatch, tmp_sessions, never, tool="mcp_bad")
    body = envelope_of(tool_msg)

    assert transport.submitted == 0
    assert body["errorCategory"] == "validation" and body["isRetryable"] is False
    assert record["status"] == "failed"


def test_h3_cancellation_propagates_and_resume_does_not_redispatch(monkeypatch, tmp_sessions):
    async def cancelled(name, arguments):
        raise asyncio.CancelledError()

    transport = FakeTransport(cancelled)
    client = MCPClientManager()
    client.sessions = {"remote": transport}
    monkeypatch.setattr(config, "ALLOWED_MCP_TOOLS", {"mcp_remote_write"})
    with pytest.raises(asyncio.CancelledError):
        run_agent(monkeypatch, [make_streamed_response(finish_reason="tool_calls", tool_calls=[
            {"id": "call_c", "name": "mcp_remote_write", "arguments": "{}"}])],
            mcp_client=client, session_id="cancel")

    saved = session.load_session("cancel")
    assert saved["execution"][0]["status"] == "running", "a cancelled call stays RUNNING on disk"

    resumed = FakeTransport(lambda n, a: None)
    client2 = MCPClientManager()
    client2.sessions = {"remote": resumed}
    run_agent(monkeypatch, [make_streamed_response(finish_reason="stop", content="ok")], mcp_client=client2,
              session_id="cancel", resume_messages=saved["messages"], resume_objective=saved["objective"],
              resume_summary=saved["summary"], resume_execution=saved["execution"])
    assert resumed.submitted == 0, "resume must not re-run the interrupted MCP call"


# ---------------------------------------------------------------------------
# H4: encoding failures never damage existing content
# ---------------------------------------------------------------------------

@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_h4_encoding_failure_leaves_existing_file_byte_for_byte_unchanged(cwd):
    target = cwd / "keep.txt"
    target.write_bytes(b"PRECIOUS\x00bytes")
    before = target.read_bytes()

    result = files.write_file_safe("keep.txt", "bad \ud800 surrogate")

    assert result["errorCategory"] == "validation" and result["isRetryable"] is True
    assert target.read_bytes() == before
    assert list(cwd.glob(".keep.txt.*.tmp")) == []


def test_h4_valid_unicode_is_written(cwd):
    result = files.write_file_safe("u.txt", "naïve — 漢字 ✓")
    assert result["isError"] is False
    assert (cwd / "u.txt").read_text(encoding="utf-8") == "naïve — 漢字 ✓"


def test_h4_unexpected_programming_error_is_not_masked(cwd, monkeypatch):
    (cwd / "p.txt").write_text("keep")

    def broken(fd):
        raise TypeError("programming defect")

    monkeypatch.setattr(files.os, "fsync", broken)
    with pytest.raises(TypeError, match="programming defect"):
        files.write_file_safe("p.txt", "new")
    assert (cwd / "p.txt").read_text() == "keep"


def test_h4_write_through_agent_loop_with_encoding_failure_keeps_file_and_runs_on(monkeypatch, cwd, tmp_sessions):
    (cwd / "data.txt").write_text("PRECIOUS")
    llm = run_agent(monkeypatch, [
        make_streamed_response(finish_reason="tool_calls", tool_calls=[
            {"id": "call_w", "name": "write_file", "arguments": json.dumps({"path": "data.txt", "content": "\ud800"})}]),
        make_streamed_response(finish_reason="stop", content="done"),
    ], session_id="h4")
    assert (cwd / "data.txt").read_text() == "PRECIOUS"
    assert len(llm.payloads) == 2


# ---------------------------------------------------------------------------
# H5: path containment by path component, including symlinks
# ---------------------------------------------------------------------------

@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    sibling = tmp_path / "work-evil"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("outside")
    monkeypatch.chdir(work)
    return work, sibling


def test_h5_sibling_prefix_collision_is_rejected_for_read_write_and_list(sandbox):
    work, sibling = sandbox
    rel = "../work-evil/secret.txt"
    assert files.read_file_safe(rel)["errorCategory"] == "permission"
    assert files.list_directory_safe("../work-evil")["errorCategory"] == "permission"
    assert files.write_file_safe(rel, "pwned")["errorCategory"] == "permission"
    assert (sibling / "secret.txt").read_text() == "outside"


def test_h5_parent_traversal_is_rejected(sandbox):
    assert files.read_file_safe("../../etc/passwd")["errorCategory"] == "permission"


def test_h5_symlink_pointing_outside_is_rejected(sandbox):
    work, sibling = sandbox
    (work / "link").symlink_to(sibling / "secret.txt")
    assert files.read_file_safe("link")["errorCategory"] == "permission"
    assert files.write_file_safe("link", "pwned")["errorCategory"] == "permission"
    assert (sibling / "secret.txt").read_text() == "outside"


def test_h5_inside_paths_are_allowed(sandbox):
    work, _ = sandbox
    absolute_inside = str(work / "nested" / "a.txt")
    assert files.write_file_safe("nested/a.txt", "one")["isError"] is False
    assert files.read_file_safe("nested/a.txt")["content"] == "one"
    assert files.read_file_safe(absolute_inside)["content"] == "one"
    assert files.write_file_safe("new-not-yet.txt", "two")["isError"] is False
    assert (work / "new-not-yet.txt").read_text() == "two"
    assert files.list_directory_safe("nested")["isError"] is False


# ---------------------------------------------------------------------------
# M-a: binary and non-UTF-8 reads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [b"\xff\xfe\x00\x01 binary", b"caf\xe9 latin1"], ids=["binary", "latin1"])
def test_ma_undecodable_read_is_a_definitive_validation_error(cwd, payload):
    (cwd / "blob.bin").write_bytes(payload)
    result = files.read_file_safe("blob.bin")
    assert result["errorCategory"] == "validation"
    assert result["isRetryable"] is False
    assert "not UTF-8" in result["content"]


def test_ma_binary_read_through_agent_loop_does_not_crash_the_run(monkeypatch, cwd, tmp_sessions):
    (cwd / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff")
    llm = run_agent(monkeypatch, [
        make_streamed_response(finish_reason="tool_calls", tool_calls=[
            {"id": "call_r", "name": "read_file", "arguments": json.dumps({"path": "img.png"})}]),
        make_streamed_response(finish_reason="stop", content="done"),
    ], session_id="ma")
    tool_msg = next(m for m in llm.payloads[1]["messages"] if m.get("role") == "tool")
    assert json.loads(tool_msg["content"])["errorCategory"] == "validation"


# ---------------------------------------------------------------------------
# M-b: contradictory active or non-completed state is rejected
# ---------------------------------------------------------------------------

def _session_with(execution_records, tool_messages):
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "Objective: g"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]}]
    messages += tool_messages
    return {"format": 4, "session_id": "x", "goal": "g", "objective": "Objective: g", "summary": None,
            "saved_at": "t", "messages": messages, "execution": execution_records}


def _rec(status, outcome=None, category=None):
    return {"tool_call_id": "c1", "tool_name": "t", "arguments": {}, "status": status,
            "outcome": outcome, "error_category": category}


def _tool(content):
    return {"role": "tool", "tool_call_id": "c1", "content": content}


ERR = json.dumps({"content": "denied", "isError": True, "errorCategory": "permission", "isRetryable": False})
OK = json.dumps({"content": "listing", "isError": False})


@pytest.mark.parametrize("label,data,should_reject", [
    ("running with result", _session_with([_rec("running")], [_tool("x")]), True),
    ("pending with result", _session_with([_rec("pending")], [_tool("x")]), True),
    ("unknown with success envelope", _session_with([_rec("unknown", OK, "unknown_outcome")], [_tool(OK)]), True),
    ("unknown with plain success text", _session_with([_rec("unknown", "Done. File written.", "unknown_outcome")],
                                                      [_tool("Done. File written.")]), True),
    ("completed with error envelope", _session_with([_rec("completed", ERR)], [_tool(ERR)]), True),
    ("failed with success envelope", _session_with([_rec("failed", OK, "validation")], [_tool(OK)]), True),
    ("unknown with interruption text", _session_with([_rec("unknown", execution.INTERRUPTED_RESULT, "unknown_outcome")],
                                                     [_tool(execution.INTERRUPTED_RESULT)]), False),
    ("failed with permission envelope", _session_with([_rec("failed", ERR, "permission")], [_tool(ERR)]), False),
    ("completed with success envelope", _session_with([_rec("completed", OK)], [_tool(OK)]), False),
    ("unknown with materialized record and no message", _session_with([_rec("unknown", execution.INTERRUPTED_RESULT,
                                                                            "unknown_outcome")], []), False),
])
def test_mb_session_consistency(label, data, should_reject):
    if should_reject:
        with pytest.raises(SessionValidationError):
            validate_session_data(copy.deepcopy(data))
    else:
        assert validate_session_data(copy.deepcopy(data)) == "structured"


# ---------------------------------------------------------------------------
# M1: a RUNNING persistence failure stops the run before dispatch
# ---------------------------------------------------------------------------

def test_m1_running_save_failure_prevents_dispatch_and_keeps_previous_state(monkeypatch, tmp_sessions):
    real_save = session.save_session
    calls = {"n": 0}

    def fail_on_running(*args, **kwargs):
        calls["n"] += 1
        records = kwargs.get("execution") or []
        if any(r.get("status") == "running" for r in records):
            raise OSError("store unavailable at RUNNING")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(session, "save_session", fail_on_running)
    dispatched = []
    monkeypatch.setattr(dispatch, "dispatch_tool", lambda *a, **k: dispatched.append(a) or "ran")
    llm = run_agent(monkeypatch, [
        make_streamed_response(finish_reason="tool_calls", tool_calls=[
            {"id": "call_m1", "name": "list_directory", "arguments": json.dumps({"path": "."})}]),
        make_streamed_response(finish_reason="stop", content="should not be requested"),
    ], session_id="m1")

    # PENDING was saved; RUNNING was not. Dispatch never happened and no second model request was made.
    assert dispatched == []
    assert len(llm.payloads) == 1
    saved = session.load_session("m1")
    assert saved is not None and saved["execution"][0]["status"] == "pending"
    assert validate_session_data(copy.deepcopy(saved)) == "structured"
