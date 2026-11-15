"""Regressions for the integration-review fixes: non-object tool arguments, and MCP failures
whose outcome is unknown. All calls are mocked; no MCP server, LLM, or shell is used."""

import asyncio
import json
import re
from types import SimpleNamespace

import pytest

from src import config, dispatch, loop
from tests.conftest import make_streamed_response, protocol_violations
from tests.test_hardening_regression import _drive


@pytest.mark.parametrize("arguments", ["null", "[]", "1", '"text"', "true"],
                         ids=["null", "array", "number", "string", "boolean"])
def test_non_object_arguments_are_answered_without_dispatch(monkeypatch, arguments):
    dispatched = []

    def dispatch_fn(name, args, *a, **k):
        dispatched.append(args)
        return "should not run"

    responses = [
        make_streamed_response(finish_reason="tool_calls",
                               tool_calls=[{"id": "call_x", "name": "list_directory", "arguments": arguments}]),
        make_streamed_response(finish_reason="stop", content="done"),
    ]
    run = _drive(monkeypatch, responses, dispatch_fn)

    assert dispatched == [], "non-object arguments must never reach dispatch"
    assert len(run.requests) == 2, "the run must continue after the rejected call"
    results = [m for m in run.requests[1]["messages"] if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in results] == ["call_x"]
    assert "must be a JSON object" in results[0]["content"]
    assert not protocol_violations(run.requests[1]["messages"]), protocol_violations(run.requests[1]["messages"])


class _SessionThatFails:
    """Transport boundary: the session raises the given error when the call is made."""

    def __init__(self, error):
        self._error = error

    async def call_tool(self, name, arguments):
        raise self._error


def _real_client_with(session):
    from mcp.types import Tool
    from src.tools.mcp_client import MCPClientManager
    client = MCPClientManager()
    client.sessions = {"remote": session}
    # The server advertises the tool (its schema passed discovery), so the call reaches the session
    client.tools_cache = {"remote": [Tool(name="write", input_schema={"type": "object", "properties": {}})]}
    return client


def test_mcp_connection_loss_is_unknown_outcome_and_not_retryable(monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_MCP_TOOLS", {"mcp_remote_write"})
    client = _real_client_with(_SessionThatFails(ConnectionError("reset after send")))
    result = asyncio.run(dispatch.dispatch_tool_async("mcp_remote_write", {"path": "x"}, client))

    assert result["isError"] is True
    assert result["errorCategory"] == "unknown_outcome"
    assert result["isRetryable"] is False
    assert "may have executed" in result["content"]


from src.tools import shell


def test_shell_missing_program_is_a_non_retryable_validation_error(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "echo")

    monkeypatch.setattr(shell.subprocess, "Popen", missing)
    monkeypatch.setattr(config, "ALLOWED_TOOLS", {"echo"})
    monkeypatch.setattr(config, "ALLOW_ALL", False)
    result = asyncio.run(shell.execute_command_async("echo hi"))

    assert result["isError"] is True
    assert result["errorCategory"] == "validation"
    assert result["isRetryable"] is False
    assert "could not start command" in result["content"]


def test_shell_spawn_exception_is_unknown_outcome_not_a_validation_error(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("unexpected spawn failure")

    monkeypatch.setattr(shell.subprocess, "Popen", broken)
    monkeypatch.setattr(config, "ALLOWED_TOOLS", {"echo"})
    monkeypatch.setattr(config, "ALLOW_ALL", False)
    event_loop = asyncio.new_event_loop()
    try:
        result = event_loop.run_until_complete(loop._dispatch_safely("run_command", {"command": "echo hi"}, None, event_loop, None))
    finally:
        event_loop.close()

    assert result["errorCategory"] == "unknown_outcome"
    assert result["isRetryable"] is False


def test_objective_is_kept_verbatim_through_repeated_compaction(monkeypatch):
    """Every compacted request still starts with the original objective, verbatim."""
    from tests.test_hardening_regression import _numbered_dispatch, _forced_compaction_config
    from tests.conftest import make_streamed_response

    responses = [make_streamed_response(finish_reason="tool_calls",
                                        tool_calls=[{"id": f"call_{i}", "name": "list_directory",
                                                     "arguments": json.dumps({"path": str(i)})}])
                 for i in range(5)]
    responses.append(make_streamed_response(finish_reason="stop", content="done"))

    run = _drive(monkeypatch, responses, _numbered_dispatch(), **_forced_compaction_config())

    assert run.summaries, "compaction must have run for this test to mean anything"
    for index, body in enumerate(run.requests):
        messages = body["messages"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"].startswith("Objective: hardening goal"), f"request {index} lost the objective"
