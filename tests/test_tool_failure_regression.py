import json

import pytest

from src import dispatch as dispatch_module


def tool_call(call_id="call_1", arguments='{"command": "echo ok"}'):
    return {
        "id": call_id,
        "name": "run_command",
        "arguments": arguments,
    }


def test_guardrail_does_not_drop_a_duplicate_call_silently(run_loop, streamed_response):
    calls = [tool_call("call_1"), tool_call("call_2")]
    first = streamed_response(finish_reason="tool_calls", tool_calls=calls)
    second = streamed_response(finish_reason="stop", content="done")

    post, dispatched = run_loop([first, second])
    results = [message for message in post.payloads[1]["messages"] if message.get("role") == "tool"]

    assert len(dispatched) == 2
    assert len(results) == 2
    assert {message["tool_call_id"] for message in results} == {"call_1", "call_2"}


def test_malformed_tool_arguments_are_not_dispatched_as_empty_object(run_loop, streamed_response):
    malformed = tool_call("call_bad", '{"command": "ls"')
    first = streamed_response(finish_reason="tool_calls", tool_calls=[malformed])
    second = streamed_response(finish_reason="stop", content="done")

    post, dispatched = run_loop([first, second])
    results = [message for message in post.payloads[1]["messages"] if message.get("role") == "tool"]

    assert dispatched == []
    assert results and results[0]["tool_call_id"] == "call_bad"
    assert "Malformed tool arguments" in results[0]["content"]


def test_middle_dispatch_exception_keeps_all_call_ids_terminal(run_loop, streamed_response):
    calls = [tool_call("call_1"), tool_call("call_2"), tool_call("call_3")]
    invocation_count = {"value": 0}

    def dispatch(name, args, *rest, **kwargs):
        invocation_count["value"] += 1
        if invocation_count["value"] == 2:
            raise dispatch_module.ToolExecutionError("uncertain external operation")
        return "completed"

    post, dispatched = run_loop([
        streamed_response(finish_reason="tool_calls", tool_calls=calls),
        streamed_response(finish_reason="stop", content="done"),
    ], dispatch_fn=dispatch)

    results = [message for message in post.payloads[1]["messages"] if message.get("role") == "tool"]
    assert len(dispatched) == 3
    assert [message["tool_call_id"] for message in results] == ["call_1", "call_2", "call_3"]
    assert "outcome is unknown" in results[1]["content"]
