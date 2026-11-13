import copy
import json
from types import SimpleNamespace

import pytest

from src import compaction, config, envelope
from src.tools import shell


def call(call_id, name="read_file", arguments=None):
    return {
        "id": call_id,
        "name": name,
        "arguments": json.dumps(arguments or {"path": "README.md"}),
    }


@pytest.mark.parametrize("calls", [[call("call_1")], [call("call_1"), call("call_2")]])
def test_assistant_tool_declarations_and_results_are_protocol_paired(run_loop, streamed_response, calls):
    first = streamed_response(finish_reason="tool_calls", tool_calls=calls)
    second = streamed_response(finish_reason="stop", content="done")

    post, _ = run_loop([first, second])
    sent = post.payloads[1]["messages"]
    assistants = [m for m in sent if m.get("role") == "assistant"]
    results = [m for m in sent if m.get("role") == "tool"]
    declared = {
        tc["id"]
        for message in assistants
        for tc in message.get("tool_calls", [])
    }
    declarations = {
        tc["id"]: tc
        for message in assistants
        for tc in message.get("tool_calls", [])
    }
    returned = {message["tool_call_id"] for message in results}

    assert assistants and all(message.get("tool_calls") for message in assistants)
    assert declared == {item["id"] for item in calls}
    assert returned == declared
    for item in calls:
        assert declarations[item["id"]]["function"]["name"] == item["name"]
        assert declarations[item["id"]]["function"]["arguments"] == item["arguments"]
    assert sent.index(assistants[0]) < sent.index(results[0])
    assert all(message["tool_call_id"] in declared for message in results)


def test_synchronous_dispatch_finishes_before_next_provider_request(run_loop, streamed_response):
    events = []

    def dispatch(tool_name, tool_input, *args, **kwargs):
        events.append("dispatch")
        return envelope.success("complete")

    first = streamed_response(finish_reason="tool_calls", tool_calls=[call("call_1")])
    second = streamed_response(finish_reason="stop", content="done")
    post, _ = run_loop([first, second], dispatch_fn=dispatch)
    events.append("second-request")

    assert events == ["dispatch", "second-request"]
    assert any(message.get("role") == "tool" for message in post.payloads[1]["messages"])


def test_dispatch_exception_is_recorded_as_unknown_outcome_and_not_retried(run_loop, streamed_response):
    """A tool that raises gets a terminal error result paired with its call ID.

    The outcome is marked unknown and non-retryable, and the tool is not dispatched again.
    Behaviour changed from the earlier audit state, where the exception propagated and left
    a pending call in the history.
    """
    dispatches = []

    def dispatch(*args, **kwargs):
        dispatches.append(args)
        from src.dispatch import ToolExecutionError
        raise ToolExecutionError("injected after dispatch began")

    first = streamed_response(finish_reason="tool_calls", tool_calls=[call("call_1")])
    second = streamed_response(finish_reason="stop", content="done")
    post, _ = run_loop([first, second], dispatch_fn=dispatch)

    assert len(dispatches) == 1, "a failed tool must not be retried automatically"
    results = [m for m in post.payloads[1]["messages"] if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in results] == ["call_1"]
    assert '"errorCategory": "unknown_outcome"' in results[0]["content"]
    assert '"isRetryable": false' in results[0]["content"]


def test_internal_dispatch_error_is_not_misreported_as_unknown_side_effect(run_loop, streamed_response):
    def dispatch(*args, **kwargs):
        raise TypeError("internal programming error")

    first = streamed_response(finish_reason="tool_calls", tool_calls=[call("call_bug")])
    with pytest.raises(TypeError, match="internal programming error"):
        run_loop([first], dispatch_fn=dispatch)


def test_background_shell_is_rejected_before_any_process_starts(monkeypatch):
    """A trailing '&' would start an unmanaged process, so it is rejected before approval or spawn."""
    import asyncio

    async def must_not_spawn(*args, **kwargs):
        raise AssertionError("a background command must not spawn a process")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", must_not_spawn)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", must_not_spawn)
    result = shell.execute_command("echo work &")

    assert result["isError"] is True
    assert result["errorCategory"] == "validation"
    assert "Unmanaged background execution is not supported" in result["content"]


def test_compaction_runs_only_after_the_tool_result_is_recorded(monkeypatch, run_loop, streamed_response):
    """Forced compaction fires at the start of the second iteration, so it must see the
    completed call group (declaration and result), never a dispatch without its result."""
    from src import prompts

    system_tokens = len(prompts.SYSTEM_PROMPT) // 4
    seen = []

    async def spy(messages, *args, **kwargs):
        seen.append(copy.deepcopy(messages))
        return messages

    monkeypatch.setattr(compaction, "compact_messages", spy)

    first = streamed_response(finish_reason="tool_calls", tool_calls=[call("call_1")])
    second = streamed_response(finish_reason="stop", content="done")
    post, calls = run_loop(
        [first, second],
        dispatch_fn=lambda name, args, *a, **k: "x" * 4000,  # ~1000 tokens, pushes the second request over budget
        MAX_TOKENS=system_tokens + 500,
        COMPACT_THRESHOLD=1.0,
    )

    assert len(calls) == 1, "the call must be dispatched exactly once"
    assert len(seen) == 1, "compaction must run exactly once, on the second iteration"
    history = seen[0]
    assert history[-1]["role"] == "tool" and history[-1]["tool_call_id"] == "call_1"
    assert any(m.get("tool_calls") for m in history if m.get("role") == "assistant")
