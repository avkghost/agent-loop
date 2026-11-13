"""B3.1 corrections: strict non-streamed tool-call structure, and strict, labelled token usage.

Validation of tool arguments and tool names stays in the loop. These tests cover only the provider's
structure and usage contract. Real httpx transport is used where the adapter path matters.
"""

import asyncio
import json

import httpx
import pytest

from src import provider as p
from src import stream as sse
from src.provider import GenerationOptions, LMStudioProvider, TokenUsage, TurnOutcome, turn_from_json

VALID_CALL = {"id": "call_1", "type": "function",
              "function": {"name": "run_command", "arguments": '{"command": "ls"}'}}


def _response(message=None, finish="tool_calls", usage=None, **extra):
    choice = {"message": {"content": "", "tool_calls": [VALID_CALL]} if message is None else message,
              "finish_reason": finish}
    body = {"choices": [choice]}
    if usage is not None:
        body["usage"] = usage
    body.update(extra)
    return body


def _malformed(turn):
    assert turn.outcome is TurnOutcome.MALFORMED, turn
    assert not turn.succeeded
    assert turn.tool_calls == [], "a malformed response must expose no tool calls"


# ---------------------------------------------------------------------------
# Non-streamed tool-call structure
# ---------------------------------------------------------------------------

def test_valid_non_streamed_assistant_response():
    turn = turn_from_json({"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]})
    assert turn.outcome is TurnOutcome.COMPLETED and turn.content == "hi" and turn.tool_calls == []


def test_valid_non_streamed_tool_call_keeps_id_and_arguments_exactly():
    args = '{"command": "echo  spaced"}'
    turn = turn_from_json(_response(message={"content": None, "tool_calls": [
        {"id": "call_xyz", "type": "function", "function": {"name": "run_command", "arguments": args}}]}))
    assert turn.outcome is TurnOutcome.TOOL_CALLS
    assert turn.tool_calls == [{"id": "call_xyz", "type": "function",
                                "function": {"name": "run_command", "arguments": args}}]


def test_tool_calls_as_a_string_is_malformed_not_a_list_of_characters():
    turn = turn_from_json(_response(message={"content": "", "tool_calls": "bad"}))
    _malformed(turn)


def test_tool_calls_as_a_dictionary_is_malformed():
    turn = turn_from_json(_response(message={"content": "", "tool_calls": VALID_CALL}))
    _malformed(turn)


@pytest.mark.parametrize("tool_calls", [42, True, 1.5])
def test_tool_calls_as_other_scalars_is_malformed(tool_calls):
    _malformed(turn_from_json(_response(message={"content": "", "tool_calls": tool_calls})))


@pytest.mark.parametrize("element", [
    "a string", 7, None, ["nested"],
    {"type": "function", "function": {"name": "n", "arguments": "{}"}},          # no id
    {"id": "", "type": "function", "function": {"name": "n", "arguments": "{}"}},  # empty id
    {"id": 5, "type": "function", "function": {"name": "n", "arguments": "{}"}},   # numeric id
    {"id": "c", "type": "image", "function": {"name": "n", "arguments": "{}"}},    # unsupported type
    {"id": "c", "type": "function"},                                                # no function object
    {"id": "c", "type": "function", "function": "run_command"},                     # function not an object
    {"id": "c", "type": "function", "function": {"arguments": "{}"}},               # no name
    {"id": "c", "type": "function", "function": {"name": "", "arguments": "{}"}},    # empty name
    {"id": "c", "type": "function", "function": {"name": "n"}},                     # no arguments
    {"id": "c", "type": "function", "function": {"name": "n", "arguments": {"a": 1}}},  # object arguments
])
def test_invalid_tool_call_elements_are_malformed(element):
    _malformed(turn_from_json(_response(message={"content": "", "tool_calls": [element]})))


def test_missing_structural_fields_at_the_message_and_choice_level_are_malformed():
    _malformed(turn_from_json({"choices": "not a list"}))
    _malformed(turn_from_json({"choices": []}))
    _malformed(turn_from_json({"choices": ["not an object"]}))
    _malformed(turn_from_json({"choices": [{"message": "a string", "finish_reason": "stop"}]}))
    _malformed(turn_from_json({"choices": [{"message": {"content": 5}, "finish_reason": "stop"}]}))
    _malformed(turn_from_json(["not", "an", "object"]))


def test_unhashable_or_non_string_finish_reason_is_malformed_not_an_exception():
    _malformed(turn_from_json({"choices": [{"message": {"content": "x"}, "finish_reason": ["stop"]}]}))
    _malformed(turn_from_json({"choices": [{"message": {"content": "x"}, "finish_reason": 3}]}))


def test_null_tool_calls_means_none_and_null_content_means_empty():
    turn = turn_from_json({"choices": [{"message": {"content": None, "tool_calls": None},
                                        "finish_reason": "stop"}]})
    assert turn.outcome is TurnOutcome.COMPLETED and turn.content == "" and turn.tool_calls == []


def test_null_inside_a_tool_call_is_malformed():
    _malformed(turn_from_json(_response(message={"content": "", "tool_calls": [
        {"id": "c", "type": "function", "function": None}]})))


def test_finish_reason_must_match_the_tool_calls():
    _malformed(turn_from_json(_response(finish="stop")))  # calls with stop
    _malformed(turn_from_json(_response(message={"content": "", "tool_calls": []}, finish="tool_calls")))


def test_application_invalid_but_structurally_valid_arguments_pass_through_for_loop_validation():
    turn = turn_from_json(_response(message={"content": "", "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "run_command", "arguments": "{not json"}}]}))
    assert turn.outcome is TurnOutcome.TOOL_CALLS
    assert turn.tool_calls[0]["function"]["arguments"] == "{not json", "arguments must reach the loop unchanged"


def test_missing_type_field_is_accepted_as_a_function_call():
    turn = turn_from_json(_response(message={"content": "", "tool_calls": [
        {"id": "c", "function": {"name": "run_command", "arguments": "{}"}}]}))
    assert turn.outcome is TurnOutcome.TOOL_CALLS and turn.tool_calls[0]["type"] == "function"


def test_non_streamed_adapter_path_applies_the_same_structure_rules():
    body = json.dumps(_response(message={"content": "", "tool_calls": "bad"})).encode()
    provider = LMStudioProvider(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=body)))

    async def run():
        try:
            return await provider.complete([{"role": "user", "content": "hi"}], [],
                                           GenerationOptions(stream=False))
        finally:
            await provider.aclose()

    _malformed(asyncio.run(run()))


# ---------------------------------------------------------------------------
# Token usage: valid, invalid, partial, contradictory, and labelled
# ---------------------------------------------------------------------------

def test_valid_reported_usage_is_reported():
    turn = turn_from_json(_response(usage={"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}))
    assert turn.usage == TokenUsage(120, 30, 150, "reported")
    assert turn.usage_error is None


def test_zero_tokens_are_valid_counts():
    turn = turn_from_json(_response(usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}))
    assert turn.usage == TokenUsage(0, 0, 0, "reported")


def test_total_is_derived_when_absent():
    turn = turn_from_json(_response(usage={"prompt_tokens": 4, "completion_tokens": 5}))
    assert turn.usage == TokenUsage(4, 5, 9, "reported")


@pytest.mark.parametrize("usage", [
    {"prompt_tokens": -1, "completion_tokens": 3},
    {"prompt_tokens": 3, "completion_tokens": -3},
    {"prompt_tokens": True, "completion_tokens": 3},
    {"prompt_tokens": 3, "completion_tokens": False},
    {"prompt_tokens": "12", "completion_tokens": 3},
    {"prompt_tokens": 1.5, "completion_tokens": 3},
    {"prompt_tokens": 3, "completion_tokens": None},
    {"prompt_tokens": 2**60, "completion_tokens": 1},           # beyond the JSON-safe range
])
def test_invalid_counts_are_unavailable_never_reported(usage):
    turn = turn_from_json(_response(usage=usage))
    assert turn.usage is None, "invalid provider counts must never become reported usage"
    assert turn.usage_error, "the rejection must be explained"
    assert turn.succeeded, "usage never fails a turn"


@pytest.mark.parametrize("usage", [
    {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 999},   # contradictory total
    {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": True},      # boolean total
    {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": -2},
])
def test_contradictory_or_invalid_totals_are_rejected(usage):
    turn = turn_from_json(_response(usage=usage))
    assert turn.usage is None and turn.usage_error


def test_missing_usage_object_is_quiet_and_unavailable():
    turn = turn_from_json(_response())
    assert turn.usage is None and turn.usage_error is None


def test_partial_usage_is_not_reported():
    turn = turn_from_json(_response(usage={"prompt_tokens": 10}))
    assert turn.usage is None and "partial usage" in turn.usage_error
    turn = turn_from_json(_response(usage={"total_tokens": 10}))
    assert turn.usage is None and turn.usage_error


def test_empty_or_non_object_usage_is_unavailable():
    assert turn_from_json(_response(usage={})).usage is None
    turn = turn_from_json(_response(usage="many"))
    assert turn.usage is None and "not an object" in turn.usage_error


def test_streamed_usage_follows_the_same_policy():
    def stream_with(usage_obj):
        lines = [b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}',
                 ("data: " + json.dumps({"usage": usage_obj})).encode(), b"data: [DONE]"]
        parser = sse.StreamParser()
        for line in lines:
            parser.feed(line)
        return p.turn_from_parser(parser)

    assert stream_with({"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}).usage == \
        TokenUsage(2, 2, 4, "reported")
    bad = stream_with({"prompt_tokens": True, "completion_tokens": 2})
    assert bad.succeeded and bad.usage is None and bad.usage_error


def test_conflicting_usage_objects_in_one_stream_are_unavailable():
    parser = sse.StreamParser()
    for line in [b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}',
                 b'data: {"usage":{"prompt_tokens":1,"completion_tokens":1}}',
                 b'data: {"usage":{"prompt_tokens":9,"completion_tokens":1}}',
                 b"data: [DONE]"]:
        parser.feed(line)
    turn = p.turn_from_parser(parser)
    assert turn.succeeded and turn.usage is None and "conflicting" in turn.usage_error


def test_repeated_identical_usage_objects_are_accepted():
    parser = sse.StreamParser()
    for line in [b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}',
                 b'data: {"usage":{"prompt_tokens":1,"completion_tokens":1}}',
                 b'data: {"usage":{"prompt_tokens":1,"completion_tokens":1}}',
                 b"data: [DONE]"]:
        parser.feed(line)
    assert p.turn_from_parser(parser).usage == TokenUsage(1, 1, 2, "reported")


def test_estimate_is_labelled_estimated_and_reported_is_labelled_reported(monkeypatch, capsys):
    from src import loop, dispatch
    from src.tools.mcp_client import MCPClientManager
    from tests import scripted_provider
    from tests.conftest import make_streamed_response

    class Reporting(scripted_provider.ScriptedProvider):
        async def complete(self, messages, tools, options, on_text=None):
            turn = await super().complete(messages, tools, options, on_text)
            if turn.succeeded and tools:
                turn.usage = TokenUsage(3, 4, 7, "reported")
            return turn

    monkeypatch.setattr(p, "make_provider", lambda: Reporting())
    monkeypatch.setattr(scripted_provider, "POST", lambda *a, **k: make_streamed_response(finish_reason="stop",
                                                                                         content="answer"))
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)

    async def run():
        await loop.agent_loop_async("goal", MCPClientManager(), asyncio.get_running_loop())

    asyncio.run(run())
    out = capsys.readouterr().out
    assert "tokens (reported): prompt 3, completion 4, total 7" in out


def test_invalid_provider_usage_is_announced_and_estimated(monkeypatch, capsys):
    from src import loop
    from src.tools.mcp_client import MCPClientManager
    from tests import scripted_provider

    body = {"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": True, "completion_tokens": 2}}
    monkeypatch.setattr(scripted_provider, "POST", lambda *a, **k: _JsonResponse(body))
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)
    monkeypatch.setattr(loop.config, "ALLOW_ALL", False)

    class NonStreaming(scripted_provider.ScriptedProvider):
        async def complete(self, messages, tools, options, on_text=None):
            return turn_from_json(body)

    monkeypatch.setattr(p, "make_provider", lambda: NonStreaming())

    async def run():
        await loop.agent_loop_async("goal", MCPClientManager(), asyncio.get_running_loop())

    asyncio.run(run())
    out = capsys.readouterr().out
    assert "provider usage not used" in out
    assert "tokens (estimated)" in out and "tokens (reported)" not in out


class _JsonResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body

    def close(self):
        return None
