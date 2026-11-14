"""Responses API adapter: offline, deterministic. Uses the preserved LM Studio probe (tests/fixtures/responses) and synthetic
SSE for cases the probe does not contain. No LM Studio request, no network."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from src import provider as p
from src import responses_provider as rp
from src.provider import GenerationOptions, TurnOutcome

FIX = Path(__file__).resolve().parent / "fixtures" / "responses"
TOOLS = [{"type": "function", "function": {"name": "get_status", "description": "Report a service status.",
                                           "parameters": {"type": "object", "properties": {"service": {"type": "string"}},
                                                          "required": ["service"], "additionalProperties": False}}}]
OPTIONS = GenerationOptions(model="qwen/qwen3.8-27b", temperature=0.2, max_tokens=2048, stream=True)
USER = [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "Check service alpha."}]


def _events(*events):
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()


def _added(index, call_id, name):
    return {"type": "response.output_item.added", "output_index": index,
            "item": {"type": "function_call", "call_id": call_id, "name": name, "arguments": ""}}


def _args_delta(index, delta):
    return {"type": "response.function_call_arguments.delta", "output_index": index, "delta": delta}


def _args_done(index, arguments):
    return {"type": "response.function_call_arguments.done", "output_index": index, "arguments": arguments}


def _item_done(index, call_id, name, arguments):
    return {"type": "response.output_item.done", "output_index": index,
            "item": {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}}


def _completed(usage=None):
    response = {"status": "completed"}
    if usage is not None:
        response["usage"] = usage
    return {"type": "response.completed", "response": response}


USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "output_tokens_details": {"reasoning_tokens": 3}}


def _parse(lines):
    parser = rp.ResponsesParser()
    for line in lines:
        parser.feed(line)
    return parser.finish()


def _turn(body: bytes, *, status=200, messages=USER, tools=TOOLS):
    def handler(request):
        return httpx.Response(status, content=body)
    provider = rp.ResponsesProvider(transport=httpx.MockTransport(handler))
    return asyncio.run(provider.complete(messages, tools, OPTIONS))


def _probe_lines():
    return (FIX / "probe_raw_sse.txt").read_text().splitlines()


# 1. complete tool call from .done only (the preserved probe)
def test_the_recorded_probe_yields_a_complete_tool_call_from_done_events_only():
    result = _parse(_probe_lines())
    assert result.outcome == TurnOutcome.TOOL_CALLS
    assert result.tool_calls == [{"id": "call_1435487479760000", "type": "function",
                                  "function": {"name": "get_status", "arguments": '{"service":"alpha"}'}}]
    assert result.finish_reason == "tool_calls"


def test_the_recorded_probe_through_the_adapter_matches_the_parser():
    turn = _turn((FIX / "probe_raw_sse.txt").read_bytes())
    assert turn.outcome == TurnOutcome.TOOL_CALLS and turn.succeeded
    assert turn.tool_calls[0]["id"] == "call_1435487479760000"
    assert json.loads(turn.tool_calls[0]["function"]["arguments"]) == {"service": "alpha"}


# 2. delta-based arguments, joined without duplication
def test_argument_deltas_are_joined_once_and_the_completion_is_not_repeated():
    body = _events(_added(0, "c1", "get_status"), _args_delta(0, '{"service"'), _args_delta(0, ':"alpha"}'),
                   _args_done(0, '{"service":"alpha"}'), _item_done(0, "c1", "get_status", '{"service":"alpha"}'),
                   _completed(USAGE))
    turn = _turn(body)
    assert turn.outcome == TurnOutcome.TOOL_CALLS
    assert turn.tool_calls[0]["function"]["arguments"] == '{"service":"alpha"}'


# 3. multiple function calls keep output order and distinct ids
def test_multiple_calls_are_returned_in_output_order_with_distinct_ids():
    body = _events(_added(1, "c-b", "get_status"), _added(0, "c-a", "get_status"),
                   _args_done(0, '{"service":"a"}'), _item_done(0, "c-a", "get_status", '{"service":"a"}'),
                   _args_done(1, '{"service":"b"}'), _item_done(1, "c-b", "get_status", '{"service":"b"}'),
                   _completed(USAGE))
    turn = _turn(body)
    assert [call["id"] for call in turn.tool_calls] == ["c-a", "c-b"]
    assert len({call["id"] for call in turn.tool_calls}) == 2


# 4. tool-call / output history round trips
def test_history_round_trips_with_stable_call_ids():
    result = _parse(_probe_lines())
    history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"},
               {"role": "assistant", "content": result.content, "tool_calls": result.tool_calls},
               {"role": "tool", "tool_call_id": result.tool_calls[0]["id"], "content": '{"content":"ok"}'}]
    instructions, items = rp.messages_to_input(history)
    assert instructions == "sys"
    call_items = [i for i in items if i.get("type") == "function_call"]
    output_items = [i for i in items if i.get("type") == "function_call_output"]
    assert call_items == [{"type": "function_call", "call_id": "call_1435487479760000", "name": "get_status",
                           "arguments": '{"service":"alpha"}'}]
    assert output_items[0]["call_id"] == call_items[0]["call_id"]
    assert output_items[0]["output"] == '{"content":"ok"}'


# 5. empty content with valid tool calls
def test_empty_text_with_valid_tool_calls_is_a_tool_turn_with_empty_content():
    body = _events(_added(0, "c1", "get_status"), _args_done(0, '{"service":"x"}'),
                   _item_done(0, "c1", "get_status", '{"service":"x"}'), _completed(USAGE))
    turn = _turn(body)
    assert turn.outcome == TurnOutcome.TOOL_CALLS and turn.content == ""


# 6. response completion
def test_a_completed_text_answer_is_a_completed_turn():
    body = _events({"type": "response.output_text.delta", "delta": "Hello"},
                   {"type": "response.output_text.delta", "delta": " there"},
                   {"type": "response.output_text.done", "text": "Hello there"}, _completed(USAGE))
    turn = _turn(body)
    assert turn.outcome == TurnOutcome.COMPLETED and turn.content == "Hello there" and turn.finish_reason == "stop"


# 7. failure and incomplete output
def test_failed_and_incomplete_responses_are_never_success():
    failed = _events({"type": "response.failed", "response": {"status": "failed", "error": {"message": "boom"}}})
    turn = _turn(failed)
    assert turn.outcome == TurnOutcome.INCOMPLETE and turn.error.kind == "response_failed" and not turn.succeeded
    cut = _events({"type": "response.incomplete", "response": {"incomplete_details": {"reason": "max_output_tokens"}}})
    assert _turn(cut).outcome == TurnOutcome.LENGTH
    other = _events({"type": "response.incomplete", "response": {"incomplete_details": {"reason": "content_filter"}}})
    assert _turn(other).outcome == TurnOutcome.INCOMPLETE


# 8. cancellation
def test_cancellation_mid_stream_propagates_and_returns_no_turn():
    class Hang(httpx.AsyncByteStream):
        def __init__(self, first):
            self._first = first
            self._gate = asyncio.Event()

        async def __aiter__(self):
            yield self._first
            await self._gate.wait()

    first = _events(_added(0, "c1", "get_status"), _args_delta(0, '{"serv'))

    def handler(request):
        return httpx.Response(200, stream=Hang(first))

    async def scenario():
        provider = rp.ResponsesProvider(transport=httpx.MockTransport(handler))
        task = asyncio.ensure_future(provider.complete(USER, TOOLS, OPTIONS))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


# 9. malformed arguments and malformed events
def test_arguments_are_passed_exactly_so_the_loop_judges_their_json():
    body = _events(_added(0, "c1", "get_status"), _args_done(0, "{bad"), _item_done(0, "c1", "get_status", "{bad"),
                   _completed(USAGE))
    turn = _turn(body)
    assert turn.outcome == TurnOutcome.TOOL_CALLS and turn.tool_calls[0]["function"]["arguments"] == "{bad"


def test_a_malformed_event_is_a_malformed_turn():
    assert _turn(b"data: {not json\n\n").outcome == TurnOutcome.MALFORMED
    assert _turn(b"data: {\"no\": \"type\"}\n\n").outcome == TurnOutcome.MALFORMED


@pytest.mark.parametrize("event", [
    {"type": "response.output_item.added", "output_index": 0, "item": "bad"},
    {"type": "response.output_item.done", "output_index": 0, "item": "bad"},
    {"type": "response.completed", "response": "bad"},
    {"type": "response.completed", "response": []},
    {"type": "response.failed", "response": "bad"},
    {"type": "response.incomplete", "response": None},
    {"type": "response.completed"},
])
def test_non_object_nested_protocol_values_are_malformed_turns(event):
    turn = _turn(_events(event))
    assert turn.outcome == TurnOutcome.MALFORMED
    assert turn.tool_calls == []
    assert not turn.succeeded


# 10. missing or contradictory identifiers
@pytest.mark.parametrize("events", [
    [{"type": "response.output_item.added", "output_index": 0,
      "item": {"type": "function_call", "name": "get_status", "arguments": ""}}, _completed()],
    [_args_delta(5, "{}"), _completed()],
    [_added(0, "dup", "get_status"), _added(1, "dup", "get_status"), _completed()],
    [_added(0, "c1", "get_status"), _args_done(0, "{}"), _item_done(0, "c1", "other", "{}"), _completed()],
    [_added(0, "c1", "get_status"), _args_delta(0, '{"a"'), _args_done(0, '{"b":1}'), _completed()],
])
def test_missing_or_contradictory_identifiers_are_malformed(events):
    turn = _turn(_events(*events))
    assert turn.outcome == TurnOutcome.MALFORMED, turn.outcome


# 11. reasoning and token usage
def test_reasoning_and_usage_are_reported_from_the_terminal_event():
    result = _parse(_probe_lines())
    assert result.usage.prompt_tokens == 331 and result.usage.completion_tokens == 25
    assert result.usage.total_tokens == 356 and result.usage.source == "reported"
    assert result.reasoning_tokens == 24 and result.reasoning_chars > 0


@pytest.mark.parametrize("usage, problem", [
    (None, "usage not reported"),
    ({"input_tokens": -1, "output_tokens": 2}, "invalid"),
    ({"input_tokens": 3, "output_tokens": 2, "total_tokens": 9}, "contradicts"),
])
def test_invalid_usage_is_not_used_and_does_not_fail_the_turn(usage, problem):
    body = _events(_added(0, "c1", "get_status"), _args_done(0, "{}"), _item_done(0, "c1", "get_status", "{}"),
                   _completed(usage))
    turn = _turn(body)
    assert turn.outcome == TurnOutcome.TOOL_CALLS and turn.usage is None and problem in turn.usage_error


# 12. nothing is executed before terminal completion
def test_a_call_is_not_returned_without_a_terminal_event():
    lines = _events(_added(0, "c1", "get_status"), _args_done(0, "{}"), _item_done(0, "c1", "get_status", "{}"))
    turn = _turn(lines)
    assert turn.outcome == TurnOutcome.INCOMPLETE and turn.tool_calls == [] and not turn.succeeded


def test_a_call_is_not_returned_when_the_response_is_incomplete_or_claims_completion_with_an_open_call():
    partial = _events(_added(0, "c1", "get_status"), _args_done(0, "{}"),
                      {"type": "response.incomplete", "response": {"incomplete_details": {"reason": "max_output_tokens"}}})
    turn = _turn(partial)
    assert turn.outcome == TurnOutcome.LENGTH and turn.tool_calls == []
    open_call = _events(_added(0, "c1", "get_status"), _args_done(0, "{}"), _completed(USAGE))
    turn = _turn(open_call)
    assert turn.outcome == TurnOutcome.INCOMPLETE and turn.tool_calls == []


def test_events_after_the_terminal_event_are_a_protocol_violation():
    body = _events(_completed(USAGE), {"type": "response.output_text.delta", "delta": "late"})
    assert _turn(body).outcome == TurnOutcome.MALFORMED


# --- translation and the opt-in boundary ----------------------------------------------------------------------

def test_tool_declarations_keep_names_and_parameter_schemas_exactly():
    converted = rp.tools_to_responses(TOOLS)
    assert converted == [{"type": "function", "name": "get_status", "description": "Report a service status.",
                          "parameters": TOOLS[0]["function"]["parameters"]}]


def test_an_unanswerable_or_duplicate_tool_result_is_refused():
    with pytest.raises(rp.TranslationError):
        rp.messages_to_input([{"role": "tool", "tool_call_id": "nope", "content": "x"}])
    declared = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}]
    with pytest.raises(rp.TranslationError):
        rp.messages_to_input(declared + [{"role": "tool", "tool_call_id": "c1", "content": "a"},
                                         {"role": "tool", "tool_call_id": "c1", "content": "b"}])


def test_outstanding_tool_calls_cannot_be_bypassed_by_a_new_turn():
    declared = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}]
    for next_message in ({"role": "user", "content": "next"}, {"role": "assistant", "content": "next"}):
        with pytest.raises(rp.TranslationError):
            rp.messages_to_input(declared + [next_message])
    with pytest.raises(rp.TranslationError):
        rp.messages_to_input(declared)


def test_multiple_calls_require_all_results_and_accept_complete_results():
    declared = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        {"id": "c2", "type": "function", "function": {"name": "g", "arguments": "{}"}},
    ]}]
    with pytest.raises(rp.TranslationError):
        rp.messages_to_input(declared + [{"role": "tool", "tool_call_id": "c1", "content": "a"}])
    _, items = rp.messages_to_input(declared + [
        {"role": "tool", "tool_call_id": "c2", "content": "b"},
        {"role": "tool", "tool_call_id": "c1", "content": "a"},
    ])
    assert [item["call_id"] for item in items if item["type"] == "function_call_output"] == ["c2", "c1"]


@pytest.mark.parametrize("tool_calls", [{}, None, "bad", 7, [None], [{"id": "c1"}],
                                        [{"id": "c1", "function": {"name": "f", "arguments": "{}"},
                                          "type": "not_function"}]])
def test_invalid_tool_call_containers_and_entries_are_rejected(tool_calls):
    with pytest.raises(rp.TranslationError):
        rp.messages_to_input([{"role": "assistant", "content": None, "tool_calls": tool_calls}])


def test_absent_and_empty_tool_calls_are_valid():
    assert rp.messages_to_input([{"role": "assistant", "content": "hello"}])[1] == [
        {"role": "assistant", "content": "hello"}]
    assert rp.messages_to_input([{"role": "assistant", "content": "hello", "tool_calls": []}])[1] == [
        {"role": "assistant", "content": "hello"}]


def test_the_request_is_stateless_and_never_names_a_previous_response():
    body = rp.build_request(USER, TOOLS, OPTIONS)
    assert body["store"] is False and "previous_response_id" not in body
    assert body["max_output_tokens"] == 2048 and body["stream"] is True and body["tool_choice"] == "auto"


def test_the_chat_completions_provider_remains_the_default_and_nothing_imports_the_adapter():
    # The suite replaces make_provider() with a scripted double, so the production factory is checked as source.
    provider_source = (Path(__file__).resolve().parents[1] / "src" / "provider.py").read_text()
    assert "def make_provider() -> ModelProvider:" in provider_source
    assert "    return LMStudioProvider()" in provider_source
    root = Path(__file__).resolve().parents[1] / "src"
    importers = [path.name for path in root.rglob("*.py") if "responses_provider" in path.read_text()
                 and path.name != "responses_provider.py"]
    assert importers == [], importers
