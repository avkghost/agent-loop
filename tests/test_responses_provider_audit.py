"""Responses adapter: audit additions. Each case must end as a MALFORMED (or refused) result with no executable call.
Offline and deterministic."""

import asyncio
import json

import httpx
import pytest

from src import responses_provider as rp
from src.provider import GenerationOptions, TurnOutcome

OPTIONS = GenerationOptions(model="m", temperature=0.2, max_tokens=8, stream=True)
CALL = {"type": "response.output_item.added", "output_index": 0,
        "item": {"type": "function_call", "call_id": "c1", "name": "get_status", "arguments": ""}}


def _turn(events):
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()
    provider = rp.ResponsesProvider(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body)))
    return asyncio.run(provider.complete([{"role": "user", "content": "hi"}], None, OPTIONS))


@pytest.mark.parametrize("event", [
    {"type": "response.output_item.added", "output_index": 0},                       # item missing
    {"type": "response.output_item.done", "output_index": 0},                        # item missing
    {"type": "response.output_item.added", "item": {"type": "function_call", "call_id": "c1", "name": "f"}},
    {"type": "response.output_item.added", "output_index": [0],                       # unhashable index
     "item": {"type": "function_call", "call_id": "c1", "name": "f", "arguments": ""}},
    {"type": "response.output_item.added", "output_index": True,                      # bool is not an index
     "item": {"type": "function_call", "call_id": "c1", "name": "f", "arguments": ""}},
    {"type": "response.output_item.added", "output_index": 0, "item": {"call_id": "c1", "name": "f"}},  # no type
])
def test_missing_or_invalid_output_items_and_indexes_are_malformed(event):
    turn = _turn([event, {"type": "response.completed", "response": {"status": "completed"}}])
    assert turn.outcome == TurnOutcome.MALFORMED and turn.tool_calls == []


@pytest.mark.parametrize("terminal", [
    {"type": "response.incomplete", "response": {"incomplete_details": "max_output_tokens"}},
    {"type": "response.incomplete", "response": {"incomplete_details": {"reason": 7}}},
    {"type": "response.failed", "response": {"error": "boom"}},
    {"type": "response.failed", "response": {"error": {"code": 1}}},
])
def test_invalid_terminal_details_are_malformed_not_incomplete(terminal):
    turn = _turn([CALL, {"type": "response.function_call_arguments.done", "output_index": 0, "arguments": "{}"},
                  {"type": "response.output_item.done", "output_index": 0,
                   "item": {"type": "function_call", "call_id": "c1", "name": "get_status", "arguments": "{}"}},
                  terminal])
    assert turn.outcome == TurnOutcome.MALFORMED and turn.tool_calls == [], turn.outcome


def test_a_valid_failed_response_without_details_is_still_incomplete_not_success():
    turn = _turn([{"type": "response.failed", "response": {"status": "failed"}}])
    assert turn.outcome == TurnOutcome.INCOMPLETE and turn.error.kind == "response_failed"


def test_usage_details_that_are_not_an_object_do_not_fail_the_turn():
    turn = _turn([{"type": "response.output_text.delta", "delta": "ok"}, {"type": "response.completed", "response": {
        "status": "completed", "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                                         "output_tokens_details": []}}}])
    assert turn.outcome == TurnOutcome.COMPLETED and turn.usage.total_tokens == 2


def test_a_tool_result_with_no_text_content_is_refused():
    declared = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}]
    with pytest.raises(rp.TranslationError):
        rp.messages_to_input(declared + [{"role": "tool", "tool_call_id": "c1", "content": None}])


def test_non_object_tool_call_entries_are_never_dropped_silently():
    with pytest.raises(rp.TranslationError):
        rp.messages_to_input([{"role": "assistant", "content": "x", "tool_calls": ["not a call"]}])
