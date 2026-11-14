"""Responses history contract: developer messages and historical argument normalization. Offline and deterministic."""

import copy
import json

import pytest

from src import responses_provider as rp
from src.loop import _provider_arguments


def _call(call_id, arguments, name="get_status"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _answered(*calls):
    """A valid turn: one assistant message with the calls, followed by a result for each call."""
    return [{"role": "assistant", "content": None, "tool_calls": list(calls)}] + [
        {"role": "tool", "tool_call_id": call["id"], "content": "r"} for call in calls]


# --- developer role ----------------------------------------------------------------------------------------

def test_a_developer_message_alone_is_a_system_role_input_item_with_no_instructions():
    assert rp.messages_to_input([{"role": "developer", "content": "Be terse."}]) == (
        None, [{"role": "system", "content": "Be terse."}])


def test_system_followed_by_developer_keeps_instructions_and_the_developer_item_separate():
    instructions, items = rp.messages_to_input([{"role": "system", "content": "sys"},
                                                {"role": "developer", "content": "dev"}])
    assert instructions == "sys"
    assert items == [{"role": "system", "content": "dev"}]


def test_a_developer_message_followed_by_a_user_message_keeps_its_order():
    _, items = rp.messages_to_input([{"role": "developer", "content": "dev"}, {"role": "user", "content": "hi"}])
    assert items == [{"role": "system", "content": "dev"}, {"role": "user", "content": "hi"}]


def test_developer_messages_in_a_valid_multi_turn_history_keep_order_and_call_pairing():
    history = [
        {"role": "developer", "content": "rule one"},
        {"role": "user", "content": "check alpha"},
        {"role": "assistant", "content": None, "tool_calls": [_call("c1", '{"service":"alpha"}')]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"content":"ok"}'},
        {"role": "developer", "content": "rule two"},
        {"role": "assistant", "content": "done"},
    ]
    _, items = rp.messages_to_input(history)
    assert [item.get("role") or item.get("type") for item in items] == [
        "system", "user", "function_call", "function_call_output", "system", "assistant"]
    assert items[2]["call_id"] == items[3]["call_id"] == "c1"
    assert items[4] == {"role": "system", "content": "rule two"}


def test_empty_developer_content_is_an_empty_system_item_as_for_user_messages():
    assert rp.messages_to_input([{"role": "developer", "content": ""}])[1] == [{"role": "system", "content": ""}]
    assert rp.messages_to_input([{"role": "developer", "content": None}])[1] == [{"role": "system", "content": ""}]


def test_malformed_developer_content_is_refused():
    with pytest.raises(rp.TranslationError):
        rp.messages_to_input([{"role": "developer", "content": 5}])


def test_a_developer_message_cannot_begin_while_calls_are_unanswered():
    history = [{"role": "assistant", "content": None, "tool_calls": [_call("c1", "{}")]},
               {"role": "developer", "content": "late"}]
    with pytest.raises(rp.TranslationError):
        rp.messages_to_input(history)


# --- historical argument normalization ---------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ('{"service":"alpha"}', '{"service":"alpha"}'),   # valid object: kept byte for byte
    ('{bad', "{}"),                                   # malformed JSON
    ("[1, 2]", "{}"),                                 # JSON array
    ('"text"', "{}"),                                 # JSON string
    ("null", "{}"),                                   # JSON null
    ("7", "{}"),                                      # JSON number
    ("{}", "{}"),                                     # already normalized
    ({"service": "alpha"}, "{}"),                     # non-text value
])
def test_historical_arguments_follow_the_provider_facing_rule(raw, expected):
    _, items = rp.messages_to_input(_answered(_call("c1", raw)))
    assert items[0]["arguments"] == expected


def test_the_rule_is_the_loop_rule_and_not_a_second_policy():
    for raw in ['{"a":1}', '{bad', "[]", "null", "3", '"s"']:
        _, items = rp.messages_to_input(_answered(_call("c", raw)))
        assert items[0]["arguments"] == _provider_arguments(raw)


def test_missing_arguments_are_an_empty_object_as_in_the_loop():
    call = {"id": "c1", "type": "function", "function": {"name": "get_status"}}
    _, items = rp.messages_to_input(_answered(call))
    assert items[0]["arguments"] == "{}"


def test_multiple_calls_with_mixed_argument_validity_are_each_normalized_by_id():
    calls = [_call("ok", '{"a":1}'), _call("bad", "{x"), _call("arr", "[1]")]
    _, items = rp.messages_to_input(_answered(*calls))
    assert {item["call_id"]: item["arguments"] for item in items if item["type"] == "function_call"} == {
        "ok": '{"a":1}', "bad": "{}", "arr": "{}"}


def test_the_source_history_is_not_mutated_by_normalization():
    history = [{"role": "assistant", "content": None, "tool_calls": [_call("bad", "{x"), _call("ok", '{"a":1}')]},
               {"role": "tool", "tool_call_id": "bad", "content": "r1"},
               {"role": "tool", "tool_call_id": "ok", "content": "r2"}]
    before = copy.deepcopy(history)
    rp.messages_to_input(history)
    assert history == before
    assert history[0]["tool_calls"][0]["function"]["arguments"] == "{x"


def test_a_persisted_and_resumed_history_translates_identically():
    history = [{"role": "developer", "content": "rule"},
               {"role": "user", "content": "go"},
               {"role": "assistant", "content": None, "tool_calls": [_call("c1", "{x"), _call("c2", '{"s":"a"}')]},
               {"role": "tool", "tool_call_id": "c1", "content": "r1"},
               {"role": "tool", "tool_call_id": "c2", "content": "r2"}]
    resumed = json.loads(json.dumps(history))  # what a saved session holds
    assert rp.messages_to_input(resumed) == rp.messages_to_input(history)
    assert rp.messages_to_input(resumed)[1][2]["arguments"] == "{}"


def test_new_calls_from_the_stream_are_not_normalized_by_the_adapter():
    # Newly generated calls keep their exact arguments; the loop judges them under its existing validation.
    import asyncio
    import httpx
    from src.provider import GenerationOptions
    body = ('data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call",'
            '"call_id":"c1","name":"f","arguments":""}}\n\n'
            'data: {"type":"response.function_call_arguments.done","output_index":0,"arguments":"{bad"}\n\n'
            'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call",'
            '"call_id":"c1","name":"f","arguments":"{bad"}}\n\n'
            'data: {"type":"response.completed","response":{"status":"completed"}}\n\n').encode()
    provider = rp.ResponsesProvider(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body)))
    turn = asyncio.run(provider.complete([{"role": "user", "content": "x"}], None,
                                         GenerationOptions(model="m", temperature=0.2, max_tokens=8, stream=True)))
    assert turn.tool_calls[0]["function"]["arguments"] == "{bad"
