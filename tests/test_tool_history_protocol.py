"""RME-1.3 provider-facing tool-call history. Scripted provider, no LLM, no network."""

import asyncio
import json

import httpx
import pytest

from src import loop, provider, session
from src.tools.mcp_client import MCPClientManager
from tests import scripted_provider
from tests.conftest import make_streamed_response


def _tool_turn(calls):
    return make_streamed_response(finish_reason="tool_calls", tool_calls=calls)


def _stop(text):
    return make_streamed_response(finish_reason="stop", content=text)


def _call(call_id, name, arguments):
    return {"id": call_id, "name": name, "arguments": arguments}


# --- 1-4: the boundary --------------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["", "{bad", "[1, 2]", '"text"', "7", "true", "null", "   ", "{\"a\": "])
def test_non_object_or_malformed_arguments_become_the_inert_object(raw):
    declared = loop._declare_tool_call({"id": "c", "function": {"name": "list_directory", "arguments": raw}})
    assert declared["function"]["arguments"] == "{}"
    json.loads(declared["function"]["arguments"])  # always valid JSON


def test_valid_object_arguments_are_kept_byte_for_byte():
    for raw in ['{"path": "."}', '{}', '{"a": 1,  "b":  [2]}', '{"nested": {"x": null}}']:
        declared = loop._declare_tool_call({"id": "c", "function": {"name": "list_directory", "arguments": raw}})
        assert declared["function"]["arguments"] == raw


def test_missing_arguments_default_to_the_inert_object():
    assert loop._declare_tool_call({"id": "c", "function": {"name": "x"}})["function"]["arguments"] == "{}"


def test_tool_name_is_kept_as_received_when_it_is_a_string():
    odd = 'list_directory(".")'
    declared = loop._declare_tool_call({"id": "c", "function": {"name": odd, "arguments": ""}})
    assert declared["function"]["name"] == odd, "the name is preserved, not inferred or corrected"


@pytest.mark.parametrize("name", ["", None, 42])
def test_missing_or_non_string_names_get_a_placeholder_that_is_not_a_real_tool(name):
    declared = loop._declare_tool_call({"id": "c", "function": {"name": name, "arguments": "{}"}})
    assert declared["function"]["name"] == loop.INVALID_CALL_NAME
    assert loop.INVALID_CALL_NAME not in {"start_job", "status_job", "collect_job", "terminate_job",
                                          "run_command", "write_file", "read_file", "list_directory"}


# --- scripted integration: rejected call, then valid history and continuation -------------------------

def _run(monkeypatch, tmp_path, replies, dispatch_calls):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)
    scripted = list(replies)
    payloads = []

    def post(*args, **kwargs):
        payloads.append(json.loads(json.dumps(kwargs["json"])))
        if not scripted:
            raise AssertionError("unexpected model request")
        return scripted.pop(0)

    monkeypatch.setattr(scripted_provider, "POST", post)

    async def recording(tool_name, tool_input, *args, **kwargs):
        dispatch_calls.append((tool_name, tool_input))
        return {"content": "ok", "isError": False}

    monkeypatch.setattr(loop.dispatch, "dispatch_tool_async", recording)

    async def go():
        return await loop.agent_loop_async("list the directory", MCPClientManager(),
                                           asyncio.get_running_loop(), session_id="protocol")
    outcome = asyncio.run(go())
    return outcome, payloads


def test_rejected_malformed_call_is_not_executed_and_the_next_request_is_valid(monkeypatch, tmp_path):
    calls = []
    outcome, payloads = _run(monkeypatch, tmp_path, [
        _tool_turn([_call("bad1", "list_directory", "")]),
        _stop("the call was rejected; I will answer without it"),
    ], calls)
    assert calls == [], "a rejected call must never be dispatched"
    assert outcome.run_status.value == "completed_unverified", "no provider failure may be fabricated"

    second = payloads[1]["messages"]
    declared = next(m for m in second if m.get("role") == "assistant")["tool_calls"][0]
    assert declared["id"] == "bad1"
    assert declared["function"]["arguments"] == "{}"
    result = next(m for m in second if m.get("role") == "tool")
    assert result["tool_call_id"] == "bad1"
    assert "Malformed tool arguments" in result["content"]

    saved = session.load_session("protocol")
    session.validate_session_data(saved)
    record = next(r for r in saved["execution"] if r["tool_call_id"] == "bad1")
    assert record["status"] == "failed" and record["arguments"] == "", "the journal keeps the original input"


def test_malformed_name_with_empty_arguments_replays_as_a_valid_protocol_history(monkeypatch, tmp_path):
    calls = []
    outcome, payloads = _run(monkeypatch, tmp_path, [
        _tool_turn([_call("149899920", 'list_directory(".")', "")]),
        _stop("continuing"),
    ], calls)
    assert calls == []
    declared = next(m for m in payloads[1]["messages"] if m.get("role") == "assistant")["tool_calls"][0]
    assert declared["function"] == {"name": 'list_directory(".")', "arguments": "{}"}
    record = session.load_session("protocol")["execution"][0]
    assert record["tool_name"] == 'list_directory(".")' and record["arguments"] == ""


def test_multiple_calls_keep_order_pairing_and_only_valid_calls_run(monkeypatch, tmp_path):
    calls = []
    outcome, payloads = _run(monkeypatch, tmp_path, [
        _tool_turn([_call("m1", "list_directory", ""), _call("ok2", "list_directory", '{"path": "."}'),
                    _call("m3", "", '{"path": "."}')]),
        _stop("done"),
    ], calls)
    assert calls == [("list_directory", {"path": "."})], "only the valid call is dispatched, exactly once"
    history = payloads[1]["messages"]
    declared = next(m for m in history if m.get("role") == "assistant")["tool_calls"]
    assert [d["id"] for d in declared] == ["m1", "ok2", "m3"]
    assert [d["function"]["arguments"] for d in declared] == ["{}", '{"path": "."}', '{"path": "."}']
    assert declared[2]["function"]["name"] == loop.INVALID_CALL_NAME
    results = [m for m in history if m.get("role") == "tool"]
    assert [r["tool_call_id"] for r in results] == ["m1", "ok2", "m3"]
    assert "Malformed" in results[0]["content"] and '"isError": false' in results[1]["content"]


def test_non_object_json_never_reaches_dispatch_or_history_as_arguments(monkeypatch, tmp_path):
    calls = []
    outcome, payloads = _run(monkeypatch, tmp_path, [
        _tool_turn([_call("a1", "list_directory", "[1, 2]"), _call("a2", "list_directory", "null")]),
        _stop("done"),
    ], calls)
    assert calls == []
    declared = next(m for m in payloads[1]["messages"] if m.get("role") == "assistant")["tool_calls"]
    assert [d["function"]["arguments"] for d in declared] == ["{}", "{}"]


def test_saved_session_with_a_rejected_malformed_call_validates(monkeypatch, tmp_path):
    calls = []
    _run(monkeypatch, tmp_path, [_tool_turn([_call("s1", "list_directory", "{bad")]), _stop("done")], calls)
    saved = session.load_session("protocol")
    assert session.validate_session_data(saved) == "structured"
    assert any(m.get("tool_calls") for m in saved["messages"] if m.get("role") == "assistant")


# --- HTTP error diagnostics ---------------------------------------------------------------------------

def _provider_with(handler):
    return provider.LMStudioProvider(url="http://test.invalid/v1/chat/completions",
                                     transport=httpx.MockTransport(handler))


def test_http_error_keeps_status_and_message_and_stores_a_bounded_excerpt():
    body = "<html>" + ("internal details " * 200) + "</html>"

    def handler(request):
        return httpx.Response(500, text=body)

    options = provider.GenerationOptions(stream=False)
    turn = asyncio.run(_provider_with(handler).complete([{"role": "user", "content": "hi"}], None, options))
    assert turn.outcome is provider.TurnOutcome.HTTP_ERROR
    assert turn.error.status == 500 and turn.error.message == "HTTP 500" and turn.error.kind == "http"
    assert turn.error.excerpt and len(turn.error.excerpt) <= provider.EXCERPT_CHARS


def test_streamed_http_error_also_stores_only_the_response_excerpt():
    def handler(request):
        return httpx.Response(503, text="model is loading")

    turn = asyncio.run(_provider_with(handler).complete([{"role": "user", "content": "hi"}], None,
                                                        provider.GenerationOptions(stream=True)))
    assert turn.error.status == 503 and turn.error.message == "HTTP 503"
    assert turn.error.excerpt == "model is loading"


def test_error_excerpt_excludes_request_material_and_controls():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(500, text="bad\x00request\n\tbody")

    turn = asyncio.run(_provider_with(handler).complete(
        [{"role": "user", "content": "SECRET PROMPT TEXT"}], None, provider.GenerationOptions(stream=False)))
    assert "SECRET PROMPT TEXT" not in (turn.error.excerpt or "")
    assert "\x00" not in turn.error.excerpt and turn.error.excerpt == "bad request body"
    assert provider._error_excerpt("") is None and provider._error_excerpt(None) is None
