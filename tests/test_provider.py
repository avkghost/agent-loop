"""B3.1: the async provider contract, the LM Studio adapter, and the async guarantees around model I/O.

The adapter tests use the real httpx client with an in-process transport, so SSE framing, timeouts,
cancellation, and connection cleanup are exercised for real. No socket is opened.
"""

import asyncio
import json

import httpx
import pytest

from src import config, dispatch, loop, provider as p
from src.provider import FinishReason, GenerationOptions, LMStudioProvider, TurnOutcome
from src.tools.mcp_client import MCPClientManager
from tests.conftest import make_streamed_response


def _sse(*events, done=True):
    lines = [f"data: {json.dumps(e)}" for e in events]
    if done:
        lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode()


def _text(content, finish=None):
    return {"choices": [{"delta": {"content": content}, "finish_reason": finish}]}


def _finish(reason):
    return {"choices": [{"delta": {}, "finish_reason": reason}]}


def _call(index, call_id=None, name=None, arguments=None):
    fn = {}
    if name is not None:
        fn["name"] = name
    if arguments is not None:
        fn["arguments"] = arguments
    frag = {"index": index, "function": fn}
    if call_id is not None:
        frag["id"] = call_id
        frag["type"] = "function"
    return {"choices": [{"delta": {"tool_calls": [frag]}, "finish_reason": None}]}


def _provider_for(body: bytes, status: int = 200, captured=None):
    def handler(request):
        if captured is not None:
            captured.append(json.loads(request.content))
        return httpx.Response(status, content=body, headers={"content-type": "text/event-stream"})
    return LMStudioProvider(transport=httpx.MockTransport(handler))


async def _complete(provider, messages=None, tools=(), options=None, on_text=None):
    """tools=() means an empty tool list; tools=None means no tool declarations at all."""
    try:
        return await provider.complete(messages or [{"role": "user", "content": "hi"}],
                                       [] if tools == () else tools,
                                       options or GenerationOptions(), on_text=on_text)
    finally:
        await provider.aclose()


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Provider contract: completed, tool-call, usage, length, and error normalization
# ---------------------------------------------------------------------------

def test_complete_assistant_response_is_a_completed_turn():
    turn = _run(_complete(_provider_for(_sse(_text("Hel"), _text("lo"), _finish("stop")))))
    assert turn.outcome is TurnOutcome.COMPLETED and turn.succeeded
    assert turn.content == "Hello" and turn.finish_reason is FinishReason.STOP
    assert turn.tool_calls == []


def test_complete_tool_call_response_keeps_ids_and_arguments_exactly():
    body = _sse(_call(0, call_id="call_abc123", name="run_command", arguments='{"command": '),
                _call(0, arguments='"ls -la"}'), _finish("tool_calls"))
    turn = _run(_complete(_provider_for(body)))
    assert turn.outcome is TurnOutcome.TOOL_CALLS
    assert turn.tool_calls[0]["id"] == "call_abc123"
    assert turn.tool_calls[0]["function"] == {"name": "run_command", "arguments": '{"command": "ls -la"}'}


def test_reported_token_usage_is_labelled_reported():
    usage = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}
    turn = _run(_complete(_provider_for(_sse(_text("ok"), _finish("stop"), {"usage": usage}))))
    assert turn.usage == p.TokenUsage(120, 30, 150, "reported")


def test_missing_usage_is_none_so_the_runtime_must_estimate_and_label_it():
    turn = _run(_complete(_provider_for(_sse(_text("ok"), _finish("stop")))))
    assert turn.usage is None


def test_length_limited_response_is_a_distinct_outcome_not_a_completion():
    turn = _run(_complete(_provider_for(_sse(_text("cut"), _finish("length")))))
    assert turn.outcome is TurnOutcome.LENGTH and not turn.succeeded


def test_http_error_status_is_normalized_and_never_a_completion():
    turn = _run(_complete(_provider_for(b"server exploded", status=500)))
    assert turn.outcome is TurnOutcome.HTTP_ERROR and turn.error.status == 500
    assert not turn.succeeded


def test_non_streamed_summary_request_omits_tools_and_needs_a_finish_reason():
    captured = []
    body = json.dumps({"choices": [{"message": {"content": "summary"}, "finish_reason": "stop"}]}).encode()
    turn = _run(_complete(_provider_for(body, captured=captured), tools=None,
                          options=GenerationOptions(stream=False, max_tokens=500, temperature=0.3)))
    assert turn.outcome is TurnOutcome.COMPLETED and turn.content == "summary"
    assert captured[0]["stream"] is False and "tools" not in captured[0]

    no_reason = json.dumps({"choices": [{"message": {"content": "partial"}}]}).encode()
    turn = _run(_complete(_provider_for(no_reason), tools=None, options=GenerationOptions(stream=False)))
    assert turn.outcome is TurnOutcome.INCOMPLETE


def test_request_payload_matches_the_existing_contract():
    captured = []
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    _run(_complete(_provider_for(_sse(_text("ok"), _finish("stop")), captured=captured), tools=tools))
    body = captured[0]
    assert body["stream"] is True and body["tools"] == tools and body["tool_choice"] == "auto"
    assert body["model"] == config.MODEL_NAME and body["max_tokens"] == config.REQUEST_MAX_TOKENS


# ---------------------------------------------------------------------------
# Streaming rules through the adapter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("body,outcome", [
    (_sse(_text("partial"), done=False), TurnOutcome.INCOMPLETE),                       # no terminator
    (_sse(_text("done")), TurnOutcome.INCOMPLETE),                                       # no finish reason
    (_sse(_text("x"), _finish("stop"), _finish("tool_calls")), TurnOutcome.MALFORMED),   # conflicting finish
    (_sse(_text("x"), _finish("stop"), done=True) + b"data: [DONE]\n\n", TurnOutcome.MALFORMED),  # duplicate [DONE]
    (_sse(_text("x"), _finish("stop")) + b'data: {"choices":[{"delta":{"content":"late"}}]}\n\n',
     TurnOutcome.MALFORMED),                                                              # data after [DONE]
    (b'data: {"choices": [\n\n', TurnOutcome.MALFORMED),                                  # malformed SSE event
    (b"this line is not sse\n\n", TurnOutcome.MALFORMED),
])
def test_stream_violations_are_never_completions(body, outcome):
    turn = _run(_complete(_provider_for(body)))
    assert turn.outcome is outcome and not turn.succeeded


def test_repeated_identical_finish_marker_is_accepted_and_documented():
    turn = _run(_complete(_provider_for(_sse(_text("ok"), _finish("stop"), _finish("stop")))))
    assert turn.outcome is TurnOutcome.COMPLETED


def test_partial_tool_arguments_yield_an_incomplete_turn_with_no_calls():
    body = _sse(_call(0, call_id="call_cut", name="run_command", arguments='{"command": "l'), done=False)
    turn = _run(_complete(_provider_for(body)))
    assert turn.outcome is TurnOutcome.INCOMPLETE
    assert turn.tool_calls == [], "a turn that never completed must expose no tool calls"


# ---------------------------------------------------------------------------
# Transport failures and cleanup
# ---------------------------------------------------------------------------

def test_connect_timeout_is_a_timeout_outcome():
    def handler(request):
        raise httpx.ConnectTimeout("connect timed out", request=request)

    turn = _run(_complete(LMStudioProvider(transport=httpx.MockTransport(handler))))
    assert turn.outcome is TurnOutcome.TIMEOUT


def test_connection_failure_is_a_transport_outcome():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    turn = _run(_complete(LMStudioProvider(transport=httpx.MockTransport(handler))))
    assert turn.outcome is TurnOutcome.TRANSPORT


class _ScriptedBody(httpx.AsyncByteStream):
    """A streamed body that yields chunks, optionally fails, and records that it was closed."""

    def __init__(self, chunks, then=None, pause=0.0, owner=None):
        self._chunks, self._then, self._pause, self._owner = chunks, then, pause, owner

    async def __aiter__(self):
        for chunk in self._chunks:
            if self._pause:
                await asyncio.sleep(self._pause)
            yield chunk
        if self._then is not None:
            raise self._then

    async def aclose(self):
        if self._owner is not None:
            self._owner.closed += 1


class _Transport(httpx.AsyncBaseTransport):
    def __init__(self, body_factory):
        self._factory = body_factory
        self.closed = 0

    async def handle_async_request(self, request):
        return httpx.Response(200, stream=self._factory(self), request=request)


def test_connection_dropped_mid_stream_is_a_transport_outcome_and_releases_the_stream():
    transport = _Transport(lambda owner: _ScriptedBody(
        [b'data: {"choices":[{"delta":{"content":"he"},"finish_reason":null}]}\n\n'],
        then=httpx.RemoteProtocolError("connection reset"), owner=owner))
    turn = _run(_complete(LMStudioProvider(transport=transport)))
    assert turn.outcome is TurnOutcome.TRANSPORT
    assert transport.closed == 1, "the response must be closed after a failure"


def test_read_timeout_mid_stream_is_a_timeout_and_releases_the_stream():
    transport = _Transport(lambda owner: _ScriptedBody(
        [b'data: {"choices":[{"delta":{"content":"he"},"finish_reason":null}]}\n\n'],
        then=httpx.ReadTimeout("read timed out"), owner=owner))
    turn = _run(_complete(LMStudioProvider(transport=transport)))
    assert turn.outcome is TurnOutcome.TIMEOUT
    assert transport.closed == 1


def test_cancellation_interrupts_a_blocked_response_and_closes_the_transport():
    async def scenario():
        started = asyncio.Event()
        transport = _Transport(lambda owner: _BlockingBody(owner, started))
        provider = LMStudioProvider(transport=transport)
        task = asyncio.ensure_future(provider.complete([{"role": "user", "content": "hi"}], [],
                                                       GenerationOptions()))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await provider.aclose()
        return transport

    transport = _run(scenario())
    assert transport.closed == 1, "cancelling a model request must close its HTTP response"


class _BlockingBody(httpx.AsyncByteStream):
    def __init__(self, owner, started):
        self._owner, self._started = owner, started

    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"he"},"finish_reason":null}]}\n\n'
        self._started.set()
        await asyncio.Event().wait()  # waits for cancellation, like a stalled server

    async def aclose(self):
        self._owner.closed += 1


def test_event_loop_stays_responsive_during_slow_provider_io():
    async def scenario():
        ticks = 0
        done = asyncio.Event()

        async def ticker():
            nonlocal ticks
            while not done.is_set():
                ticks += 1
                await asyncio.sleep(0.01)

        transport = _Transport(lambda owner: _ScriptedBody(
            [b'data: {"choices":[{"delta":{"content":"a"},"finish_reason":null}]}\n\n'] * 15
            + [b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n', b"data: [DONE]\n\n"],
            pause=0.02, owner=owner))
        tick_task = asyncio.ensure_future(ticker())
        turn = await _complete(LMStudioProvider(transport=transport))
        done.set()
        await tick_task
        return turn, ticks

    turn, ticks = _run(scenario())
    assert turn.outcome is TurnOutcome.COMPLETED
    assert ticks >= 15, "other tasks must keep running while the model streams"


# ---------------------------------------------------------------------------
# Runtime integration: retries, cancellation, usage, compaction, and tool safety
# ---------------------------------------------------------------------------

def _drive_agent(monkeypatch, provider_obj, dispatch_fn=None, **config_overrides):
    monkeypatch.setattr(p, "make_provider", lambda: provider_obj)
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)
    for name, value in config_overrides.items():
        monkeypatch.setattr(config, name, value)
    calls = []

    async def recording(tool_name, tool_input, *args, **kwargs):
        calls.append((tool_name, tool_input))
        return dispatch_fn(tool_name, tool_input) if dispatch_fn else "ok"

    monkeypatch.setattr(dispatch, "dispatch_tool_async", recording)
    return calls


def test_retry_backoff_does_not_block_other_tasks(monkeypatch, capsys):
    from tests import scripted_provider
    replies = [None, None]  # two timeouts, then a completed answer

    def post(url, json=None, timeout=None, stream=False):
        if replies:
            replies.pop()
            raise scripted_provider.requests.exceptions.Timeout("slow model")
        return make_streamed_response(finish_reason="stop", content="done")

    monkeypatch.setattr(scripted_provider, "POST", post)
    monkeypatch.setattr(config, "LM_STUDIO_RETRY_BASE", 0.25)
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)

    async def scenario():
        ticks = 0
        done = asyncio.Event()

        async def ticker():
            nonlocal ticks
            while not done.is_set():
                ticks += 1
                await asyncio.sleep(0.01)

        tick_task = asyncio.ensure_future(ticker())
        event_loop = asyncio.get_running_loop()
        await loop.agent_loop_async("goal", MCPClientManager(), event_loop)
        done.set()
        await tick_task
        return ticks

    ticks = _run(scenario())
    assert ticks >= 40, f"backoff of 0.25s and 0.5s must leave other tasks running (ticks={ticks})"


def test_cancelling_the_run_during_a_blocked_model_stream_dispatches_nothing_and_closes_transport(monkeypatch):
    async def scenario():
        started = asyncio.Event()
        transport = _Transport(lambda owner: _BlockingBody(owner, started))
        calls = _drive_agent(monkeypatch, LMStudioProvider(transport=transport))
        task = asyncio.ensure_future(loop.agent_loop_async(
            "goal", MCPClientManager(), asyncio.get_running_loop(), session_id="cancel-stream"))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return transport, calls

    transport, calls = _run(scenario())
    assert transport.closed == 1
    assert calls == [], "a cancelled model stream must not dispatch anything"


async def _agent_run():
    await loop.agent_loop_async("goal", MCPClientManager(), asyncio.get_running_loop())


def test_completed_turn_reports_reported_usage_and_estimate_is_labelled(monkeypatch, capsys):
    from tests import scripted_provider
    usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}

    def post(url, json=None, timeout=None, stream=False):
        return make_streamed_response(finish_reason="stop", content="answer")

    monkeypatch.setattr(scripted_provider, "POST", post)
    _drive_agent(monkeypatch, scripted_provider.ScriptedProvider())
    _run(_agent_run())
    assert "tokens (estimated)" in capsys.readouterr().out

    lines = [b'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":null}]}\n\n',
             b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
             ("data: " + json.dumps({"usage": usage}) + "\n\n").encode(), b"data: [DONE]\n\n"]

    class Streamed:
        def raise_for_status(self):
            return None

        def iter_lines(self):
            return iter(lines)

        def close(self):
            return None

    monkeypatch.setattr(scripted_provider, "POST", lambda url, json=None, timeout=None, stream=False: Streamed())
    _run(_agent_run())
    assert "tokens (reported): prompt 7, completion 3, total 10" in capsys.readouterr().out


def test_summary_failure_is_reported_not_silent(monkeypatch, capsys):
    from tests import scripted_provider
    from src import compaction

    def post(url, json=None, timeout=None, stream=False):
        raise scripted_provider.requests.exceptions.ConnectionError("summarizer down")

    monkeypatch.setattr(scripted_provider, "POST", post)
    state = compaction.CompactionState(objective="Objective: x")
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Objective: x"}]
    messages += [{"role": "assistant", "content": f"step {i} " * 40} for i in range(12)]
    result = _run(compaction.compact_messages(messages, state=state))
    assert state.last_failure and "transport" in state.last_failure
    assert result, "the bounded fallback still produces a valid history"


def test_no_provider_task_is_left_running_after_a_request_or_a_cancellation():
    async def scenario():
        before = set(asyncio.all_tasks())
        turn = await _complete(_provider_for(_sse(_text("ok"), _finish("stop"))))
        await asyncio.sleep(0)
        leftover_after_success = set(asyncio.all_tasks()) - before
        started = asyncio.Event()
        transport = _Transport(lambda owner: _BlockingBody(owner, started))
        provider = LMStudioProvider(transport=transport)
        task = asyncio.ensure_future(provider.complete([{"role": "user", "content": "hi"}], [], GenerationOptions()))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await provider.aclose()
        await asyncio.sleep(0)
        leftover_after_cancel = set(asyncio.all_tasks()) - before - {asyncio.current_task()}
        return turn, leftover_after_success, leftover_after_cancel

    turn, after_success, after_cancel = _run(scenario())
    assert turn.outcome is TurnOutcome.COMPLETED
    assert after_success == set() and after_cancel == set()
