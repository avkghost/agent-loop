"""B3.0: a streamed model turn completes only when the protocol says so.

Scripted streams only: no network, no LM Studio, no MCP. The parser tests call src/stream.py directly, and
the loop tests drive the real agent loop with scripted responses and count tool dispatches.
"""

import asyncio
import json

import pytest
import requests

from src import stream
from tests.conftest import make_streamed_response


def _flat(text):
    """Renderer output wraps long messages; compare the words, not the line breaks."""
    return " ".join(text.split())


def _data(obj):
    return "data: " + json.dumps(obj)


def _text(content, finish=None):
    return _data({"choices": [{"delta": {"content": content}, "finish_reason": finish}]})


def _finish(reason):
    return _data({"choices": [{"delta": {}, "finish_reason": reason}]})


def _call_fragment(index, call_id=None, name=None, arguments=None):
    function = {}
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments
    fragment = {"index": index, "function": function}
    if call_id is not None:
        fragment["id"] = call_id
        fragment["type"] = "function"
    return _data({"choices": [{"delta": {"tool_calls": [fragment]}, "finish_reason": None}]})


DONE = "data: [DONE]"


class ScriptedStream:
    """A streamed response that yields the given lines, then optionally fails part-way through."""

    def __init__(self, lines, then=None):
        self._lines = lines
        self._then = then
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        for line in self._lines:
            yield line
        if self._then is not None:
            raise self._then

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# Parser: the protocol decides completion
# ---------------------------------------------------------------------------

def test_valid_assistant_completion_is_accepted():
    result = stream.parse_events([_text("Hel"), _text("lo"), _finish("stop"), DONE])
    assert result.content == "Hello"
    assert result.finish_reason == "stop"
    assert result.tool_calls == []


def test_valid_tool_call_completion_keeps_id_name_and_joined_arguments():
    result = stream.parse_events([
        _call_fragment(0, call_id="call_1", name="run_command", arguments='{"command": '),
        _call_fragment(0, arguments='"ls"}'),
        _finish("tool_calls"),
        DONE,
    ])
    assert result.finish_reason == "tool_calls"
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call["id"] == "call_1"
    assert call["function"] == {"name": "run_command", "arguments": '{"command": "ls"}'}


def test_stream_ending_after_partial_text_is_incomplete():
    with pytest.raises(stream.StreamIncomplete):
        stream.parse_events([_text("I will now"), _text(" check the")])


def test_stream_ending_during_tool_call_arguments_is_incomplete():
    with pytest.raises(stream.StreamIncomplete):
        stream.parse_events([_call_fragment(0, call_id="call_1", name="run_command", arguments='{"command": "l')])


def test_complete_looking_arguments_without_terminator_are_still_incomplete():
    # The arguments parse as JSON, but the turn was never finished: they must not be trusted
    with pytest.raises(stream.StreamIncomplete):
        stream.parse_events([_call_fragment(0, call_id="call_1", name="run_command",
                                            arguments='{"command": "echo hi"}')])


def test_missing_required_terminator_is_incomplete_even_with_a_finish_reason():
    with pytest.raises(stream.StreamIncomplete, match=r"\[DONE\]"):
        stream.parse_events([_text("done"), _finish("stop")])


def test_missing_finish_reason_is_incomplete_even_with_the_terminator():
    with pytest.raises(stream.StreamIncomplete, match="finish reason"):
        stream.parse_events([_text("done"), DONE])


def test_malformed_json_event_is_malformed_not_skipped():
    with pytest.raises(stream.StreamMalformed):
        stream.parse_events([_text("ok"), "data: {not json", _finish("stop"), DONE])


@pytest.mark.parametrize("line", [
    "garbage without a field",
    _data(["a", "list", "is", "not", "an", "event"]),
    _data({"choices": "not a list"}),
    _data({"choices": [{"delta": {"content": 5}, "finish_reason": None}]}),
])
def test_structurally_invalid_events_are_malformed(line):
    with pytest.raises(stream.StreamMalformed):
        stream.parse_events([line, _finish("stop"), DONE])


def test_conflicting_finish_reasons_are_malformed():
    with pytest.raises(stream.StreamMalformed, match="conflicting"):
        stream.parse_events([_finish("stop"), _finish("tool_calls"), DONE])


def test_unrecognized_finish_reason_is_malformed():
    with pytest.raises(stream.StreamMalformed, match="unrecognized"):
        stream.parse_events([_finish("content_filter"), DONE])


def test_tool_calls_with_stop_and_stop_free_tool_turns_are_malformed():
    with pytest.raises(stream.StreamMalformed):
        stream.parse_events([_call_fragment(0, call_id="c", name="n", arguments="{}"), _finish("stop"), DONE])
    with pytest.raises(stream.StreamMalformed):
        stream.parse_events([_text("no calls here"), _finish("tool_calls"), DONE])


def test_changing_tool_call_identity_mid_call_is_malformed():
    with pytest.raises(stream.StreamMalformed, match="id changed"):
        stream.parse_events([_call_fragment(0, call_id="c1", name="n", arguments="{}"),
                             _call_fragment(0, call_id="c2"), _finish("tool_calls"), DONE])


def test_missing_tool_call_id_is_reported_as_missing_not_invented():
    result = stream.parse_events([_call_fragment(0, name="run_command", arguments="{}"),
                                  _finish("tool_calls"), DONE])
    assert result.tool_calls[0]["id"] is None, "the parser must never invent a tool_call_id"


def test_length_limit_is_well_formed_but_reported_as_length():
    result = stream.parse_events([_text("cut"), _finish("length"), DONE])
    assert result.finish_reason == "length"


def test_keep_alive_comments_event_lines_and_bytes_are_accepted():
    lines = [b": keep-alive", b"event: message", _text("hi").encode(), b"", _finish("stop").encode(), b"data: [DONE]"]
    result = stream.parse_events(lines)
    assert result.content == "hi"


def test_content_deltas_are_reported_as_they_arrive():
    seen = []
    stream.parse_events([_text("a"), _text("b"), _finish("stop"), DONE], on_text=seen.append)
    assert seen == ["a", "b"]


def test_completion_shape_preserves_the_loop_contract():
    completion = stream.to_completion(stream.parse_events([_text("x"), _finish("stop"), DONE]))
    choice = completion["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"] == {"content": "x"}


# ---------------------------------------------------------------------------
# Loop: incomplete turns cause no dispatch and no completion
# ---------------------------------------------------------------------------

def test_valid_completion_runs_once_and_completes(run_loop, capsys):
    post, calls = run_loop([make_streamed_response(finish_reason="stop", content="all done")])
    out = capsys.readouterr().out
    assert len(post.payloads) == 1
    assert calls == []
    assert "all done" in out and "completed successfully" in out


def test_valid_tool_call_completion_dispatches_exactly_once(run_loop):
    post, calls = run_loop([
        make_streamed_response(finish_reason="tool_calls", tool_calls=[
            {"id": "call_ok", "name": "run_command", "arguments": json.dumps({"command": "echo hi"})}]),
        make_streamed_response(finish_reason="stop", content="done"),
    ], dispatch_fn=lambda *a, **k: "hi")
    assert [name for name, _ in calls] == ["run_command"]
    assert calls[0][1] == {"command": "echo hi"}


def test_partial_text_then_end_of_stream_is_not_a_completed_goal(run_loop, capsys):
    partial = ScriptedStream([_text("I will now"), _text(" check the")])
    post, calls = run_loop([partial])
    out = capsys.readouterr().out
    assert calls == []
    assert "Model stream incomplete" in out
    assert "completed successfully" not in out
    assert partial.closed, "the response must be released"


def test_stream_ending_during_tool_arguments_dispatches_nothing(run_loop, capsys):
    cut = ScriptedStream([_call_fragment(0, call_id="call_cut", name="run_command",
                                         arguments='{"command": "echo hi"}')])
    post, calls = run_loop([cut])
    out = capsys.readouterr().out
    assert calls == [], "complete-looking arguments of an unfinished turn must never run"
    assert "Model stream incomplete" in out
    assert "completed successfully" not in out


def test_missing_terminator_on_a_tool_turn_dispatches_nothing(run_loop, capsys):
    lines = [_call_fragment(0, call_id="call_nt", name="run_command", arguments='{"command": "echo hi"}'),
             _finish("tool_calls")]
    post, calls = run_loop([ScriptedStream(lines)])
    assert calls == []
    assert "Model stream incomplete" in capsys.readouterr().out


def test_malformed_event_inside_a_complete_looking_stream_dispatches_nothing(run_loop, capsys):
    lines = [_call_fragment(0, call_id="call_bad", name="run_command", arguments='{"command": "echo hi"}'),
             "data: {broken", _finish("tool_calls"), DONE]
    post, calls = run_loop([ScriptedStream(lines)])
    assert calls == []
    assert "Model stream malformed" in capsys.readouterr().out


def test_transport_error_during_streaming_dispatches_nothing(run_loop, capsys):
    broken = ScriptedStream([_call_fragment(0, call_id="call_tr", name="run_command", arguments="{}")],
                            then=requests.exceptions.ChunkedEncodingError("connection reset"))
    post, calls = run_loop([broken])
    out = capsys.readouterr().out
    assert calls == []
    assert "Transport error during the model stream" in _flat(out)
    assert "completed successfully" not in out
    assert broken.closed


def test_cancellation_during_streaming_propagates_and_dispatches_nothing(run_loop):
    cancelled = ScriptedStream([_call_fragment(0, call_id="call_cx", name="run_command", arguments="{}")],
                               then=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        run_loop([cancelled])
    assert cancelled.closed


def test_length_limited_turn_is_discarded_and_stops_without_completion(run_loop, capsys):
    post, calls = run_loop([make_streamed_response(finish_reason="length", content="partial answer")])
    out = capsys.readouterr().out
    assert calls == []
    assert len(post.payloads) == 1, "a length-limited turn is not retried or continued here"
    assert "cut off by the length limit" in _flat(out)
    assert "completed successfully" not in out


def test_a_timeout_before_any_output_is_still_retried(run_loop, monkeypatch):
    timed_out = ScriptedStream([], then=requests.exceptions.Timeout("slow"))
    post, calls = run_loop([timed_out, make_streamed_response(finish_reason="stop", content="ok")],
                           LM_STUDIO_RETRY_BASE=0)
    assert len(post.payloads) == 2
    assert calls == []
