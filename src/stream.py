"""Completion rules for one streamed model turn (OpenAI-compatible server-sent events).

A turn is complete only when the protocol says so: the stream carries a finish reason, and the stream
ends with the [DONE] terminator. Anything else is a failure, never a normal stop:

- StreamIncomplete: the stream ended without its terminator or without a finish reason.
- StreamMalformed: an event could not be parsed, or the events contradict one another.

A stream that completes with finish_reason "length" is well formed but not a successful turn. The loop
reports it and discards it. Tool-call arguments are never judged here: a complete stream passes them on
unchanged, and the loop validates them exactly as before.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

FINISH_STOP = "stop"
FINISH_TOOL_CALLS = "tool_calls"
FINISH_LENGTH = "length"
_KNOWN_FINISH = {FINISH_STOP, FINISH_TOOL_CALLS, FINISH_LENGTH}
_SSE_IGNORED_FIELDS = {"event", "id", "retry"}


class StreamError(Exception):
    """The turn did not complete. Nothing from it may be appended to history or dispatched."""

    category = "malformed"


class StreamIncomplete(StreamError):
    category = "incomplete"


class StreamMalformed(StreamError):
    category = "malformed"


@dataclass
class StreamResult:
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""


def parse_events(lines: Iterable[Any], on_text: Optional[Callable[[str], None]] = None) -> StreamResult:
    """Parse one complete server-sent event stream, or raise a StreamError.

    Each item is one line, as bytes or str. on_text receives content deltas as they arrive.
    """
    parts: List[str] = []
    calls: Dict[int, Dict[str, Any]] = {}
    finish: Optional[str] = None
    terminated = False

    for raw in lines:
        if raw is None:
            continue
        try:
            line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        except UnicodeDecodeError as e:
            raise StreamMalformed(f"stream is not valid UTF-8: {e}") from e
        line = line.rstrip("\r\n")
        if not line or line.startswith(":"):
            continue  # blank separator or comment (keep-alive)
        field_name, _, value = line.partition(":")
        if field_name in _SSE_IGNORED_FIELDS:
            continue
        if field_name != "data":
            raise StreamMalformed(f"unexpected stream line: {line[:60]!r}")
        if value.startswith(" "):
            value = value[1:]
        if value == "[DONE]":
            terminated = True
            break

        try:
            chunk = json.loads(value)
        except ValueError as e:
            raise StreamMalformed(f"invalid JSON event: {e}") from e
        if not isinstance(chunk, dict):
            raise StreamMalformed("stream event is not an object")
        choices = chunk.get("choices")
        if choices is None:
            continue  # for example a usage-only event
        if not isinstance(choices, list):
            raise StreamMalformed("stream event choices is not a list")
        if not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            raise StreamMalformed("stream choice is not an object")

        delta = choice.get("delta", {})
        if not isinstance(delta, dict):
            raise StreamMalformed("stream delta is not an object")
        text = delta.get("content")
        if text is not None:
            if not isinstance(text, str):
                raise StreamMalformed("stream content is not a string")
            if text:
                parts.append(text)
                if on_text is not None:
                    on_text(text)
        for fragment in delta.get("tool_calls") or []:
            _merge_tool_call_fragment(calls, fragment)

        reason = choice.get("finish_reason")
        if reason is not None:
            if not isinstance(reason, str):
                raise StreamMalformed("finish_reason is not a string")
            if finish is not None and reason != finish:
                raise StreamMalformed(f"conflicting finish reasons: {finish!r} and {reason!r}")
            finish = reason

    if not terminated:
        raise StreamIncomplete("the stream ended before its [DONE] terminator")
    if finish is None:
        raise StreamIncomplete("the stream ended without a finish reason")
    if finish not in _KNOWN_FINISH:
        raise StreamMalformed(f"unrecognized finish reason {finish!r}")

    tool_calls = [calls[index] for index in sorted(calls)]
    if finish == FINISH_TOOL_CALLS and not tool_calls:
        raise StreamMalformed("finish reason tool_calls without any tool call")
    if finish == FINISH_STOP and tool_calls:
        raise StreamMalformed("tool calls with finish reason stop")
    return StreamResult(content="".join(parts), tool_calls=tool_calls, finish_reason=finish)


def _merge_tool_call_fragment(calls: Dict[int, Dict[str, Any]], fragment: Any) -> None:
    if not isinstance(fragment, dict):
        raise StreamMalformed("tool call fragment is not an object")
    index = fragment.get("index", 0)
    if isinstance(index, bool) or not isinstance(index, int):
        raise StreamMalformed("tool call index is not an integer")
    entry = calls.setdefault(index, {"id": None, "type": "function", "function": {"name": None, "arguments": ""}})

    call_id = fragment.get("id")
    if call_id is not None:
        if not isinstance(call_id, str):
            raise StreamMalformed("tool call id is not a string")
        if entry["id"] is None:
            entry["id"] = call_id
        elif entry["id"] != call_id:
            raise StreamMalformed("tool call id changed within one call")

    function = fragment.get("function") or {}
    if not isinstance(function, dict):
        raise StreamMalformed("tool call function is not an object")
    name = function.get("name")
    if name is not None:
        if not isinstance(name, str):
            raise StreamMalformed("tool call name is not a string")
        if entry["function"]["name"] is None:
            entry["function"]["name"] = name
        elif entry["function"]["name"] != name:
            raise StreamMalformed("tool call name changed within one call")
    arguments = function.get("arguments")
    if arguments is not None:
        if not isinstance(arguments, str):
            raise StreamMalformed("tool call arguments are not a string")
        entry["function"]["arguments"] += arguments


def to_completion(result: StreamResult) -> Dict[str, Any]:
    """The completion shape the loop consumes (one choice, with message and finish_reason)."""
    message: Dict[str, Any] = {"content": result.content}
    if result.tool_calls:
        message["tool_calls"] = result.tool_calls
    return {"choices": [{"message": message, "finish_reason": result.finish_reason}]}
