"""Opt-in LM Studio Responses API adapter (prototype). The Chat Completions provider remains the default; nothing in the
runtime imports this module, and make_provider() is unchanged.

Contract: complete() returns the same ProviderTurn the Chat Completions provider returns, so the loop, the journal,
authorization and verification see no difference. Requests are stateless: the full history is sent every time
(store is false, and previous_response_id is never used).

Tool-call integrity: a function call is returned only when its output item is complete and the response reached
response.completed. A partial call is never returned. Arguments are passed through exactly as received; their JSON is
judged later by the loop. Reasoning text is observational: it is counted, never parsed for instructions.
"""

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from . import config
from . import provider as p
from .loop import _provider_arguments
from .provider import FinishReason, GenerationOptions, ProviderError, ProviderTurn, TokenUsage, TurnOutcome


class TranslationError(ValueError):
    """The history cannot be expressed in the Responses input format. Nothing is sent."""


class _Protocol(Exception):
    """The stream violates the protocol: malformed, contradictory, or missing identifiers."""


def tools_to_responses(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Chat tool declarations become top-level Responses function tools. Names and parameter schemas are kept."""
    if tools is None:
        return None
    converted = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(function, dict):
            raise TranslationError("a tool declaration is not a function object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise TranslationError("a tool declaration has no name")
        item: Dict[str, Any] = {"type": "function", "name": name,
                                "parameters": copy.deepcopy(function.get("parameters", {"type": "object", "properties": {}}))}
        if "description" in function:
            item["description"] = function["description"]
        converted.append(item)
    return converted


def _validated_call(call: Any, declared: set, index: int) -> Tuple[str, str, str]:
    """Structure of one historical call. Its arguments are normalized with the provider-facing rule the Chat Completions
    history uses (loop._provider_arguments): a JSON object is kept byte for byte, anything else becomes "{}". The source
    message is not changed. Structure is still validated; nothing is silently dropped."""
    if not isinstance(call, dict):
        raise TranslationError(f"message {index} has a tool call that is not an object")
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise TranslationError(f"message {index} has a tool call without a valid id")
    if call_id in declared:
        raise TranslationError(f"call id {call_id!r} is declared more than once")
    if "type" in call and call["type"] != "function":
        raise TranslationError(f"call {call_id!r} has an invalid type")
    function = call.get("function")
    if not isinstance(function, dict):
        raise TranslationError(f"call {call_id!r} has no function object")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise TranslationError(f"call {call_id!r} has no valid function name")
    return call_id, name, _provider_arguments(function.get("arguments", "{}"))


def messages_to_input(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Chat history becomes (instructions, input items), in order.

    - A tool result must answer exactly one outstanding call: declared earlier and not yet answered.
    - No user, system or assistant message may begin while calls remain unanswered. Several calls in one assistant
      turn are valid, and so are several results.
    - The history may not end with unanswered calls: an invalid partial sequence is never sent.
    - tool_calls, when the key is present, must be a list. Missing means no calls. An empty list is valid.
    - developer messages keep their order and text as system-role input items (see the developer mapping above).
    - historical arguments are normalized with the provider-facing rule; the source messages are not changed.
    """
    if not isinstance(messages, list):
        raise TranslationError("history must be a list of messages")
    instructions: List[str] = []
    items: List[Dict[str, Any]] = []
    declared: set = set()
    outstanding: List[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TranslationError(f"message {index} is not an object")
        role = message.get("role")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise TranslationError(f"message {index} content must be text")
        if role != "tool" and outstanding:
            raise TranslationError(f"message {index} ({role}) begins while calls {outstanding} are unanswered")
        if role == "system":
            instructions.append(content or "")
        elif role == "developer":
            # Developer instructions keep their position and text. Direct developer input is not established for
            # LM Studio's Responses endpoint, so they are sent as a system-role input item rather than merged.
            items.append({"role": "system", "content": content or ""})
        elif role == "user":
            items.append({"role": "user", "content": content or ""})
        elif role == "assistant":
            if content:
                items.append({"role": "assistant", "content": content})
            if "tool_calls" in message:
                calls = message["tool_calls"]
                if not isinstance(calls, list):
                    raise TranslationError(f"message {index} tool_calls must be a list")
                for call in calls:
                    call_id, name, arguments = _validated_call(call, declared, index)
                    declared.add(call_id)
                    outstanding.append(call_id)
                    items.append({"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments})
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(content, str):
                raise TranslationError(f"tool result {index} content must be text")
            if call_id not in outstanding:
                raise TranslationError(f"tool result {index} answers no outstanding call ({call_id!r})")
            outstanding.remove(call_id)
            items.append({"type": "function_call_output", "call_id": call_id, "output": content})
        else:
            raise TranslationError(f"message {index} has an unsupported role {role!r}")
    if outstanding:
        raise TranslationError(f"history ends with unanswered calls {outstanding}")
    return ("\n\n".join(instructions) or None), items


def build_request(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]],
                  options: GenerationOptions) -> Dict[str, Any]:
    instructions, items = messages_to_input(messages)
    body: Dict[str, Any] = {"model": options.model, "input": items,
                            "max_output_tokens": options.max_tokens, "stream": True, "store": False}
    if options.temperature is not None:  # omitted when unset, as in the Chat Completions request
        body["temperature"] = options.temperature
    if instructions is not None:
        body["instructions"] = instructions
    declared = tools_to_responses(tools)
    if declared is not None:
        body["tools"] = declared
        body["tool_choice"] = "auto"
    return body


@dataclass
class _Call:
    output_index: int
    call_id: str
    name: str
    deltas: List[str] = field(default_factory=list)
    done_arguments: Optional[str] = None
    completed: bool = False


@dataclass
class ResponsesResult:
    outcome: TurnOutcome
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Optional[TokenUsage] = None
    usage_error: Optional[str] = None
    error: Optional[ProviderError] = None
    reasoning_chars: int = 0
    reasoning_tokens: Optional[int] = None

    def to_turn(self) -> ProviderTurn:
        return ProviderTurn(outcome=self.outcome, content=self.content, tool_calls=self.tool_calls,
                            finish_reason=FinishReason(self.finish_reason) if self.finish_reason else None,
                            usage=self.usage, error=self.error, usage_error=self.usage_error)


def _usage(response: Dict[str, Any]) -> Tuple[Optional[TokenUsage], Optional[str], Optional[int]]:
    """Usage is informational: a bad usage object is reported, never allowed to fail or fabricate a turn."""
    if "usage" not in response:
        return None, "usage not reported", None
    raw = response["usage"]
    if not isinstance(raw, dict):
        return None, "usage is not an object", None
    counts = {"input": raw.get("input_tokens"), "output": raw.get("output_tokens"), "total": raw.get("total_tokens")}
    for name, value in counts.items():
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)
                                  or not 0 <= value <= p.MAX_TOKEN_COUNT):
            return None, f"{name} token count is invalid", None
    reasoning = None
    details = raw.get("output_tokens_details")
    if isinstance(details, dict):
        value = details.get("reasoning_tokens")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            reasoning = value
    if counts["input"] is None or counts["output"] is None:
        return None, "usage is incomplete", reasoning
    if counts["total"] is not None and counts["total"] != counts["input"] + counts["output"]:
        return None, "usage total contradicts its parts", reasoning
    return TokenUsage.reported(counts["input"], counts["output"]), None, reasoning


def _output_index(event: Dict[str, Any]) -> int:
    value = event.get("output_index")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _Protocol("an event has a missing or invalid output_index")
    return value


def _object_field(event: Dict[str, Any], key: str, what: str) -> Dict[str, Any]:
    """A required nested object: a missing field and a present-but-invalid field are both protocol violations."""
    if key not in event:
        raise _Protocol(f"{what} is missing")
    value = event[key]
    if not isinstance(value, dict):
        raise _Protocol(f"{what} is not an object")
    return value


def _text_field(event: Dict[str, Any], key: str, what: str) -> str:
    value = event.get(key)
    if not isinstance(value, str):
        raise _Protocol(f"{what} is not text")
    return value


class ResponsesParser:
    """Feeds SSE lines, validates every event, and returns a result only after a terminal event. Nothing is returned
    for a function call that is incomplete, and nothing is returned from a stream without a terminal event."""

    def __init__(self, on_text: Optional[Callable[[str], None]] = None):
        self._on_text = on_text
        self._text: List[str] = []
        self._text_done: Optional[str] = None
        self._calls: Dict[int, _Call] = {}
        self._call_ids: set = set()
        self._reasoning_chars = 0
        self._terminal: Optional[Tuple[str, Dict[str, Any]]] = None

    def feed(self, raw_line: str) -> None:
        line = raw_line.rstrip("\r\n")
        if not line or line.startswith((":", "event:", "id:", "retry:")):
            return  # separators and comments carry no event, even after the terminal event
        if self._terminal is not None:
            raise _Protocol("an event arrived after the terminal event")
        if not line.startswith("data:"):
            raise _Protocol("unexpected line in the stream")
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        try:
            event = json.loads(payload)
        except ValueError as exc:
            raise _Protocol(f"an event is not JSON: {exc}") from exc
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise _Protocol("an event has no type")
        self._apply(event)

    def _call(self, event: Dict[str, Any]) -> _Call:
        index = _output_index(event)
        call = self._calls.get(index)
        if call is None:
            raise _Protocol(f"an event refers to output {index}, which is not a function call")
        return call

    def _function_item(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The output item of an event, or None when it is not a function call. The item must be an object with a
        string type; a non-function item carries no tool state and is ignored."""
        item = _object_field(event, "item", "an output item")
        item_type = item.get("type")
        if not isinstance(item_type, str):
            raise _Protocol("an output item has no type")
        return item if item_type == "function_call" else None

    def _apply(self, event: Dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "response.output_text.delta":
            delta = _text_field(event, "delta", "a text delta")
            self._text.append(delta)
            if self._on_text is not None:
                self._on_text(delta)
        elif kind == "response.output_text.done":
            self._text_done = _text_field(event, "text", "a text completion")
        elif kind == "response.reasoning_text.delta":
            self._reasoning_chars += len(_text_field(event, "delta", "a reasoning delta"))  # observational only
        elif kind == "response.output_item.added":
            item = self._function_item(event)
            if item is None:
                return
            index = _output_index(event)
            call_id, name = item.get("call_id"), item.get("name")
            if index in self._calls:
                raise _Protocol("a function call has a repeated output index")
            if not isinstance(call_id, str) or not call_id or call_id in self._call_ids:
                raise _Protocol("a function call has a missing, invalid or repeated call id")
            if not isinstance(name, str) or not name:
                raise _Protocol("a function call has no valid name")
            self._calls[index] = _Call(index, call_id, name)
            self._call_ids.add(call_id)
        elif kind == "response.function_call_arguments.delta":
            call = self._call(event)
            delta = _text_field(event, "delta", "an argument delta")
            if call.completed:
                raise _Protocol("an argument delta arrives after its call completed")
            call.deltas.append(delta)
        elif kind == "response.function_call_arguments.done":
            call = self._call(event)
            arguments = _text_field(event, "arguments", "an argument completion")
            if call.done_arguments is not None:
                raise _Protocol("an argument completion is repeated")
            if call.deltas and "".join(call.deltas) != arguments:
                raise _Protocol("argument deltas contradict the completed arguments")
            call.done_arguments = arguments
        elif kind == "response.output_item.done":
            item = self._function_item(event)
            if item is None:
                return
            call = self._call(event)
            if item.get("call_id") != call.call_id or item.get("name") != call.name:
                raise _Protocol("a completed call contradicts its declared id or name")
            arguments = item.get("arguments", call.done_arguments) if "arguments" in item else call.done_arguments
            if not isinstance(arguments, str):
                raise _Protocol("a completed call has no text arguments")
            if call.done_arguments is not None and call.done_arguments != arguments:
                raise _Protocol("a completed call contradicts its argument completion")
            if call.deltas and "".join(call.deltas) != arguments:
                raise _Protocol("a completed call contradicts its argument deltas")
            call.done_arguments = arguments
            call.completed = True
        elif kind == "response.completed":
            self._terminal = ("completed", _object_field(event, "response", "the response"))
        elif kind == "response.incomplete":
            self._terminal = ("incomplete", _object_field(event, "response", "the response"))
        elif kind == "response.failed":
            self._terminal = ("failed", _object_field(event, "response", "the response"))
        elif kind == "error":
            self._terminal = ("failed", event)
        # Other event types (created, in_progress, content_part, ...) carry no tool or answer state.

    def finish(self) -> ResponsesResult:
        if self._terminal is None:
            return ResponsesResult(TurnOutcome.INCOMPLETE, error=ProviderError(
                "incomplete", "the stream ended before a terminal event", None, None),
                reasoning_chars=self._reasoning_chars)
        kind, response = self._terminal
        usage, usage_error, reasoning_tokens = _usage(response)
        base = dict(usage=usage, usage_error=usage_error, reasoning_chars=self._reasoning_chars,
                    reasoning_tokens=reasoning_tokens)
        if kind == "failed":
            message = "the response failed"
            if "error" in response:
                error = response["error"]
                if not isinstance(error, dict) or not isinstance(error.get("message"), str):
                    raise _Protocol("the failure carries no valid error object")
                message = error["message"]
            return ResponsesResult(TurnOutcome.INCOMPLETE, error=ProviderError("response_failed", message, None, None),
                                   **base)
        if kind == "incomplete":
            reason = None
            if "incomplete_details" in response:
                details = response["incomplete_details"]
                if not isinstance(details, dict):
                    raise _Protocol("incomplete_details is not an object")
                if "reason" in details:
                    reason = details["reason"]
                    if not isinstance(reason, str):
                        raise _Protocol("incomplete_details.reason is not text")
            outcome = TurnOutcome.LENGTH if reason == "max_output_tokens" else TurnOutcome.INCOMPLETE
            return ResponsesResult(outcome, error=ProviderError("incomplete", f"incomplete: {reason}", None, None),
                                   **base)
        calls = sorted(self._calls.values(), key=lambda call: call.output_index)
        if any(not call.completed for call in calls):
            # A response that claims completion while a call is still open is contradictory: nothing is returned.
            return ResponsesResult(TurnOutcome.INCOMPLETE, error=ProviderError(
                "incomplete", "a function call did not complete", None, None), **base)
        if self._text_done is not None and self._text and "".join(self._text) != self._text_done:
            return ResponsesResult(TurnOutcome.MALFORMED, error=ProviderError(
                "malformed", "text deltas contradict the text completion", None, None), **base)
        content = "".join(self._text) if self._text else (self._text_done or "")
        tool_calls = [{"id": call.call_id, "type": "function",
                       "function": {"name": call.name, "arguments": call.done_arguments}} for call in calls]
        if tool_calls:
            return ResponsesResult(TurnOutcome.TOOL_CALLS, content=content, tool_calls=tool_calls,
                                   finish_reason="tool_calls", **base)
        return ResponsesResult(TurnOutcome.COMPLETED, content=content, finish_reason="stop", **base)


class ResponsesProvider:
    """The opt-in provider. Streams every request; the non-streamed form is not supported by this prototype."""

    def __init__(self, url: Optional[str] = None, *, transport: Optional[httpx.AsyncBaseTransport] = None):
        self._url = url or config.LM_STUDIO_URL.replace("/v1/chat/completions", "/v1/responses")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=config.LM_STUDIO_CONNECT_TIMEOUT, read=config.LM_STUDIO_TIMEOUT,
                                  write=10.0, pool=5.0),
            transport=transport)
        self.last_result: Optional[ResponsesResult] = None  # observational; the loop never reads it

    async def complete(self, messages, tools, options: GenerationOptions,
                       on_text: Optional[Callable[[str], None]] = None) -> ProviderTurn:
        if not options.stream:
            raise ValueError("the Responses adapter streams every request")
        try:
            body = build_request(messages, tools, options)
        except TranslationError as exc:
            return p.failure(TurnOutcome.MALFORMED, "malformed", f"history cannot be sent: {exc}")
        parser = ResponsesParser(on_text)
        try:
            async with self._client.stream("POST", self._url, json=body) as response:
                if response.status_code >= 400:
                    await response.aread()
                    return p.failure(TurnOutcome.HTTP_ERROR, "http", f"HTTP {response.status_code}",
                                     response.status_code, p._error_excerpt(response.text))
                async for line in response.aiter_lines():
                    parser.feed(line)
                result = parser.finish()
        except _Protocol as exc:
            return p.failure(TurnOutcome.MALFORMED, "malformed", str(exc))
        except httpx.TimeoutException as exc:
            return p.failure(TurnOutcome.TIMEOUT, "timeout", str(exc) or type(exc).__name__)
        except httpx.TransportError as exc:
            return p.failure(TurnOutcome.TRANSPORT, "transport", str(exc) or type(exc).__name__)
        self.last_result = result
        return result.to_turn()

    async def aclose(self) -> None:
        await self._client.aclose()


def make_responses_provider() -> ResponsesProvider:
    """Explicit, opt-in construction. The runtime's make_provider() does not call this."""
    return ResponsesProvider()
