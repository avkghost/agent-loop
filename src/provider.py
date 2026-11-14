"""Model provider boundary: typed turns, and the LM Studio adapter (async HTTP with SSE streaming).

The agent loop and the compaction code talk only to ModelProvider. They never see HTTP, SSE framing, or
transport exception classes. Every outcome of a model request is returned as a ProviderTurn, and
cancellation propagates as asyncio.CancelledError. A failure is never a completed turn.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import httpx

from . import config
from . import stream as sse


class FinishReason(str, Enum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"


class TurnOutcome(str, Enum):
    COMPLETED = "completed"        # finish_reason stop: an assistant answer
    TOOL_CALLS = "tool_calls"      # finish_reason tool_calls: calls to validate and dispatch
    LENGTH = "length"              # cut off by the length limit: well formed, not a finished answer
    INCOMPLETE = "incomplete"      # the stream ended before its terminator or finish reason
    MALFORMED = "malformed"        # a protocol violation
    TRANSPORT = "transport"        # connection or I/O failure
    TIMEOUT = "timeout"            # a connect, read, or write timeout
    HTTP_ERROR = "http_error"      # the server answered with an error status


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    source: str  # "reported" by the provider, or "estimated" by the runtime

    @classmethod
    def reported(cls, prompt: int, completion: int, total: Optional[int] = None) -> "TokenUsage":
        return cls(prompt, completion, prompt + completion if total is None else total, "reported")

    @classmethod
    def estimated(cls, prompt: int, completion: int) -> "TokenUsage":
        return cls(prompt, completion, prompt + completion, "estimated")


EXCERPT_CHARS = 200  # bounded diagnostic excerpt of an HTTP error body; never the request


def _error_excerpt(text: Any) -> Optional[str]:
    """The first EXCERPT_CHARS of a server error body with control characters removed. Only the response is
    read here: request bodies, prompts, and headers (including credentials) are never passed in."""
    if not isinstance(text, str) or not text:
        return None
    cleaned = "".join(ch if ch.isprintable() or ch in " \t" else " " for ch in text[:EXCERPT_CHARS * 4])
    return " ".join(cleaned.split())[:EXCERPT_CHARS] or None


@dataclass(frozen=True)
class ProviderError:
    kind: str
    message: str
    status: Optional[int] = None
    # Diagnostic only. Kept on the error object; the user-facing message does not include it.
    excerpt: Optional[str] = None


@dataclass(frozen=True)
class GenerationOptions:
    model: str = config.MODEL_NAME
    temperature: float = 0.2
    max_tokens: int = config.REQUEST_MAX_TOKENS
    stream: bool = True


@dataclass
class ProviderTurn:
    outcome: TurnOutcome
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[FinishReason] = None
    usage: Optional[TokenUsage] = None
    error: Optional[ProviderError] = None
    usage_error: Optional[str] = None  # why provider usage was not used; never fails the turn

    @property
    def succeeded(self) -> bool:
        return self.outcome in (TurnOutcome.COMPLETED, TurnOutcome.TOOL_CALLS)


class ModelProvider(Protocol):
    async def complete(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]],
                       options: GenerationOptions,
                       on_text: Optional[Callable[[str], None]] = None) -> ProviderTurn:
        """One model request. tools=None omits tool declarations (used for summaries)."""

    async def aclose(self) -> None:
        """Release transport resources."""


def build_payload(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]],
                  options: GenerationOptions) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": options.model,
        "messages": messages,
        "temperature": options.temperature,
        "max_tokens": options.max_tokens,
        "stream": options.stream,
    }
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return payload


def failure(outcome: TurnOutcome, kind: str, message: str, status: Optional[int] = None,
            excerpt: Optional[str] = None) -> ProviderTurn:
    return ProviderTurn(outcome=outcome, error=ProviderError(kind, message, status, excerpt))


_OUTCOME_FOR_FINISH = {
    sse.FINISH_STOP: TurnOutcome.COMPLETED,
    sse.FINISH_TOOL_CALLS: TurnOutcome.TOOL_CALLS,
    sse.FINISH_LENGTH: TurnOutcome.LENGTH,
}


MAX_TOKEN_COUNT = 2**53 - 1  # JSON-safe integer range; larger values cannot be real token counts


def _token_count(value: Any) -> Optional[int]:
    """A nonnegative integer token count, or None. Booleans, floats, strings, and negatives are not counts."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0 or value > MAX_TOKEN_COUNT:
        return None
    return value


def usage_from(raw: Any) -> Tuple[Optional[TokenUsage], Optional[str]]:
    """(usage, problem). Usage is reported only when its counts are valid and unambiguous.

    Policy (usage never fails a turn):
    - No usage object: (None, None). The caller estimates, labelled as an estimate.
    - Not an object, a count that is not a nonnegative integer, or a total that contradicts
      prompt_tokens + completion_tokens: (None, problem). The caller estimates and says why.
    - prompt_tokens and completion_tokens both valid: reported. total_tokens is accepted when it is valid
      and consistent, and derived when absent.
    - Only one of prompt_tokens and completion_tokens: (None, problem). A total alone does not give the split,
      so partial usage is not reported.
    """
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        return None, "usage is not an object"
    prompt_raw = raw.get("prompt_tokens")
    completion_raw = raw.get("completion_tokens")
    total_raw = raw.get("total_tokens")
    if prompt_raw is None and completion_raw is None:
        if total_raw is None:
            return None, None  # an empty object carries no counts
        return None, "only total_tokens was reported"
    if prompt_raw is None or completion_raw is None:
        return None, "partial usage: both prompt_tokens and completion_tokens are required"
    prompt, completion = _token_count(prompt_raw), _token_count(completion_raw)
    if prompt is None or completion is None:
        return None, "token counts must be nonnegative integers"
    if total_raw is None:
        return TokenUsage.reported(prompt, completion), None
    total = _token_count(total_raw)
    if total is None:
        return None, "total_tokens must be a nonnegative integer"
    if total != prompt + completion:
        return None, f"total_tokens {total} contradicts prompt_tokens + completion_tokens = {prompt + completion}"
    return TokenUsage.reported(prompt, completion, total), None


def usage_from_events(events: List[Any]) -> Tuple[Optional[TokenUsage], Optional[str]]:
    """Usage from every usage object a stream carried. Repeats of one object are fine; conflicts are not."""
    if not events:
        return None, None
    distinct: List[Any] = []
    for event in events:
        if event not in distinct:
            distinct.append(event)
    if len(distinct) > 1:
        return None, "the stream carried conflicting usage objects"
    return usage_from(distinct[0])


def turn_from_stream_error(error: sse.StreamError) -> ProviderTurn:
    outcome = TurnOutcome.INCOMPLETE if isinstance(error, sse.StreamIncomplete) else TurnOutcome.MALFORMED
    return failure(outcome, error.category, str(error))


def turn_from_parser(parser: sse.StreamParser) -> ProviderTurn:
    try:
        result = parser.finish()
    except sse.StreamError as e:
        return turn_from_stream_error(e)
    usage, problem = usage_from_events(parser.usage_events)
    return ProviderTurn(
        outcome=_OUTCOME_FOR_FINISH[result.finish_reason],
        content=result.content,
        tool_calls=result.tool_calls,
        finish_reason=FinishReason(result.finish_reason),
        usage=usage,
        usage_error=problem,
    )


class _MalformedResponse(Exception):
    pass


def _validated_tool_call(raw: Any, index: int) -> Dict[str, Any]:
    """Structure only: application validation (JSON arguments, known tool names) stays in the loop.

    A call must be an object with an id (string), type "function" (or omitted), and a function object
    with a name (non-empty string) and arguments (string). Nothing is invented or coerced.
    """
    if not isinstance(raw, dict):
        raise _MalformedResponse(f"tool call {index} is not an object")
    if raw.get("type", "function") != "function":
        raise _MalformedResponse(f"tool call {index} has an unsupported type")
    call_id = raw.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise _MalformedResponse(f"tool call {index} has no valid id")
    function = raw.get("function")
    if not isinstance(function, dict):
        raise _MalformedResponse(f"tool call {index} has no function object")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise _MalformedResponse(f"tool call {index} has no valid function name")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise _MalformedResponse(f"tool call {index} arguments must be a string")
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def turn_from_json(data: Any) -> ProviderTurn:
    """A non-streamed completion. Same completion rule as a stream: the finish reason must be present."""
    try:
        if not isinstance(data, dict):
            return failure(TurnOutcome.MALFORMED, "malformed", "response is not an object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return failure(TurnOutcome.MALFORMED, "malformed", "response choices must be a non-empty list")
        choice = choices[0]
        if not isinstance(choice, dict):
            return failure(TurnOutcome.MALFORMED, "malformed", "response choice is not an object")
        message = choice.get("message")
        if not isinstance(message, dict):
            return failure(TurnOutcome.MALFORMED, "malformed", "response message is not an object")
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            return failure(TurnOutcome.MALFORMED, "malformed", "response content is not a string")
        reason = choice.get("finish_reason")
        if reason is None:
            return failure(TurnOutcome.INCOMPLETE, "incomplete", "response has no finish reason")
        if not isinstance(reason, str) or reason not in _OUTCOME_FOR_FINISH:
            return failure(TurnOutcome.MALFORMED, "malformed", f"unrecognized finish reason {reason!r}")
        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            raw_calls = []  # null means no tool calls
        if not isinstance(raw_calls, list):
            return failure(TurnOutcome.MALFORMED, "malformed", "tool_calls must be a list")
        calls = [_validated_tool_call(raw, index) for index, raw in enumerate(raw_calls)]
        if reason == sse.FINISH_STOP and calls or reason == sse.FINISH_TOOL_CALLS and not calls:
            return failure(TurnOutcome.MALFORMED, "malformed", "finish reason does not match the tool calls")
        usage, problem = usage_from(data.get("usage"))
        return ProviderTurn(
            outcome=_OUTCOME_FOR_FINISH[reason], content=content, tool_calls=calls,
            finish_reason=FinishReason(reason), usage=usage, usage_error=problem)
    except _MalformedResponse as e:
        return failure(TurnOutcome.MALFORMED, "malformed", str(e))


class LMStudioProvider:
    """LM Studio's OpenAI-compatible chat endpoint over httpx. Every request has explicit timeouts."""

    def __init__(self, url: str = config.LM_STUDIO_URL, *, transport: Optional[httpx.AsyncBaseTransport] = None):
        self._url = url
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=config.LM_STUDIO_CONNECT_TIMEOUT, read=config.LM_STUDIO_TIMEOUT,
                                  write=10.0, pool=5.0),
            transport=transport,
        )

    async def complete(self, messages, tools, options, on_text=None) -> ProviderTurn:
        payload = build_payload(messages, tools, options)
        try:
            if not options.stream:
                response = await self._client.post(self._url, json=payload)
                if response.status_code >= 400:
                    return failure(TurnOutcome.HTTP_ERROR, "http", f"HTTP {response.status_code}", response.status_code,
                                   _error_excerpt(response.text))
                try:
                    data = response.json()
                except ValueError as e:
                    return failure(TurnOutcome.MALFORMED, "malformed", f"response is not JSON: {e}")
                return turn_from_json(data)

            async with self._client.stream("POST", self._url, json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    return failure(TurnOutcome.HTTP_ERROR, "http", f"HTTP {response.status_code}", response.status_code,
                                   _error_excerpt(response.text))
                parser = sse.StreamParser(on_text)
                async for line in response.aiter_lines():
                    parser.feed(line)
                return turn_from_parser(parser)
        except sse.StreamError as e:
            return turn_from_stream_error(e)
        except httpx.TimeoutException as e:
            return failure(TurnOutcome.TIMEOUT, "timeout", str(e) or type(e).__name__)
        except httpx.TransportError as e:
            return failure(TurnOutcome.TRANSPORT, "transport", str(e) or type(e).__name__)

    async def aclose(self) -> None:
        await self._client.aclose()


def make_provider() -> ModelProvider:
    """The provider the runtime uses. Tests replace this factory; no test may reach a real endpoint."""
    return LMStudioProvider()
