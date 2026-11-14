"""Regression cases that were not covered by the other regression files.

1. Short history: the first request carries exactly one system message.
2. Loop-level summarizer failure: compaction falls back, the run completes, history stays valid.
3. Third identical call: answered with a 'Skipped' result and never dispatched.
4. Crash after dispatch: the unresolved call is on disk and no outcome is recorded.
"""

from tests import scripted_provider  # noqa: E402
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from src import config, dispatch, loop, prompts, session
from tests.conftest import protocol_violations

OBS_CHARS = 4000


class SimulatedCrash(BaseException):
    pass


def _call(call_id, path="."):
    return {"id": call_id, "name": "list_directory", "arguments": json.dumps({"path": path})}


def _run(monkeypatch, responses, dispatch_fn, *, summarizer_fails=False, session_id=None):
    """Drive agent_loop_async directly so the summarizer can be failed separately from the loop."""
    payloads = []

    def post(*args, **kwargs):
        body = copy.deepcopy(kwargs["json"])
        if "tools" not in body:  # compaction summarizer call (no tools key)
            payloads.append(("summary", body))
            if summarizer_fails:
                raise ConnectionError("summarizer unavailable")
            return _Json({"choices": [{"message": {"content": "Earlier work summarized."}}]})
        payloads.append(("loop", body))
        return responses.pop(0)

    monkeypatch.setattr(scripted_provider, "POST", post)
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)
    monkeypatch.setattr(dispatch, "dispatch_tool", dispatch_fn)
    async def _dispatch_async(tool_name, tool_input, mcp_client=None, event_loop=None, allowed_tools=None,
                                   authorization=None):
        return dispatch_fn(tool_name, tool_input)
    monkeypatch.setattr(dispatch, "dispatch_tool_async", _dispatch_async)
    ev = asyncio.new_event_loop()
    try:
        ev.run_until_complete(loop.agent_loop_async(
            "gap goal", SimpleNamespace(get_openai_tools=lambda: []), ev, session_id=session_id))
    finally:
        ev.close()
    return [body for kind, body in payloads if kind == "loop"], [b for k, b in payloads if k == "summary"]


class _Json:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_short_history_sends_exactly_one_system_message(monkeypatch, streamed_response):
    """Regression: the inline slice used to duplicate the system message on short histories."""
    monkeypatch.setattr(config, "COMPACT_THRESHOLD", 1.0)
    monkeypatch.setattr(config, "MAX_TOKENS", 1)  # every request is over budget
    responses = [streamed_response(finish_reason="stop", content="hi")]

    requests, _ = _run(monkeypatch, responses, lambda *a, **k: "unused")

    system_count = sum(1 for m in requests[0]["messages"] if m.get("role") == "system")
    assert system_count == 1


def _three_iteration_responses(streamed_response):
    return [
        streamed_response(finish_reason="tool_calls", tool_calls=[_call("call_0", "a")]),
        streamed_response(finish_reason="tool_calls", tool_calls=[_call("call_A", "b"), _call("call_B", "c")]),
        streamed_response(finish_reason="stop", content="all done"),
    ]


def _force_compaction_on_third_request(monkeypatch):
    system_tokens = len(prompts.SYSTEM_PROMPT) // 4
    monkeypatch.setattr(config, "COMPACT_THRESHOLD", 1.0)
    monkeypatch.setattr(config, "MAX_TOKENS", system_tokens + 2000)


def _numbered_dispatch():
    counter = {"n": 0}

    def dispatch_fn(name, args, *a, **k):
        counter["n"] += 1
        return f"result {counter['n']}: " + "x" * OBS_CHARS

    return dispatch_fn


def test_summarizer_failure_during_loop_falls_back_and_completes(monkeypatch, streamed_response):
    _force_compaction_on_third_request(monkeypatch)
    responses = _three_iteration_responses(streamed_response)

    requests, summaries = _run(monkeypatch, responses, _numbered_dispatch(), summarizer_fails=True)

    assert len(summaries) == 1, "compaction should have attempted to summarize once"
    assert len(requests) == 3
    final = requests[2]["messages"]
    assert not any("[Context Summary]" in (m.get("content") or "") for m in final)
    assert not protocol_violations(final), protocol_violations(final)


def test_third_identical_call_is_answered_as_skipped_and_not_dispatched(monkeypatch, streamed_response):
    dispatched = []

    def dispatch_fn(name, args, *a, **k):
        dispatched.append(args)
        return "same listing"

    responses = [
        streamed_response(finish_reason="tool_calls", tool_calls=[_call("call_1"), _call("call_2"), _call("call_3")]),
        streamed_response(finish_reason="stop", content="done"),
    ]
    requests, _ = _run(monkeypatch, responses, dispatch_fn)

    assert len(dispatched) == 2
    results = [m for m in requests[1]["messages"] if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in results] == ["call_1", "call_2", "call_3"]
    assert "Skipped" in results[2]["content"]
    assert not protocol_violations(requests[1]["messages"])


def test_crash_after_dispatch_persists_unresolved_call_without_outcome(monkeypatch, streamed_response, tmp_sessions):
    def dispatch_fn(name, args, *a, **k):
        raise SimulatedCrash("killed after effect")

    responses = [streamed_response(finish_reason="tool_calls", tool_calls=[_call("call_X", "out")])]
    with pytest.raises(SimulatedCrash):
        _run(monkeypatch, responses, dispatch_fn, session_id="sid-gap")

    saved = session.load_session("sid-gap")["messages"]
    assert saved[-1]["role"] == "assistant"
    assert saved[-1]["tool_calls"][0]["id"] == "call_X"
    assert not any(m.get("role") == "tool" for m in saved)
