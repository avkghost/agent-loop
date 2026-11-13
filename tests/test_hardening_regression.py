"""Hardening regressions: background contract, persistence (M5, M9), dispatch exceptions, summarization.

All LLM, tool and process interaction is mocked. Nothing here starts a real process, reaches a
network, or touches the repository outside pytest's tmp_path.
"""

import asyncio
import copy
import json
import re
from types import SimpleNamespace

import pytest
import requests

from src import config, dispatch, envelope, loop, prompts, session
from src.tools import shell
from tests.conftest import protocol_violations


class SimulatedCrash(BaseException):
    """Stands in for a process kill. BaseException so no application handler catches it."""


class ProcessControl(BaseException):
    """Stands in for cancellation or a non-tool control-flow signal."""


class JSONResponse:
    def __init__(self, payload=None, error=None):
        self._payload = payload
        self._error = error

    def raise_for_status(self):
        return None

    def json(self):
        if self._error is not None:
            raise self._error
        return self._payload


class CrashingResponse:
    """Accepts the request, then dies while the streamed reply is being read."""

    def raise_for_status(self):
        return None

    def iter_lines(self):
        raise SimulatedCrash("killed while waiting for the model reply")


def _call(call_id, name="list_directory", path="."):
    return {"id": call_id, "name": name, "arguments": json.dumps({"path": path})}


def _drive(monkeypatch, responses, dispatch_fn, *, summarizer=None, session_id=None,
           resume_messages=None, **config_overrides):
    """Run agent_loop_async with scripted model replies. Summary calls are routed separately."""
    loop_posts, summary_posts = [], []

    def post(*args, **kwargs):
        body = copy.deepcopy(kwargs["json"])
        if "tools" not in body:  # compaction summarizer request (no tools key)
            summary_posts.append(body)
            if summarizer is None:
                return JSONResponse({"choices": [{"message": {"content": "Earlier work summarized."}}]})
            return summarizer(body)
        loop_posts.append(body)
        return responses.pop(0)

    monkeypatch.setattr(loop.requests, "post", post)
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)
    monkeypatch.setattr(dispatch, "dispatch_tool", dispatch_fn)
    async def _dispatch_async(tool_name, tool_input, mcp_client=None, event_loop=None, allowed_tools=None):
        return dispatch_fn(tool_name, tool_input)
    monkeypatch.setattr(dispatch, "dispatch_tool_async", _dispatch_async)
    for name, value in config_overrides.items():
        monkeypatch.setattr(config, name, value)

    event_loop = asyncio.new_event_loop()
    try:
        event_loop.run_until_complete(loop.agent_loop_async(
            "hardening goal", SimpleNamespace(get_openai_tools=lambda: []), event_loop,
            session_id=session_id, resume_messages=resume_messages))
    finally:
        event_loop.close()
    return SimpleNamespace(requests=loop_posts, summaries=summary_posts)


def _tool_messages(messages):
    return [m for m in messages if m.get("role") == "tool"]


# ---------------------------------------------------------------------------
# A. Background execution contract
# ---------------------------------------------------------------------------

def test_agent_prompt_advertises_managed_jobs_and_no_shell_job_control():
    # B2 changed this contract on purpose: managed jobs are advertised through their tools.
    # Shell-level job control (the `jobs` builtin, `kill %1`) is still not offered.
    assert "`jobs`" not in prompts.SYSTEM_PROMPT
    assert "kill %1" not in prompts.SYSTEM_PROMPT
    for tool in ("start_job", "status_job", "collect_job", "terminate_job"):
        assert tool in prompts.SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# B1 (M5). Atomic session writes, observed through the filesystem
# ---------------------------------------------------------------------------

PREVIOUS = [{"role": "user", "content": "previous state"}]
NEXT = [{"role": "user", "content": "new state"}]


def test_interrupted_overwrite_keeps_the_previous_session_readable(tmp_sessions, monkeypatch):
    """Previous valid file is untouched when the new write dies after writing a prefix."""
    session.save_session("sid", "goal", PREVIOUS)
    before = (tmp_sessions / "sid.json").read_bytes()

    def partial_dump(obj, fp, indent=None):
        fp.write('{"session_id": "sid", "messages": [')
        raise OSError("injected write interruption")

    monkeypatch.setattr(session.json, "dump", partial_dump)
    with pytest.raises(OSError, match="injected"):
        session.save_session("sid", "goal", NEXT)

    assert (tmp_sessions / "sid.json").read_bytes() == before
    assert session.load_session("sid")["messages"] == PREVIOUS
    assert not (tmp_sessions / "sid.json.tmp").exists(), "temporary file must be cleaned up"


def test_failed_replace_keeps_the_previous_session_and_removes_the_temp_file(tmp_sessions, monkeypatch):
    session.save_session("sid", "goal", PREVIOUS)

    def failing_replace(src, dst):
        raise OSError("injected failure before replacement")

    monkeypatch.setattr(session.os, "replace", failing_replace)
    with pytest.raises(OSError, match="before replacement"):
        session.save_session("sid", "goal", NEXT)

    assert session.load_session("sid")["messages"] == PREVIOUS
    assert not (tmp_sessions / "sid.json.tmp").exists()


def test_successful_overwrite_replaces_the_previous_session(tmp_sessions):
    session.save_session("sid", "goal", PREVIOUS)
    session.save_session("sid", "goal", NEXT)

    assert session.load_session("sid")["messages"] == NEXT
    assert not (tmp_sessions / "sid.json.tmp").exists()


# ---------------------------------------------------------------------------
# B2 (M9). A completed tool result is saved before the next model request
# ---------------------------------------------------------------------------

def test_completed_result_survives_interruption_of_the_next_request_and_is_not_redispatched(
        monkeypatch, tmp_sessions):
    dispatched = []

    def dispatch_fn(name, args, *a, **k):
        dispatched.append(name)
        return "listing"

    from tests.conftest import make_streamed_response

    responses = [
        make_streamed_response(finish_reason="tool_calls", tool_calls=[_call("call_1")]),
        CrashingResponse(),  # the second request dies while its reply is being read
    ]
    with pytest.raises(SimulatedCrash):
        _drive(monkeypatch, responses, dispatch_fn, session_id="sid-m9")

    saved = session.load_session("sid-m9")["messages"]
    results = _tool_messages(saved)
    assert [m["tool_call_id"] for m in results] == ["call_1"], "completed result missing from disk"
    assert results[0]["content"] == "listing"
    assert dispatched == ["list_directory"]

    # Resume: the completed call must not be dispatched again
    resumed = _drive(monkeypatch, [make_streamed_response(finish_reason="stop", content="done")],
                     dispatch_fn, session_id="sid-m9", resume_messages=saved)
    assert dispatched == ["list_directory"], "resume re-dispatched a completed call"
    assert not protocol_violations(resumed.requests[0]["messages"])


# ---------------------------------------------------------------------------
# C. Tool dispatch exceptions
# ---------------------------------------------------------------------------

def test_middle_call_failure_still_pairs_every_call_and_runs_the_rest(monkeypatch):
    from tests.conftest import make_streamed_response

    def dispatch_fn(name, args, *a, **k):
        if args == {"path": "b"}:
            raise dispatch.ToolExecutionError("middle call failed")
        return f"ok:{args['path']}"

    responses = [
        make_streamed_response(finish_reason="tool_calls",
                               tool_calls=[_call("call_a", path="a"), _call("call_b", path="b"), _call("call_c", path="c")]),
        make_streamed_response(finish_reason="stop", content="done"),
    ]
    run = _drive(monkeypatch, responses, dispatch_fn)

    results = _tool_messages(run.requests[1]["messages"])
    assert [m["tool_call_id"] for m in results] == ["call_a", "call_b", "call_c"]
    assert results[0]["content"] == "ok:a"
    assert '"errorCategory": "unknown_outcome"' in results[1]["content"]
    assert '"isRetryable": false' in results[1]["content"]
    assert results[2]["content"] == "ok:c"
    assert not protocol_violations(run.requests[1]["messages"])


def test_base_exception_from_dispatch_propagates_and_leaves_the_call_pending(monkeypatch, tmp_sessions):
    from tests.conftest import make_streamed_response

    def dispatch_fn(name, args, *a, **k):
        raise ProcessControl("cancelled")

    responses = [make_streamed_response(finish_reason="tool_calls", tool_calls=[_call("call_p", "write_file")])]
    with pytest.raises(ProcessControl):
        _drive(monkeypatch, responses, dispatch_fn, session_id="sid-cancel")

    saved = session.load_session("sid-cancel")["messages"]
    assert saved[-1]["tool_calls"][0]["id"] == "call_p"
    assert not any(m.get("role") == "tool" for m in saved), "no outcome may be invented for a cancelled call"


def test_keyboard_interrupt_during_dispatch_is_not_swallowed(monkeypatch):
    from tests.conftest import make_streamed_response

    def dispatch_fn(name, args, *a, **k):
        raise KeyboardInterrupt()

    responses = [make_streamed_response(finish_reason="tool_calls", tool_calls=[_call("call_k")])]
    with pytest.raises(KeyboardInterrupt):
        _drive(monkeypatch, responses, dispatch_fn)


def test_interrupted_call_resumes_as_interrupted_without_redispatch(monkeypatch, tmp_sessions):
    from tests.conftest import make_streamed_response

    dispatched = []

    def dispatch_fn(name, args, *a, **k):
        dispatched.append(name)
        raise ProcessControl("cancelled")

    with pytest.raises(ProcessControl):
        _drive(monkeypatch, [make_streamed_response(finish_reason="tool_calls", tool_calls=[_call("call_w", "write_file")])],
               dispatch_fn, session_id="sid-resume-c")
    saved = session.load_session("sid-resume-c")["messages"]

    run = _drive(monkeypatch, [make_streamed_response(finish_reason="stop", content="ok")],
                 dispatch_fn, session_id="sid-resume-c", resume_messages=saved)

    assert dispatched == ["write_file"], "interrupted call must not run again on resume"
    results = _tool_messages(run.requests[0]["messages"])
    assert [m["tool_call_id"] for m in results] == ["call_w"]
    assert "Interrupted" in results[0]["content"]
    assert not protocol_violations(run.requests[0]["messages"])


# ---------------------------------------------------------------------------
# D. Summarization failures, oversized output, repeated compaction
# ---------------------------------------------------------------------------

def _system_tokens():
    return len(prompts.SYSTEM_PROMPT) // 4


def _forced_compaction_config():
    """Threshold lies between the second and third request's estimated size (see test_compaction_regression)."""
    return dict(COMPACT_THRESHOLD=1.0, MAX_TOKENS=_system_tokens() + 2000)


def _three_iteration_responses():
    from tests.conftest import make_streamed_response

    return [
        make_streamed_response(finish_reason="tool_calls", tool_calls=[_call("call_0", path="a")]),
        make_streamed_response(finish_reason="tool_calls", tool_calls=[_call("call_A", path="b"), _call("call_B", path="c")]),
        make_streamed_response(finish_reason="stop", content="all done"),
    ]


def _numbered_dispatch():
    counter = {"n": 0}

    def dispatch_fn(name, args, *a, **k):
        counter["n"] += 1
        return f"result {counter['n']}: " + "x" * 4000

    return dispatch_fn


def _failing_summarizer(exc):
    def summarizer(body):
        raise exc
    return summarizer


def _invalid_payload_summarizer(body):
    return JSONResponse({"unexpected": True})


def _empty_summarizer(body):
    return JSONResponse({"choices": [{"message": {"content": ""}}]})


def _oversized_summarizer(body):
    limit_chars = int(config.MAX_TOKENS * 0.2 * 4)
    return JSONResponse({"choices": [{"message": {"content": "x" * (limit_chars + 400)}}]})


@pytest.mark.parametrize("summarizer", [
    _failing_summarizer(requests.exceptions.Timeout("summarizer timed out")),
    _failing_summarizer(requests.exceptions.ConnectionError("transport down")),
    _invalid_payload_summarizer,
    _empty_summarizer,
    _oversized_summarizer,
], ids=["timeout", "transport", "invalid-payload", "empty", "oversized"])
def test_summarizer_failure_falls_back_to_protocol_valid_truncation(monkeypatch, summarizer):
    dispatch_fn = _numbered_dispatch()
    calls = {"n": 0}

    def counting(dispatch_fn_):
        def wrapper(*a, **k):
            calls["n"] += 1
            return dispatch_fn_(*a, **k)
        return wrapper

    run = _drive(monkeypatch, _three_iteration_responses(), counting(dispatch_fn),
                 summarizer=summarizer, **_forced_compaction_config())

    assert len(run.summaries) == 1, "compaction should have attempted exactly one summary"
    assert len(run.requests) == 3, "the run must continue after the failed summary"
    assert calls["n"] == 3, "a failed summary must not cause any tool to be dispatched again"
    final = run.requests[2]["messages"]
    assert sum(1 for m in final if m.get("role") == "system") == 1
    assert not any("[Context Summary]" in (m.get("content") or "") for m in final)
    assert not protocol_violations(final), protocol_violations(final)


def test_summary_within_budget_is_used(monkeypatch):
    run = _drive(monkeypatch, _three_iteration_responses(), _numbered_dispatch(),
                 **_forced_compaction_config())

    final = run.requests[2]["messages"]
    assert final[1] == {"role": "user", "content": "Objective: hardening goal\n\n[Context Summary]\nEarlier work summarized."}
    assert not protocol_violations(final), protocol_violations(final)


def test_summary_does_not_create_adjacent_user_messages(monkeypatch):
    from src import compaction

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "latest request"},
        {"role": "assistant", "content": "latest answer"},
        {"role": "user", "content": "final request"},
        {"role": "assistant", "content": "final answer"},
    ]

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "summary"}}]}

    monkeypatch.setattr(compaction.requests, "post", lambda *args, **kwargs: Response())
    import asyncio
    event_loop = asyncio.new_event_loop()
    try:
        result = event_loop.run_until_complete(compaction.compact_messages(messages, event_loop))
    finally:
        event_loop.close()

    roles = [message["role"] for message in result]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    assert result[1]["content"].startswith("old\n\n[Context Summary]\nsummary")
    assert "latest request" in result[1]["content"]


def test_repeated_compaction_keeps_every_request_valid_and_terminates(monkeypatch):
    from tests.conftest import make_streamed_response

    responses = []
    for i in range(6):
        responses.append(make_streamed_response(finish_reason="tool_calls", tool_calls=[_call(f"call_{i}", path=str(i))]))
    responses.append(make_streamed_response(finish_reason="stop", content="done"))

    run = _drive(monkeypatch, responses, _numbered_dispatch(), **_forced_compaction_config())

    assert len(run.requests) == 7, "the run must reach the final stop reply"
    assert len(run.summaries) <= len(run.requests), "at most one summary per iteration: no compaction loop"
    for index, body in enumerate(run.requests):
        problems = protocol_violations(body["messages"])
        assert not problems, f"request {index}: {problems}"
