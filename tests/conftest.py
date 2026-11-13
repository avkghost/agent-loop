import asyncio
import copy
import inspect
import json
from types import SimpleNamespace

import pytest

from src import envelope, loop


class FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        return None

    def iter_lines(self):
        return iter(self._lines)


def make_streamed_response(finish_reason="stop", content=None, tool_calls=None):
    delta = {}
    if content:
        delta["content"] = content
    if tool_calls:
        delta["tool_calls"] = [
            {
                "index": index,
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["arguments"],
                },
            }
            for index, call in enumerate(tool_calls)
        ]

    chunks = [
        {
            "choices": [
                {
                    "delta": delta,
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ]
        },
    ]
    lines = [f"data: {json.dumps(chunk)}" for chunk in chunks]
    lines.append("data: [DONE]")
    return FakeResponse(lines)


@pytest.fixture
def streamed_response():
    return make_streamed_response


class ScriptedPost:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def __call__(self, *args, **kwargs):
        self.payloads.append(copy.deepcopy(kwargs["json"]))
        if not self.responses:
            raise AssertionError("Unexpected additional model request")
        return self.responses.pop(0)


@pytest.fixture
def fake_post(monkeypatch):
    def install(responses):
        post = ScriptedPost(responses)
        monkeypatch.setattr(loop.requests, "post", post)
        return post

    return install


def _async_dispatch(sync_fn):
    """Awaitable stand-in for dispatch_tool_async that routes every local tool to a sync fake."""
    async def dispatch_async(tool_name, tool_input, mcp_client=None, event_loop=None, allowed_tools=None):
        return sync_fn(tool_name, tool_input)
    return dispatch_async


@pytest.fixture
def run_loop(monkeypatch):
    def run(
        responses,
        dispatch_fn=None,
        goal="test goal",
        session_id=None,
        resume_messages=None,
        resume_objective=None,
        resume_summary=None,
        resume_execution=None,
        **config_overrides,
    ):
        post = ScriptedPost(responses)
        monkeypatch.setattr(loop.requests, "post", post)
        monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)

        calls = []

        if dispatch_fn is None:
            def dispatch_fn(tool_name, tool_input, *args, **kwargs):
                return envelope.success("tool result")

        def recording_dispatch(tool_name, tool_input, *args, **kwargs):
            calls.append((tool_name, copy.deepcopy(tool_input)))
            return dispatch_fn(tool_name, tool_input, *args, **kwargs)

        monkeypatch.setattr(loop.dispatch, "dispatch_tool", recording_dispatch)
        monkeypatch.setattr(loop.dispatch, "dispatch_tool_async", _async_dispatch(recording_dispatch))
        run.last_post = post
        run.last_calls = calls
        for name, value in config_overrides.items():
            monkeypatch.setattr(loop.config, name, value)

        event_loop = asyncio.new_event_loop()
        try:
            parameters = inspect.signature(loop.agent_loop_async).parameters
            extra = {}
            if "session_id" in parameters:
                extra["session_id"] = session_id
                extra["resume_messages"] = resume_messages
                extra["resume_objective"] = resume_objective
                extra["resume_summary"] = resume_summary
                extra["resume_execution"] = resume_execution
            event_loop.run_until_complete(
                loop.agent_loop_async(
                    goal,
                    SimpleNamespace(get_openai_tools=lambda: []),
                    event_loop,
                    **extra,
                )
            )
        finally:
            event_loop.close()

        return post, calls

    return run


@pytest.fixture
def tmp_sessions(monkeypatch, tmp_path):
    from src import session

    monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path)
    return tmp_path


def protocol_violations(messages):
    """Tool-calling protocol check: each result follows its declaration and nothing is left unanswered."""
    problems, declared, open_ids = [], set(), []
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant":
            if open_ids:
                problems.append(f"msg {i}: assistant follows unanswered calls {open_ids}")
            open_ids = []
            for tc in m.get("tool_calls") or []:
                declared.add(tc["id"])
                open_ids.append(tc["id"])
        elif role == "tool":
            tid = m.get("tool_call_id")
            if tid not in declared:
                problems.append(f"msg {i}: tool result {tid!r} has no declaration")
            elif tid in open_ids:
                open_ids.remove(tid)
            else:
                problems.append(f"msg {i}: tool result {tid!r} is outside its call group")
        elif open_ids:
            problems.append(f"msg {i}: {role} message while calls {open_ids} are unanswered")
            open_ids = []
    if open_ids:
        problems.append(f"end: unanswered calls {open_ids}")
    return problems


def pytest_configure(config):
    config.addinivalue_line("markers", "real_process: runs harmless real subprocesses (see test_shell_hardening)")
