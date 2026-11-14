"""B3.4 session isolation and authorization hardening.

Scripted model, real bounded local commands (uname, true, echo), fake MCP tools. No LLM, no network.
"""

import asyncio
import json

import pytest

from src import config, dispatch, envelope, loop, runs, session
from src.authorization import SessionAuthorization
from src.runtime import AgentRuntime, RuntimeShutdownRefused, SessionBusy
from tests import scripted_provider
from tests.test_session_ownership import _ScriptedPost, _saved, _stop, _tool_content, _tool_turn


class FakeMCP:
    """Bounded stand-in for an MCP client: one tool, no servers, no network."""

    def __init__(self):
        self.calls = []
        self.tools_cache = {}

    async def connect_all(self, config_data):
        return None

    async def close_all(self):
        return None

    def get_openai_tools(self):
        return []

    async def call_tool(self, tool_name, tool_input):
        self.calls.append(tool_name)
        return envelope.success(f"{tool_name} ran")


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def script(monkeypatch):
    def install(turns):
        post = _ScriptedPost(turns)
        monkeypatch.setattr(scripted_provider, "POST", post)
        return post
    return install


@pytest.fixture
def answers(monkeypatch):
    """Scripted answers to the interactive permission prompt. An unexpected prompt fails the test."""
    queue = []

    def set_answers(*keys):
        queue[:] = list(keys)

    def next_key(prompt, valid_keys):
        assert queue, "an unexpected permission prompt was shown"
        return queue.pop(0)

    monkeypatch.setattr(loop, "_get_single_key_input", next_key)
    return set_answers


@pytest.fixture
def rt():
    runtime = AgentRuntime()
    yield runtime
    try:
        runtime.shutdown()
    except RuntimeShutdownRefused:
        pass


def _run(rt, sess, goal):
    return rt.event_loop.run_until_complete(sess.run_goal(goal))


def _ask(call_id, command):
    return _tool_turn("run_command", {"command": command}, call_id)


# --- 1-3: search state ---------------------------------------------------------------------------

def test_search_state_is_session_local_and_the_global_flag_is_gone(rt, script, answers, tmp_sessions):
    assert not hasattr(config, "HAS_SEARCHED_CODEBASE")
    a = rt.open_session("search-a")
    b = rt.open_session("search-b")
    script([_ask("g1", "grep -r nothing ."), _stop("searched")])
    _run(rt, a, "search in A")
    assert a.authorization.has_searched is True
    assert b.authorization.has_searched is False

    script([_tool_turn("write_file", {"path": "new_in_b.txt", "content": "x"}, "w1"), _stop("wrote")])
    _run(rt, b, "create a file in B")
    assert "No codebase search" in _tool_content(_saved("search-b")["messages"], "w1")


def test_search_state_persists_across_goals_in_one_session(rt, script, answers, tmp_sessions):
    sess = rt.open_session("search-same")
    script([_ask("g1", "grep -r nothing ."), _stop("searched")])
    _run(rt, sess, "search")
    script([_tool_turn("write_file", {"path": "later.txt", "content": "x"}, "w2"), _stop("wrote")])
    _run(rt, sess, "create after searching")
    assert "No codebase search" not in _tool_content(_saved("search-same")["messages"], "w2")


def test_new_and_resumed_sessions_start_without_inherited_search(rt, script, answers, tmp_sessions):
    sess = rt.open_session("search-resume")
    script([_ask("g1", "grep -r nothing ."), _stop("searched")])
    _run(rt, sess, "search")
    sess.close()
    rt.shutdown()

    rt2 = AgentRuntime()
    try:
        resumed = rt2.resume_session("search-resume")
        assert resumed.authorization.has_searched is False, "resume must not fabricate a prior search"
        fresh = rt2.open_session("search-fresh")
        assert fresh.authorization.has_searched is False
    finally:
        rt2.shutdown()


def test_refused_search_is_not_counted():
    from src.tools import shell
    auth = SessionAuthorization(commands=[])  # grep is not on the CLI policy
    early, _, _ = shell.validate_command("grep -r x .", auth)
    assert early["requiresPermission"] is True and early["command"] == "grep -r x ."
    assert auth.has_searched is False, "a refused request must not count as a search"

    auth.grant("command_exact", "grep -r x .")
    early, spawn, _ = shell.validate_command("grep -r x .", auth)
    assert early is None and spawn == ["grep", "-r", "x", "."]
    assert auth.has_searched is True


# --- 3: authorization granularity -----------------------------------------------------------------

def test_interactive_grants_cannot_create_cli_base_name_policy():
    auth = SessionAuthorization()
    with pytest.raises(ValueError, match="CLI policy"):
        auth.grant("command", "curl")
    with pytest.raises(ValueError, match="CLI policy"):
        auth.grant_once("command", "curl")


def test_cli_allowlist_is_a_snapshot_and_is_broad_by_design(monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_TOOLS", {"uname"})
    auth = SessionAuthorization.from_config()
    assert auth.allows_command("uname", "uname -a") is True
    assert auth.allows_command("uname", "uname") is True
    assert auth.allows_command("true", "true") is False

    monkeypatch.setattr(config, "ALLOWED_TOOLS", {"uname", "true"})
    assert auth.allows_command("true", "true") is False, "a later config change must not widen this session"


def test_cli_allowlist_narrowing_is_not_widened_by_interactive_approval(rt, script, answers, tmp_sessions, monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_TOOLS", {"echo"})
    sess = rt.open_session("narrow")
    answers("2")  # approve "true" for this session
    script([_ask("n1", "true"), _stop("ran")])
    _run(rt, sess, "approve true")
    assert "true" not in sess.authorization.snapshot()["command"]
    assert "true" in sess.authorization.snapshot()["command_exact"]


# --- 4-6: shell approval matching and one-time grants --------------------------------------------

def test_one_time_approval_cannot_authorize_a_second_invocation(rt, script, answers, tmp_sessions):
    sess = rt.open_session("once-twice")
    answers("1")
    script([_ask("o1", "uname -s"), _stop("first")])
    _run(rt, sess, "first use")
    assert "uname -s" not in sess.authorization.snapshot()["command_exact"]
    assert sess.authorization.snapshot()["once"] == []

    answers("3")  # the second request is asked again, and denied
    script([_ask("o2", "uname -s"), _stop("second")])
    _run(rt, sess, "same command again")
    assert "denied" in _tool_content(_saved("once-twice")["messages"], "o2").lower()


def test_same_executable_with_different_arguments_is_not_implicitly_approved(rt, script, answers, tmp_sessions):
    sess = rt.open_session("args")
    answers("2")
    script([_ask("a1", "uname -s"), _stop("ok")])
    _run(rt, sess, "approve uname -s")

    answers("3")
    script([_ask("a2", "uname -m"), _stop("second")])
    _run(rt, sess, "uname -m is not approved")
    assert "denied" in _tool_content(_saved("args")["messages"], "a2").lower()


def test_session_approval_matching_is_exact_text_after_trimming(rt, script, answers, tmp_sessions):
    sess = rt.open_session("exact")
    auth = sess.authorization
    auth.grant("command_exact", "uname -s")
    assert auth.allows_command("uname", "uname -s") is True
    assert auth.allows_command("uname", "uname  -s") is False, "whitespace inside the command is significant"
    assert auth.allows_command("uname", "uname -s -a") is False
    assert auth.allows_command("uname", "uname -m") is False
    sess.close()


def test_curl_approval_for_one_set_of_arguments_does_not_approve_another(rt, script, answers, tmp_sessions):
    auth = SessionAuthorization()
    auth.grant("command_exact", "curl -I https://example.invalid/a")
    assert auth.allows_command("curl", "curl -I https://example.invalid/a") is True
    assert auth.allows_command("curl", "curl https://example.invalid/b") is False


def test_shell_operator_approval_is_exact_and_one_time(rt, script, answers, tmp_sessions):
    sess = rt.open_session("pipes")
    answers("1")
    script([_ask("p1", "echo hi | cat"), _stop("ok")])
    _run(rt, sess, "pipe once")
    assert "hi" in _tool_content(_saved("pipes")["messages"], "p1")
    assert sess.authorization.snapshot()["shell"] == []

    answers("3")
    script([_ask("p2", "echo hi | cat"), _stop("again")])
    _run(rt, sess, "same pipe again")
    assert "denied" in _tool_content(_saved("pipes")["messages"], "p2").lower()


# --- 7: session A grant is denied in session B -----------------------------------------------------

def test_session_a_grant_is_denied_in_session_b(rt, script, answers, tmp_sessions):
    a = rt.open_session("grant-a")
    b = rt.open_session("grant-b")
    answers("2")
    script([_ask("x1", "uname -s"), _stop("ok")])
    _run(rt, a, "approve in A")

    answers("3")
    script([_ask("x2", "uname -s"), _stop("try in B")])
    _run(rt, b, "same command in B")
    assert "denied" in _tool_content(_saved("grant-b")["messages"], "x2").lower()
    assert "uname -s" in a.authorization.snapshot()["command_exact"]
    assert "uname -s" not in b.authorization.snapshot()["command_exact"]


# --- 8-9: one-time grant and error paths -------------------------------------------------------------

def test_one_time_grant_is_discarded_when_the_approved_attempt_fails_validation(rt, script, answers, tmp_sessions, monkeypatch):
    sess = rt.open_session("validation-fail")
    real = dispatch.dispatch_tool_async
    seen = {"n": 0}

    async def first_permission_then_validation_error(*args, **kwargs):
        seen["n"] += 1
        if seen["n"] == 2:  # the approved re-dispatch fails validation before reaching the grant check
            return envelope.error_validation("simulated validation failure", fixable=True)
        return await real(*args, **kwargs)

    monkeypatch.setattr(loop.dispatch, "dispatch_tool_async", first_permission_then_validation_error)
    answers("1")
    script([_ask("v1", "uname -s"), _stop("reported")])
    _run(rt, sess, "approved attempt fails validation")
    assert sess.authorization.snapshot()["once"] == [], "a failed approved attempt must not leave a grant"

    answers("3")
    script([_ask("v2", "uname -s"), _stop("again")])
    _run(rt, sess, "the next request is asked again")
    assert "denied" in _tool_content(_saved("validation-fail")["messages"], "v2").lower()


def test_one_time_grant_is_discarded_when_the_approved_attempt_is_cancelled(rt, script, answers, tmp_sessions, monkeypatch):
    sess = rt.open_session("cancel-once")
    real = dispatch.dispatch_tool_async
    seen = {"n": 0}

    async def first_permission_then_cancel(*args, **kwargs):
        seen["n"] += 1
        if seen["n"] == 2:
            raise asyncio.CancelledError()
        return await real(*args, **kwargs)

    monkeypatch.setattr(loop.dispatch, "dispatch_tool_async", first_permission_then_cancel)
    answers("1")
    script([_ask("c1", "uname -s"), _stop("never")])
    with pytest.raises(asyncio.CancelledError):
        _run(rt, sess, "cancelled while approved")
    assert sess.authorization.snapshot()["once"] == []
    assert sess.closed is False

    answers("3")
    script([_ask("c2", "uname -s"), _stop("after")])
    _run(rt, sess, "a later request is asked again")
    assert "denied" in _tool_content(_saved("cancel-once")["messages"], "c2").lower()


def test_one_time_grant_is_discarded_when_the_approved_attempt_raises(rt, script, answers, tmp_sessions, monkeypatch):
    sess = rt.open_session("raise-once")
    real = dispatch.dispatch_tool_async
    seen = {"n": 0}

    async def first_permission_then_raise(*args, **kwargs):
        seen["n"] += 1
        if seen["n"] == 2:
            raise RuntimeError("injected dispatch failure")
        return await real(*args, **kwargs)

    monkeypatch.setattr(loop.dispatch, "dispatch_tool_async", first_permission_then_raise)
    answers("1")
    script([_ask("r1", "uname -s"), _stop("never")])
    with pytest.raises(RuntimeError):
        _run(rt, sess, "raised while approved")
    assert sess.authorization.snapshot()["once"] == []


# --- 10: batched calls with mixed permissions ----------------------------------------------------------

def test_batched_calls_with_mixed_permissions_are_decided_independently(rt, script, answers, tmp_sessions):
    """An allowed call that precedes a denied one still runs. A user denial skips the calls after it in the same
    turn, which is the existing batch rule and is kept."""
    sess = rt.open_session("batch")
    answers("3")
    script([make_streamed_response_batch([
        {"id": "b1", "name": "run_command", "arguments": json.dumps({"command": "echo batch-ok"})},
        {"id": "b2", "name": "run_command", "arguments": json.dumps({"command": "uname -s"})},
        {"id": "b3", "name": "run_command", "arguments": json.dumps({"command": "echo after-denial"})},
    ]), _stop("done")])
    _run(rt, sess, "mixed batch")
    messages = _saved("batch")["messages"]
    assert "batch-ok" in _tool_content(messages, "b1")
    assert "denied" in _tool_content(messages, "b2").lower()
    skipped = _tool_content(messages, "b3")
    assert "Skipped" in skipped and "after-denial" not in skipped
    assert [m["tool_call_id"] for m in messages if m.get("role") == "tool"] == ["b1", "b2", "b3"]


def make_streamed_response_batch(calls):
    from tests.conftest import make_streamed_response
    return make_streamed_response(finish_reason="tool_calls", tool_calls=calls)


# --- 11: MCP authorization isolation -----------------------------------------------------------------

def test_mcp_grant_is_session_local_and_one_time_is_not_persistent(rt, script, answers, tmp_sessions):
    fake_a, fake_b = FakeMCP(), FakeMCP()
    a = rt.open_session("mcp-a", mcp_client=fake_a)
    b = rt.open_session("mcp-b", mcp_client=fake_b)

    answers("2")
    script([_tool_turn("mcp_remote_write", {"path": "x"}, "m1"), _stop("ok")])
    _run(rt, a, "approve the MCP tool for A")
    assert fake_a.calls == ["mcp_remote_write"]

    script([_tool_turn("mcp_remote_write", {"path": "y"}, "m2"), _stop("again")])
    _run(rt, a, "A uses its session approval, without a prompt")
    assert fake_a.calls == ["mcp_remote_write", "mcp_remote_write"]

    answers("3")
    script([_tool_turn("mcp_remote_write", {"path": "z"}, "m3"), _stop("b")])
    _run(rt, b, "B has no grant")
    assert fake_b.calls == []
    assert "denied" in _tool_content(_saved("mcp-b")["messages"], "m3").lower()

    answers("1")
    script([_tool_turn("mcp_other", {}, "m4"), _stop("once")])
    _run(rt, b, "one-time approval for another tool")
    assert fake_b.calls == ["mcp_other"]
    assert b.authorization.snapshot()["once"] == []


# --- 12: session isolation of a shared runtime -----------------------------------------------------------

def test_refused_shutdown_and_retry_preserve_grants_and_isolation(rt, script, answers, tmp_sessions):
    a = rt.open_session("iso-a")
    b = rt.open_session("iso-b")
    answers("2")
    script([_ask("i1", "uname -s"), _stop("ok")])
    _run(rt, a, "grant in A")

    script([_tool_turn("start_job", {"command": 'python3 -u -c "import time; time.sleep(30)"', "lifetime_seconds": 60}, "i2"),
            _stop("started")])
    task = rt.event_loop.create_task(b.run_goal("busy in B"))
    for _ in range(100):
        rt.event_loop.run_until_complete(asyncio.sleep(0))
        if b.busy:
            break
    with pytest.raises(RuntimeShutdownRefused):
        rt.shutdown()
    assert "uname -s" in a.authorization.snapshot()["command_exact"], "a refused shutdown erases nothing"
    assert "uname -s" not in b.authorization.snapshot()["command_exact"]
    rt.event_loop.run_until_complete(task)

    assert rt.shutdown() == []


# --- 13-14: close cleanup -------------------------------------------------------------------------------

def test_session_close_discards_its_grants_and_search_state(rt, script, answers, tmp_sessions):
    sess = rt.open_session("closing-state")
    answers("2")
    script([_ask("k1", "uname -s"), _stop("ok")])
    _run(rt, sess, "grant then search")
    script([_ask("k2", "grep -r nothing ."), _stop("searched")])
    _run(rt, sess, "search")
    assert sess.authorization.has_searched is True
    assert sess.close() == []
    assert sess not in rt.sessions()

    fresh = rt.resume_session("closing-state")
    assert fresh.authorization is not sess.authorization
    assert fresh.authorization.has_searched is False
    assert "uname -s" not in fresh.authorization.snapshot()["command_exact"], "grants are not persisted"
    fresh.close()


def test_closed_session_cannot_be_used_to_start_a_goal(rt, script, tmp_sessions):
    sess = rt.open_session("dead")
    sess.close()
    from src.runtime import SessionClosed
    with pytest.raises(SessionClosed):
        asyncio.run(sess.run_goal("after close"))


def test_one_time_grant_is_consumed_by_its_first_check_even_without_the_loop_cleanup():
    """The authorization object itself must refuse a second use, not rely on the loop to clean up."""
    auth = SessionAuthorization()
    auth.grant_once("command_exact", "uname -s")
    assert auth.allows_command("uname", "uname -s") is True
    assert auth.allows_command("uname", "uname -s") is False

    auth.grant_once("shell", "echo a | cat")
    assert auth.allows("shell", "echo a | cat") is True
    assert auth.allows("shell", "echo a | cat") is False

    auth.grant_once("mcp", "mcp_tool")
    assert auth.allows("mcp", "mcp_tool") is True
    assert auth.allows("mcp", "mcp_tool") is False
