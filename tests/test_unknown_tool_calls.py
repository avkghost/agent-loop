"""A tool call naming a tool that was not offered for the run is refused before its arguments are read.

It is never dispatched, never reaches a permission gate, and is answered exactly once. Valid names keep their existing
argument handling. Offline only: the model is scripted, and dispatch is recorded rather than run.
"""

import json

import pytest

from src import loop, session
from tests.conftest import make_streamed_response

UNKNOWN_BARE = 'list_directory(".")'
UNKNOWN_QUOTED = 'list_directory "."'


def _calls_turn(calls):
    return make_streamed_response(finish_reason="tool_calls", tool_calls=[
        {"id": cid, "name": name, "arguments": arguments} for cid, name, arguments in calls])


def _stop():
    return make_streamed_response(finish_reason="stop", content="done")


def _text(reply):
    """The observation text the model receives (the reply itself is a JSON envelope)."""
    return json.loads(reply["content"])["content"]


def _tool_messages(post):
    return [m for m in post.payloads[1]["messages"] if m.get("role") == "tool"]


@pytest.fixture
def no_permission_gate(monkeypatch):
    """Any permission prompt for a call in these tests is a failure."""
    def refuse(*args, **kwargs):
        raise AssertionError("a permission prompt was requested")
    monkeypatch.setattr(loop, "_get_single_key_input", refuse)
    return refuse


# --- unknown names are refused before their arguments are read ---------------------------------------------------

def test_unknown_name_with_empty_arguments_is_refused_with_its_exact_name(run_loop, tmp_sessions, no_permission_gate):
    post, dispatched = run_loop([_calls_turn([("c1", UNKNOWN_BARE, "")]), _stop()], session_id="unknown-empty")
    assert dispatched == []
    [reply] = _tool_messages(post)
    assert reply["tool_call_id"] == "c1"
    assert f"Unknown tool {UNKNOWN_BARE!r}" in _text(reply)
    assert "Malformed" not in _text(reply), "the name, not the arguments, is the reported fault"


def test_unknown_name_containing_call_syntax_is_refused_with_its_exact_text(run_loop, tmp_sessions, no_permission_gate):
    post, dispatched = run_loop([_calls_turn([("c1", UNKNOWN_QUOTED, "{}")]), _stop()], session_id="unknown-quoted")
    assert dispatched == []
    [reply] = _tool_messages(post)
    assert f"Unknown tool {UNKNOWN_QUOTED!r}" in _text(reply)


def test_unknown_name_is_journalled_failed_and_the_history_stays_valid(run_loop, tmp_sessions, no_permission_gate):
    run_loop([_calls_turn([("c1", UNKNOWN_BARE, "")]), _stop()], session_id="unknown-journal")
    saved = session.load_session("unknown-journal")
    session.validate_session_data(saved)
    [record] = [r for r in saved["execution"] if r["tool_call_id"] == "c1"]
    assert record["status"] == "failed" and record["error_category"] == "validation"
    assert record["tool_name"] == UNKNOWN_BARE, "the journal keeps the exact received name"


def test_a_valid_call_followed_by_an_unknown_call_in_one_turn(run_loop, tmp_sessions, no_permission_gate):
    post, dispatched = run_loop([
        _calls_turn([("c1", "list_directory", json.dumps({"path": "."})), ("c2", UNKNOWN_BARE, "")]),
        _stop(),
    ], session_id="mixed-turn")
    assert [name for name, _ in dispatched] == ["list_directory"], "only the offered call is dispatched"
    replies = {m["tool_call_id"]: m["content"] for m in _tool_messages(post)}
    assert set(replies) == {"c1", "c2"}
    assert "Unknown tool" in replies["c2"] and "Unknown tool" not in replies["c1"]
    saved = session.load_session("mixed-turn")
    session.validate_session_data(saved)
    statuses = {r["tool_call_id"]: r["status"] for r in saved["execution"]}
    assert statuses == {"c1": "completed", "c2": "failed"}


def test_every_declared_call_id_receives_exactly_one_reply(run_loop, tmp_sessions, no_permission_gate):
    calls = [("a", "list_directory", json.dumps({"path": "."})), ("b", UNKNOWN_BARE, ""),
             ("c", "list_directory", "{}"), ("d", UNKNOWN_QUOTED, "{}")]
    post, dispatched = run_loop([_calls_turn(calls), _stop()], session_id="one-reply")
    replies = [m["tool_call_id"] for m in _tool_messages(post)]
    assert sorted(replies) == ["a", "b", "c", "d"], "each declared id is answered exactly once"
    assert len(dispatched) == 2
    session.validate_session_data(session.load_session("one-reply"))


# --- unknown names never reach dispatch or permission ------------------------------------------------------------

def test_unknown_mcp_style_name_never_reaches_the_permission_gate(run_loop, tmp_sessions, monkeypatch):
    """An unlisted mcp_ name used to route to the MCP gate and ask for approval. It is refused before that."""
    gate_calls = []
    monkeypatch.setattr(loop.dispatch, "_mcp_gate", lambda *a, **k: gate_calls.append(a) or None)
    post, dispatched = run_loop([_calls_turn([("c1", "mcp_context7_invented", "{}")]), _stop()],
                                session_id="mcp-unknown")
    assert dispatched == [] and gate_calls == []
    assert "Unknown tool 'mcp_context7_invented'" in _tool_messages(post)[0]["content"]


def test_unknown_names_are_never_passed_to_the_dispatcher(run_loop, tmp_sessions, monkeypatch, no_permission_gate):
    reached = []
    real = loop._dispatch_safely

    async def recording(tool_name, *args, **kwargs):
        reached.append(tool_name)
        return await real(tool_name, *args, **kwargs)

    monkeypatch.setattr(loop, "_dispatch_safely", recording)
    run_loop([_calls_turn([("c1", UNKNOWN_BARE, ""), ("c2", "list_directory", '{"path": "."}')]), _stop()],
             session_id="no-dispatch")
    assert reached == ["list_directory"]


# --- valid names keep their existing argument handling -----------------------------------------------------------

def test_a_valid_name_with_malformed_arguments_keeps_the_argument_error(run_loop, tmp_sessions, no_permission_gate):
    post, dispatched = run_loop([_calls_turn([("c1", "list_directory", "")]), _stop()], session_id="valid-bad-args")
    assert dispatched == []
    [reply] = _tool_messages(post)
    assert "Malformed tool arguments" in reply["content"]
    saved = session.load_session("valid-bad-args")
    session.validate_session_data(saved)
    [record] = [r for r in saved["execution"] if r["tool_call_id"] == "c1"]
    assert record["status"] == "failed" and record["arguments"] == "", "the journal keeps the original input"


def test_an_empty_name_keeps_the_no_function_name_error(run_loop, tmp_sessions, no_permission_gate):
    post, dispatched = run_loop([_calls_turn([("c1", "", "{}")]), _stop()], session_id="empty-name")
    assert dispatched == []
    assert "Tool call has no function name" in _tool_messages(post)[0]["content"]


def test_a_known_tool_outside_this_runs_offered_set_keeps_its_dispatch_answer(run_loop, tmp_sessions, monkeypatch):
    """A tool the runtime knows, but this run does not offer (a scenario or skill allowlist), is not an unknown name.
    It goes to the existing dispatch path, which answers it; the unknown-tool diagnostic does not apply to it."""
    from src.tools import get_all_tools
    offered_only = [t for t in get_all_tools(None) if t["function"]["name"] == "list_directory"]
    monkeypatch.setattr(loop, "get_all_tools", lambda mcp_client=None: offered_only)
    post, dispatched = run_loop([_calls_turn([("c1", "start_job", "{}")]), _stop()], session_id="known-outside")
    assert [name for name, _ in dispatched] == ["start_job"], "the existing dispatch path decides this call"
    assert "Unknown tool" not in _text(_tool_messages(post)[0])
