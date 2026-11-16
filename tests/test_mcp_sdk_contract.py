"""The MCP SDK contract: tool schemas, call results, and journal outcomes, using the real mcp.types classes.

The stub server is a local child process over stdio (tests/mcp_stub_server.py). Nothing here calls a network service
or a model: model requests are scripted, and the stub is the only MCP server.
"""

import asyncio
import copy
import importlib.metadata
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent, Tool

from src import config, envelope, loop, runs, session
from src.acceptance import AcceptanceCriterion
from src.authorization import SessionAuthorization
from src.tools.mcp_client import MCPClientManager, MCPContractError, offered_tools, tool_parameters

STUB = Path(__file__).resolve().parent / "mcp_stub_server.py"
STUB_CONFIG = {"mcpServers": {"stub": {"command": sys.executable, "args": [str(STUB)]}}}
RESOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to look up."},
        "libraryName": {"type": "string", "description": "Official library name."},
    },
    "required": ["query", "libraryName"],
}


def _tool(name, schema):
    return Tool(name=name, description=f"{name} tool", input_schema=schema)


# --- tool schema: what the model is offered -------------------------------------------------------------------------

def test_required_string_arguments_are_offered_exactly_as_declared():
    client = MCPClientManager()
    client.tools_cache = {"context7": [_tool("resolve-library-id", RESOLVE_SCHEMA)]}
    offered = client.get_openai_tools()
    assert offered == [{"type": "function", "function": {
        "name": "mcp_context7_resolve-library-id",
        "description": "resolve-library-id tool",
        "parameters": RESOLVE_SCHEMA,
    }}]


def test_nested_schema_with_references_is_preserved_and_not_aliased():
    schema = {
        "type": "object",
        "properties": {"options": {"$ref": "#/$defs/Options"}},
        "required": ["options"],
        "$defs": {"Options": {"type": "object", "properties": {"depth": {"type": "integer"}},
                              "required": ["depth"]}},
    }
    tool = _tool("configure", schema)
    parameters = tool_parameters(tool)
    assert parameters == schema
    parameters["properties"]["options"]["$ref"] = "changed"
    assert tool.input_schema["properties"]["options"]["$ref"] == "#/$defs/Options", "the SDK object must not change"


def test_a_dialect_marker_is_not_passed_to_the_model():
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "properties": {}}
    assert "$schema" not in tool_parameters(_tool("t", schema))


def test_an_object_without_properties_takes_no_arguments_and_is_not_a_contract_error():
    assert tool_parameters(_tool("ping", {"type": "object"})) == {"type": "object", "properties": {}, "required": []}


@pytest.mark.parametrize("tool, reason", [
    (SimpleNamespace(name="no-schema"), "no input_schema"),
    (SimpleNamespace(name="none-schema", input_schema=None), "not a JSON object"),
    (SimpleNamespace(name="string-schema", input_schema="{}"), "not a JSON object"),
    (SimpleNamespace(name="array", input_schema={"type": "array"}), "type 'object'"),
    (SimpleNamespace(name="untyped", input_schema={"properties": {}}), "type 'object'"),
    (SimpleNamespace(name="bad-props", input_schema={"type": "object", "properties": ["query"]}), "properties"),
    (SimpleNamespace(name="bad-required", input_schema={"type": "object", "required": "query"}), "required"),
    (SimpleNamespace(name="bad-required-item", input_schema={"type": "object", "required": [1]}), "required"),
], ids=lambda value: value.name if isinstance(value, SimpleNamespace) else None)
def test_a_missing_or_malformed_schema_is_a_contract_error_never_an_empty_parameter_list(tool, reason):
    with pytest.raises(MCPContractError, match=reason):
        tool_parameters(tool)


def test_offered_tools_excludes_and_reports_the_broken_tool_and_keeps_the_rest(capsys):
    good = _tool("lookup", RESOLVE_SCHEMA)
    broken = SimpleNamespace(name="broken")
    assert offered_tools([broken, good]) == [good]
    assert "Not offered to the model" in capsys.readouterr().out


def test_the_installed_sdk_still_exposes_the_attributes_this_code_reads():
    assert "input_schema" in Tool.model_fields
    assert "is_error" in CallToolResult.model_fields
    assert importlib.metadata.version("mcp").startswith("2.3."), "the pinned SDK version changed"


# --- call results: classified from the SDK's is_error ------------------------------------------------------------------

def test_a_successful_result_is_a_success():
    body = envelope.mcp_result_envelope("mcp_s_t", CallToolResult(content=[TextContent(type="text", text="ok")],
                                                                  is_error=False))
    assert body["isError"] is False and body["content"] == "ok" and body["errorCategory"] is None


def test_an_empty_successful_result_is_a_success_with_empty_content():
    body = envelope.mcp_result_envelope("mcp_s_t", CallToolResult(content=[], is_error=False))
    assert body["isError"] is False and body["content"] == ""


def test_a_non_text_content_item_is_represented_not_dropped():
    body = envelope.mcp_result_envelope("mcp_s_t", CallToolResult(
        content=[ImageContent(type="image", data="AAAA", mime_type="image/png")], is_error=False))
    assert body["content"] == "[image content omitted]" and body["isError"] is False


def test_an_error_result_is_unknown_outcome_and_keeps_the_server_message():
    body = envelope.mcp_result_envelope("mcp_s_t", CallToolResult(
        content=[TextContent(type="text", text="quota exceeded")], is_error=True))
    assert body["isError"] is True
    assert body["errorCategory"] == "mcp_error"
    assert body["isRetryable"] is False
    assert body["content"] == "quota exceeded"
    assert body["description"]


@pytest.mark.parametrize("result", [
    None,
    {"content": "a dict is not an SDK result", "isError": False},
    SimpleNamespace(content=[]),                       # no is_error flag at all: never a success by default
    SimpleNamespace(content=[], is_error="no"),
    SimpleNamespace(content=[], isError=False),        # the old attribute name is not read
], ids=["none", "dict", "no-flag", "non-bool-flag", "old-name"])
def test_a_result_without_a_recognizable_flag_is_unknown_never_success(result):
    body = envelope.mcp_result_envelope("mcp_s_t", result)
    assert body["isError"] is True
    assert body["errorCategory"] == "unknown_outcome" and body["isRetryable"] is False
    assert "malformed" in body["content"]


# --- stub server over real stdio -------------------------------------------------------------------------------------

@pytest.fixture
def stub_client():
    event_loop = asyncio.new_event_loop()
    manager = MCPClientManager()
    try:
        event_loop.run_until_complete(manager.connect_all(STUB_CONFIG))
        yield manager, event_loop
    finally:
        event_loop.run_until_complete(manager.close_all())
        event_loop.close()


def _call(stub, name, arguments):
    manager, event_loop = stub
    return event_loop.run_until_complete(manager.call_tool(f"mcp_stub_{name}", arguments))


def test_the_stub_offers_its_required_arguments_to_the_model(stub_client):
    manager, _ = stub_client
    offered = {tool["function"]["name"]: tool["function"]["parameters"] for tool in manager.get_openai_tools()}
    assert offered["mcp_stub_lookup"]["required"] == ["query", "library_name"]
    assert set(offered["mcp_stub_lookup"]["properties"]) == {"query", "library_name"}


def test_the_stub_offers_nested_arguments_with_their_references(stub_client):
    manager, _ = stub_client
    configure = next(t["function"]["parameters"] for t in manager.get_openai_tools()
                     if t["function"]["name"] == "mcp_stub_configure")
    assert configure["properties"]["options"] == {"$ref": "#/$defs/Options"}
    assert configure["$defs"]["Options"]["required"] == ["depth", "label"]


def test_a_call_with_missing_required_arguments_is_a_failed_result_from_the_sdk(stub_client):
    body = _call(stub_client, "lookup", {})
    assert body["isError"] is True and body["errorCategory"] == "mcp_error" and body["isRetryable"] is False
    assert "validation" in body["content"].lower()


def test_a_valid_call_succeeds(stub_client):
    body = _call(stub_client, "lookup", {"query": "routes", "library_name": "Axum"})
    assert body["isError"] is False and body["content"] == "Axum: routes"


def test_a_nested_argument_call_succeeds(stub_client):
    body = _call(stub_client, "configure", {"name": "svc", "options": {"depth": 2, "label": "x"}})
    assert body["isError"] is False and body["content"] == "svc:2:x"


def test_a_tool_error_keeps_the_server_message_and_is_not_retried(stub_client):
    body = _call(stub_client, "fail", {"reason": "quota"})
    assert body["isError"] is True and body["errorCategory"] == "mcp_error" and body["isRetryable"] is False
    assert "remote refused: quota" in body["content"]


def test_real_stdio_timeout_after_commit_is_unknown_and_not_retried(stub_client, tmp_path, monkeypatch):
    effect_path = tmp_path / "effect.txt"
    monkeypatch.setattr(config, "MCP_TOOL_TIMEOUT", 0.05)

    body = _call(stub_client, "hang_after_commit", {"effect_path": str(effect_path)})

    assert body["isError"] is True
    assert body["errorCategory"] == "unknown_outcome"
    assert body["isRetryable"] is False
    assert "timed out after 0.05 seconds" in body["content"]
    assert effect_path.read_text(encoding="utf-8") == "committed"


# --- the run: offered schema, and accurate journal and evidence outcomes -------------------------------------------

@pytest.fixture
def run_stub_agent(fake_post, tmp_sessions, monkeypatch, stub_client):
    manager, _ = stub_client
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)

    def run(replies, session_id):
        post = fake_post(replies)
        # The stub's stdio session is bound to the loop that connected it, so the run uses that same loop, as the
        # runtime does.
        _, event_loop = stub_client
        event_loop.run_until_complete(loop.agent_loop_async(
            "lookup docs", manager, event_loop, session_id=session_id,
            authorization=SessionAuthorization(mcp_tools=["mcp_stub_lookup"])))
        return post

    return run


def _tool_call_reply(arguments):
    from tests.conftest import make_streamed_response
    return make_streamed_response(finish_reason="tool_calls", tool_calls=[
        {"id": "call_1", "name": "mcp_stub_lookup", "arguments": json.dumps(arguments)}])


def _stop_reply():
    from tests.conftest import make_streamed_response
    return make_streamed_response(finish_reason="stop", content="done")


def test_the_run_offers_the_schema_and_journals_a_missing_argument_call_as_unknown(run_stub_agent, tmp_sessions):
    post = run_stub_agent([_tool_call_reply({}), _stop_reply()], session_id="mcp-missing-arg")

    offered = next(t["function"] for t in post.payloads[0]["tools"] if t["function"]["name"] == "mcp_stub_lookup")
    assert offered["parameters"]["required"] == ["query", "library_name"]

    tool_message = next(m for m in post.payloads[1]["messages"] if m.get("role") == "tool")
    body = json.loads(tool_message["content"])
    assert body["isError"] is True and body["errorCategory"] == "mcp_error"

    saved = session.load_session("mcp-missing-arg")
    assert saved["execution"][0]["status"] == "unknown", "the server dispatched the call; its outcome is not established"
    assert saved["evidence"][0]["outcome"] == "unknown", "evidence must match the journal"


def test_the_run_journals_a_valid_call_as_completed(run_stub_agent, tmp_sessions):
    run_stub_agent([_tool_call_reply({"query": "routes", "library_name": "Axum"}), _stop_reply()],
                   session_id="mcp-valid")
    saved = session.load_session("mcp-valid")
    assert saved["execution"][0]["status"] == "completed"
    assert saved["evidence"][0]["outcome"] == "succeeded"


# --- excluded and unadvertised tools are never sent -----------------------------------------------------------------

class _SpySession:
    """A stand-in for the SDK session: it records every name that reaches the remote side."""

    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append(name)
        return CallToolResult(content=[TextContent(type="text", text="ran")], is_error=False)


class _EffectThenFailureSession:
    """A disposable remote action that commits once, then loses its response."""

    def __init__(self, effect_db, failure):
        self.effect_db = effect_db
        self.failure = failure
        self.calls = 0

    async def call_tool(self, name, arguments):
        self.calls += 1
        with sqlite3.connect(self.effect_db) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS effects (key TEXT PRIMARY KEY)")
            connection.execute("INSERT OR IGNORE INTO effects(key) VALUES (?)", ("submitted-once",))
        if self.failure == "timeout":
            raise asyncio.TimeoutError()
        raise ConnectionResetError("connection lost after submission")


def _advertising_spy(*tools):
    client = MCPClientManager()
    spy = _SpySession()
    client.sessions = {"srv": spy}
    client.tools_cache = {"srv": offered_tools(list(tools))}
    return client, spy


def _effect_failure_client(effect_db, failure):
    client = MCPClientManager()
    remote = _EffectThenFailureSession(effect_db, failure)
    client.sessions = {"srv": remote}
    client.tools_cache = {"srv": [_tool("mutate", {"type": "object", "properties": {}})]}
    return client, remote


@pytest.mark.parametrize("failure", ["timeout", "disconnect"])
def test_submitted_effect_is_unknown_and_never_retried(tmp_path, failure):
    client, remote = _effect_failure_client(tmp_path / "effects.sqlite", failure)

    body = asyncio.run(client.call_tool("mcp_srv_mutate", {}))

    assert body["isError"] is True
    assert body["errorCategory"] == "unknown_outcome"
    assert body["isRetryable"] is False
    assert remote.calls == 1
    with sqlite3.connect(tmp_path / "effects.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM effects").fetchone()[0] == 1


def test_an_excluded_tool_is_rejected_before_the_session_while_advertised_tools_still_run():
    client, spy = _advertising_spy(_tool("lookup", RESOLVE_SCHEMA), SimpleNamespace(name="broken"))
    rejected = asyncio.run(client.call_tool("mcp_srv_broken", {}))
    assert rejected["isError"] is True and rejected["errorCategory"] == "validation"
    assert rejected["isRetryable"] is False and "not an advertised tool" in rejected["content"]
    assert spy.calls == [], "an excluded tool must never reach the remote side"
    accepted = asyncio.run(client.call_tool("mcp_srv_lookup", {"query": "q", "libraryName": "L"}))
    assert accepted["isError"] is False and spy.calls == ["lookup"]


def test_a_name_the_server_never_listed_is_rejected_before_the_session():
    client, spy = _advertising_spy(_tool("lookup", RESOLVE_SCHEMA))
    body = asyncio.run(client.call_tool("mcp_srv_invented", {}))
    assert body["errorCategory"] == "validation" and spy.calls == []


def test_an_unlisted_name_on_the_real_stub_is_a_local_rejection_not_a_remote_error(stub_client):
    body = _call(stub_client, "invented", {})
    assert body["errorCategory"] == "validation", "the server would have answered mcp_error; this never reached it"
    assert body["isRetryable"] is False


# --- the run: local rejections are FAILED, remote errors are UNKNOWN, and verification respects both ----------------

def _tool_reply(name, arguments):
    from tests.conftest import make_streamed_response
    return make_streamed_response(finish_reason="tool_calls", tool_calls=[
        {"id": "call_1", "name": name, "arguments": json.dumps(arguments)}])


def _stop():
    from tests.conftest import make_streamed_response
    return make_streamed_response(finish_reason="stop", content="done")


def _run_agent(fake_post, client, event_loop, replies, session_id, allowed, criteria=None, cycles=1):
    post = fake_post(replies)
    kwargs = {"criteria": criteria, "verification_cycles": cycles} if criteria is not None else {}
    outcome = event_loop.run_until_complete(loop.agent_loop_async(
        "lookup docs", client, event_loop, session_id=session_id,
        authorization=SessionAuthorization(mcp_tools=allowed), **kwargs))
    return outcome, post


@pytest.fixture
def no_guidelines(monkeypatch, tmp_sessions):
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)


def test_a_call_to_an_excluded_tool_is_journaled_failed_and_evidenced_failed(fake_post, no_guidelines, monkeypatch):
    client, spy = _advertising_spy(_tool("lookup", RESOLVE_SCHEMA), SimpleNamespace(name="broken"))
    event_loop = asyncio.new_event_loop()
    try:
        _, post = _run_agent(fake_post, client, event_loop, [_tool_reply("mcp_srv_broken", {}), _stop()],
                             "excluded-call", allowed=["mcp_srv_broken"])
    finally:
        event_loop.close()
    tool_message = next(m for m in post.payloads[1]["messages"] if m.get("role") == "tool")
    assert json.loads(tool_message["content"])["errorCategory"] == "validation"
    saved = session.load_session("excluded-call")
    assert saved["execution"][0]["status"] == "failed"
    assert saved["evidence"][0]["outcome"] == "failed"
    assert spy.calls == [], "the excluded tool was never sent"


def test_a_remote_error_is_journaled_unknown_with_matching_evidence(stub_client, fake_post, no_guidelines):
    manager, event_loop = stub_client
    _, post = _run_agent(fake_post, manager, event_loop, [_tool_reply("mcp_stub_fail", {"reason": "quota"}), _stop()],
                         "remote-error", allowed=["mcp_stub_fail"])
    tool_message = next(m for m in post.payloads[1]["messages"] if m.get("role") == "tool")
    body = json.loads(tool_message["content"])
    assert body["isError"] is True and body["errorCategory"] == "mcp_error" and body["isRetryable"] is False
    assert "remote refused: quota" in body["content"]
    saved = session.load_session("remote-error")
    assert saved["execution"][0]["status"] == "unknown"
    assert saved["evidence"][0]["outcome"] == "unknown"


def _file_criterion(tmp_path):
    (tmp_path / "calc.py").write_text("x = 1\n", encoding="utf-8")
    return [AcceptanceCriterion(criterion_id="present", run_id="", criterion_type="file_content", expected=True,
                                origin="user", parameters={"path": "calc.py", "contains": "x = 1"})]


def test_an_unknown_remote_outcome_withholds_verified_completion(stub_client, fake_post, no_guidelines,
                                                                 tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager, event_loop = stub_client
    outcome, _ = _run_agent(fake_post, manager, event_loop, [_tool_reply("mcp_stub_fail", {"reason": "quota"}), _stop()],
                            "remote-blocks", allowed=["mcp_stub_fail"], criteria=_file_criterion(tmp_path))
    assert outcome.run_status is runs.RunStatus.NEEDS_ATTENTION
    assert "unknown" in outcome.reason


def test_the_same_run_with_a_successful_remote_call_is_verified(stub_client, fake_post, no_guidelines, tmp_path,
                                                                monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager, event_loop = stub_client
    outcome, _ = _run_agent(fake_post, manager, event_loop,
                            [_tool_reply("mcp_stub_lookup", {"query": "q", "library_name": "L"}), _stop()],
                            "remote-control", allowed=["mcp_stub_lookup"], criteria=_file_criterion(tmp_path))
    assert outcome.run_status is runs.RunStatus.COMPLETED_VERIFIED
