"""B3.6 tool registry contract: deterministic schemas, refused registrations, and unchanged authorization."""

import asyncio
import dataclasses

import pytest

from src import dispatch, envelope, jobs
from src.authorization import SessionAuthorization
from src.tools import get_local_tools, local_registry
from src.tools import shell
from src.tools.registry import FILE, GIT, JOB, MCP, SHELL, RegistryError, ToolDescriptor, ToolRegistry

PROVIDER_ORDER = ["start_job", "status_job", "collect_job", "terminate_job",
                  "run_command", "write_file", "read_file", "list_directory",
                  "edit_file", "git_status", "git_diff"]


def _schema(name):
    return {"type": "function", "function": {"name": name, "description": "d", "parameters": {"type": "object"}}}


def test_provider_visible_order_and_names_are_stable():
    assert [tool["function"]["name"] for tool in get_local_tools()] == PROVIDER_ORDER
    assert local_registry().names() == PROVIDER_ORDER


def test_routing_kinds_match_the_tool_families():
    registry = local_registry()
    assert registry.kind_of("run_command") == SHELL
    assert registry.kind_of("read_file") == FILE
    assert registry.kind_of("write_file") == FILE
    assert registry.kind_of("list_directory") == FILE
    assert registry.kind_of("edit_file") == FILE
    assert registry.kind_of("git_status") == GIT
    assert registry.kind_of("git_diff") == GIT
    for name in ("start_job", "status_job", "collect_job", "terminate_job"):
        assert registry.kind_of(name) == JOB
    assert registry.kind_of("mcp_anything") == MCP


def test_job_registrations_agree_with_the_jobs_module():
    job_names = {name for name in local_registry().names() if local_registry().kind_of(name) == JOB}
    assert job_names == set(jobs.TOOL_NAMES)


def test_registry_is_built_once_and_is_finalized():
    registry = local_registry()
    assert registry is local_registry()
    assert registry.finalized
    with pytest.raises(RegistryError, match="finalized"):
        registry.register(ToolDescriptor("late_tool", FILE, _schema("late_tool")))


def test_duplicate_registration_is_rejected_deterministically():
    registry = ToolRegistry()
    registry.register(ToolDescriptor("alpha", FILE, _schema("alpha")))
    with pytest.raises(RegistryError, match="duplicate tool registration: alpha"):
        registry.register(ToolDescriptor("alpha", SHELL, _schema("alpha")))


@pytest.mark.parametrize("name, kind, message", [
    ("", FILE, "non-empty"),
    ("mcp_fake", FILE, "reserved for MCP"),
    ("tool", "plugin", "unknown tool kind"),
])
def test_invalid_descriptors_are_refused(name, kind, message):
    with pytest.raises(RegistryError, match=message):
        ToolDescriptor(name, kind, _schema(name))


def test_schema_name_must_match_the_descriptor_name():
    with pytest.raises(RegistryError, match="does not match"):
        ToolDescriptor("alpha", FILE, _schema("beta"))


def test_descriptors_are_immutable_and_schemas_are_copies():
    descriptor = local_registry().get("read_file")
    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.kind = SHELL
    schemas = get_local_tools()
    schemas[0]["function"]["name"] = "tampered"
    schemas.pop()
    assert [tool["function"]["name"] for tool in get_local_tools()] == PROVIDER_ORDER


def test_unknown_and_non_string_names_resolve_to_nothing():
    registry = local_registry()
    assert registry.kind_of("delete_everything") is None
    assert registry.kind_of(None) is None
    assert registry.kind_of(42) is None
    assert registry.get(["run_command"]) is None


def test_unknown_tool_is_rejected_before_any_handler_runs(monkeypatch):
    calls = []
    monkeypatch.setattr(shell, "execute_command_async", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(shell, "execute_command", lambda *a, **k: calls.append(a))
    result = asyncio.run(dispatch.dispatch_tool_async("delete_everything", {"path": "/"}))
    assert result["isError"] is True and "Unknown tool" in result["content"]
    assert calls == []


def test_non_string_tool_name_is_an_error_envelope_not_an_exception():
    result = dispatch.dispatch_tool(None, {})
    assert result["isError"] is True and "Unknown tool" in result["content"]


def test_registry_lookup_does_not_bypass_shell_authorization():
    """run_command is routed by the registry, but an unlisted command still needs a grant."""
    auth = SessionAuthorization()
    result = asyncio.run(dispatch.dispatch_tool_async(
        "run_command", {"command": "uname -s"}, authorization=auth))
    assert result.get("requiresPermission") is True
    assert result["kind"] == "command"


def test_registry_lookup_does_not_bypass_the_mcp_gate():
    class FakeMCP:
        calls = []

        async def call_tool(self, name, tool_input):
            self.calls.append(name)
            return envelope.success("ran")

    fake = FakeMCP()
    denied = asyncio.run(dispatch.dispatch_tool_async(
        "mcp_remote_write", {"path": "x"}, mcp_client=fake, authorization=SessionAuthorization()))
    assert denied.get("requiresPermission") is True and fake.calls == []

    granted = SessionAuthorization()
    granted.grant("mcp", "mcp_remote_write")
    allowed = asyncio.run(dispatch.dispatch_tool_async(
        "mcp_remote_write", {"path": "x"}, mcp_client=fake, authorization=granted))
    assert allowed["content"] == "ran" and fake.calls == ["mcp_remote_write"]


def test_file_tool_containment_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    escaped = asyncio.run(dispatch.dispatch_tool_async("read_file", {"path": "../outside.txt"}))
    assert escaped["isError"] is True
    assert (tmp_path / "in.txt").exists() is False
    written = asyncio.run(dispatch.dispatch_tool_async("write_file", {"path": "in.txt", "content": "ok"},
                                                       authorization=SessionAuthorization()))
    assert written["isError"] is False and (tmp_path / "in.txt").read_text() == "ok"


def test_handlers_keep_no_session_state_in_the_registry(monkeypatch, tmp_path):
    """Two sessions with different search state must not influence each other through the registry."""
    monkeypatch.chdir(tmp_path)
    searched = SessionAuthorization()
    searched.note_search()
    fresh = SessionAuthorization()
    in_searched = asyncio.run(dispatch.dispatch_tool_async(
        "write_file", {"path": "a.txt", "content": "x"}, authorization=searched))
    in_fresh = asyncio.run(dispatch.dispatch_tool_async(
        "write_file", {"path": "b.txt", "content": "x"}, authorization=fresh))
    assert "No codebase search" not in in_searched["content"]
    assert "No codebase search" in in_fresh["content"]
    assert local_registry().get("write_file").schema["function"]["name"] == "write_file"


def test_mutating_a_looked_up_descriptor_cannot_alter_the_shared_registry():
    looked_up = local_registry().get("read_file")
    looked_up.schema["function"]["name"] = "TAMPERED"
    assert local_registry().get("read_file").schema["function"]["name"] == "read_file"
    assert local_registry().names() == PROVIDER_ORDER


def test_mutating_the_schema_passed_at_registration_cannot_alter_the_registry():
    source = _schema("gamma")
    registry = ToolRegistry()
    registry.register(ToolDescriptor("gamma", FILE, source))
    source["function"]["name"] = "TAMPERED"
    assert registry.get("gamma").schema["function"]["name"] == "gamma"


def test_every_registered_name_has_a_dispatch_handler():
    """A registered tool with no handler would silently fall through to 'Unknown tool'. Catch that here."""
    import tempfile
    import os
    for name in local_registry().names():
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                result = asyncio.run(dispatch.dispatch_tool_async(name, {}, authorization=SessionAuthorization()))
            finally:
                os.chdir(cwd)
        text = result.get("content", "") if isinstance(result, dict) else ""
        assert "Unknown tool" not in text, f"{name} is registered but has no dispatch handler"


def test_skill_allowlist_denies_before_execution_for_every_kind(monkeypatch, tmp_path):
    """Authorization before execution: a skill allowlist that excludes a tool stops it before any handler runs."""
    monkeypatch.chdir(tmp_path)
    ran = []

    async def record_spawn(*args, **kwargs):
        ran.append("shell")
        return {}

    monkeypatch.setattr(shell, "_run", record_spawn)

    class FakeMCP:
        async def call_tool(self, name, tool_input):
            ran.append("mcp")
            return envelope.success("ran")

    cases = [("run_command", {"command": "true"}), ("write_file", {"path": "f.txt", "content": "x"}),
             ("start_job", {"command": "true"}), ("mcp_remote_write", {"path": "x"})]
    for name, args in cases:
        result = asyncio.run(dispatch.dispatch_tool_async(name, args, mcp_client=FakeMCP(),
                                                          allowed_tools=["read_file"],
                                                          authorization=SessionAuthorization(allow_all=True)))
        assert result["isError"] is True and "not permitted" in result["content"], name
    assert ran == [] and not (tmp_path / "f.txt").exists()


def test_authorization_gates_are_consulted_before_execution_for_each_routed_kind(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    ran = []

    async def record_spawn(*args, **kwargs):  # the real validation runs first; this only sees an approved spawn
        ran.append("shell")
        return {}

    monkeypatch.setattr(shell, "_run", record_spawn)

    class FakeMCP:
        async def call_tool(self, name, tool_input):
            ran.append("mcp")
            return envelope.success("ran")

    # Shell: an unlisted command returns a permission request and does not run
    asyncio.run(dispatch.dispatch_tool_async("run_command", {"command": "uname -s"},
                                             authorization=SessionAuthorization()))
    # MCP: an ungranted tool returns a permission request and is not called
    asyncio.run(dispatch.dispatch_tool_async("mcp_remote_write", {"path": "x"}, mcp_client=FakeMCP(),
                                             authorization=SessionAuthorization()))
    assert ran == []


def test_provider_visible_schemas_match_the_golden_snapshot():
    """Schema drift is a provider-visible change. It must be deliberate: update the fixture with the change."""
    import json
    from pathlib import Path
    golden = json.loads((Path(__file__).parent / "fixtures" / "provider_schemas.json").read_text())
    assert get_local_tools() == golden
