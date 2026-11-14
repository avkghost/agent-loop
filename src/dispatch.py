"""Tool dispatch and permission gating."""

import asyncio
import json
from typing import Dict, Any, Optional

from . import config, envelope
from .authorization import SessionAuthorization
from .envelope import ToolExecutionError  # re-exported: dispatch.ToolExecutionError
from .tools import shell, files
from . import jobs
from .tools.mcp_client import MCPClientManager


def _allowed_tools_denial(tool_name: str, allowed_tools: Optional[list]) -> Optional[Dict[str, Any]]:
    """Two-layer defense in depth: skills may restrict which tools they can call."""
    if allowed_tools and isinstance(allowed_tools, list) and len(allowed_tools) > 0:
        if tool_name not in allowed_tools:
            return envelope.error_permission(
                f"Tool '{tool_name}' is not permitted for this skill (allowed: {', '.join(allowed_tools)})"
            )
    return None


def _mcp_gate(tool_name: str, tool_input: Dict[str, Any], mcp_client: Optional[MCPClientManager],
              authorization: Optional[SessionAuthorization] = None):
    """Checks shared by both dispatch entry points. Returns a result to send back, or None to proceed."""
    if not mcp_client:
        return envelope.error_validation(
            "MCP tool requested but MCP client not initialized",
            fixable=False
        )
    if authorization is None:
        authorization = SessionAuthorization.from_config()
    if not authorization.allows("mcp", tool_name):
        return {
            "requiresPermission": True,
            "kind": "mcp_tool",
            "identifier": tool_name,
            "arguments": tool_input
        }
    return None


async def dispatch_tool_async(tool_name: str, tool_input: Dict[str, Any], mcp_client: Optional[MCPClientManager] = None,
                              event_loop: Optional[asyncio.AbstractEventLoop] = None,
                              allowed_tools: Optional[list] = None,
                              authorization: Optional[SessionAuthorization] = None) -> Dict[str, Any]:
    """Awaitable dispatch for use inside the running agent loop.

    MCP calls are awaited directly on the current event loop, so the transport is reached
    without nesting run_until_complete. Local tools run synchronously, exactly as in dispatch_tool.
    authorization is the calling session's policy; None means the CLI snapshot with no kept grants.
    """
    if tool_name == "run_command":
        # Awaited directly so a running command never blocks the agent's event loop
        denial = _allowed_tools_denial(tool_name, allowed_tools)
        if denial:
            return denial
        command = tool_input.get("command", "")
        if not command:
            return envelope.error_validation(
                "Error: run_command requires a 'command' argument",
                fixable=False
            )
        return await shell.execute_command_async(command, tool_input.get("timeout_seconds"), authorization)

    if tool_name in jobs.TOOL_NAMES:
        denial = _allowed_tools_denial(tool_name, allowed_tools)
        if denial:
            return denial
        return await jobs.call_tool(tool_name, tool_input, authorization)

    if not (isinstance(tool_name, str) and tool_name.startswith("mcp_")):
        return dispatch_tool(tool_name, tool_input, mcp_client, event_loop, allowed_tools=allowed_tools,
                             authorization=authorization)

    denial = _allowed_tools_denial(tool_name, allowed_tools)
    if denial:
        return denial
    gate = _mcp_gate(tool_name, tool_input, mcp_client, authorization)
    if gate is not None:
        return gate
    return await mcp_client.call_tool(tool_name, tool_input)


def dispatch_tool(tool_name: str, tool_input: Dict[str, Any], mcp_client: Optional[MCPClientManager] = None,
                  event_loop: Optional[asyncio.AbstractEventLoop] = None, allowed_tools: Optional[list] = None,
                  authorization: Optional[SessionAuthorization] = None) -> Dict[str, Any]:
    """Dispatch a tool call to the appropriate handler.

    Args:
        tool_name: Name of the tool to call
        tool_input: Input parameters for the tool
        mcp_client: Optional MCP client for MCP tools
        event_loop: Optional event loop for async operations
        allowed_tools: Optional list of allowed tool names. None or empty list means no restriction.
        authorization: The calling session's policy (None: CLI snapshot, no kept grants)
    """

    denial = _allowed_tools_denial(tool_name, allowed_tools)
    if denial:
        return denial

    # Handle local tools
    if tool_name == "run_command":
        command = tool_input.get("command", "")
        if not command:
            return envelope.error_validation(
                "Error: run_command requires a 'command' argument",
                fixable=False
            )

        result = shell.execute_command(command, tool_input.get("timeout_seconds"), authorization)

        # Check for permission request sentinel
        if isinstance(result, dict) and result.get("requiresPermission"):
            return result

        return result

    elif tool_name == "read_file":
        path = tool_input.get("path", "")

        if not path:
            return envelope.error_validation(
                "Error: read_file requires a 'path' argument",
                fixable=False
            )

        return files.read_file_safe(path)

    elif tool_name == "list_directory":
        path = tool_input.get("path", ".")

        return files.list_directory_safe(path)

    elif tool_name == "write_file":
        path = tool_input.get("path", "")
        content = tool_input.get("content", "")

        if not path:
            return envelope.error_validation(
                "Error: write_file requires a 'path' argument",
                fixable=False
            )

        return files.write_file_safe(path, content)

    # Handle MCP tools
    elif tool_name in jobs.TOOL_NAMES:
        return jobs.call_tool_sync(tool_name, tool_input, authorization)

    elif tool_name.startswith("mcp_"):
        gate = _mcp_gate(tool_name, tool_input, mcp_client, authorization)
        if gate is not None:
            return gate

        # Synchronous callers only. Inside the agent loop use dispatch_tool_async, because the
        # loop already runs an event loop and run_until_complete cannot be nested.
        if not event_loop:
            return envelope.error_validation(
                "Event loop not available for MCP tool call",
                fixable=False
            )

        return event_loop.run_until_complete(mcp_client.call_tool(tool_name, tool_input))

    else:
        return envelope.error_validation(
            f"Unknown tool: {tool_name}",
            fixable=False,
            description="The tool name is not recognized."
        )
