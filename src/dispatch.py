"""Tool dispatch and permission gating."""

import asyncio
import json
from typing import Dict, Any, Optional

from . import config, envelope
from .tools import shell, files
from .tools.mcp_client import MCPClientManager


def dispatch_tool(tool_name: str, tool_input: Dict[str, Any], mcp_client: Optional[MCPClientManager] = None,
                  event_loop: Optional[asyncio.AbstractEventLoop] = None, allowed_tools: Optional[list] = None) -> Dict[str, Any]:
    """Dispatch a tool call to the appropriate handler.

    Args:
        tool_name: Name of the tool to call
        tool_input: Input parameters for the tool
        mcp_client: Optional MCP client for MCP tools
        event_loop: Optional event loop for async operations
        allowed_tools: Optional list of allowed tool names. None or empty list means no restriction.
    """

    # Check allowed-tools enforcement (two-layer defense in depth)
    if allowed_tools and isinstance(allowed_tools, list) and len(allowed_tools) > 0:
        if tool_name not in allowed_tools:
            return envelope.error_permission(
                f"Tool '{tool_name}' is not permitted for this skill (allowed: {', '.join(allowed_tools)})"
            )

    # Handle local tools
    if tool_name == "run_command":
        command = tool_input.get("command", "")
        if not command:
            return envelope.error_validation(
                "Error: run_command requires a 'command' argument",
                fixable=False
            )

        result = shell.execute_command(command)

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
    elif tool_name.startswith("mcp_"):
        if not mcp_client:
            return envelope.error_validation(
                f"MCP tool requested but MCP client not initialized",
                fixable=False
            )

        # Check if the MCP tool is allowed
        if tool_name not in config.ALLOWED_MCP_TOOLS:
            if not config.ALLOW_ALL:
                # Return permission request sentinel
                return {
                    "requiresPermission": True,
                    "kind": "mcp_tool",
                    "identifier": tool_name,
                    "arguments": tool_input
                }

        # Call the MCP tool async via the event loop
        if not event_loop:
            return envelope.error_validation(
                "Event loop not available for MCP tool call",
                fixable=False
            )

        try:
            result = event_loop.run_until_complete(
                mcp_client.call_tool(tool_name, tool_input)
            )
            return result
        except Exception as e:
            return envelope.error_transient(
                f"Error calling MCP tool: {str(e)}",
                description="Failed to dispatch the MCP tool call."
            )

    else:
        return envelope.error_validation(
            f"Unknown tool: {tool_name}",
            fixable=False,
            description="The tool name is not recognized."
        )
