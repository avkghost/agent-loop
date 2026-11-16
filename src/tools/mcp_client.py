"""MCP client for connecting to external MCP servers."""

import asyncio
import contextlib
import copy
import json
from contextlib import AsyncExitStack
from typing import Dict, Any, Optional

from .. import config, envelope

_MISSING = object()


class MCPContractError(ValueError):
    """An MCP tool does not provide the schema Agent Loop needs. The tool is not offered to the model."""


def tool_parameters(tool: Any) -> Dict[str, Any]:
    """The parameter schema for an SDK tool (mcp.types.Tool), copied from its input_schema.

    Nothing is invented. A missing, non-object, or malformed schema raises MCPContractError instead of becoming an
    empty parameter list. A declared object without properties means the tool takes no arguments.
    """
    name = getattr(tool, "name", "<unnamed>")
    schema = getattr(tool, "input_schema", _MISSING)
    if schema is _MISSING:
        raise MCPContractError(f"tool {name}: the SDK tool has no input_schema attribute")
    if not isinstance(schema, dict):
        raise MCPContractError(f"tool {name}: input_schema is not a JSON object")
    if schema.get("type") != "object":
        raise MCPContractError(f"tool {name}: input_schema must declare type 'object'")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise MCPContractError(f"tool {name}: input_schema properties must be an object")
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise MCPContractError(f"tool {name}: input_schema required must be a list of property names")
    parameters = copy.deepcopy(schema)
    parameters.pop("$schema", None)  # a dialect marker, not a parameter; the provider does not need it
    parameters["properties"] = copy.deepcopy(properties)
    parameters["required"] = list(required)
    return parameters


def offered_tools(listed: Any) -> list:
    """The discovered tools that satisfy the schema contract. Each excluded tool is reported, never silently dropped."""
    accepted = []
    for tool in listed:
        try:
            tool_parameters(tool)
        except MCPContractError as exc:
            print(f"  ⚠️ Not offered to the model: {exc}")
            continue
        accepted.append(tool)
    return accepted


class MCPClientManager:
    """Manages connections to MCP servers and tool discovery."""

    def __init__(self):
        """Initialize the MCP client manager."""
        self.sessions = {}  # server_name -> ClientSession
        self.tools_cache = {}  # server_name -> list of tools
        self.exit_stack = None  # AsyncExitStack for managing context managers

    async def connect_all(self, config: Dict[str, Any]) -> None:
        """Connect to all configured MCP servers."""
        mcp_servers = config.get("mcpServers", {})

        if not mcp_servers:
            print("ℹ️ No MCP servers configured (mcp_servers.json is empty or missing)")
            return

        # Try importing the MCP SDK
        try:
            from mcp import ClientSession
            from mcp.client.stdio import stdio_client, StdioServerParameters
        except ImportError:
            print("⚠️ MCP SDK not available. Run: pip install mcp")
            print("   Proceeding with local tools only.")
            return

        # Create the exit stack for managing all server contexts
        self.exit_stack = contextlib.AsyncExitStack()
        await self.exit_stack.__aenter__()

        for server_name, server_config in mcp_servers.items():
            try:
                print(f"🔗 Connecting to MCP server: {server_name}")

                # Determine transport type
                if server_config.get("type") == "http" or "url" in server_config:
                    # HTTP/SSE transport — not yet implemented
                    print(f"  ⚠️ Skipping {server_name}: HTTP transport not yet implemented")
                    continue

                # Stdio transport (command + args)
                command = server_config.get("command")
                args = server_config.get("args", [])

                if not command:
                    print(f"  ⚠️ Skipping {server_name}: stdio server requires 'command'")
                    continue

                try:
                    # Create stdio transport
                    stdio_params = StdioServerParameters(command=command, args=args)

                    # Enter the stdio_client context and get read/write streams
                    transport = await self.exit_stack.enter_async_context(
                        stdio_client(stdio_params)
                    )
                    read_stream, write_stream = transport

                    # Create and initialize session
                    session = ClientSession(read_stream, write_stream)
                    await self.exit_stack.enter_async_context(session)
                    await session.initialize()

                    # Discover tools. A tool whose schema breaks the contract is reported and not offered to the model.
                    response = await session.list_tools()
                    listed = response.tools if hasattr(response, "tools") else []
                    tools = offered_tools(listed)

                    self.sessions[server_name] = session
                    self.tools_cache[server_name] = tools

                    print(f"  ✅ Connected. Discovered {len(tools)} tool(s)")

                except Exception as e:
                    print(f"  ⚠️ Connection failed: {e}")
                    continue

            except Exception as e:
                print(f"  ⚠️ Unexpected error: {e}")
                continue

    def get_openai_tools(self) -> list:
        """Get OpenAI-format tool schemas for all discovered MCP tools."""
        tools = []

        for server_name, server_tools in self.tools_cache.items():
            for tool in server_tools:
                # The tool's declared schema, copied whole, is what the model sees. Tools whose schema does not
                # satisfy the contract were already excluded at discovery, so this does not fail here.
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": f"mcp_{server_name}_{tool.name}",
                        "description": tool.description or f"Tool from {server_name}",
                        "parameters": tool_parameters(tool),
                    }
                }
                tools.append(openai_tool)

        return tools

    async def call_tool(self, prefixed_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool and return the result in our envelope format."""
        # Parse the server and tool name from the prefix (mcp_{server}_{tool})
        parts = prefixed_name.split("_", 2)  # Split on first two underscores
        if len(parts) != 3 or parts[0] != "mcp":
            return envelope.error_validation(
                f"Invalid MCP tool name format: {prefixed_name}",
                fixable=False
            )

        server_name = parts[1]
        tool_name = parts[2]

        # Get the session
        session = self.sessions.get(server_name)
        if not session:
            return envelope.error_validation(
                f"No connection to MCP server: {server_name}",
                fixable=False
            )

        # Only tools that passed schema discovery are callable. A tool the server did not list, or one excluded for a
        # schema that breaks the contract, is rejected here, before anything is sent to the server.
        if tool_name not in {tool.name for tool in self.tools_cache.get(server_name, [])}:
            return envelope.error_validation(
                f"MCP tool {prefixed_name} is not an advertised tool of server {server_name}; it was not sent.",
                fixable=False,
                description="The tool was not listed by the server or did not pass schema discovery. It was not called.",
            )

        # Everything below happens after the call may have been submitted, so any failure
        # leaves the external outcome unknown. Such failures are never retried automatically.
        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments),
                timeout=config.MCP_TOOL_TIMEOUT
            )
        except asyncio.TimeoutError:
            return envelope.error_unknown_outcome(
                f"MCP tool {prefixed_name} timed out after {config.MCP_TOOL_TIMEOUT} seconds. "
                "It may have executed; it was not retried.",
                description="The call was submitted and no response arrived in time.",
            )
        except Exception as e:  # CancelledError and other BaseExceptions propagate untouched
            return envelope.error_unknown_outcome(
                f"MCP tool {prefixed_name} failed during transport ({type(e).__name__}). "
                "It may have executed; it was not retried.",
                description="The connection failed after the call may have been submitted.",
            )

        return envelope.mcp_result_envelope(prefixed_name, result)

    async def close_all(self) -> None:
        """Close all MCP server connections gracefully."""
        try:
            if self.exit_stack:
                # Close all contexts in the stack
                await self.exit_stack.__aexit__(None, None, None)
        except RuntimeError as e:
            # Handle async context manager errors during cleanup
            # This can happen when connections fail mid-setup
            if "cancel scope" not in str(e).lower():
                raise
            # Silently ignore cancel scope errors during cleanup
        except Exception as e:
            # Log but don't raise - we're cleaning up anyway
            print(f"⚠️  Warning during MCP cleanup: {type(e).__name__}: {e}")
        finally:
            # Always clear references
            self.sessions.clear()
            self.tools_cache.clear()
            self.exit_stack = AsyncExitStack()
