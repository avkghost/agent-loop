"""MCP client for connecting to external MCP servers."""

import asyncio
import contextlib
import json
from contextlib import AsyncExitStack
from typing import Dict, Any, Optional

from .. import envelope


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

                    # Discover tools
                    response = await session.list_tools()
                    tools = response.tools if hasattr(response, "tools") else []

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
                # Transform MCP tool schema to OpenAI format
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": f"mcp_{server_name}_{tool.name}",
                        "description": tool.description or f"Tool from {server_name}",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                }

                # Map input schema to parameters if present
                if hasattr(tool, "inputSchema") and tool.inputSchema:
                    schema = tool.inputSchema
                    if isinstance(schema, dict):
                        openai_tool["function"]["parameters"]["properties"] = schema.get("properties", {})
                        openai_tool["function"]["parameters"]["required"] = schema.get("required", [])

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

        # Call the tool with timeout
        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments),
                timeout=30.0
            )

            # Normalize the result
            return envelope.normalize_mcp_result({
                "content": result.content if hasattr(result, "content") else str(result),
                "isError": result.isError if hasattr(result, "isError") else False,
            })

        except asyncio.TimeoutError:
            return envelope.error_transient(
                f"MCP tool {prefixed_name} timed out after 30 seconds.",
                description="The MCP server took too long to respond."
            )

        except Exception as e:
            return envelope.error_transient(
                f"MCP tool error: {str(e)}",
                description="Error communicating with the MCP server."
            )

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
