"""A local MCP server over stdio, for offline tests. It runs as a child process and makes no network calls.

Tools:
  lookup(query, library_name)   two required strings
  configure(name, options)      a nested object argument (options: depth, label)
  fail(reason)                  always returns an MCP error result carrying the reason
"""

import asyncio
from pathlib import Path
from typing import TypedDict

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError


class Options(TypedDict):
    depth: int
    label: str


server = MCPServer("agent-loop-stub")


@server.tool()
def lookup(query: str, library_name: str) -> str:
    """Look up documentation for a library."""
    return f"{library_name}: {query}"


@server.tool()
def configure(name: str, options: Options) -> str:
    """Apply nested options to a named target."""
    return f"{name}:{options['depth']}:{options['label']}"


@server.tool()
def fail(reason: str) -> str:
    """Always fails with the given reason."""
    raise ToolError(f"remote refused: {reason}")  # an anticipated failure: its message reaches the model


@server.tool()
async def hang_after_commit(effect_path: str) -> str:
    """Commit a disposable effect, then delay long enough for the client timeout."""
    Path(effect_path).write_text("committed", encoding="utf-8")
    await asyncio.sleep(10)
    return "late response"


if __name__ == "__main__":
    server.run("stdio")
