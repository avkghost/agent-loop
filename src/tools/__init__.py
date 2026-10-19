"""Tool definitions and schemas."""

def get_local_tools() -> list:
    """Return the OpenAI-format schemas for local tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": """Execute a shell command from the allowlist.

Purpose: Run allowlisted shell commands (grep, rg, find, python3, npm, etc.) without shell interpretation.

Inputs:
- command (string): The exact command to run. Arguments are passed literally via shlex.split, so no pipes, redirection, or globbing via shell — e.g., "grep -r 'pattern' ." or "python3 script.py".

Examples:
- Search codebase: run_command with "grep -r 'function_name' ."
- Run a script: run_command with "python3 analysis.py"
- List files: run_command with "find . -name '*.py' -type f"

Edge cases & limitations:
- No shell builtins (cd, echo with redirection, etc.) — these are passed to the subprocess literally and fail.
- Timeout: commands exceeding 15 seconds are terminated.
- Some commands may require human approval (1/2/3 escalation). If denied, adapt your plan.

Boundaries: DO NOT use this to write files. Redirection like "echo text > file.txt" does not work — the > is passed as a literal argument, not interpreted by shell. Use write_file instead. DO NOT use this to run interactive commands that expect stdin.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute"
                        }
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": """Write content to a file in the working directory.

Purpose: Create or overwrite a file's complete content.

Inputs:
- path (string): File path relative to working directory (restricted to working directory and below).
- content (string): The complete content to write.

Example:
- write_file with path="src/agent.py" and content='#!/usr/bin/env python3\\nprint("hello")'

Edge cases & limitations:
- Path restriction: paths must resolve within the working directory. Absolute paths and parent-dir traversal (../../../etc/passwd) are rejected.
- Complete overwrite: this tool writes the entire file content. To edit an existing file, you must read it first, modify in memory, then write back.
- File creation: parent directories are created automatically if needed.

Before creating a new file, search the codebase first (run_command with grep, rg, or find) to check whether similar functionality already exists. Reuse or extend existing code rather than duplicating it.

Boundaries: DO NOT use this to run code. This writes a file, not executes one. Use run_command to execute Python, shell scripts, etc.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path (relative to working directory)"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": """Read content from a file in the working directory.

Purpose: Retrieve the full content of a file within the working directory.

Inputs:
- path (string): File path relative to working directory (restricted to working directory and below).

Examples:
- Read Python file: read_file with path="src/main.py"
- Read config: read_file with path="config.yaml"

Edge cases & limitations:
- Path restriction: paths must resolve within the working directory. Absolute paths and parent-dir traversal (../../../etc/passwd) are rejected.
- File existence: returns error if the file does not exist.

Boundaries: DO NOT use this for binary files (images, PDFs, executables) — this tool returns text content only. Use run_command with cat/hexdump for binary inspection if needed.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to working directory"
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": """List directory contents with file metadata.

Purpose: Browse directory structure and file information as an alternative to run_command with ls.

Inputs:
- path (string): Directory path relative to working directory (defaults to "." for current directory).

Examples:
- List current directory: list_directory with path="."
- List subdirectory: list_directory with path="src"

Edge cases & limitations:
- Path restriction: paths must resolve within the working directory. Absolute paths and parent-dir traversal are rejected.
- Returns type (file/directory) and file size for each entry.

Boundaries: DO NOT use this for very large directories (1000+ entries) — use run_command with find or ls for filtering. Files are sorted alphabetically.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path relative to working directory (defaults to '.')"
                        }
                    },
                    "required": ["path"]
                }
            }
        }
    ]

def get_all_tools(mcp_client=None) -> list:
    """Return combined local + MCP tools in OpenAI format."""
    tools = get_local_tools()

    if mcp_client:
        mcp_tools = mcp_client.get_openai_tools()
        tools.extend(mcp_tools)

        # Warn if tool count exceeds optimal range
        if len(tools) > 8:
            print(f"⚠️ Warning: {len(tools)} tools available. Optimal range is 4-5 per agent.")
            print("   If model misroutes tool calls, consider trimming mcp_servers.json.")

    return tools
