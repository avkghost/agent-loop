"""Main agent loop and orchestration."""

import asyncio
import json
import requests
from pathlib import Path
from typing import Optional

from . import config, envelope, prompts, dispatch, render
from .tools import get_all_tools
from .tools.mcp_client import MCPClientManager


def _get_single_key_input(prompt: str, valid_keys: list) -> str:
    """Get single key input without requiring Enter (Unix/Linux/macOS)."""
    import sys
    import termios
    import tty

    print(prompt, end='', flush=True)

    try:
        # Save terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            # Set terminal to raw mode (no buffering, no echo)
            tty.setraw(fd)
            # Read one character
            ch = sys.stdin.read(1)

            # Validate input
            if ch in valid_keys:
                print(ch)  # Echo the character
                return ch
            elif ch == '\x03':  # Ctrl+C
                print("\n❌ Interrupted - Denying.")
                return "3"
            else:
                print()  # New line
                return _get_single_key_input(prompt, valid_keys)  # Retry
        finally:
            # Restore terminal settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except (ImportError, termios.error, OSError):
        # Fallback for Windows or non-TTY environments
        return input(prompt).strip() or "3"


def load_project_guidelines() -> Optional[str]:
    """Load project-specific AGENTS.md rules if they exist."""
    agents_md = Path("AGENTS.md")
    if agents_md.exists():
        try:
            with open(agents_md) as f:
                content = f.read()
            return content
        except Exception as e:
            print(f"⚠️  Warning: Could not load AGENTS.md: {e}")
    return None


def get_token_count(text):
    """Simple heuristic: 4 characters ~= 1 token."""
    return len(text) // 4


async def agent_loop_async(user_goal: str, mcp_client: MCPClientManager, event_loop: asyncio.AbstractEventLoop,
                            system_prompt_override: Optional[str] = None, tool_allowlist_override: Optional[list] = None) -> None:
    """Main agent loop (async version for MCP support).

    Args:
        user_goal: The objective for the agent
        mcp_client: MCP client manager for external tools
        event_loop: Asyncio event loop
        system_prompt_override: Optional custom system prompt (overrides default)
        tool_allowlist_override: Optional list of allowed tool names (filters available tools)
    """

    # Use override system prompt if provided, otherwise use default
    system_prompt = system_prompt_override if system_prompt_override else prompts.SYSTEM_PROMPT

    # Load project-specific AGENTS.md rules if they exist
    project_guidelines = load_project_guidelines()
    if project_guidelines:
        print(f"📋 Loaded project guidelines from AGENTS.md")
        system_prompt = f"{project_guidelines}\n\n{system_prompt}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Objective: {user_goal}"}
    ]

    # Merge local and MCP tools
    all_tools = get_all_tools(mcp_client)

    # Filter tools by allowlist if provided
    if tool_allowlist_override:
        tools = [t for t in all_tools if t.get('function', {}).get('name') in tool_allowlist_override]
    else:
        tools = all_tools

    render.print_banner(user_goal, len(tools))

    # Track previous tool call for "No Progress" guardrail
    previous_tool_call = None

    for iteration in range(1, config.MAX_ITERATIONS + 1):
        render.print_iteration_header(iteration, config.MAX_ITERATIONS)

        # --- CONTEXT COMPACTION ---
        current_tokens = sum(get_token_count(msg.get('content', '') or json.dumps(msg)) for msg in messages)

        if current_tokens > (config.MAX_TOKENS * config.COMPACT_THRESHOLD):
            after_tokens = sum(get_token_count(msg.get('content', '') or json.dumps(msg)) for msg in (messages[:1] + messages[-4:]))
            render.print_compaction_notice(current_tokens, after_tokens)
            messages = messages[:1] + messages[-4:]

        # 1. Query LM Studio
        payload = {
            "model": config.MODEL_NAME,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
            "max_tokens": config.REQUEST_MAX_TOKENS,  # Per-request limit; MAX_TOKENS is total conversation budget
            "stream": True  # Enable streaming
        }

        # Try with retries on timeout
        result = None
        last_error = None
        full_content = ""

        for attempt in range(1 + config.LM_STUDIO_RETRIES):
            try:
                if attempt > 0:
                    wait_time = config.LM_STUDIO_RETRY_BASE * (config.LM_STUDIO_RETRY_BACKOFF ** (attempt - 1))
                    render.print_retry_notice(attempt, config.LM_STUDIO_RETRIES, wait_time)
                    import time
                    time.sleep(wait_time)

                response = requests.post(
                    config.LM_STUDIO_URL,
                    json=payload,
                    timeout=(config.LM_STUDIO_CONNECT_TIMEOUT, config.LM_STUDIO_TIMEOUT),
                    stream=True
                )
                response.raise_for_status()

                # Stream the response line by line
                print("\n🤖 Assistant: ", end="", flush=True)
                full_content = ""
                tool_calls_dict = {}  # Merge tool_calls by index
                finish_reason = "stop"

                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8') if isinstance(line, bytes) else line
                        if line.startswith("data: "):
                            chunk_str = line[6:]  # Remove "data: " prefix
                            if chunk_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(chunk_str)
                                # Extract content and tool_calls from streaming response
                                if "choices" in chunk and chunk["choices"]:
                                    choice = chunk["choices"][0]
                                    if "delta" in choice:
                                        delta = choice["delta"]
                                        if "content" in delta and delta["content"]:
                                            content = delta["content"]
                                            full_content += content
                                            print(content, end="", flush=True)
                                        if "tool_calls" in delta and delta["tool_calls"]:
                                            for tc in delta["tool_calls"]:
                                                idx = tc.get("index", 0)
                                                if idx not in tool_calls_dict:
                                                    tool_calls_dict[idx] = tc
                                                else:
                                                    # Merge delta with existing tool_call
                                                    existing = tool_calls_dict[idx]
                                                    if "function" in tc and "function" in existing:
                                                        if "arguments" in tc["function"]:
                                                            existing["function"]["arguments"] += tc["function"]["arguments"]
                                    if "finish_reason" in choice and choice["finish_reason"]:
                                        finish_reason = choice["finish_reason"]
                            except json.JSONDecodeError:
                                continue

                print()  # New line after streaming completes

                # Convert tool_calls dict back to list, sorted by index
                tool_calls = [tool_calls_dict[idx] for idx in sorted(tool_calls_dict.keys())]

                # Build result object with both content and tool_calls
                message = {"content": full_content}
                if tool_calls:
                    message["tool_calls"] = tool_calls

                result = {
                    "choices": [{
                        "message": message,
                        "finish_reason": finish_reason
                    }]
                }
                break  # Success, exit retry loop

            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt == config.LM_STUDIO_RETRIES:
                    render.print_llm_error(f"LM Studio timeout after {attempt + 1} attempts: {str(e)}")
                    break
            except Exception as e:
                last_error = e
                render.print_llm_error(str(e))
                break  # Don't retry on non-timeout errors

        if result is None:
            break

        # Get the assistant's response
        if not result.get("choices") or not result["choices"]:
            render.print_empty_response()
            break

        choice = result["choices"][0]
        finish_reason = choice.get("finish_reason")
        message = choice.get("message", {})

        # Print assistant's response
        if message.get("content"):
            render.print_assistant_message(message['content'])

        # Append assistant message to history
        messages.append({"role": "assistant", "content": message.get("content", "")})

        # 2. Check for completion
        if finish_reason == "stop":
            render.print_completion()
            break

        # 3. Handle tool calls
        if finish_reason == "tool_calls" and message.get("tool_calls"):
            tool_calls = message.get("tool_calls", [])

            for tool_call in tool_calls:
                tool_id = tool_call.get("id")
                tool_name = tool_call.get("function", {}).get("name")

                # Parse tool arguments
                try:
                    arguments_str = tool_call.get("function", {}).get("arguments", "{}")
                    tool_input = json.loads(arguments_str)
                except json.JSONDecodeError as e:
                    tool_input = {}
                    observation = envelope.error_validation(
                        f"Error: Malformed tool arguments: {str(e)}",
                        fixable=False
                    )

                if tool_name and tool_input is not None:
                    observation_str = None  # Reset for this tool call iteration
                    render.print_tool_call(tool_name, tool_input)

                    observation = dispatch.dispatch_tool(tool_name, tool_input, mcp_client, event_loop, allowed_tools=tool_allowlist_override)

                    # --- HANDLE PERMISSION REQUEST ---
                    if isinstance(observation, dict) and observation.get("requiresPermission"):
                        kind = observation.get("kind", "command")
                        identifier = observation.get("identifier", "")

                        # In ALLOW_ALL mode, auto-approve all permissions
                        if config.ALLOW_ALL:
                            render.print_permission_result("✅ ALLOW_ALL mode: Approved.")
                            choice_input = "2"  # Session approval
                        else:
                            # Request permission with single-key input
                            render.print_permission_request(kind, identifier)
                            choice_input = _get_single_key_input("Allow this? (1. Yes, only | 2. Yes, this session | 3. No): ", valid_keys=['1', '2', '3'])

                        if choice_input == "1":
                            render.print_permission_result("✅ Allowed for this use only.")
                            # Re-execute without permission gate
                            if kind in ["command", "shell_command", "mcp_tool"]:
                                # Mark shell commands as approved for this attempt
                                if kind == "shell_command":
                                    from .tools.shell import mark_shell_command_approved
                                    mark_shell_command_approved(observation.get("command", ""))
                                observation = dispatch.dispatch_tool(tool_name, tool_input, mcp_client, event_loop, allowed_tools=tool_allowlist_override)
                            # Convert envelope dict to JSON string for the model
                            if isinstance(observation, dict) and "isError" in observation:
                                observation_str = envelope.serialize_for_model(observation)
                            else:
                                observation_str = str(observation)
                            render.print_observation(observation_str)

                        elif choice_input == "2":
                            render.print_permission_result("✅ Added to allowed list for this session.")
                            if kind == "command":
                                config.ALLOWED_TOOLS.add(identifier)
                            elif kind == "mcp_tool":
                                config.ALLOWED_MCP_TOOLS.add(identifier)
                            elif kind == "shell_command":
                                from .tools.shell import mark_shell_command_approved
                                mark_shell_command_approved(observation.get("command", ""))

                            # Re-execute now that it's in the allowed list
                            observation = dispatch.dispatch_tool(tool_name, tool_input, mcp_client, event_loop, allowed_tools=tool_allowlist_override)
                            # Convert envelope dict to JSON string for the model
                            if isinstance(observation, dict) and "isError" in observation:
                                observation_str = envelope.serialize_for_model(observation)
                            else:
                                observation_str = str(observation)
                            render.print_observation(observation_str)

                        else:
                            render.print_permission_result("❌ Denied.")
                            observation = envelope.error_permission("Command/tool denied by user.")

                    # Convert envelope dict to JSON string for the model (if not already converted above)
                    if observation_str is None:
                        if isinstance(observation, dict) and "isError" in observation:
                            observation_str = envelope.serialize_for_model(observation)
                        else:
                            observation_str = str(observation)
                        # Only render if not already rendered during permission handling
                        render.print_observation(observation_str)

                    # --- GUARDRAIL: No Progress Detection ---
                    current_tool_call = (tool_name, str(tool_input), observation_str[:100])
                    if current_tool_call == previous_tool_call:
                        render.print_guardrail_stop()
                        break
                    previous_tool_call = current_tool_call

                    # Append tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": observation_str
                    })

        elif finish_reason != "stop":
            render.print_finish_reason_warning(finish_reason)
            break
    else:
        render.print_max_iterations_reached()


def agent_loop(user_goal: str, system_prompt_override: Optional[str] = None, tool_allowlist_override: Optional[list] = None) -> None:
    """Synchronous entry point that sets up async context.

    Args:
        user_goal: The objective for the agent
        system_prompt_override: Optional custom system prompt (overrides default)
        tool_allowlist_override: Optional list of allowed tool names (filters available tools)
    """
    # Initialize MCP client manager
    mcp_client = MCPClientManager()

    # Load MCP server configuration
    try:
        import json
        from pathlib import Path
        config_path = Path("mcp_servers.json")
        if config_path.exists():
            with open(config_path) as f:
                mcp_config = json.load(f)
        else:
            mcp_config = {}
    except Exception as e:
        print(f"⚠️ Could not load MCP configuration: {e}")
        mcp_config = {}

    # Set up event loop and connect to MCP servers
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)

    try:
        # Connect to MCP servers
        event_loop.run_until_complete(mcp_client.connect_all(mcp_config))

        # Run the agent loop
        event_loop.run_until_complete(agent_loop_async(user_goal, mcp_client, event_loop, system_prompt_override, tool_allowlist_override))

    finally:
        # Close MCP connections
        event_loop.run_until_complete(mcp_client.close_all())
        event_loop.close()
