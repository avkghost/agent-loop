"""Main agent loop and orchestration."""

import asyncio
import json
import uuid
import requests
from pathlib import Path
from typing import Optional

from . import config, envelope, prompts, dispatch, render, compaction, session, execution, jobs, stream
from .tools import get_all_tools
from .tools.mcp_client import MCPClientManager

INTERRUPTED_RESULT = execution.INTERRUPTED_RESULT


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


def check_goal(goal: str) -> str:
    """Validate the goal without assigning structural meaning to its text."""
    return goal


def _declare_tool_call(tool_call):
    """Normalize a streamed tool call into the declaration shape sent back to the provider."""
    function = tool_call.get("function", {})
    return {
        "id": tool_call.get("id"),
        "type": "function",
        "function": {
            "name": function.get("name"),
            "arguments": function.get("arguments", "{}"),
        },
    }


def _validate_tool_call_batch(tool_calls, journal):
    """Validate provider call identifiers before mutating history or journal state."""
    seen = set()
    for tool_call in tool_calls:
        call_id = tool_call.get("id")
        if not isinstance(call_id, str) or not call_id:
            return False, "Model returned a tool call without a valid tool_call_id."
        if call_id in seen or journal.get(call_id) is not None:
            return False, f"Model returned duplicate tool_call_id: {call_id}"
        seen.add(call_id)
    return True, None


def _answer_interrupted_calls(messages):
    """On resume, answer trailing tool calls that have no recorded result.

    They are never re-executed: the tool may already have run before the interruption.
    """
    last = None
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            last = i
    if last is None:
        return messages
    answered = {m.get("tool_call_id") for m in messages[last + 1:] if m.get("role") == "tool"}
    for tc in messages[last]["tool_calls"]:
        if tc.get("id") not in answered:
            messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": INTERRUPTED_RESULT})
    return messages


async def _dispatch_safely(tool_name, tool_input, mcp_client, event_loop, allowed_tools):
    """Dispatch one call inside the running loop. An explicit tool failure becomes a terminal result.

    Only ToolExecutionError is converted. Other exceptions are programming defects and propagate.
    BaseException (KeyboardInterrupt, SystemExit, asyncio cancellation) propagates too. The tool is never
    retried, because it may already have had a side effect. event_loop is accepted for compatibility;
    the await runs on the loop that is already active.
    """
    try:
        return await dispatch.dispatch_tool_async(tool_name, tool_input, mcp_client, event_loop, allowed_tools=allowed_tools)
    except dispatch.ToolExecutionError as e:
        return envelope.error_unknown_outcome(
            f"Tool '{tool_name}' raised {type(e).__name__}: {e}. Its outcome is unknown and it was not retried. "
            "Verify its effects before relying on it."
        )


def _serialize_observation(observation):
    """Convert a tool observation into the string sent back to the model."""
    if isinstance(observation, dict) and "isError" in observation:
        return envelope.serialize_for_model(observation)
    return str(observation)


def _recover_execution_messages(messages, journal):
    """Materialize persisted terminal outcomes without redispatching calls."""
    declared = {
        tool_call.get("id")
        for message in messages
        if message.get("role") == "assistant"
        for tool_call in message.get("tool_calls", [])
    }
    recorded = {
        message.get("tool_call_id")
        for message in messages
        if message.get("role") == "tool"
    }
    for record in journal.records:
        if record.status in (
            execution.ToolCallStatus.COMPLETED,
            execution.ToolCallStatus.FAILED,
            execution.ToolCallStatus.UNKNOWN,
        ) and record.tool_call_id in declared and record.tool_call_id not in recorded:
            messages.append({
                "role": "tool",
                "tool_call_id": record.tool_call_id,
                "content": record.outcome,
            })
    return messages


async def agent_loop_async(user_goal: str, mcp_client: MCPClientManager, event_loop: asyncio.AbstractEventLoop,
                            system_prompt_override: Optional[str] = None, tool_allowlist_override: Optional[list] = None,
                            session_id: Optional[str] = None, resume_messages: Optional[list] = None,
                            resume_objective: Optional[str] = None, resume_summary: Optional[str] = None,
                            resume_execution: Optional[list] = None) -> None:
    """Run one agent session. Managed jobs started here belong to this session and are terminated when it ends."""
    owner = session_id or f"run-{uuid.uuid4().hex}"
    token = jobs.bind_owner(owner)
    try:
        await _agent_loop_run(user_goal, mcp_client, event_loop, system_prompt_override, tool_allowlist_override,
                              session_id=session_id, resume_messages=resume_messages,
                              resume_objective=resume_objective, resume_summary=resume_summary,
                              resume_execution=resume_execution)
    finally:
        try:
            await jobs.end_session(owner)
        finally:
            jobs.unbind_owner(token)


async def _agent_loop_run(user_goal: str, mcp_client: MCPClientManager, event_loop: asyncio.AbstractEventLoop,
                            system_prompt_override: Optional[str] = None, tool_allowlist_override: Optional[list] = None,
                            session_id: Optional[str] = None, resume_messages: Optional[list] = None,
                            resume_objective: Optional[str] = None, resume_summary: Optional[str] = None,
                            resume_execution: Optional[list] = None) -> None:
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

    check_goal(user_goal)
    if resume_objective is not None:
        objective = resume_objective
    elif resume_messages:
        objective = next(
            (m.get("content") for m in resume_messages if m.get("role") == "user"),
            f"Objective: {user_goal}",
        )
    else:
        objective = f"Objective: {user_goal}"
    state = compaction.CompactionState(objective=objective, summary=resume_summary)
    journal = execution.ExecutionJournal.from_dicts(
        resume_execution if resume_execution is not None else []
    )
    if resume_messages is not None:
        messages = compaction.restore_structured_state(list(resume_messages), state)
        journal.recover_unresolved(INTERRUPTED_RESULT)
        messages = _recover_execution_messages(messages, journal)
        messages = _answer_interrupted_calls(messages)
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Objective: {user_goal}"}
        ]

    def _persist():
        """Save the history after every appended message, so an interruption loses nothing recorded."""
        if session_id is None:
            return True
        try:
            session.save_session(
                session_id, user_goal, messages,
                objective=state.objective,
                summary=state.summary,
                execution=journal.to_dicts(),
            )
            return True
        except OSError as e:
            print(f"⚠️  Could not save session {session_id}: {e}. Durability is unknown; execution will stop without retrying.")
            return False

    if not _persist():
        return

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
            # Summarize the middle of the history; compact_messages falls back to hard truncation on failure
            try:
                messages = await compaction.compact_messages(
                    messages, event_loop, mcp_client, state=state
                )
                if not _persist():
                    return
            except compaction.CannotCompactError as e:
                print(f"\n❌ {e} Stopping: the run cannot continue within the context budget.")
                break
            after_tokens = sum(get_token_count(msg.get('content', '') or json.dumps(msg)) for msg in messages)
            render.print_compaction_notice(current_tokens, after_tokens)

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

        # Try with retries on timeout. A turn is complete only if its stream says so (see src/stream.py).
        # Retries happen only before the turn's tool calls are dispatched, which is always the case here.
        result = None
        last_error = None

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
                try:
                    response.raise_for_status()
                    print("\n🤖 Assistant: ", end="", flush=True)
                    parsed = stream.parse_events(
                        response.iter_lines(), on_text=lambda text: print(text, end="", flush=True))
                finally:
                    response.close()

                print()  # New line after streaming completes
                result = stream.to_completion(parsed)
                break  # Complete, well-formed turn

            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt == config.LM_STUDIO_RETRIES:
                    render.print_llm_error(f"LM Studio timeout after {attempt + 1} attempts: {str(e)}")
                    break
            except stream.StreamError as e:
                # Incomplete or malformed: the turn is discarded, nothing is appended, and nothing is dispatched
                last_error = e
                render.print_llm_error(
                    f"Model stream {e.category}: {e}. The turn was discarded and no tool was run. "
                    "The run stops without claiming completion.")
                break
            except requests.exceptions.RequestException as e:
                # Transport failure, including a connection that drops mid-stream
                last_error = e
                render.print_llm_error(
                    f"Transport error during the model stream: {e}. The turn was discarded and no tool was run.")
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

        if finish_reason == stream.FINISH_LENGTH:
            # Complete on the wire but cut off by the length limit: not a finished answer, so it is not kept
            render.print_llm_error(
                "Model response was cut off by the length limit (finish_reason=length). The turn was discarded "
                "and no tool was run. The run stops without claiming completion.")
            break

        if message.get("tool_calls"):
            valid_calls, call_error = _validate_tool_call_batch(message["tool_calls"], journal)
            if not valid_calls:
                render.print_llm_error(call_error)
                break

        # Print assistant's response
        if message.get("content"):
            render.print_assistant_message(message['content'])

        # Append assistant message to history
        assistant_msg = {"role": "assistant", "content": message.get("content", "")}
        if message.get("tool_calls"):
            # Keep the declarations so tool results can be paired with them (OpenAI protocol)
            assistant_msg["tool_calls"] = [_declare_tool_call(tc) for tc in message["tool_calls"]]
        messages.append(assistant_msg)
        if message.get("tool_calls"):
            for declared_call in assistant_msg["tool_calls"]:
                function = declared_call.get("function", {})
                journal.register(
                    declared_call.get("id"),
                    function.get("name") or "",
                    function.get("arguments", "{}"),
                )
        if not _persist():
            return

        # 2. Check for completion
        if finish_reason == "stop":
            render.print_completion()
            break

        # 3. Handle tool calls
        if finish_reason == "tool_calls" and message.get("tool_calls"):
            tool_calls = message.get("tool_calls", [])

            for tool_index, tool_call in enumerate(tool_calls):
                tool_id = tool_call.get("id")
                tool_name = tool_call.get("function", {}).get("name")

                # Parse tool arguments
                parse_error = None
                try:
                    arguments_str = tool_call.get("function", {}).get("arguments", "{}")
                    tool_input = json.loads(arguments_str)
                except json.JSONDecodeError as e:
                    tool_input = None
                    parse_error = envelope.error_validation(
                        f"Error: Malformed tool arguments: {str(e)}",
                        fixable=False
                    )
                if parse_error is None and not isinstance(tool_input, dict):
                    # Valid JSON that is not an object (null, [], 1, "x") cannot be tool arguments
                    tool_input = None
                    parse_error = envelope.error_validation(
                        "Error: Tool arguments must be a JSON object", fixable=False)

                # Invalid or nameless calls are answered with an error and never dispatched
                if parse_error is not None or not tool_name:
                    observation = parse_error or envelope.error_validation(
                        "Error: Tool call has no function name", fixable=False)
                    observation_str = _serialize_observation(observation)
                    render.print_observation(observation_str)
                    journal.transition(tool_id, execution.ToolCallStatus.FAILED, observation_str, "validation")
                    messages.append({"role": "tool", "tool_call_id": tool_id, "content": observation_str})
                    if not _persist():
                        return
                    continue

                if tool_name and tool_input is not None:
                    observation_str = None  # Reset for this tool call iteration
                    render.print_tool_call(tool_name, tool_input)

                    try:
                        journal.transition(tool_id, execution.ToolCallStatus.RUNNING)
                    except execution.ExecutionStateError:
                        raise
                    if not _persist():
                        return

                    observation = await _dispatch_safely(tool_name, tool_input, mcp_client, event_loop, tool_allowlist_override)

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
                                observation = await _dispatch_safely(tool_name, tool_input, mcp_client, event_loop, tool_allowlist_override)
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
                            observation = await _dispatch_safely(tool_name, tool_input, mcp_client, event_loop, tool_allowlist_override)
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
                    is_repeat = current_tool_call == previous_tool_call
                    previous_tool_call = current_tool_call

                    # Append tool result to messages (every dispatched call gets one)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": observation_str
                    })
                    if isinstance(observation, dict) and observation.get("isError"):
                        category = observation.get("errorCategory")
                        terminal_status = (
                            execution.ToolCallStatus.UNKNOWN
                            if category in ("unknown_outcome", "transient")
                            else execution.ToolCallStatus.FAILED
                        )
                        journal.transition(tool_id, terminal_status, observation_str, category)
                    else:
                        journal.transition(tool_id, execution.ToolCallStatus.COMPLETED, observation_str)
                    if not _persist():
                        return

                    if is_repeat:
                        render.print_guardrail_stop()
                        # Declared calls that will not run still need a result to keep the group valid
                        for skipped in tool_calls[tool_index + 1:]:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": skipped.get("id"),
                                "content": execution.SKIPPED_RESULT
                            })
                            journal.transition(
                                skipped.get("id"), execution.ToolCallStatus.FAILED,
                                execution.SKIPPED_RESULT,
                                "guardrail",
                            )
                        if not _persist():
                            return
                        break

        elif finish_reason != "stop":
            render.print_finish_reason_warning(finish_reason)
            break
    else:
        render.print_max_iterations_reached()


def agent_loop(user_goal: str, system_prompt_override: Optional[str] = None, tool_allowlist_override: Optional[list] = None,
               session_id: Optional[str] = None, resume_messages: Optional[list] = None,
               resume_objective: Optional[str] = None, resume_summary: Optional[str] = None,
               resume_execution: Optional[list] = None) -> None:
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
        event_loop.run_until_complete(agent_loop_async(user_goal, mcp_client, event_loop, system_prompt_override,
                                                       tool_allowlist_override, session_id=session_id,
                                                       resume_messages=resume_messages,
                                                       resume_objective=resume_objective,
                                                       resume_summary=resume_summary,
                                                       resume_execution=resume_execution))

    finally:
        # Close MCP connections
        event_loop.run_until_complete(mcp_client.close_all())
        event_loop.close()
