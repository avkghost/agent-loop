"""Shell command execution tool with permission escalation for shell features."""

import shlex
import subprocess
from typing import Dict, Any

from .. import config, envelope

# Track approved shell commands for the session
_APPROVED_SHELL_COMMANDS = set()

def _has_shell_operators(command_str: str) -> bool:
    """Check if command uses shell operators (pipes, redirects, logical operators)."""
    shell_operators = ["|", "&&", "||", ">", ">>", "<"]
    return any(op in command_str for op in shell_operators)

def execute_command(command_str: str) -> Dict[str, Any]:
    """Execute a shell command, with permission escalation for shell features."""
    global _APPROVED_SHELL_COMMANDS

    command_str = command_str.strip()

    if not command_str:
        return envelope.error_validation(
            "Error: Empty command.",
            fixable=False,
            description="A command must be provided."
        )

    # Check if command uses shell operators
    uses_shell = _has_shell_operators(command_str)

    # If shell operators detected and not approved, request permission
    if uses_shell and command_str not in _APPROVED_SHELL_COMMANDS:
        if not config.ALLOW_ALL:
            return {
                "requiresPermission": True,
                "kind": "shell_command",
                "identifier": "shell_operators",
                "command": command_str,
                "description": "This command uses shell operators (pipes |, redirects >, logical operators &&/||). Approval required."
            }
        else:
            print(f"⚠️ ALLOW_ALL MODE: Executing shell command with operators")
            _APPROVED_SHELL_COMMANDS.add(command_str)

    # For non-shell commands, parse and check base command
    if not uses_shell:
        try:
            tokens = shlex.split(command_str)
        except ValueError as e:
            return envelope.error_validation(
                f"Error: Invalid command syntax (unbalanced quotes): {str(e)}",
                fixable=True,
                description="Check the command for unbalanced quotes or escaped characters."
            )

        if not tokens:
            return envelope.error_validation(
                "Error: Empty command.",
                fixable=False,
                description="A command must be provided."
            )

        base_cmd = tokens[0]

        # Track if a codebase search command is being used
        if base_cmd in config.SEARCH_COMMANDS:
            config.HAS_SEARCHED_CODEBASE = True

        # Check if command is allowed
        if base_cmd not in config.ALLOWED_TOOLS:
            if config.ALLOW_ALL:
                print(f"⚠️ ALLOW_ALL MODE: Executing unauthorized command: {base_cmd}")
            else:
                # Return permission request sentinel (not an error envelope)
                return {
                    "requiresPermission": True,
                    "kind": "command",
                    "identifier": base_cmd,
                    "arguments": tokens[1:] if len(tokens) > 1 else []
                }

    # Check if command runs in background (ends with &)
    is_background = command_str.rstrip().endswith('&')

    # Execute the command (with shell=True for shell operators, False otherwise)
    try:
        if is_background:
            # Background execution: don't wait for completion
            result = subprocess.Popen(
                command_str if uses_shell else shlex.split(command_str),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=uses_shell
            )
            pid = result.pid
            return envelope.success(
                f"[Background process started - PID: {pid}]\n"
                f"Process is running. Use 'jobs' to list or 'kill {pid}' to stop."
            )
        else:
            # Foreground execution: wait for completion
            result = subprocess.run(
                command_str if uses_shell else shlex.split(command_str),
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute hard timeout (use 'timeout' command for finer control)
                shell=uses_shell
            )
            output = result.stdout + result.stderr

            # Distinguish between "command ran successfully and produced no output" vs error
            if output.strip():
                return envelope.success(output)
            else:
                return envelope.success("[Command completed with no visual output]")
    except subprocess.TimeoutExpired:
        return envelope.error_transient(
            "Error: Command timed out after 15 seconds.",
            description="The command took too long to execute. Try a simpler command or check for infinite loops."
        )
    except Exception as e:
        return envelope.error_transient(
            f"Error: {type(e).__name__}: {str(e)}",
            description="An error occurred while executing the command."
        )

def mark_shell_command_approved(command_str: str) -> None:
    """Mark a shell command as approved for the session."""
    global _APPROVED_SHELL_COMMANDS
    _APPROVED_SHELL_COMMANDS.add(command_str)

def clear_approved_commands() -> None:
    """Clear the approved commands list (useful for testing)."""
    global _APPROVED_SHELL_COMMANDS
    _APPROVED_SHELL_COMMANDS.clear()
