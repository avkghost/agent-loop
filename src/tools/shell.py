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
    if is_background:
        # Strip the trailing '&' so it is not passed to the program as an argument
        command_str = command_str.rstrip()[:-1].rstrip()
        uses_shell = _has_shell_operators(command_str)

    # Execute the command (with shell=True for shell operators, False otherwise)
    try:
        if is_background:
            # Background execution: start the process and return without waiting.
            # Output is discarded (DEVNULL) so a full pipe cannot block the process;
            # commands that need output should redirect it to a file themselves.
            result = subprocess.Popen(
                command_str if uses_shell else shlex.split(command_str),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=uses_shell
            )
            pid = result.pid
            return envelope.success(
                f"[Background process started - PID: {pid}]\n"
                f"Started only: completion is not tracked, and the output is discarded "
                f"unless the command redirects it to a file."
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
    except (FileNotFoundError, PermissionError, NotADirectoryError) as e:
        # The program could not be started, so nothing ran and retrying is pointless
        return envelope.error_validation(
            f"Error: could not start command: {e}",
            fixable=False
        )
    except subprocess.TimeoutExpired:
        # The process was stopped, but partial effects are possible, so the outcome is unknown
        raise envelope.ToolExecutionError(
            "Command timed out after 120 seconds and was stopped. It may have partly run; it was not retried."
        )
    except Exception as e:
        raise envelope.ToolExecutionError(
            f"Command failed with {type(e).__name__}: {e}. It may have partly run; it was not retried."
        ) from e

def mark_shell_command_approved(command_str: str) -> None:
    """Mark a shell command as approved for the session."""
    global _APPROVED_SHELL_COMMANDS
    _APPROVED_SHELL_COMMANDS.add(command_str)

def clear_approved_commands() -> None:
    """Clear the approved commands list (useful for testing)."""
    global _APPROVED_SHELL_COMMANDS
    _APPROVED_SHELL_COMMANDS.clear()
