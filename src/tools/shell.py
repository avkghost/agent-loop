"""Foreground shell execution: deadline-driven, bounded, and scoped to its own process group.

Contract:
- Foreground only. An unquoted, unescaped background '&' is rejected, and so are detaching
  commands (nohup, setsid, disown, screen, tmux, at, batch). Quoted or escaped '&', '&&',
  '2>&1', and '&>' are ordinary syntax.
- Unsupported grammar is rejected, not interpreted: unquoted subshells '( )', backticks,
  command substitution, and heredocs. Unbalanced quotes are malformed input.
- Plain commands run without a shell. Commands with shell operators (pipes, redirection,
  &&, ||) run through /bin/sh only after one-time approval.
- stdin is /dev/null, so a command cannot consume the REPL's input.
- Each command runs in its own session (POSIX). Process-group signals are sent only while the
  group leader is unreaped, so its PGID cannot be reused. Group membership is read from the
  process table, not inferred from open pipes.
- Output is captured up to CAPTURE_LIMIT_BYTES per stream. The model and journal see at most
  the first and last MODEL_STREAM_BYTES of each stream.
- POSIX only (Linux and macOS). Other platforms are rejected rather than given weaker guarantees.
- Group membership requires `ps` (procps). Without it, commands are refused before starting.
"""

import asyncio
import atexit
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from .. import config, envelope

DEFAULT_TIMEOUT_SECONDS = 120
MIN_TIMEOUT_SECONDS = 1
MAX_TIMEOUT_SECONDS = 600
GRACE_SECONDS = 5.0             # SIGTERM -> SIGKILL escalation window
KILL_CONFIRM_SECONDS = 5.0      # time to observe the group empty after SIGKILL
CAPTURE_LIMIT_BYTES = 1 << 20   # memory bound per stream
MODEL_STREAM_BYTES = 16 << 10   # head and tail kept per stream for the model and journal
_READ_CHUNK = 64 << 10
_POLL_SECONDS = 0.02

# Track approved shell commands for the session
_APPROVED_SHELL_COMMANDS = set()

_DETACHING_COMMANDS = {"nohup", "setsid", "disown", "screen", "tmux", "at", "batch"}
_DOUBLE_QUOTE_ESCAPES = set('"\\$`\n')

# Leaders whose stop could not be confirmed. They stay referenced, identity held, until reap_unreaped()
# reaps them as verified-empty zombies, or shutdown_retained() runs at exit. Bounded by MAX_RETAINED_LEADERS.
_UNREAPED: List[subprocess.Popen] = []


# ---------------------------------------------------------------------------
# Command syntax
# ---------------------------------------------------------------------------

def _unquoted(command: str) -> List[Tuple[int, str]]:
    """(index, char) for each character outside quotes that is not escaped.

    Follows POSIX sh quoting: single quotes are literal; double quotes honour backslash only
    before $ ` " \\ and newline; outside quotes a backslash escapes the next character.
    Raises ValueError for an unbalanced quote or a trailing backslash.
    """
    chars: List[Tuple[int, str]] = []
    quote = None
    i, n = 0, len(command)
    while i < n:
        c = command[i]
        if quote == "'":
            if c == "'":
                quote = None
        elif quote == '"':
            if c == "\\" and i + 1 < n and command[i + 1] in _DOUBLE_QUOTE_ESCAPES:
                i += 2
                continue
            if c == '"':
                quote = None
        else:
            if c == "\\":
                if i + 1 >= n:
                    raise ValueError("trailing backslash")
                i += 2
                continue
            if c in "'\"":
                quote = c
            else:
                chars.append((i, c))
        i += 1
    if quote is not None:
        raise ValueError("unbalanced quote")
    return chars


def _syntax_rejection(command: str) -> Optional[Dict[str, Any]]:
    """Reject background execution, unsupported grammar, and malformed input before any approval."""
    try:
        chars = _unquoted(command)
    except ValueError as e:
        return envelope.error_validation(f"Error: malformed command ({e}).", fixable=True)

    for _, c in chars:
        if c in "()`":
            return envelope.error_validation(
                "Error: subshells, command substitution, and backticks are not supported. "
                "Use plain commands or pipes.",
                fixable=False,
            )
    for (i, c), (j, d) in zip(chars, chars[1:]):
        if c == "<" and d == "<" and j == i + 1:
            return envelope.error_validation("Error: heredocs are not supported.", fixable=False)

    skip = set()
    for i, c in chars:
        if c != "&" or i in skip:
            continue
        nxt = command[i + 1] if i + 1 < len(command) else ""
        prv = command[i - 1] if i > 0 else ""
        if nxt == "&":
            skip.add(i + 1)  # logical AND
            continue
        if prv in "<>" or nxt == ">":
            continue  # 2>&1, >&2, &>
        return envelope.error_validation(
            "Unmanaged background execution is not supported. Use foreground execution; "
            "managed background jobs will be available through start_job.",
            fixable=False,
        )

    segment_start = 0
    segments = []
    for i, c in chars:
        if c in "|;&\n":
            segments.append(command[segment_start:i])
            segment_start = i + 1
    segments.append(command[segment_start:])
    for segment in segments:
        try:
            words = shlex.split(segment)
        except ValueError:
            continue
        if words and os.path.basename(words[0]) in _DETACHING_COMMANDS:
            return envelope.error_validation(
                f"Error: '{words[0]}' detaches processes and is not supported; use foreground execution.",
                fixable=False,
            )
    return None


def _has_shell_operators(command_str: str) -> bool:
    """Check if command uses shell operators (pipes, redirects, logical operators)."""
    shell_operators = ["|", "&&", "||", ">", ">>", "<"]
    return any(op in command_str for op in shell_operators)


def _timeout_error(timeout_seconds: Any) -> Optional[Dict[str, Any]]:
    if timeout_seconds is None:
        return None
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not (
            MIN_TIMEOUT_SECONDS <= timeout_seconds <= MAX_TIMEOUT_SECONDS):
        return envelope.error_validation(
            f"Error: timeout_seconds must be an integer from {MIN_TIMEOUT_SECONDS} to {MAX_TIMEOUT_SECONDS}.",
            fixable=True,
        )
    return None


# ---------------------------------------------------------------------------
# Process identity and group stop
# ---------------------------------------------------------------------------
# Identity rule: a command's PGID equals its leader's PID. While the leader is unreaped its PID,
# and therefore its PGID, stays reserved, so group signals are safe. Once the leader is reaped,
# by this module or by anything else in the process, the PGID may be reused and is never signalled.
#
# Group membership comes from `ps` (procps). A failed or unavailable listing means "unknown",
# never "empty". Commands are refused up front when `ps` is missing.

RUNNING = "running"
EXITED = "exited"               # zombie: exited but unreaped, so its PGID is still reserved
IDENTITY_LOST = "identity-lost" # reaped by something other than this module; exit status unknown
_IDENTITY_LOST_SENTINEL = "identity-lost"  # Popen.returncode value: Popen then never signals, polls, or warns

_SETTLED = "settled"
_TIMED_OUT = "timed_out"
_CANCELLED = "cancelled"

MAX_RETAINED_LEADERS = 8        # unconfirmed leaders kept for cleanup; new commands are refused past this
SHUTDOWN_SECONDS = 2.0          # best-effort wait for retained groups at interpreter exit

_identity_losses = 0            # leaders reaped outside this module (diagnostics)
_CLEANUP_TASKS: set = set()     # cleanup tasks in flight; referenced so they cannot be dropped

LOST_MESSAGE = ("Command's process identity was lost: another part of this process reaped it, so its exit "
                "status and whether its processes stopped are unknown. Its process group was not signalled "
                "and may still be running. It may have partly run; it was not retried.")


def _signal_group(pgid: int, sig: int) -> bool:
    """Signal a process group we hold the leader of. False if the signal could not be delivered."""
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return True  # nothing left in the group
    except PermissionError:
        return False


def _mark_lost(proc: subprocess.Popen) -> None:
    global _identity_losses
    _identity_losses += 1
    proc.returncode = _IDENTITY_LOST_SENTINEL
    print(f"agent-loop: lost track of a shell process group (leader pid {proc.pid}); "
          "its processes may still be running.", file=sys.stderr)


def _observe_leader(proc: subprocess.Popen) -> str:
    """RUNNING, EXITED, or IDENTITY_LOST. Never reaps: WNOWAIT keeps an exited leader a zombie."""
    if proc.returncode == _IDENTITY_LOST_SENTINEL:
        return IDENTITY_LOST
    if proc.returncode is not None:
        return EXITED  # reaped by this module; callers check returncode before signalling
    try:
        waited = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        _mark_lost(proc)  # someone else reaped it: the PGID is no longer ours to judge
        return IDENTITY_LOST
    return RUNNING if waited is None else EXITED


def _identity_held(proc: subprocess.Popen) -> bool:
    """True while the leader is unreaped and its PGID is therefore ours to signal."""
    return proc.returncode is None and _observe_leader(proc) != IDENTITY_LOST


def _exit_status(proc: subprocess.Popen) -> Optional[int]:
    """Exit code, or -signal, of an exited leader, read without reaping. None if identity is lost."""
    if proc.returncode is not None and proc.returncode != _IDENTITY_LOST_SENTINEL:
        return proc.returncode
    if _observe_leader(proc) != EXITED:
        return None
    try:
        info = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)
    except ChildProcessError:
        _mark_lost(proc)
        return None
    return info.si_status if info.si_code == os.CLD_EXITED else -info.si_status


def _group_members(pgid: int) -> Optional[List[int]]:
    """Non-zombie processes in the group, read from the process table. None if unavailable."""
    try:
        out = subprocess.run(
            ["ps", "-A", "-o", "pid=", "-o", "pgid=", "-o", "stat="],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    members = []
    for line in out.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[1] == str(pgid) and not fields[2].startswith("Z"):
            members.append(int(fields[0]))
    return members


async def _wait_group_empty(proc: subprocess.Popen, seconds: float) -> Optional[bool]:
    """True: group empty and leader an exited zombie. False: still populated at the deadline.
    None: identity or membership cannot be established. A failed check is never treated as empty."""
    deadline = time.monotonic() + seconds
    while True:
        obs = _observe_leader(proc)
        if obs == IDENTITY_LOST:
            return None
        members = await asyncio.to_thread(_group_members, proc.pid)
        if members is None:
            return None
        if not members and obs == EXITED:
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(_POLL_SECONDS)


async def _stop_group(proc: subprocess.Popen, grace: Optional[float] = None,
                      kill_confirm: Optional[float] = None) -> bool:
    """Stop the command's process group. True only when the group is confirmed empty.

    Signals are sent only while the leader's identity is held. A confirmed-empty group's zombie
    leader is reaped here. Otherwise the leader stays unreaped for reap_unreaped(), and nothing is
    claimed.
    """
    if not _identity_held(proc):
        return False
    grace = GRACE_SECONDS if grace is None else grace
    kill_confirm = KILL_CONFIRM_SECONDS if kill_confirm is None else kill_confirm
    pgid = proc.pid
    _signal_group(pgid, signal.SIGTERM)
    state = await _wait_group_empty(proc, grace)
    if state is not True and _identity_held(proc):
        _signal_group(pgid, signal.SIGKILL)
        state = await _wait_group_empty(proc, kill_confirm)
    if state is True:
        proc.wait()  # an exited zombie: returns at once
        return True
    return False


def _retain(proc: subprocess.Popen) -> None:
    """Keep an unreaped leader (identity held) for later cleanup. A live handle is never dropped."""
    if _identity_held(proc) and proc not in _UNREAPED:
        _UNREAPED.append(proc)


def _reap_verified() -> None:
    """Reap retained leaders that are zombies with verifiably empty groups. Never signals."""
    for proc in list(_UNREAPED):
        if not _identity_held(proc):
            _UNREAPED.remove(proc)  # reaped or lost; a loss was already reported
            continue
        if _observe_leader(proc) == EXITED and _group_members(proc.pid) == []:
            proc.wait()
            _UNREAPED.remove(proc)


def reap_unreaped() -> None:
    """Opportunistic cleanup of retained leaders. Synchronous and bounded by MAX_RETAINED_LEADERS.

    Reaps only verified-empty zombies. For a group still populated while its identity is held,
    SIGKILL is retried. Nothing is signalled once identity is lost, and nothing is discarded.
    """
    _reap_verified()
    for proc in list(_UNREAPED):
        if _identity_held(proc):
            _signal_group(proc.pid, signal.SIGKILL)


def shutdown_retained(seconds: float = SHUTDOWN_SECONDS) -> List[int]:
    """Runtime exit: SIGKILL retained groups whose identity is still held, then wait briefly for
    them to be confirmed gone. Returns the pids still unconfirmed; they are left to the OS."""
    for proc in list(_UNREAPED):
        if _identity_held(proc):
            _signal_group(proc.pid, signal.SIGKILL)
    deadline = time.monotonic() + seconds
    while True:
        _reap_verified()
        if not _UNREAPED or time.monotonic() >= deadline:
            break
        time.sleep(_POLL_SECONDS)
    return [proc.pid for proc in _UNREAPED]


def _at_exit() -> None:
    remaining = shutdown_retained()
    if remaining:
        print(f"agent-loop: {len(remaining)} shell process group(s) could not be confirmed stopped "
              f"at exit (leader pids {remaining}).", file=sys.stderr)


atexit.register(_at_exit)


async def _wait_settled(proc: subprocess.Popen, readers, seconds: float) -> str:
    """_SETTLED: leader exited and both output pipes at EOF. _TIMED_OUT: deadline passed.
    IDENTITY_LOST: the leader was reaped elsewhere. Pipe EOF never proves group identity."""
    deadline = time.monotonic() + seconds
    while True:
        obs = _observe_leader(proc)
        if obs == IDENTITY_LOST:
            return IDENTITY_LOST
        if obs == EXITED and all(r.done() for r in readers):
            return _SETTLED
        if time.monotonic() >= deadline:
            return _TIMED_OUT
        await asyncio.sleep(_POLL_SECONDS)


async def _complete(coro):
    """Run cleanup to completion even if the caller is cancelled (or interrupted) meanwhile.

    The cleanup is a separate task, so cancelling the caller cannot interrupt it. Repeated
    cancellation only ends the wait; the cancellation is re-raised after the cleanup has finished.
    """
    task = asyncio.ensure_future(coro)
    _CLEANUP_TASKS.add(task)
    task.add_done_callback(_CLEANUP_TASKS.discard)
    interrupt = None
    while not task.done():
        try:
            await asyncio.wait({task})
        except BaseException as exc:  # cancellation or KeyboardInterrupt: keep waiting, re-raise later
            interrupt = exc
    if interrupt is not None:
        raise interrupt
    return task.result()


# ---------------------------------------------------------------------------
# Output capture
# ---------------------------------------------------------------------------

async def _open_reader(pipe):
    """A StreamReader over a subprocess pipe. The returned transport owns and closes the pipe."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    transport, _ = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), pipe)
    return reader, transport


async def _read_bounded(stream) -> tuple:
    """Drain a pipe to EOF while keeping at most CAPTURE_LIMIT_BYTES (head and tail).

    Draining continues past the limit so the child never blocks on a full pipe.
    """
    half = CAPTURE_LIMIT_BYTES // 2
    head, tail, total = bytearray(), bytearray(), 0
    while True:
        chunk = await stream.read(_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        room = half - len(head)
        if room > 0:
            head += chunk[:room]
            chunk = chunk[room:]
        if chunk:
            tail += chunk
            overflow = len(tail) - half
            if overflow > 0:
                del tail[:overflow]
    return bytes(head), bytes(tail), total


async def _discard(readers, transports) -> None:
    for task in readers:
        if not task.done():
            task.cancel()
    await asyncio.gather(*readers, return_exceptions=True)
    for transport in transports:
        transport.close()


def _stream_text(captured: tuple) -> str:
    """Model-facing text for one stream: everything if small, otherwise its head and tail."""
    head, tail, total = captured
    data = head + tail
    if total == len(data) and len(data) <= 2 * MODEL_STREAM_BYTES:
        return _decode(data)
    return (_decode(data[:MODEL_STREAM_BYTES])
            + f"\n[output truncated: {total} bytes; showing first and last {MODEL_STREAM_BYTES} bytes]\n"
            + _decode(data[-MODEL_STREAM_BYTES:]))


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _format_result(returncode: int, out: tuple, err: tuple) -> str:
    stdout_text = _stream_text(out)
    stderr_text = _stream_text(err)
    body = stdout_text
    if stderr_text:
        body += ("\n" if body and not body.endswith("\n") else "") + "--- stderr ---\n" + stderr_text
    if not body.strip():
        body = "[Command completed with no visual output]"
    if returncode > 0:
        body += f"\n[exit code {returncode}]"
    elif returncode < 0:
        body += f"\n[terminated by signal {-returncode}]"
    return body


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

async def _terminate(proc: subprocess.Popen, readers, transports, outcome: str, timeout: int):
    """Stop the group, collect output, release pipes, and keep any unreaped leader. One unit of work."""
    try:
        if outcome == _CANCELLED:
            await _stop_group(proc)
            return None
        if outcome == IDENTITY_LOST:
            raise envelope.ToolExecutionError(LOST_MESSAGE)
        if outcome == _TIMED_OUT:
            confirmed = await _stop_group(proc)
            if confirmed:
                raise envelope.ToolExecutionError(
                    f"Command timed out after {timeout} seconds; its process group was "
                    "confirmed stopped. Processes that left the group are not tracked and may still run. "
                    "It may have partly run; it was not retried."
                )
            raise envelope.ToolExecutionError(
                f"Command timed out after {timeout} seconds; its termination could not "
                "be confirmed. It may still be running. Not retried."
            )
        # Leader exited and output reached EOF. Read the status before any reap, then sweep leftovers
        status = _exit_status(proc)
        if status is None:
            raise envelope.ToolExecutionError(LOST_MESSAGE)
        swept = await _stop_group(proc)
        text = _format_result(status, readers[0].result(), readers[1].result())
        if not swept:
            text += "\n[warning: could not confirm that the command's process group is empty]"
        return envelope.success(text)
    finally:
        await _discard(readers, transports)
        _retain(proc)



async def _run(spawn: List[str], uses_shell: bool, timeout: int) -> Dict[str, Any]:
    """Run one foreground command. `spawn` is [shell_string] when uses_shell, else argv."""
    reap_unreaped()
    if len(_UNREAPED) >= MAX_RETAINED_LEADERS:
        return envelope.error_validation(
            f"Error: {len(_UNREAPED)} earlier command(s) could not be confirmed stopped, so nothing new was "
            "started. Their processes are still tracked for cleanup.",
            fixable=False,
        )
    if shutil.which("ps") is None:
        return envelope.error_validation(
            "Error: shell execution requires 'ps' (procps) to verify process groups; nothing was started.",
            fixable=False,
        )
    try:
        proc = subprocess.Popen(
            spawn[0] if uses_shell else spawn,
            shell=uses_shell,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # own process group, so its tree can be signalled
        )
    except OSError as e:
        # The program could not be started, so nothing ran
        return envelope.error_validation(f"Error: could not start command: {e}", fixable=False)
    except Exception as e:
        raise envelope.ToolExecutionError(
            f"Command could not be started cleanly ({type(e).__name__}); it may have started. Not retried."
        ) from e

    out_reader, out_transport = await _open_reader(proc.stdout)
    err_reader, err_transport = await _open_reader(proc.stderr)
    transports = (out_transport, err_transport)
    readers = [asyncio.ensure_future(_read_bounded(out_reader)),
               asyncio.ensure_future(_read_bounded(err_reader))]

    try:
        outcome = await _wait_settled(proc, readers, timeout)
    except BaseException as original:
        # Cancellation, KeyboardInterrupt, SystemExit: the cleanup runs to completion, then the original propagates
        try:
            await _complete(_terminate(proc, readers, transports, _CANCELLED, timeout))
        except BaseException:
            pass  # a cancellation that arrived during cleanup is already reflected in `original`'s propagation
        raise original

    return await _complete(_terminate(proc, readers, transports, outcome, timeout))


def validate_command(command_str: str) -> Tuple[Optional[Dict[str, Any]], Optional[List[str]], bool]:
    """Validation shared by foreground commands and managed jobs. Starts nothing.

    Returns (early_result, spawn, uses_shell). early_result is a validation error or a permission
    request to return unchanged; otherwise spawn is the Popen target (argv, or [shell_string]).
    """
    global _APPROVED_SHELL_COMMANDS

    if os.name != "posix":
        return envelope.error_validation(
            "Error: shell execution requires a POSIX system (Linux or macOS).",
            fixable=False,
        ), None, False

    command_str = (command_str or "").strip()
    if not command_str:
        return envelope.error_validation(
            "Error: Empty command.",
            fixable=False,
            description="A command must be provided."
        ), None, False

    rejected = _syntax_rejection(command_str)
    if rejected:
        return rejected, None, False

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
            }, None, False
        else:
            print(f"⚠️ ALLOW_ALL MODE: Executing shell command with operators")
            _APPROVED_SHELL_COMMANDS.add(command_str)

    if uses_shell:
        return None, [command_str], True

    # For non-shell commands, parse and check base command
    try:
        tokens = shlex.split(command_str)
    except ValueError as e:
        return envelope.error_validation(
            f"Error: Invalid command syntax (unbalanced quotes): {str(e)}",
            fixable=True,
            description="Check the command for unbalanced quotes or escaped characters."
        ), None, False

    if not tokens:
        return envelope.error_validation(
            "Error: Empty command.",
            fixable=False,
            description="A command must be provided."
        ), None, False

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
            }, None, False
    return None, tokens, False


async def execute_command_async(command_str: str, timeout_seconds: Optional[int] = None) -> Dict[str, Any]:
    """Run a foreground command without blocking the event loop."""
    if os.name != "posix" or not (command_str or "").strip():
        early, _, _ = validate_command(command_str)
        return early

    timeout_error = _timeout_error(timeout_seconds)
    if timeout_error:
        return timeout_error
    timeout = DEFAULT_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds

    early, spawn, uses_shell = validate_command(command_str)
    if early is not None:
        return early
    return await _run(spawn, uses_shell, timeout)


def execute_command(command_str: str, timeout_seconds: Optional[int] = None) -> Dict[str, Any]:
    """Synchronous entry point for callers outside the agent loop. Never call it from inside one."""
    return asyncio.run(execute_command_async(command_str, timeout_seconds))


def mark_shell_command_approved(command_str: str) -> None:
    """Mark a shell command as approved for the session."""
    global _APPROVED_SHELL_COMMANDS
    _APPROVED_SHELL_COMMANDS.add(command_str)


def clear_approved_commands() -> None:
    """Clear the approved commands list (useful for testing)."""
    global _APPROVED_SHELL_COMMANDS
    _APPROVED_SHELL_COMMANDS.clear()
