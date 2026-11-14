"""Read-only Git inspection: git_status and git_diff.

The model supplies no Git arguments. Each tool runs one fixed command, without a shell, from the workspace root:

  git_status   status --porcelain=v1 -z --branch --untracked-files=normal --no-renames
  git_diff     diff [HEAD] --no-ext-diff --no-textconv --no-color --no-renames [-- <path>]

Properties:
  * the only model-supplied value is an optional path. It goes through _contained_target, and it is passed after
    "--" with --literal-pathspecs. Any other argument is refused;
  * the workspace must be the top level of its Git repository, so nothing outside it is reported on;
  * Git runs against a private copy of the repository index, made in a temporary directory that is removed when the
    inspection ends. Git may refresh its index cache while it reads; that refresh lands in the copy, so the
    repository's own index is never rewritten. No commit, stage, reset, restore or clean is ever issued;
  * user, system and global Git configuration are disabled, so only the repository's own config is read, and
    core.fsmonitor is forced off. No pager runs; external diff and textconv helpers are disabled;
  * bounded: stdout is capped at GIT_MAX_OUTPUT_BYTES (the rest is discarded and reported), and each command is
    stopped after GIT_TIMEOUT_SECONDS;
  * cancellable: the dispatcher sets a cancel event when its caller is cancelled. The poll loop then signals only the
    Git process group it started, reaps the leader, and the temporary index is removed before the caller's
    cancellation completes;
  * no authorization is consulted and none is granted. Git is not on the shell allowlist, so run_command still
    asks for approval before it runs git.
"""

import contextlib
import os
import selectors
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

from .. import envelope
from .files import _contained_target
from .shell import _signal_group

GIT_TIMEOUT_SECONDS = 10
GIT_MAX_OUTPUT_BYTES = 64 * 1024
_STDERR_KEEP_BYTES = 2048
_POLL_SECONDS = 0.05  # how quickly a cancel request is noticed while git runs
_GLOBAL_OPTIONS = ("--no-optional-locks", "--literal-pathspecs", "--no-pager", "-c", "core.fsmonitor=false")
_ACCEPTED_ARGUMENTS = {"git_status": frozenset(), "git_diff": frozenset({"path"})}


class _Run(NamedTuple):
    returncode: Optional[int]  # None when the command was stopped
    stdout: bytes
    truncated: bool            # stdout exceeded GIT_MAX_OUTPUT_BYTES
    stderr: str
    stop: Optional[str]        # None, "timeout", or "cancelled"


def _git_executable() -> Optional[str]:
    return shutil.which("git")


def _git_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    env.update(extra or {})
    return env


def _terminate_and_reap(proc: subprocess.Popen) -> None:
    """Kill the process group this command leads, then reap the leader. The group is ours: it was started in its own
    session, so its id is the leader's pid and no other process shares it."""
    if not _signal_group(proc.pid, signal.SIGKILL):
        proc.kill()
    proc.wait()


def _run_git(args, cwd: Path, executable: str, extra_env: Optional[Dict[str, str]] = None,
             cancel: Optional[threading.Event] = None) -> _Run:
    argv = [executable, *_GLOBAL_OPTIONS, *args]
    proc = subprocess.Popen(argv, cwd=str(cwd), env=_git_env(extra_env), stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ, "out")
    selector.register(proc.stderr, selectors.EVENT_READ, "err")
    out, err = bytearray(), bytearray()
    truncated = False
    stop: Optional[str] = None
    deadline = time.monotonic() + GIT_TIMEOUT_SECONDS
    try:
        while selector.get_map() and not truncated:
            if cancel is not None and cancel.is_set():
                stop = "cancelled"
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stop = "timeout"
                break
            for key, _ in selector.select(min(remaining, _POLL_SECONDS)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                if key.data == "out":
                    out += chunk[: GIT_MAX_OUTPUT_BYTES + 1 - len(out)]
                    if len(out) > GIT_MAX_OUTPUT_BYTES:
                        truncated = True
                        break
                elif len(err) < _STDERR_KEEP_BYTES:
                    err += chunk[: _STDERR_KEEP_BYTES - len(err)]
        returncode: Optional[int] = None
        if stop is None and not truncated:
            while proc.poll() is None:
                if cancel is not None and cancel.is_set():
                    stop = "cancelled"
                    break
                if time.monotonic() >= deadline:
                    stop = "timeout"
                    break
                time.sleep(_POLL_SECONDS)
            else:
                returncode = proc.returncode
        if truncated or stop is not None:
            _terminate_and_reap(proc)
    finally:
        if proc.poll() is None:
            _terminate_and_reap(proc)
        selector.close()
        proc.stdout.close()
        proc.stderr.close()
    stderr_text = bytes(err).decode("utf-8", errors="replace").strip()
    return _Run(returncode, bytes(out[:GIT_MAX_OUTPUT_BYTES]), truncated, stderr_text, stop)


def _stopped_error(run: _Run, what: str) -> Dict[str, Any]:
    if run.stop == "cancelled":
        return envelope.error_transient(f"{what} was cancelled; the Git process was stopped and reaped.",
                                        description="Nothing was changed.")
    return envelope.error_transient(
        f"Error: {what} did not finish within {GIT_TIMEOUT_SECONDS} seconds; the Git process was stopped.",
        description="Nothing was changed. Retry only if the repository is expected to respond quickly.")


def _first_line(text: str) -> str:
    return text.splitlines()[0] if text else "no details"


@contextlib.contextmanager
def _inspection(cancel: Optional[threading.Event] = None):
    """Yield (context, error). context holds the executable, the workspace root and the environment for Git.

    The workspace must be the top level of a repository. Git reads a private copy of the repository index, made in a
    temporary directory that is removed when the block ends, including when the inspection is cancelled.
    """
    if os.name != "posix":
        yield None, envelope.error_validation(
            "Error: Git inspection requires a POSIX system (Linux or macOS).", fixable=False)
        return
    executable = _git_executable()
    if executable is None:
        yield None, envelope.error_validation("Error: git is not installed or not on PATH.", fixable=False)
        return

    cwd = Path.cwd().resolve()
    run = _run_git(["rev-parse", "--show-toplevel"], cwd, executable, cancel=cancel)
    if run.stop:
        yield None, _stopped_error(run, "Locating the Git repository")
        return
    if run.returncode != 0:
        yield None, envelope.error_validation(
            f"Error: the working directory is not a usable Git repository ({_first_line(run.stderr)}).",
            fixable=False,
            description="git_status and git_diff inspect a repository whose top level is the working directory.")
        return
    if Path(run.stdout.decode("utf-8", errors="replace").strip()).resolve() != cwd:
        yield None, envelope.error_validation(
            "Error: the working directory is inside a Git repository but is not its top level. "
            "git_status and git_diff only inspect a repository whose top level is the working directory.",
            fixable=False)
        return

    with tempfile.TemporaryDirectory(prefix="agent-loop-git-") as scratch:
        index_copy = os.path.join(scratch, "index")
        run = _run_git(["rev-parse", "--path-format=absolute", "--git-path", "index"], cwd, executable, cancel=cancel)
        if run.stop:
            yield None, _stopped_error(run, "Locating the repository index")
            return
        if run.returncode != 0:
            yield None, envelope.error_validation(
                f"Error: could not locate the repository index ({_first_line(run.stderr)}).", fixable=False)
            return
        source = Path(run.stdout.decode("utf-8", errors="replace").strip())
        if source.is_file():
            shutil.copyfile(source, index_copy)
        context = {"executable": executable, "cwd": cwd, "env": {"GIT_INDEX_FILE": index_copy}}
        yield context, None


def git_status_safe(cancel: Optional[threading.Event] = None) -> Dict[str, Any]:
    with _inspection(cancel) as (ctx, error):
        if error:
            return error
        run = _run_git(["status", "--porcelain=v1", "-z", "--branch", "--untracked-files=normal", "--no-renames"],
                       ctx["cwd"], ctx["executable"], ctx["env"], cancel)
        if run.stop:
            return _stopped_error(run, "git status")
        if run.truncated:
            return envelope.success(_format_status(run.stdout, truncated=True))
        if run.returncode != 0:
            return envelope.error_validation(f"Error: git status failed ({_first_line(run.stderr)}).", fixable=False)
        return envelope.success(_format_status(run.stdout, truncated=False))


def _format_status(raw: bytes, truncated: bool) -> str:
    records = raw.decode("utf-8", errors="replace").split("\0")
    if truncated:
        records = records[:-1]  # the final record may have been cut short
    branch, entries = None, []
    for record in records:
        if record.startswith("## "):
            branch = record[3:]
        elif len(record) > 3:
            entries.append(f"{record[:2]} {record[3:]}")
    lines = [f"Branch: {branch or '(unknown)'}"]
    if entries:
        lines.append(f"{len(entries)} change(s); XY is index status then worktree status:")
        lines.extend(entries)
    else:
        lines.append("Working tree clean.")
    if truncated:
        lines.append(f"[output truncated at {GIT_MAX_OUTPUT_BYTES} bytes; this listing is incomplete]")
    return "\n".join(lines)


def git_diff_safe(path_str: Optional[str] = None, cancel: Optional[threading.Event] = None) -> Dict[str, Any]:
    with _inspection(cancel) as (ctx, error):
        if error:
            return error
        cwd, executable, env = ctx["cwd"], ctx["executable"], ctx["env"]

        verify = _run_git(["rev-parse", "--verify", "--quiet", "HEAD"], cwd, executable, env, cancel)
        if verify.stop:
            return _stopped_error(verify, "Locating HEAD")
        base = "HEAD" if verify.returncode == 0 else None

        args = ["diff", "--no-ext-diff", "--no-textconv", "--no-color", "--no-renames"]
        if base:
            args.append(base)
        if path_str is not None:
            if not isinstance(path_str, str) or not path_str:
                return envelope.error_validation("Error: path must be a non-empty string.", fixable=True)
            target, path_error = _contained_target(path_str)
            if path_error:
                return path_error
            args += ["--", target.relative_to(cwd).as_posix()]

        run = _run_git(args, cwd, executable, env, cancel)
        if run.stop:
            return _stopped_error(run, "git diff")
        if not run.truncated and run.returncode != 0:
            return envelope.error_validation(f"Error: git diff failed ({_first_line(run.stderr)}).", fixable=False)

        compared = "HEAD (staged and unstaged changes)" if base \
            else "the index (no commits yet, so staged changes are not part of the comparison)"
        header = f"Comparing the working tree with {compared}."
        body = run.stdout.decode("utf-8", errors="replace")
        if run.truncated:
            body += f"\n[output truncated at {GIT_MAX_OUTPUT_BYTES} bytes; the diff is incomplete]"
        elif not body:
            body = "No differences."
        return envelope.success(f"{header}\n{body}")


def call(tool_name: str, tool_input: Dict[str, Any], cancel: Optional[threading.Event] = None) -> Dict[str, Any]:
    """Dispatch entry point. Unexpected arguments are refused, so no Git option can be smuggled in."""
    if tool_name not in _ACCEPTED_ARGUMENTS:
        return envelope.error_validation(f"Unknown Git tool: {tool_name}", fixable=False)
    unexpected = sorted(set(tool_input) - _ACCEPTED_ARGUMENTS[tool_name])
    if unexpected:
        return envelope.error_validation(
            f"Error: {tool_name} does not accept: {', '.join(unexpected)}. Nothing was run.", fixable=True)
    if tool_name == "git_status":
        return git_status_safe(cancel)
    return git_diff_safe(tool_input.get("path"), cancel)
