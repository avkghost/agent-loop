"""Managed background jobs: runtime-scoped, session-owned, explicitly controlled.

Design:
- Jobs run on a worker thread with its own event loop (job_runtime.py). The agent loop makes blocking
  model requests on its own loop, so pipe draining, lifetime watchdogs, and termination do not depend on it.
- Each job is owned by one agent session (the session_id bound by agent_loop_async, or a per-run id).
  Every tool checks ownership, and a foreign or unknown id gets the same answer.
- A job lives until it exits, is terminated, its owning session ends, or the runtime shuts down.
  Jobs are never persisted and never restarted.
- Process identity and group stop reuse the B1 primitives in tools/shell.py. Output is drained
  continuously into a bounded buffer (job_output.py).

Job states: STARTING, RUNNING, COMPLETED, FAILED, TERMINATED, UNKNOWN.
- COMPLETED: the leader exited with status 0. FAILED: nonzero exit, or killed by an unrequested signal.
- TERMINATED: a stop was requested (by the model, the lifetime limit, a session end, or shutdown) and the
  process group was confirmed empty.
- UNKNOWN: a stop could not be confirmed, startup failed without confirmed cleanup, or another party
  reaped the leader. Processes may still be running. UNKNOWN is never reported as COMPLETED or FAILED.
  It moves to TERMINATED only when a later check confirms the group is empty.

Module boundaries: this module holds job records, their state machine, process lifecycle, and the
tool adapters. job_output.py holds bounded buffers and cursors. job_runtime.py holds the worker thread
and the runtime lifecycle gate. None of them imports this module.
"""

import asyncio
import atexit
import concurrent.futures
import contextvars
import json
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional

from . import config, envelope
from .authorization import SessionAuthorization
from .job_output import OutputBuffer, READ_CHUNK, decode_cursor, encode_cursor
from .job_runtime import CLOSED, RuntimeGate, RuntimeRefused, WorkerRuntime
from .tools import shell as _shell

STARTING = "starting"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
TERMINATED = "terminated"
UNKNOWN = "unknown"

TERMINAL_STATES = frozenset({COMPLETED, FAILED, TERMINATED})
_ALLOWED_TRANSITIONS = {
    STARTING: {RUNNING, FAILED, UNKNOWN},
    RUNNING: {COMPLETED, FAILED, TERMINATED, UNKNOWN},
    UNKNOWN: {TERMINATED},  # only a later verified stop; never COMPLETED or FAILED
    COMPLETED: set(),
    FAILED: set(),
    TERMINATED: set(),
}

TOOL_NAMES = frozenset({"start_job", "status_job", "collect_job", "terminate_job"})

_JOB_ID = re.compile(r"^job_[0-9a-f]{32}$")
_POLL_SECONDS = 0.02
_DRAIN_SECONDS = 1.0
_JANITOR_SECONDS = 1.0
_MAX_COMMAND_CHARS = 2000
_MAX_DIAGNOSTICS = 50

# Leader observations come from the B1 process-identity primitives
_LEADER_EXITED = _shell.EXITED
_LEADER_LOST = _shell.IDENTITY_LOST

owner_var: contextvars.ContextVar = contextvars.ContextVar("agent_loop_job_owner", default=None)


def _grace() -> Dict[str, float]:
    return {"grace": config.JOB_GRACE_SECONDS, "kill_confirm": config.JOB_KILL_CONFIRM_SECONDS}


# ---------------------------------------------------------------------------
# Job record
# ---------------------------------------------------------------------------

class _Job:
    def __init__(self, owner: str, command: str, lifetime: float):
        self.job_id = "job_" + uuid.uuid4().hex
        self.owner = owner
        self.command = command
        self.proc: Optional[subprocess.Popen] = None  # attached immediately after Popen succeeds
        self.state = STARTING
        self.reason = ""
        self.exit_status: Optional[int] = None
        self.group = "running"  # running | empty | unconfirmed | lost
        self.created = time.monotonic()
        self.deadline = self.created + lifetime
        self.finished: Optional[float] = None
        self.stdout = OutputBuffer(config.JOB_OUTPUT_BUFFER_BYTES)
        self.stderr = OutputBuffer(config.JOB_OUTPUT_BUFFER_BYTES)
        self.readers: List[asyncio.Future] = []
        self.transports: List[object] = []
        self.lock = asyncio.Lock()  # serializes startup, exit, termination, and identity loss
        self.supervisor: Optional[asyncio.Future] = None
        self.watchdog: Optional[asyncio.Future] = None
        self.stop_requested: Optional[str] = None

    def set_state(self, new: str, reason: str) -> None:
        if new not in _ALLOWED_TRANSITIONS[self.state]:
            raise RuntimeError(f"invalid job transition {self.state} -> {new}")
        self.state = new
        self.reason = reason
        if new != RUNNING and self.finished is None:
            self.finished = time.monotonic()


async def _pump(reader: asyncio.StreamReader, stream: OutputBuffer) -> None:
    """Drain one pipe to EOF into its bounded buffer. Never stops early, so the child never blocks."""
    while True:
        chunk = await reader.read(READ_CHUNK)
        if not chunk:
            return
        stream.append(chunk)


# ---------------------------------------------------------------------------
# Manager (lives on the worker loop; its state is touched only there)
# ---------------------------------------------------------------------------

class _Manager:
    def __init__(self):
        self.jobs: Dict[str, _Job] = {}
        self.closing = False
        self.retained: List[_Job] = []  # jobs whose stop is unconfirmed; the leader stays unreaped
        self.tasks: set = set()
        self.diagnostics: deque = deque(maxlen=_MAX_DIAGNOSTICS)

    # -- tracked tasks and their failures -----------------------------------

    def _spawn(self, coro, label: str, job: Optional[_Job] = None) -> asyncio.Future:
        task = asyncio.ensure_future(coro)
        self.tasks.add(task)
        task.add_done_callback(lambda done: self._task_finished(done, label, job))
        return task

    def _task_finished(self, task: asyncio.Future, label: str, job: Optional[_Job]) -> None:
        self.tasks.discard(task)
        if task.cancelled():
            return  # expected: shutdown or a finished job cancels its own helpers
        error = task.exception()  # retrieves the exception, so asyncio never reports it as lost
        if error is None:
            return
        self._diagnose(label, error)
        if job is None:
            return
        if label == "terminate":
            if job.state == RUNNING:
                # The stop itself failed: its outcome is unknown, so never report it as stopped
                job.group = "unconfirmed"
                self._retain(job)
                job.set_state(UNKNOWN, f"termination failed ({type(error).__name__}); processes may still be running")
            return
        if job.state == RUNNING:
            # A critical helper failed (output capture, lifetime enforcement, or leader supervision). The job
            # can no longer be managed safely, so stop it rather than leave it running unmanaged.
            self._spawn(self.terminate(job, f"{label} failed ({type(error).__name__})"), "terminate", job)

    def _diagnose(self, label: str, error: BaseException) -> None:
        entry = {"task": label, "error": type(error).__name__, "message": str(error)[:200]}
        self.diagnostics.append(entry)
        print(f"agent-loop: job task '{label}' failed: {entry['error']}: {entry['message']}", file=sys.stderr)

    def start_background(self) -> None:
        self._spawn(self._janitor_loop(), "janitor")

    async def _janitor_loop(self) -> None:
        """Periodic cleanup independent of agent activity. A failing pass is recorded and the loop continues."""
        while True:
            await asyncio.sleep(_JANITOR_SECONDS)
            try:
                await self._retry_retained()
                self._housekeeping()
            except Exception as e:
                self._diagnose("janitor", e)

    # -- tool operations ----------------------------------------------------

    async def handle(self, tool: str, owner: str, args: Dict[str, Any],
                     authorization: Optional[SessionAuthorization] = None) -> Dict[str, Any]:
        if tool == "start_job":
            return await self._start(owner, args["command"], args.get("lifetime_seconds"), authorization)
        job = self._owned(owner, args["job_id"])
        if job is None:
            return _not_found()
        if tool == "status_job":
            return envelope.success(json.dumps(self._describe(job)))
        if tool == "collect_job":
            return self._collect(job, args.get("cursor"))
        if tool == "terminate_job":
            await self.terminate(job, "terminated on request")
            return envelope.success(json.dumps(self._describe(job)))
        return envelope.error_validation(f"Unknown job tool: {tool}", fixable=False)

    def _owned(self, owner: str, job_id: str) -> Optional[_Job]:
        job = self.jobs.get(job_id)
        return job if job is not None and job.owner == owner else None

    async def _start(self, owner: str, command: str, lifetime: Optional[int],
                     authorization: Optional[SessionAuthorization] = None) -> Dict[str, Any]:
        if self.closing:
            return envelope.error_validation(
                "Managed jobs are unavailable: the runtime is shutting down. No job was started.", fixable=False)
        self._housekeeping()
        active = [j for j in self.jobs.values() if j.state in (STARTING, RUNNING)]
        if len(active) >= config.JOB_MAX_CONCURRENT:
            return envelope.error_validation(
                f"Error: {len(active)} jobs are already running (limit {config.JOB_MAX_CONCURRENT}). "
                "Terminate one first. No job was started.", fixable=True)
        if len(self.retained) >= config.JOB_MAX_RETAINED:
            return envelope.error_validation(
                f"Error: {len(self.retained)} earlier job(s) could not be confirmed stopped, so no new job was "
                "started. They are still tracked for cleanup.", fixable=False)
        early, spawn, uses_shell = _shell.validate_command(command, authorization)
        if early is not None:
            return early
        if shutil.which("ps") is None:
            return envelope.error_validation(
                "Error: managed jobs require 'ps' (procps) to verify process groups; nothing was started.",
                fixable=False)

        job = _Job(owner, command, config.JOB_DEFAULT_LIFETIME_SECONDS if lifetime is None else lifetime)
        try:
            job.proc = subprocess.Popen(
                spawn[0] if uses_shell else spawn,
                shell=uses_shell,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as e:
            return envelope.error_validation(f"Error: job could not be started: {e}. Nothing was started.",
                                             fixable=False)
        except Exception as e:
            return envelope.error_unknown_outcome(
                f"Job start raised {type(e).__name__}; it may have started and is not tracked. Not retried.")

        # Ownership is established here: the process is attached to its job before any await or fallible step
        try:
            self._register(job)
        except Exception as e:
            return await self._abort_startup(job, registered=False, cause=e)

        async with job.lock:  # no finalization can run while the pipes are attached
            try:
                await self._attach_readers(job)
            except Exception as e:
                return await self._abort_startup(job, registered=True, cause=e)
            job.set_state(RUNNING, "running")
            job.supervisor = self._spawn(self._supervise(job), "supervisor", job)
            job.watchdog = self._spawn(self._watchdog(job), "watchdog", job)
        return envelope.success(json.dumps(self._describe(job)))

    def _register(self, job: _Job) -> None:
        self.jobs[job.job_id] = job

    async def _attach_readers(self, job: _Job) -> None:
        out_reader, out_transport = await _shell._open_reader(job.proc.stdout)
        job.transports.append(out_transport)
        err_reader, err_transport = await _shell._open_reader(job.proc.stderr)
        job.transports.append(err_transport)
        job.readers = [self._spawn(_pump(out_reader, job.stdout), "pump", job),
                       self._spawn(_pump(err_reader, job.stderr), "pump", job)]

    async def _abort_startup(self, job: _Job, registered: bool, cause: BaseException) -> Dict[str, Any]:
        """A started process could not be fully attached. Release what exists, then stop its group."""
        await _shell._discard(job.readers, tuple(job.transports))
        for pipe in (job.proc.stdout, job.proc.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()
        name = type(cause).__name__
        confirmed = await _shell._complete(_shell._stop_group(job.proc, **_grace()))
        if confirmed:
            job.group = "empty"
            job.set_state(FAILED, f"startup failed ({name}); its process group was stopped")
            if registered:
                self.jobs.pop(job.job_id, None)  # never expose a partially initialized job
            return envelope.error_validation(
                f"Error: job could not be started ({name}); its process group was stopped. "
                "Nothing is left running.", fixable=False)
        if _shell._observe_leader(job.proc) == _LEADER_LOST:
            job.group = "lost"
            job.set_state(UNKNOWN, f"startup failed ({name}); process identity was lost, so the group is unknown")
        else:
            job.group = "unconfirmed"
            job.set_state(UNKNOWN, f"startup failed ({name}) and its process group could not be confirmed stopped")
            self._retain(job)  # the unreaped leader stays referenced for cleanup
        # The id is returned only when the job is registered: an unregistered job is not authorized to anyone
        payload = {
            "state": UNKNOWN,
            "message": (f"Job startup failed ({name}); its processes could not be confirmed stopped and may still "
                        "be running. Use status_job or terminate_job with job_id to inspect or stop it. "
                        "Not retried."),
        }
        if registered:
            payload["job_id"] = job.job_id
        return envelope.error_unknown_outcome(json.dumps(payload))

    def _collect(self, job: _Job, cursor: Optional[str]) -> Dict[str, Any]:
        so, se = 0, 0
        if cursor is not None:
            position = decode_cursor(cursor, job.job_id)
            if position is None or position[0] > job.stdout.end or position[1] > job.stderr.end:
                return _invalid_cursor()
            so, se = position
        out, so2, out_dropped, out_more = job.stdout.read(so, _shell.MODEL_STREAM_BYTES)
        err, se2, err_dropped, err_more = job.stderr.read(se, _shell.MODEL_STREAM_BYTES)
        payload = self._describe(job)
        payload.update({
            "cursor": encode_cursor(job.job_id, so2, se2),
            "stdout": out.decode("utf-8", errors="replace"),
            "stderr": err.decode("utf-8", errors="replace"),
            "stdout_dropped_bytes": out_dropped,
            "stderr_dropped_bytes": err_dropped,
            "truncated": bool(out_dropped or err_dropped),
            "more": out_more or err_more,
        })
        return envelope.success(json.dumps(payload))

    def _describe(self, job: _Job) -> Dict[str, Any]:
        remaining = max(0.0, job.deadline - time.monotonic()) if job.state in (STARTING, RUNNING) else 0.0
        return {
            "job_id": job.job_id,
            "state": job.state,
            "reason": job.reason,
            "exit_status": job.exit_status,
            "group": job.group,
            "command": job.command,
            "lifetime_remaining_seconds": round(remaining, 1),
            "stdout_bytes_written": job.stdout.end,
            "stderr_bytes_written": job.stderr.end,
        }

    # -- lifecycle ----------------------------------------------------------

    async def _supervise(self, job: _Job) -> None:
        """Watch the leader until it exits or its identity is lost. Termination paths take over from there."""
        while job.state == RUNNING:
            observed = _shell._observe_leader(job.proc)
            if observed == _LEADER_LOST:
                async with job.lock:
                    if job.state == RUNNING:
                        await self._finish_lost(job)
                return
            if observed == _LEADER_EXITED:
                async with job.lock:
                    if job.state == RUNNING:
                        await self._finish_exited(job)
                return
            await asyncio.sleep(_POLL_SECONDS)

    async def _finish_exited(self, job: _Job) -> None:
        status = _shell._exit_status(job.proc)
        if status is None:
            return await self._finish_lost(job)
        # The leader's status is known. Stop any same-group leftovers while the zombie reserves the PGID.
        confirmed = await _shell._stop_group(job.proc, **_grace())
        await self._drain(job)
        job.exit_status = status
        suffix = ""
        if confirmed:
            job.group = "empty"
        elif _shell._observe_leader(job.proc) == _LEADER_LOST:
            job.group = "lost"
            suffix = "; the group state is unknown"
        else:
            job.group = "unconfirmed"
            suffix = "; processes left in its group could not be confirmed stopped"
            self._retain(job)
        if status == 0:
            job.set_state(COMPLETED, "exited with code 0" + suffix)
        elif status > 0:
            job.set_state(FAILED, f"exited with code {status}" + suffix)
        else:
            job.set_state(FAILED, f"terminated by signal {-status} (not requested)" + suffix)
        self._cancel_watchdog(job)

    async def _finish_lost(self, job: _Job) -> None:
        await _shell._discard(job.readers, tuple(job.transports))
        job.group = "lost"
        job.set_state(UNKNOWN, "process identity was lost: another part of the process reaped the leader. "
                               "Its exit status and group state are unknown; the group was not signalled.")
        self._cancel_watchdog(job)

    async def terminate(self, job: _Job, reason: str) -> None:
        async with job.lock:
            if job.state == UNKNOWN and job.proc is not None and _shell._identity_held(job.proc):
                await self._retry_stop(job, reason)  # an earlier stop was unconfirmed: try again, bounded
                return
            if job.state != RUNNING:
                return
            job.stop_requested = reason
            confirmed = await _shell._stop_group(job.proc, **_grace())
            await _shell._discard(job.readers, tuple(job.transports))
            self._cancel_watchdog(job)
            if confirmed:
                job.group = "empty"
                rc = job.proc.returncode
                job.exit_status = rc if isinstance(rc, int) else None
                job.set_state(TERMINATED, f"{reason}; process group confirmed stopped")
            elif _shell._observe_leader(job.proc) == _LEADER_LOST:
                job.group = "lost"
                job.set_state(UNKNOWN, f"{reason}; process identity was lost during termination, so the group "
                                       "may still be running")
            else:
                job.group = "unconfirmed"
                self._retain(job)
                job.set_state(UNKNOWN, f"{reason}; termination could not be confirmed, and its processes may "
                                       "still be running")

    async def _retry_stop(self, job: _Job, reason: str) -> None:
        confirmed = await _shell._complete(_shell._stop_group(job.proc, **_grace()))
        if confirmed:
            job.group = "empty"
            rc = job.proc.returncode
            job.exit_status = rc if isinstance(rc, int) else None
            self._forget(job)
            job.set_state(TERMINATED, f"{reason}; process group confirmed stopped on retry")
        else:
            job.group = "unconfirmed"
            self._retain(job)
            job.reason = f"{job.reason}; {reason}: termination still could not be confirmed"

    async def _watchdog(self, job: _Job) -> None:
        await asyncio.sleep(max(0.0, job.deadline - time.monotonic()))
        await self.terminate(job, "lifetime limit reached")

    def _cancel_watchdog(self, job: _Job) -> None:
        if job.watchdog is not None and job.watchdog is not asyncio.current_task():
            job.watchdog.cancel()

    async def _drain(self, job: _Job) -> None:
        if job.readers:
            await asyncio.wait(job.readers, timeout=_DRAIN_SECONDS)
        await _shell._discard(job.readers, tuple(job.transports))

    # -- retained leaders and retention -------------------------------------

    def _retain(self, job: _Job) -> None:
        if job.proc.returncode is None and job not in self.retained and _shell._identity_held(job.proc):
            self.retained.append(job)

    def _forget(self, job: _Job) -> None:
        if job in self.retained:
            self.retained.remove(job)

    async def _retry_retained(self) -> None:
        """Reap verified-empty zombies, retry SIGKILL for populated groups, and resolve UNKNOWN jobs.

        Membership inspection runs off the worker loop, and identity is re-checked after every await.
        """
        for job in list(self.retained):
            proc = job.proc
            if not _shell._identity_held(proc):
                self._forget(job)  # reaped or lost; a loss was already reported
                continue
            if _shell._observe_leader(proc) == _LEADER_EXITED:
                members = await asyncio.to_thread(_shell._group_members, proc.pid)
                if members == [] and _shell._identity_held(proc):
                    proc.wait()  # a zombie with a verifiably empty group: reaping cannot hit another group
                    self._forget(job)
                    job.group = "empty"
                    if job.state == UNKNOWN and job.stop_requested:
                        job.set_state(TERMINATED, f"{job.stop_requested}; the group was verified empty later")
                    continue
            if _shell._identity_held(proc):
                _shell._signal_group(proc.pid, signal.SIGKILL)

    async def _finalize_retained(self) -> None:
        """Runtime shutdown: one bounded, cancellation-resistant stop attempt for each retained leader."""
        async def finalize(job: _Job) -> None:
            if not _shell._identity_held(job.proc):
                self._forget(job)
                return
            confirmed = await _shell._complete(_shell._stop_group(job.proc, **_grace()))
            if confirmed:
                job.group = "empty"
                self._forget(job)
                if job.state == UNKNOWN:
                    job.set_state(TERMINATED, f"{job.reason}; verified stopped at shutdown")

        await asyncio.gather(*(finalize(job) for job in list(self.retained)), return_exceptions=True)

    def _housekeeping(self) -> None:
        now = time.monotonic()
        for job_id, job in list(self.jobs.items()):
            if job.state in TERMINAL_STATES and now - job.finished > config.JOB_RETENTION_SECONDS:
                del self.jobs[job_id]
        terminal = sorted((j for j in self.jobs.values() if j.state in TERMINAL_STATES), key=lambda j: j.finished)
        for job in terminal[:max(0, len(terminal) - config.JOB_RETENTION_MAX)]:
            del self.jobs[job.job_id]

    # -- shutdown -----------------------------------------------------------

    async def shutdown(self, owner: Optional[str], reason: str) -> List[Dict[str, Any]]:
        """Terminate jobs in scope. owner=None is runtime shutdown: refuse new jobs, stop everything, and
        stop background tasks. Returns diagnostics for jobs whose stop remains unconfirmed."""
        if owner is None:
            self.closing = True
        in_scope = (lambda j: owner is None or j.owner == owner)
        targets = [j for j in self.jobs.values() if in_scope(j) and j.state in (STARTING, RUNNING)]
        if targets:
            await asyncio.gather(*(self.terminate(j, reason) for j in targets), return_exceptions=True)
        if owner is None:
            await self._finalize_retained()
        else:
            await self._retry_retained()
        unresolved = [{"job_id": j.job_id, "state": j.state, "leader_pid": j.proc.pid, "reason": j.reason}
                      for j in self.jobs.values() if j.state == UNKNOWN and in_scope(j)]
        if owner is None:
            tasks = list(self.tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        return unresolved


def _not_found() -> Dict[str, Any]:
    # One answer for unknown, foreign, and malformed ids, so ids cannot be probed
    return envelope.error_validation("No job with that id belongs to this session.", fixable=True)


def _invalid_cursor() -> Dict[str, Any]:
    return envelope.error_validation("Error: cursor is not valid for this job.", fixable=True)


# ---------------------------------------------------------------------------
# Runtime (worker thread plus manager) and its lifecycle
# ---------------------------------------------------------------------------

_STOP_SECONDS = 5.0  # how long a shutdown waits for the worker thread to exit


class _JobRuntime:
    def __init__(self):
        self.worker = WorkerRuntime("agent-loop-jobs")
        self.manager = _Manager()
        self.unresolved: List[Dict[str, Any]] = []
        self.worker.submit(self._start())

    async def _start(self) -> None:
        self.manager.start_background()

    def worker_alive(self) -> bool:
        return self.worker.thread.is_alive()


_GATE = RuntimeGate(_JobRuntime)


def _refusal(error: RuntimeRefused) -> Dict[str, Any]:
    return envelope.error_validation(str(error), fixable=False)


def _validate_input(tool: str, args: Any) -> Optional[Dict[str, Any]]:
    """Explicit input schemas: required and allowed keys, types, and ranges. None when valid."""
    if not isinstance(args, dict):
        return envelope.error_validation(f"Error: {tool} arguments must be an object.", fixable=True)
    required, allowed = {
        "start_job": ({"command"}, {"command", "lifetime_seconds"}),
        "status_job": ({"job_id"}, {"job_id"}),
        "collect_job": ({"job_id"}, {"job_id", "cursor"}),
        "terminate_job": ({"job_id"}, {"job_id"}),
    }[tool]
    unknown = set(args) - allowed
    if unknown:
        return envelope.error_validation(f"Error: {tool} does not accept {sorted(unknown)}.", fixable=True)
    missing = required - set(args)
    if missing:
        return envelope.error_validation(f"Error: {tool} requires {sorted(missing)}.", fixable=True)
    if tool == "start_job":
        command = args["command"]
        if not isinstance(command, str) or not command.strip() or len(command) > _MAX_COMMAND_CHARS:
            return envelope.error_validation(
                f"Error: command must be a non-empty string of at most {_MAX_COMMAND_CHARS} characters.", fixable=True)
        if "lifetime_seconds" in args:
            value = args["lifetime_seconds"]
            if isinstance(value, bool) or not isinstance(value, int) or not (
                    1 <= value <= config.JOB_MAX_LIFETIME_SECONDS):
                return envelope.error_validation(
                    f"Error: lifetime_seconds must be an integer from 1 to {config.JOB_MAX_LIFETIME_SECONDS}.",
                    fixable=True)
    else:
        if not isinstance(args["job_id"], str) or not _JOB_ID.match(args["job_id"]):
            return _not_found()
        if "cursor" in args and (not isinstance(args["cursor"], str) or len(args["cursor"]) > 256):
            return _invalid_cursor()
    return None


def bind_owner(owner: str) -> contextvars.Token:
    """Bind the owning session for the current agent run. Returns a token for unbind_owner."""
    return owner_var.set(owner)


def unbind_owner(token: contextvars.Token) -> None:
    owner_var.reset(token)


async def call_tool(tool_name: str, tool_input: Any,
                    authorization: Optional[SessionAuthorization] = None) -> Dict[str, Any]:
    """Awaitable entry used by the agent loop. The worker does the work; a caller's cancellation only
    stops this caller from waiting. It never interrupts a start, a termination, or its cleanup."""
    owner = owner_var.get()
    if owner is None:
        return envelope.error_validation("Managed jobs are available only inside an agent run.", fixable=False)
    invalid = _validate_input(tool_name, tool_input)
    if invalid is not None:
        return invalid
    try:
        runtime = _GATE.acquire()
    except RuntimeRefused as e:
        return _refusal(e)
    future = runtime.worker.submit(runtime.manager.handle(tool_name, owner, tool_input, authorization))
    if future is None:
        return _refusal(RuntimeRefused("Managed jobs are unavailable: the runtime has shut down."))
    return await asyncio.shield(asyncio.wrap_future(future))


def call_tool_sync(tool_name: str, tool_input: Any,
                   authorization: Optional[SessionAuthorization] = None) -> Dict[str, Any]:
    """Blocking entry for synchronous callers. Never call it from the worker thread."""
    owner = owner_var.get()
    if owner is None:
        return envelope.error_validation("Managed jobs are available only inside an agent run.", fixable=False)
    invalid = _validate_input(tool_name, tool_input)
    if invalid is not None:
        return invalid
    try:
        runtime = _GATE.acquire()
    except RuntimeRefused as e:
        return _refusal(e)
    future = runtime.worker.submit(runtime.manager.handle(tool_name, owner, tool_input, authorization))
    if future is None:
        return _refusal(RuntimeRefused("Managed jobs are unavailable: the runtime has shut down."))
    return future.result()


def _session_timeout_entry(owner: str, timeout: float) -> Dict[str, Any]:
    return {"job_id": None, "state": "session", "leader_pid": None,
            "reason": f"job cleanup for session {owner} did not finish within {timeout} seconds; "
                      "its jobs may still be running"}


def _report_unresolved(unresolved: List[Dict[str, Any]]) -> None:
    for entry in unresolved:
        if entry["job_id"] is None:
            print(f"agent-loop: {entry['reason']}")
        else:
            print(f"agent-loop: job {entry['job_id']} could not be confirmed stopped at session close "
                  f"(leader pid {entry['leader_pid']}); its processes may still be running.")


def close_session_jobs(owner: str, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
    """Terminate the jobs owned by one session. Bounded: a stop that is not finished in time is reported
    as unresolved, and the stop itself keeps running on the worker. Other sessions' jobs are untouched."""
    timeout = config.JOB_SHUTDOWN_TIMEOUT_SECONDS if timeout is None else timeout
    runtime = _GATE.running()
    if runtime is None:
        return []
    future = runtime.worker.submit(runtime.manager.shutdown(owner, "owning session ended"))
    if future is None:
        return []
    try:
        unresolved = list(future.result(timeout=timeout))
    except concurrent.futures.TimeoutError:
        unresolved = [_session_timeout_entry(owner, timeout)]
    _report_unresolved(unresolved)
    return unresolved


async def end_session(owner: str) -> List[Dict[str, Any]]:
    """Terminate the owner's jobs when a one-shot run ends. Bounded, and the stop is always attempted.
    Returns the diagnostics for stops that could not be confirmed."""
    runtime = _GATE.running()
    if runtime is None:
        return []
    future = runtime.worker.submit(runtime.manager.shutdown(owner, "owning session ended"))
    if future is None:
        return []
    timeout = config.JOB_SHUTDOWN_TIMEOUT_SECONDS
    try:
        unresolved = list(await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), timeout))
    except asyncio.TimeoutError:
        unresolved = [_session_timeout_entry(owner, timeout)]
    _report_unresolved(unresolved)
    return unresolved


def runtime_state() -> Optional[str]:
    """The job runtime's state (running, closing, closed), or None before first use."""
    return _GATE.state()


def shutdown_runtime(timeout: Optional[float] = None) -> List[Dict[str, Any]]:
    """Explicit runtime shutdown. Idempotent, bounded, and safe under concurrency.

    One attempt at a time: it refuses new work, terminates running jobs, finalizes retained leaders,
    and stops the worker. The gate becomes CLOSED only when the worker thread has actually stopped. An
    attempt that times out leaves the runtime closing, still owning its worker, and a later call retries.
    Callers that find an attempt in progress wait for it, and report success only if it completed.
    Returns diagnostics for everything that remains unconfirmed.
    """
    timeout = config.JOB_SHUTDOWN_TIMEOUT_SECONDS if timeout is None else timeout
    runtime, performs = _GATE.begin_shutdown()
    if runtime is None:
        return []
    if not performs:
        if not _GATE.wait_attempt(timeout):
            print(f"agent-loop: job shutdown did not finish within {timeout} seconds.")
        if _GATE.state() == CLOSED:
            return list(runtime.unresolved)
        return list(runtime.unresolved) + [_incomplete_entry()]

    completed = False
    manager_done = False
    unresolved: List[Dict[str, Any]] = []
    try:
        future = runtime.worker.submit(runtime.manager.shutdown(None, "runtime shutdown"))
        if future is None:
            manager_done = True  # the worker loop has already stopped; nothing is left to shut down
        else:
            try:
                unresolved = future.result(timeout=timeout)
                manager_done = True
            except concurrent.futures.TimeoutError:
                print(f"agent-loop: job shutdown did not finish within {timeout} seconds; remaining jobs may still "
                      "be running. The runtime stays closed to new work until a later shutdown completes.")
            except Exception as e:
                print(f"agent-loop: job shutdown failed: {type(e).__name__}: {e}")
        if manager_done:
            completed = runtime.worker.stop(timeout=_STOP_SECONDS)
            if not completed:
                print("agent-loop: the job worker thread did not stop; the runtime stays closed to new work.")
    finally:
        if manager_done:
            runtime.unresolved = list(unresolved)
        _GATE.end_shutdown(completed)
    if not completed:
        unresolved = list(unresolved) + [_incomplete_entry()]
    for entry in unresolved:
        if entry["job_id"] is None:
            print(f"agent-loop: {entry['reason']}")
        else:
            print(f"agent-loop: job {entry['job_id']} could not be confirmed stopped at shutdown "
                  f"(leader pid {entry['leader_pid']}); its processes may still be running.")
    return unresolved


def _incomplete_entry() -> Dict[str, Any]:
    return {"job_id": None, "state": "runtime", "leader_pid": None,
            "reason": "shutdown incomplete: the job worker thread has not stopped; new work stays refused"}


def reopen_runtime() -> None:
    """Explicitly permit a new runtime after a completed shutdown. Nothing reopens implicitly."""
    _GATE.reopen()


def _inspect() -> Optional[_JobRuntime]:
    """The current runtime object, in any state. For diagnostics and tests."""
    return _GATE.current()


def _reset_for_tests() -> None:
    shutdown_runtime(timeout=10)
    if _GATE.state() == CLOSED or _GATE.state() is None:
        reopen_runtime()


def _at_exit() -> None:
    try:
        shutdown_runtime()
    except Exception as e:  # never let cleanup raise during interpreter exit
        print(f"agent-loop: job shutdown failed at exit: {e}")


atexit.register(_at_exit)
