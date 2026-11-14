"""Runtime, session, and goal ownership.

AgentRuntime  shared infrastructure for its sessions: an event loop, session registration, and coordinated
              shutdown. The managed-job worker is process-wide (jobs.py), so at most one AgentRuntime may be
              live at a time; otherwise one runtime's shutdown could stop another runtime's jobs.
AgentSession  one stable session_id that owns the conversation, the execution journal, run history and
              evidence (session format 5), session-local authorization, session-owned background jobs, and
              a closed state. Goals run one at a time.
AgentRun      one goal execution. It is not a separate object: the run is recorded by loop.py as a RunOutcome
              (runs.py) and saved in the session document, so there is no second persistence model.

Provider lifetime: each run still creates and closes its own provider in loop.agent_loop_async. Reusing one
provider across runs is not done in this phase.
MCP lifetime: each session connects its MCP servers when it opens and closes them when it closes. A session
never shares an MCP connection with another session.
"""

import asyncio
import json
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from . import jobs, loop, runs, session
from .authorization import SessionAuthorization
from .tools.mcp_client import MCPClientManager

MCP_CONFIG_PATH = Path("mcp_servers.json")

_claim_lock = threading.Lock()
_live_runtime: Optional["AgentRuntime"] = None


class SessionError(RuntimeError):
    """Base class for refused session operations."""


class SessionClosed(SessionError):
    """The session (or its runtime) is closed; no new goal can start."""


class SessionBusy(SessionError):
    """A goal is already running in this session. Sessions run goals one at a time."""


class SessionStateUnknown(SessionError):
    """The last run's outcome is not confirmed on disk, so the session refuses new goals."""


class RuntimeShutdownRefused(SessionError):
    """Runtime shutdown was refused before anything was closed. ``busy`` lists the sessions running goals."""

    def __init__(self, message: str, busy: Iterable[str] = ()):
        super().__init__(message)
        self.busy = list(busy)


# Runtime states. Only CLOSED and CLOSED_UNCONFIRMED are final; the event loop and job worker are released
# only when the runtime reaches one of them.
OPEN = "open"
SHUTTING_DOWN = "shutting_down"
CLEANUP_PENDING = "cleanup_pending"
CLOSED = "closed"
CLOSED_UNCONFIRMED = "closed_unconfirmed"
_FINAL_STATES = frozenset({CLOSED, CLOSED_UNCONFIRMED})


def load_mcp_config(path: Path = MCP_CONFIG_PATH) -> Dict[str, Any]:
    """The MCP server configuration file, or an empty configuration when it is absent."""
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ Could not load MCP configuration: {e}")
    return {}


def _diagnostic(reason: str) -> Dict[str, Any]:
    return {"job_id": None, "state": "session", "leader_pid": None, "reason": reason}


def _state_from_saved(data: Dict[str, Any]) -> Dict[str, Any]:
    """Restored state for a saved session, using the same rules as the one-shot resume command."""
    kind = session.validate_session_data(data)
    prior_runs, prior_evidence, legacy_history = session.resume_history(data)
    messages = data.get("messages", [])
    if kind in ("structured", "structured_legacy"):
        objective = data.get("objective")
        summary = data.get("summary")
        execution = data.get("execution", [])
    else:
        objective = next((m.get("content") for m in messages if m.get("role") == "user"), None)
        if not isinstance(objective, str):
            objective = None
        summary = None
        execution = None
    return {
        "goal": data.get("goal", ""),
        "messages": messages,
        "objective": objective,
        "summary": summary,
        "execution": execution,
        "runs": list(prior_runs),
        "evidence": list(prior_evidence),
        "legacy_history": bool(legacy_history),
    }


class AgentRuntime:
    """Process-level owner for sessions. Use as a context manager or call shutdown()."""

    def __init__(self, mcp_config: Optional[Dict[str, Any]] = None,
                 mcp_client_factory: Callable[[], Any] = MCPClientManager):
        global _live_runtime
        with _claim_lock:
            if _live_runtime is not None:
                raise RuntimeError("another AgentRuntime is live; shut it down first, because the job runtime "
                                   "is process-wide")
            # Creating a runtime explicitly permits a new job runtime after a completed shutdown
            if jobs.runtime_state() == jobs.CLOSED:
                jobs.reopen_runtime()
            _live_runtime = self
        self.event_loop = asyncio.new_event_loop()
        self._mcp_config = dict(mcp_config or {})
        self._mcp_factory = mcp_client_factory
        self._sessions: Dict[str, "AgentSession"] = {}
        self._state_lock = threading.Lock()  # held only for short state checks, never across awaits
        self._state = OPEN
        self._unresolved: List[Dict[str, Any]] = []

    def __enter__(self) -> "AgentRuntime":
        return self

    def __exit__(self, *exc_info) -> None:
        self.shutdown()

    @property
    def state(self) -> str:
        return self._state

    @property
    def closed(self) -> bool:
        """True only in a final state. CLEANUP_PENDING is not closed: it still owns resources."""
        return self._state in _FINAL_STATES

    @property
    def accepting(self) -> bool:
        return self._state == OPEN

    def open_session(self, session_id: Optional[str] = None, mcp_client: Any = None,
                     authorization: Optional[SessionAuthorization] = None) -> "AgentSession":
        """Create a new session. A new session never takes over an id that already has saved state."""
        self._require_open()
        sid = session_id or str(uuid.uuid4())
        self._require_unregistered(sid)
        if session.load_session(sid) is not None:
            raise SessionError(f"session {sid} already has saved state; resume it instead of overwriting it")
        return self._attach(sid, None, mcp_client, authorization)

    def resume_session(self, session_id: str, mcp_client: Any = None,
                       authorization: Optional[SessionAuthorization] = None) -> "AgentSession":
        """Reopen saved durable state (formats 4 and 5). Live jobs and earlier grants are not restored."""
        self._require_open()
        self._require_unregistered(session_id)
        data = session.load_session(session_id)
        if data is None:
            raise SessionError(f"no saved session {session_id}")
        return self._attach(session_id, _state_from_saved(data), mcp_client, authorization)

    def sessions(self) -> List["AgentSession"]:
        return list(self._sessions.values())

    def shutdown(self) -> List[Dict[str, Any]]:
        """Shut the runtime down. Returns the diagnostics for anything that could not be confirmed.

        Refusal: if any session is running a goal, raises RuntimeShutdownRefused before anything is closed. The
        runtime stays usable, and the caller can cancel or finish the goal and shut down again.
        Partial failure: a session or job cleanup that is not confirmed leaves the runtime CLEANUP_PENDING, with
        its event loop and job worker still owned. A later shutdown() retries only what remains.
        Idempotent once final.
        """
        global _live_runtime
        with self._state_lock:
            if self._state in _FINAL_STATES:
                return list(self._unresolved)
            if self._state == SHUTTING_DOWN:
                raise RuntimeShutdownRefused("runtime shutdown is already in progress")
            busy = [sid for sid, active in self._sessions.items() if active.busy]
            if busy:
                raise RuntimeShutdownRefused(
                    f"runtime shutdown refused: session(s) {', '.join(busy)} still running a goal. "
                    "Finish or cancel the goal, then shut down again. Nothing was closed.", busy=busy)
            # Committed: no session may start a goal from here on, including one scheduled before this point
            self._state = SHUTTING_DOWN
            for active in self._sessions.values():
                active.begin_closing()

        unresolved: List[Dict[str, Any]] = []
        try:
            for active in list(self._sessions.values()):
                try:
                    unresolved.extend(active.close())
                except Exception as e:
                    reason = f"session {active.session_id} close failed: {type(e).__name__}: {e}"
                    print(f"agent-loop: {reason}")
                    unresolved.append(_diagnostic(reason))
            if not unresolved:
                # Every owned session is confirmed closed, so the shared job runtime can be stopped
                unresolved.extend(jobs.shutdown_runtime())
        except BaseException:
            self._set_state(CLEANUP_PENDING)
            raise

        self._unresolved = list(unresolved)
        job_gate = jobs.runtime_state()
        if unresolved and job_gate != jobs.CLOSED:
            self._set_state(CLEANUP_PENDING)  # retryable: the job worker is still owned
        elif unresolved:
            self._set_state(CLOSED_UNCONFIRMED)  # job stops that a stopped worker cannot retry in this process
        else:
            self._set_state(CLOSED)
        if self._state in _FINAL_STATES:
            self.event_loop.close()
            with _claim_lock:
                if _live_runtime is self:
                    _live_runtime = None
        return list(unresolved)

    def _set_state(self, state: str) -> None:
        with self._state_lock:
            self._state = state

    # internal ---------------------------------------------------------------

    def _require_open(self) -> None:
        if not self.accepting:
            raise SessionClosed(f"the runtime is {self._state}; no session can be opened")

    def _require_unregistered(self, sid: str) -> None:
        if sid in self._sessions:
            raise SessionError(f"session {sid} is already open in this runtime")

    def _attach(self, sid: str, state: Optional[Dict[str, Any]], mcp_client: Any,
                authorization: Optional[SessionAuthorization]) -> "AgentSession":
        if mcp_client is None:
            mcp_client = self._mcp_factory()
            try:
                self.event_loop.run_until_complete(mcp_client.connect_all(self._mcp_config))
            except BaseException:
                try:
                    self.event_loop.run_until_complete(mcp_client.close_all())
                except Exception as close_error:
                    print(f"agent-loop: MCP cleanup after a failed connect also failed: {close_error}")
                raise
        if authorization is None:
            authorization = SessionAuthorization.from_config()
        active = AgentSession(self, sid, mcp_client, authorization, state)
        self._sessions[sid] = active
        return active

    def _forget(self, sid: str) -> None:
        self._sessions.pop(sid, None)


class AgentSession:
    """A stable session that runs goals sequentially. Create through AgentRuntime.open_session()."""

    def __init__(self, runtime: AgentRuntime, session_id: str, mcp_client: Any,
                 authorization: SessionAuthorization, state: Optional[Dict[str, Any]]):
        self._runtime = runtime
        self.session_id = session_id
        self._mcp = mcp_client
        self.authorization = authorization
        self._state = state  # None until the first goal has been saved
        self.goal: Optional[str] = state["goal"] if state else None
        self._busy = False
        self._closing = False     # set when a close or runtime shutdown is committed: no new goal may start
        self._closed = False      # final: every owned resource is confirmed released
        self._jobs_closed = False
        self._mcp_closed = False
        self._durable = True

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def closing(self) -> bool:
        return self._closing

    @property
    def busy(self) -> bool:
        return self._busy

    def begin_closing(self) -> None:
        """Refuse new goals from now on. Called under the runtime's state lock before any cleanup runs."""
        self._closing = True

    @property
    def run_records(self) -> List[Dict[str, Any]]:
        """Persisted run records for this session, in order. Previous outcomes are never rewritten."""
        return list(self._state["runs"]) if self._state else []

    async def run_goal(self, goal: str) -> "runs.RunOutcome":
        """Run one goal in this session. Cancellation interrupts the run, not the session."""
        # Check and mark busy in one step with no await in between, under the state lock
        with self._runtime._state_lock:
            self._check_usable()
            self._busy = True
        continuation = self._state is not None
        session_goal = self.goal if self.goal is not None else goal
        before = len(self._state["runs"]) if self._state else 0
        outcome = None
        try:
            outcome = await loop.agent_loop_async(
                goal, self._mcp, self._runtime.event_loop,
                session_id=self.session_id, owner=self.session_id, keep_jobs=True,
                authorization=self.authorization, session_goal=session_goal, continuation=continuation,
                **self._resume_kwargs())
        finally:
            self._busy = False
            self._reconcile(outcome, before, session_goal)
        return outcome

    def close(self) -> List[Dict[str, Any]]:
        """Close the session: stop its jobs (bounded) and close its MCP connections.

        Returns the diagnostics for anything not yet confirmed. An unconfirmed session stays closing (refusing
        goals) and keeps the resources it still owns, so calling close() again retries only what is pending.
        Confirmed steps are not repeated, so no process is stopped or connection closed twice.
        """
        if self._closed:
            return []
        if self._busy:
            raise SessionBusy("a goal is still running; cancel or finish it before closing the session")
        self._closing = True
        problems: List[Dict[str, Any]] = []
        if not self._jobs_closed:
            try:
                job_problems = jobs.close_session_jobs(self.session_id)
            except Exception as e:
                job_problems = [_diagnostic(f"job cleanup for session {self.session_id} failed: "
                                            f"{type(e).__name__}: {e}")]
                print(f"agent-loop: {job_problems[0]['reason']}")
            if job_problems:
                problems.extend(job_problems)
            else:
                self._jobs_closed = True
        if not self._mcp_closed:
            try:
                self._runtime.event_loop.run_until_complete(self._mcp.close_all())
                self._mcp_closed = True
            except Exception as e:
                reason = f"MCP close for session {self.session_id} failed: {type(e).__name__}: {e}"
                print(f"agent-loop: {reason}")
                problems.append(_diagnostic(reason))
        if problems:
            return problems
        self._closed = True
        self._runtime._forget(self.session_id)
        return []

    # internal ---------------------------------------------------------------

    def _check_usable(self) -> None:
        if self._closed or self._closing:
            raise SessionClosed(f"session {self.session_id} is closed or closing; no new goal can start")
        if not self._runtime.accepting:
            raise SessionClosed(f"session {self.session_id} cannot start a goal: the runtime is "
                                f"{self._runtime.state}")
        if self._busy:
            raise SessionBusy(f"session {self.session_id} is already running a goal")
        if not self._durable:
            raise SessionStateUnknown(
                f"session {self.session_id} refuses new goals: the last run's outcome is not confirmed on disk")

    def _resume_kwargs(self) -> Dict[str, Any]:
        if self._state is None:
            return {}
        state = self._state
        return {
            "resume_messages": state["messages"],
            "resume_objective": state["objective"],
            "resume_summary": state["summary"],
            "resume_execution": state["execution"],
            "resume_runs": state["runs"],
            "resume_evidence": state["evidence"],
            "legacy_history": state["legacy_history"],
        }

    def _reconcile(self, outcome: Optional["runs.RunOutcome"], before: int, session_goal: str) -> None:
        """Reload what is durably saved after a run. Never raises, so it cannot mask the run's exception."""
        try:
            data = session.load_session(self.session_id)
            if data is None:
                raise session.SessionLoadError("missing", "the session file is missing after a run")
            state = _state_from_saved(data)
        except (OSError, session.SessionValidationError) as e:
            self._durable = False
            print(f"agent-loop: session {self.session_id} could not be reloaded after a run: {e}")
            return
        self._state = state
        if self.goal is None:
            self.goal = session_goal
        # Judge the raw saved records. The restored view marks a stale "running" record as interrupted, and
        # that would hide a final save that never reached disk.
        saved_runs = data.get("runs") or []
        last = saved_runs[-1] if saved_runs else None
        confirmed = (
            last is not None
            and last.get("status") != runs.RUNNING
            and len(saved_runs) == before + 1
            and (outcome is None or last.get("run_id") == outcome.run_id)
        )
        if not confirmed:
            self._durable = False
            print(f"agent-loop: session {self.session_id}: the last run's outcome is not confirmed on disk; "
                  "this session will refuse new goals.")
