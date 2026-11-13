"""Session persistence for resuming interrupted runs."""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from .compaction import SUMMARY_MARKER, RECENT_USER_MARKER
from .execution import ExecutionJournal, ExecutionStateError, ToolCallStatus, INTERRUPTED_RESULT, SKIPPED_RESULT


SESSIONS_DIR = Path.home() / ".agent_loop" / "sessions"
# Format 4 persists objective, replaceable summary, and durable execution records independently.
SESSION_FORMAT = 4


class SessionValidationError(ValueError):
    """The persisted session cannot be resumed safely."""


class SessionLoadError(OSError):
    """A session file exists but cannot be read or decoded safely."""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def validate_session_data(data: Dict[str, Any]) -> str:
    """Validate a session and return ``structured`` or ``legacy``."""
    if not isinstance(data, dict):
        raise SessionValidationError("session must be a JSON object")

    stored_format = data.get("format")
    messages = data.get("messages")
    if not isinstance(messages, list) or any(
        not isinstance(message, dict) or not isinstance(message.get("role"), str)
        for message in messages
    ):
        raise SessionValidationError("session messages must be a list of role-bearing objects")

    if stored_format == SESSION_FORMAT:
        required = ("session_id", "goal", "objective", "summary", "messages", "execution", "saved_at")
        missing = [field for field in required if field not in data]
        if missing:
            raise SessionValidationError(
                f"format 4 is missing required field(s): {', '.join(missing)}"
            )
        if not isinstance(data.get("session_id"), str) or not isinstance(data.get("goal"), str):
            raise SessionValidationError("format 4 requires string session_id and goal")
        if not isinstance(data.get("saved_at"), str):
            raise SessionValidationError("format 4 requires a string saved_at")
        if not isinstance(data.get("objective"), str):
            raise SessionValidationError("format 4 requires a string objective")
        if not messages or messages[0].get("role") != "system":
            raise SessionValidationError("format 4 messages must begin with a system message")
        if data.get("summary") is not None and not isinstance(data.get("summary"), str):
            raise SessionValidationError("format 4 summary must be a string or null")
        try:
            ExecutionJournal.from_dicts(data.get("execution"))
        except ExecutionStateError as exc:
            raise SessionValidationError(f"invalid execution state: {exc}") from exc
        _validate_execution_messages(data)
        return "structured"

    if stored_format == 3:
        # Format 3 has structured compaction state but no durable execution journal.
        required = ("session_id", "goal", "objective", "summary", "messages", "saved_at")
        missing = [field for field in required if field not in data]
        if missing:
            raise SessionValidationError(
                f"format 3 is missing required field(s): {', '.join(missing)}"
            )
        if not isinstance(data.get("objective"), str) or data.get("summary") is not None and not isinstance(data.get("summary"), str):
            raise SessionValidationError("format 3 has invalid structured state")
        if not messages or messages[0].get("role") != "system":
            raise SessionValidationError("format 3 messages must begin with a system message")
        return "structured_legacy"

    if stored_format not in (None, 2):
        raise SessionValidationError(f"unsupported session format: {stored_format!r}")

    # Marker-bearing legacy histories cannot be separated safely. They are rejected,
    # never migrated by guessing at arbitrary user or model text.
    if any(
        marker in str(message.get("content") or "")
        for message in messages
        for marker in (SUMMARY_MARKER, RECENT_USER_MARKER)
    ):
        raise SessionValidationError(
            "legacy session contains ambiguous compaction markers and cannot be resumed safely"
        )
    return "legacy"


def _ensure_dir():
    """Ensure sessions directory exists."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _is_error_envelope(content: Any) -> bool:
    """True when a stored tool result is a serialized error envelope (isError: true)."""
    if not isinstance(content, str):
        return False
    try:
        parsed = json.loads(content)
    except ValueError:
        return False
    return isinstance(parsed, dict) and parsed.get("isError") is True


def _is_error_result(content: Any) -> bool:
    """True for results the runtime writes for non-completed calls.

    These are serialized error envelopes, the interruption text, or the guardrail skip text.
    Anything else on a FAILED or UNKNOWN record is inconsistent with that status.
    """
    return _is_error_envelope(content) or content in (INTERRUPTED_RESULT, SKIPPED_RESULT)


def _validate_execution_messages(data: Dict[str, Any]) -> None:
    """Validate retained tool groups against the authoritative execution journal."""
    journal = ExecutionJournal.from_dicts(data.get("execution"))
    declared = {}
    results = {}
    for message in data.get("messages", []):
        if message.get("role") == "assistant" and "tool_calls" in message:
            calls = message.get("tool_calls")
            if not isinstance(calls, list):
                raise SessionValidationError("format 4 assistant tool_calls must be a list")
            for call in calls:
                if not isinstance(call, dict):
                    raise SessionValidationError("format 4 assistant tool_call must be an object")
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id or call_id in declared:
                    raise SessionValidationError("format 4 contains duplicate or invalid assistant tool_call_id")
                if "type" in call and call.get("type") != "function":
                    raise SessionValidationError(f"format 4 tool_call {call_id} has invalid type")
                function = call.get("function")
                if not isinstance(function, dict):
                    raise SessionValidationError(f"format 4 tool_call {call_id} requires a function object")
                if not isinstance(function.get("name"), str) or not function.get("name"):
                    raise SessionValidationError(f"format 4 tool_call {call_id} requires a function name")
                if not isinstance(function.get("arguments"), str):
                    raise SessionValidationError(f"format 4 tool_call {call_id} requires string function arguments")
                declared[call_id] = call
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id in results:
                raise SessionValidationError(f"format 4 contains duplicate tool result for {call_id}")
            results[call_id] = message.get("content")

    for call_id, content in results.items():
        record = journal.get(call_id)
        if record is None:
            raise SessionValidationError(f"format 4 tool result has no execution record: {call_id}")
        if call_id not in declared:
            raise SessionValidationError(f"format 4 tool result has no retained assistant declaration: {call_id}")
        if record.status in (ToolCallStatus.PENDING, ToolCallStatus.RUNNING):
            raise SessionValidationError(
                f"format 4 active call {call_id} ({record.status.value}) already has a tool result"
            )
        if record.outcome is not None and content != record.outcome:
            raise SessionValidationError(f"format 4 tool result conflicts with execution record: {call_id}")
        if record.status == ToolCallStatus.COMPLETED and _is_error_envelope(content):
            raise SessionValidationError(f"format 4 completed call {call_id} has an error result")
        if record.status in (ToolCallStatus.UNKNOWN, ToolCallStatus.FAILED) and not _is_error_result(content):
            raise SessionValidationError(
                f"format 4 {record.status.value} call {call_id} has a success-looking result"
            )

    for call_id in declared:
        if journal.get(call_id) is None:
            raise SessionValidationError(f"format 4 assistant declaration has no execution record: {call_id}")


def save_session(session_id: str, goal: str, messages: List[Dict[str, Any]],
                 objective: Optional[str] = None, summary: Optional[str] = None,
                 execution: Optional[List[Dict[str, Any]]] = None) -> None:
    """Save session state to disk."""
    _ensure_dir()

    # ``goal`` remains display/CLI metadata for compatibility; ``objective`` is authoritative.
    stored_objective = objective if objective is not None else f"Objective: {goal}"
    stored_execution = execution if execution is not None else []
    try:
        ExecutionJournal.from_dicts(stored_execution)
    except ExecutionStateError as exc:
        raise SessionValidationError(f"invalid execution state: {exc}") from exc

    session_data = {
        "format": SESSION_FORMAT,
        "session_id": session_id,
        "goal": goal,
        "objective": stored_objective,
        "summary": summary,
        "messages": messages,
        "execution": stored_execution,
        "saved_at": datetime.now().isoformat(),
    }

    session_path = SESSIONS_DIR / f"{session_id}.json"
    # Write to a temp file in the same directory, then atomically replace the target.
    # An interrupted write leaves the previous session file intact.
    tmp_path = session_path.with_name(session_path.name + ".tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(session_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, session_path)
    except BaseException:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Load session state from disk."""
    session_path = SESSIONS_DIR / f"{session_id}.json"

    try:
        if not session_path.exists():
            return None
    except OSError as e:
        raise SessionLoadError("read_error", f"could not inspect session {session_id}: {e}") from e

    try:
        with open(session_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise SessionLoadError("invalid_json", f"invalid JSON in session {session_id}: {e}") from e
    except OSError as e:
        raise SessionLoadError("read_error", f"could not read session {session_id}: {e}") from e


def list_sessions() -> List[Dict[str, Any]]:
    """List all saved sessions."""
    _ensure_dir()

    sessions = []
    for session_file in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            with open(session_file) as f:
                data = json.load(f)
            sessions.append({
                "session_id": data.get("session_id", session_file.stem),
                "goal": data.get("goal", ""),
                "saved_at": data.get("saved_at", ""),
                "message_count": len(data.get("messages", [])),
            })
        except Exception:
            continue

    return sessions


def delete_session(session_id: str) -> bool:
    """Delete a saved session."""
    session_path = SESSIONS_DIR / f"{session_id}.json"

    try:
        session_path.unlink()
        return True
    except Exception:
        return False
