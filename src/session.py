"""Session persistence for resuming interrupted runs."""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from .compaction import SUMMARY_MARKER, RECENT_USER_MARKER
from .execution import ExecutionJournal, ExecutionStateError, ToolCallStatus, INTERRUPTED_RESULT, SKIPPED_RESULT
from . import runs as run_records
from . import acceptance, verification


SESSIONS_DIR = Path.home() / ".agent_loop" / "sessions"
# Format 4 persists objective, replaceable summary, and durable execution records independently.
# Format 5 adds run outcomes and evidence records. Format 4 sessions still load; their run history is
# explicitly unrecorded (legacy_history), never reconstructed.
SESSION_FORMAT = 5
VERIFICATION_SESSION_FORMAT = 6
RUN_HISTORY_FIELDS = ("runs", "evidence", "legacy_history")


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

    if stored_format == 5:
        _validate_format_five(data, messages)
        return "structured"

    if stored_format == VERIFICATION_SESSION_FORMAT:
        _validate_format_six(data, messages)
        return "structured"

    if stored_format == 4:
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


def _validate_format_five(data: Dict[str, Any], messages: List[Any]) -> None:
    """Format 4 rules apply to the conversation and journal; run history is validated against them."""
    required = ("session_id", "goal", "objective", "summary", "messages", "execution", "saved_at",
                *RUN_HISTORY_FIELDS)
    missing = [field for field in required if field not in data]
    if missing:
        raise SessionValidationError(f"format 5 is missing required field(s): {', '.join(missing)}")
    if not isinstance(data.get("session_id"), str) or not isinstance(data.get("goal"), str):
        raise SessionValidationError("format 5 requires string session_id and goal")
    if not isinstance(data.get("saved_at"), str) or not isinstance(data.get("objective"), str):
        raise SessionValidationError("format 5 requires string saved_at and objective")
    if not messages or messages[0].get("role") != "system":
        raise SessionValidationError("format 5 messages must begin with a system message")
    if data.get("summary") is not None and not isinstance(data.get("summary"), str):
        raise SessionValidationError("format 5 summary must be a string or null")
    if not isinstance(data.get("legacy_history"), bool):
        raise SessionValidationError("format 5 legacy_history must be a boolean")
    try:
        journal = ExecutionJournal.from_dicts(data.get("execution"))
    except ExecutionStateError as exc:
        raise SessionValidationError(f"invalid execution state: {exc}") from exc
    _validate_execution_messages(data)
    try:
        run_records.validate_run_history(data.get("runs"), data.get("evidence"))
    except run_records.RunRecordError as exc:
        raise SessionValidationError(f"invalid run history: {exc}") from exc
    if data.get("format") == SESSION_FORMAT and any(
            item.get("status") == run_records.RunStatus.COMPLETED_VERIFIED.value
            for item in data.get("runs", [])):
        raise SessionValidationError("completed_verified requires session format 6 verification state")
    _validate_evidence_references(data, journal)


def _validate_format_six(data: Dict[str, Any], messages: List[Any]) -> None:
    """Format 5 plus immutable, run-owned criteria and verification observations."""
    _validate_format_five(data, messages)
    required = {"criteria", "verification_results"}
    if not required.issubset(data):
        raise SessionValidationError("format 6 is missing acceptance-criteria fields")
    try:
        criteria = acceptance.validate_criteria(data["criteria"])
        verification.validate_results(data["verification_results"], criteria)
    except (ValueError, TypeError) as exc:
        raise SessionValidationError(f"invalid format 6 verification state: {exc}") from exc
    criterion_ids = {criterion.criterion_id for criterion in criteria}
    results = verification.validate_results(data["verification_results"], criteria)
    run_statuses = {raw.get("run_id"): raw.get("status") for raw in data.get("runs", [])}
    try:
        verification.validate_attempt_history(results, criteria, run_statuses)
    except ValueError as exc:
        raise SessionValidationError(f"invalid verification attempt history: {exc}") from exc
    for raw in data.get("runs", []):
        if raw.get("status") == run_records.RunStatus.COMPLETED_VERIFIED.value:
            run_id = raw.get("run_id")
            # Only the latest attempt for each required criterion decides; earlier failures stay in history
            latest = verification.VerificationEngine.current(item for item in results if item.run_id == run_id)
            passed = {criterion_id for criterion_id, item in latest.items()
                      if item.status == verification.VerificationStatus.PASSED.value}
            required_for_run = {criterion.criterion_id for criterion in criteria if criterion.run_id == run_id and criterion.trusted}
            if not required_for_run or not required_for_run.issubset(passed):
                raise SessionValidationError(f"verified run {run_id} lacks passed trusted criteria")


def _validate_evidence_references(data: Dict[str, Any], journal: ExecutionJournal) -> None:
    """A runtime observation must match the journal record it cites: same run's call, and the same status."""
    expected = {"succeeded": "completed", "failed": "failed", "unknown": "unknown"}
    for raw in data["evidence"]:
        record = run_records.EvidenceRecord.from_dict(raw)
        if record.provenance != run_records.Provenance.RUNTIME_OBSERVATION.value:
            continue
        call = journal.get(record.journal_call_id)
        if call is None:
            raise SessionValidationError(
                f"evidence {record.evidence_id} cites a journal record that does not exist")
        if call.status.value != expected[record.outcome]:
            raise SessionValidationError(
                f"evidence {record.evidence_id} claims {record.outcome} but its journal record is {call.status.value}")


def resume_history(data: Dict[str, Any]) -> tuple:
    """(runs, evidence, legacy_history) for a session being resumed.

    Format 5 history is returned with interrupted runs recovered. Older sessions have no recorded history,
    so they return an empty history and legacy_history=True. Nothing is reconstructed for them.
    """
    if data.get("format") in (5, VERIFICATION_SESSION_FORMAT):
        return run_records.recover_interrupted(data["runs"]), list(data["evidence"]), bool(data["legacy_history"])
    return [], [], True


def _ensure_dir():
    """Ensure sessions directory exists."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _fsync_directory(directory: Path) -> None:
    """Persist directory metadata after an atomic rename."""
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


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
    label = f"session format {data.get('format')}"
    declared = {}
    results = {}
    for message in data.get("messages", []):
        if message.get("role") == "assistant" and "tool_calls" in message:
            calls = message.get("tool_calls")
            if not isinstance(calls, list):
                raise SessionValidationError(f"{label} assistant tool_calls must be a list")
            for call in calls:
                if not isinstance(call, dict):
                    raise SessionValidationError(f"{label} assistant tool_call must be an object")
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id or call_id in declared:
                    raise SessionValidationError(f"{label} contains duplicate or invalid assistant tool_call_id")
                if "type" in call and call.get("type") != "function":
                    raise SessionValidationError(f"{label} tool_call {call_id} has invalid type")
                function = call.get("function")
                if not isinstance(function, dict):
                    raise SessionValidationError(f"{label} tool_call {call_id} requires a function object")
                if not isinstance(function.get("name"), str) or not function.get("name"):
                    raise SessionValidationError(f"{label} tool_call {call_id} requires a function name")
                if not isinstance(function.get("arguments"), str):
                    raise SessionValidationError(f"{label} tool_call {call_id} requires string function arguments")
                declared[call_id] = call
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id in results:
                raise SessionValidationError(f"{label} contains duplicate tool result for {call_id}")
            results[call_id] = message.get("content")

    for call_id, content in results.items():
        record = journal.get(call_id)
        if record is None:
            raise SessionValidationError(f"{label} tool result has no execution record: {call_id}")
        if call_id not in declared:
            raise SessionValidationError(f"{label} tool result has no retained assistant declaration: {call_id}")
        if record.status in (ToolCallStatus.PENDING, ToolCallStatus.RUNNING):
            raise SessionValidationError(
                f"{label} active call {call_id} ({record.status.value}) already has a tool result"
            )
        if record.outcome is not None and content != record.outcome:
            raise SessionValidationError(f"{label} tool result conflicts with execution record: {call_id}")
        if record.status == ToolCallStatus.COMPLETED and _is_error_envelope(content):
            raise SessionValidationError(f"{label} completed call {call_id} has an error result")
        if record.status in (ToolCallStatus.UNKNOWN, ToolCallStatus.FAILED) and not _is_error_result(content):
            raise SessionValidationError(
                f"{label} {record.status.value} call {call_id} has a success-looking result"
            )

    for call_id in declared:
        if journal.get(call_id) is None:
            raise SessionValidationError(f"{label} assistant declaration has no execution record: {call_id}")


def build_candidate(session_id: str, goal: str, messages: List[Dict[str, Any]],
                    objective: Optional[str] = None, summary: Optional[str] = None,
                    execution: Optional[List[Dict[str, Any]]] = None,
                    runs: Optional[List[Dict[str, Any]]] = None,
                    evidence: Optional[List[Dict[str, Any]]] = None,
                    legacy_history: bool = False,
                    criteria: Optional[List[Dict[str, Any]]] = None,
                    verification_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """The complete format-5 session document that would be written."""
    # ``goal`` remains display/CLI metadata for compatibility; ``objective`` is authoritative.
    document = {
        "format": VERIFICATION_SESSION_FORMAT if criteria is not None or verification_results is not None else SESSION_FORMAT,
        "session_id": session_id,
        "goal": goal,
        "objective": objective if objective is not None else f"Objective: {goal}",
        "summary": summary,
        "messages": messages,
        "execution": execution if execution is not None else [],
        "runs": runs if runs is not None else [],
        "evidence": evidence if evidence is not None else [],
        "legacy_history": bool(legacy_history),
        "saved_at": datetime.now().isoformat(),
    }
    if document["format"] == VERIFICATION_SESSION_FORMAT:
        document["criteria"] = criteria if criteria is not None else []
        document["verification_results"] = verification_results if verification_results is not None else []
    return document


def save_session(session_id: str, goal: str, messages: List[Dict[str, Any]],
                 objective: Optional[str] = None, summary: Optional[str] = None,
                 execution: Optional[List[Dict[str, Any]]] = None,
                 runs: Optional[List[Dict[str, Any]]] = None,
                 evidence: Optional[List[Dict[str, Any]]] = None,
                 legacy_history: bool = False,
                 criteria: Optional[List[Dict[str, Any]]] = None,
                 verification_results: Optional[List[Dict[str, Any]]] = None) -> None:
    """Save session state to disk (format 5 or format 6 when verification state is present).

    The candidate is validated with the loader's own rules before anything touches the disk. An invalid
    candidate raises SessionValidationError and leaves the previously saved file exactly as it was.
    """
    _ensure_dir()
    candidate = build_candidate(session_id, goal, messages, objective=objective, summary=summary,
                                execution=execution, runs=runs, evidence=evidence,
                                legacy_history=legacy_history, criteria=criteria,
                                verification_results=verification_results)
    validate_session_data(candidate)

    session_path = SESSIONS_DIR / f"{session_id}.json"
    # Write to a temp file in the same directory, then atomically replace the target.
    # An interrupted write leaves the previous session file intact.
    tmp_path = session_path.with_name(session_path.name + ".tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(candidate, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, session_path)
        _fsync_directory(SESSIONS_DIR)
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
