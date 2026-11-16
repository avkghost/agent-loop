"""Durable tool-call lifecycle state for conservative session recovery."""

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Dict, List, Optional


# Result texts the runtime writes itself. Session validation uses them to tell legitimate
# non-completed results apart from invented ones.
INTERRUPTED_RESULT = ("Interrupted before a result was recorded. The tool may or may not have run, "
                      "so it was not re-executed. Verify its effects before relying on it.")
SKIPPED_RESULT = "Skipped: not executed because the duplicate-call guardrail stopped this turn."
_RUN_ID = re.compile(r"^run_[0-9a-f]{32}$")


class ToolCallStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ExecutionStateError(ValueError):
    """An execution record or lifecycle transition is invalid."""


@dataclass
class ToolExecutionRecord:
    tool_call_id: str
    tool_name: str
    arguments: Any
    status: ToolCallStatus = ToolCallStatus.PENDING
    outcome: Optional[str] = None
    error_category: Optional[str] = None
    origin_run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        record = {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "status": self.status.value,
            "outcome": self.outcome,
            "error_category": self.error_category,
        }
        if self.origin_run_id is not None:
            record["origin_run_id"] = self.origin_run_id
        return record

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolExecutionRecord":
        if not isinstance(data, dict):
            raise ExecutionStateError("execution record must be an object")
        call_id = data.get("tool_call_id")
        name = data.get("tool_name")
        arguments = data.get("arguments")
        try:
            status = ToolCallStatus(data.get("status"))
        except ValueError as exc:
            raise ExecutionStateError("execution record has an invalid status") from exc
        if not isinstance(call_id, str) or not call_id:
            raise ExecutionStateError("execution record requires a tool_call_id")
        if not isinstance(name, str):
            raise ExecutionStateError("execution record has invalid tool metadata")
        outcome = data.get("outcome")
        category = data.get("error_category")
        origin_run_id = data.get("origin_run_id")
        if outcome is not None and not isinstance(outcome, str):
            raise ExecutionStateError("execution outcome must be a string or null")
        if category is not None and not isinstance(category, str):
            raise ExecutionStateError("execution error_category must be a string or null")
        if origin_run_id is not None and (not isinstance(origin_run_id, str) or not _RUN_ID.fullmatch(origin_run_id)):
            raise ExecutionStateError("execution record has an invalid origin_run_id")
        if status in (ToolCallStatus.COMPLETED, ToolCallStatus.FAILED, ToolCallStatus.UNKNOWN) and outcome is None:
            raise ExecutionStateError("terminal execution records require an outcome")
        return cls(call_id, name, arguments, status, outcome, category, origin_run_id)


class ExecutionJournal:
    """Small in-memory journal serialized inside the atomic session envelope."""

    def __init__(self, records: Optional[List[ToolExecutionRecord]] = None):
        self.records = records or []
        self._by_id = {}
        for record in self.records:
            if record.tool_call_id in self._by_id:
                raise ExecutionStateError("duplicate tool_call_id in execution journal")
            self._by_id[record.tool_call_id] = record

    @classmethod
    def from_dicts(cls, records: Any) -> "ExecutionJournal":
        if not isinstance(records, list):
            raise ExecutionStateError("execution must be a list")
        return cls([ToolExecutionRecord.from_dict(record) for record in records])

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [record.to_dict() for record in self.records]

    def register(self, call_id: str, name: str, arguments: Any,
                 origin_run_id: Optional[str] = None) -> ToolExecutionRecord:
        if call_id in self._by_id:
            raise ExecutionStateError(f"duplicate tool_call_id: {call_id}")
        if origin_run_id is not None and (not isinstance(origin_run_id, str) or not _RUN_ID.fullmatch(origin_run_id)):
            raise ExecutionStateError("execution record has an invalid origin_run_id")
        record = ToolExecutionRecord(call_id, name, arguments, origin_run_id=origin_run_id)
        self.records.append(record)
        self._by_id[call_id] = record
        return record

    def transition(self, call_id: str, status: ToolCallStatus, outcome: Optional[str] = None,
                   error_category: Optional[str] = None) -> ToolExecutionRecord:
        record = self._by_id.get(call_id)
        if record is None:
            raise ExecutionStateError(f"unknown tool_call_id: {call_id}")
        valid = {
            ToolCallStatus.PENDING: {ToolCallStatus.RUNNING, ToolCallStatus.FAILED, ToolCallStatus.UNKNOWN},
            ToolCallStatus.RUNNING: {ToolCallStatus.COMPLETED, ToolCallStatus.FAILED, ToolCallStatus.UNKNOWN},
            ToolCallStatus.COMPLETED: set(),
            ToolCallStatus.FAILED: set(),
            ToolCallStatus.UNKNOWN: set(),
        }
        if status not in valid[record.status]:
            raise ExecutionStateError(f"invalid transition {record.status.value} -> {status.value}")
        if status in (ToolCallStatus.COMPLETED, ToolCallStatus.FAILED, ToolCallStatus.UNKNOWN) and outcome is None:
            raise ExecutionStateError("terminal transition requires an outcome")
        record.status = status
        record.outcome = outcome
        record.error_category = error_category
        return record

    def recover_unresolved(self, outcome: str) -> List[ToolExecutionRecord]:
        recovered = []
        for record in self.records:
            if record.status in (ToolCallStatus.PENDING, ToolCallStatus.RUNNING, ToolCallStatus.UNKNOWN):
                if record.status != ToolCallStatus.UNKNOWN:
                    self.transition(record.tool_call_id, ToolCallStatus.UNKNOWN, outcome, "unknown_outcome")
                elif record.outcome is None:
                    record.outcome = outcome
                recovered.append(record)
        return recovered

    def get(self, call_id: str) -> Optional[ToolExecutionRecord]:
        return self._by_id.get(call_id)
