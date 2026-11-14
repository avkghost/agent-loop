"""Run outcomes and evidence records: typed, bounded, validated, and owned by exactly one run.

Schema names (persisted in session format 5):
- Run record, "agent-loop.run/1": one goal execution inside a session.
- Evidence record, "agent-loop.evidence/1": one observation about a run. Evidence never establishes task
  success by itself, and a model's words are never evidence.

Semantics of the run status (what the runtime can actually establish):
- completed_unverified: the model stopped normally. Nothing was verified.
- needs_attention: the runtime stopped because durable state could not be saved, so a human must check.
- blocked: the user refused a requested action. The run stops rather than override that decision.
- unknown: the runtime cannot establish whether an external effect happened, or the run ended in an
  unexpected error. Nothing here may be read as success.
- interrupted: cancellation, an interrupt, or a process that ended before the run recorded an outcome.
- provider_failed: the model request failed, or its response was incomplete, malformed, or empty.
- limit_reached: the iteration budget or the context budget, or the response length limit, was hit.
- completed_verified: reserved for the verification gate. The runtime does not emit it in this version.
- running: persisted only while a run is open. It never appears as a final outcome.
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

RUN_SCHEMA = "agent-loop.run/1"
EVIDENCE_SCHEMA = "agent-loop.evidence/1"

MAX_GOAL_CHARS = 500
MAX_REASON_CHARS = 300
MAX_FINAL_MESSAGE_CHARS = 2000
MAX_EVIDENCE_PER_RUN = 500
MAX_EVIDENCE_SUMMARY_CHARS = 200
MAX_ITERATION_COUNT = 1_000_000
# Aggregate bounds per session. Deliberately not a retention policy: exceeding them is an explicit validation
# error on save and on load. Nothing is truncated, deleted, or archived.
MAX_RUNS_PER_SESSION = 500
MAX_EVIDENCE_PER_SESSION = 5000
RUNNING = "running"


class RunStatus(str, Enum):
    COMPLETED_UNVERIFIED = "completed_unverified"
    COMPLETED_VERIFIED = "completed_verified"
    NEEDS_ATTENTION = "needs_attention"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"
    INTERRUPTED = "interrupted"
    PROVIDER_FAILED = "provider_failed"
    LIMIT_REACHED = "limit_reached"


RESERVED_STATUSES = frozenset({RunStatus.COMPLETED_VERIFIED})
_RUNTIME_STATUSES = frozenset(s.value for s in RunStatus if s not in RESERVED_STATUSES)
_PERSISTED_STATUSES = _RUNTIME_STATUSES | {RUNNING} | {RunStatus.COMPLETED_VERIFIED.value}


class Provenance(str, Enum):
    RUNTIME_OBSERVATION = "runtime_observation"
    MODEL_CLAIM = "model_claim"


class EvidenceOutcome(str, Enum):
    SUCCEEDED = "succeeded"   # only from a completed journal record
    FAILED = "failed"         # only from a failed journal record
    UNKNOWN = "unknown"       # from an unknown journal record: never a success
    CLAIM = "claim"           # a model statement; never a verification


class RunRecordError(ValueError):
    """A run or evidence record is invalid, oversized, or inconsistent."""


_RUN_ID = re.compile(r"^run_[0-9a-f]{32}$")
_EVIDENCE_ID = re.compile(r"^ev_[0-9a-f]{32}$")


def new_run_id() -> str:
    return "run_" + uuid.uuid4().hex


def new_evidence_id() -> str:
    return "ev_" + uuid.uuid4().hex


def _now() -> str:
    return datetime.now().isoformat()


def _text(value: Any, field_name: str, limit: int, allow_none: bool = False) -> Optional[str]:
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise RunRecordError(f"{field_name} must be a string")
    if len(value) > limit:
        raise RunRecordError(f"{field_name} exceeds {limit} characters")
    return value


def _timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise RunRecordError(f"{field_name} must be an ISO timestamp string")
    try:
        return datetime.fromisoformat(value)
    except ValueError as e:
        raise RunRecordError(f"{field_name} is not an ISO timestamp") from e


def _count(value: Any, field_name: str, limit: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= limit:
        raise RunRecordError(f"{field_name} must be an integer from 0 to {limit}")
    return value


def _check_keys(data: Any, required: set, optional: set, what: str) -> None:
    if not isinstance(data, dict):
        raise RunRecordError(f"{what} must be an object")
    unknown = set(data) - required - optional
    if unknown:
        raise RunRecordError(f"{what} has unexpected field(s): {', '.join(sorted(unknown))}")
    missing = required - set(data)
    if missing:
        raise RunRecordError(f"{what} is missing field(s): {', '.join(sorted(missing))}")


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    goal: str
    status: str            # a RunStatus value, or "running" while the run is open
    reason: str
    started_at: str
    finished_at: Optional[str]
    iteration_count: int
    final_message: Optional[str] = None
    evidence_ids: tuple = field(default_factory=tuple)
    evidence_dropped: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not _RUN_ID.match(self.run_id):
            raise RunRecordError("run_id must look like run_<32 hex>")
        _text(self.goal, "goal", MAX_GOAL_CHARS)
        if self.status not in _PERSISTED_STATUSES:
            raise RunRecordError(f"invalid run status {self.status!r}")
        _text(self.reason, "reason", MAX_REASON_CHARS)
        started = _timestamp(self.started_at, "started_at")
        if self.status == RUNNING:
            if self.finished_at is not None:
                raise RunRecordError("a running run cannot have finished_at")
        else:
            if self.finished_at is None:
                raise RunRecordError("a finished run requires finished_at")
            if _timestamp(self.finished_at, "finished_at") < started:
                raise RunRecordError("finished_at is before started_at")
        _count(self.iteration_count, "iteration_count", MAX_ITERATION_COUNT)
        _text(self.final_message, "final_message", MAX_FINAL_MESSAGE_CHARS, allow_none=True)
        if not isinstance(self.evidence_ids, tuple) or len(self.evidence_ids) > MAX_EVIDENCE_PER_RUN:
            raise RunRecordError("evidence_ids must be a bounded list")
        for evidence_id in self.evidence_ids:
            if not isinstance(evidence_id, str) or not _EVIDENCE_ID.match(evidence_id):
                raise RunRecordError("evidence_ids must contain evidence identifiers")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise RunRecordError("evidence_ids must be unique")
        _count(self.evidence_dropped, "evidence_dropped", MAX_EVIDENCE_PER_RUN * 1000)

    @property
    def run_status(self) -> RunStatus:
        return RunStatus(self.status)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": RUN_SCHEMA,
            "run_id": self.run_id,
            "goal": self.goal,
            "status": self.status,
            "reason": self.reason,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "iteration_count": self.iteration_count,
            "final_message": self.final_message,
            "evidence_ids": list(self.evidence_ids),
            "evidence_dropped": self.evidence_dropped,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RunOutcome":
        required = {"schema", "run_id", "goal", "status", "reason", "started_at", "finished_at",
                    "iteration_count", "final_message", "evidence_ids", "evidence_dropped"}
        _check_keys(data, required, set(), "run record")
        if data["schema"] != RUN_SCHEMA:
            raise RunRecordError(f"unsupported run schema {data['schema']!r}")
        ids = data["evidence_ids"]
        if not isinstance(ids, list):
            raise RunRecordError("evidence_ids must be a list")
        return cls(
            run_id=data["run_id"], goal=data["goal"], status=data["status"], reason=data["reason"],
            started_at=data["started_at"], finished_at=data["finished_at"],
            iteration_count=data["iteration_count"], final_message=data["final_message"],
            evidence_ids=tuple(ids), evidence_dropped=data["evidence_dropped"],
        )


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    run_id: str
    source: str                 # "tool_call": the only source the runtime records in this version
    provenance: str             # Provenance value
    outcome: str                # EvidenceOutcome value
    journal_call_id: Optional[str]
    summary: str
    recorded_at: str
    criterion_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_id, str) or not _EVIDENCE_ID.match(self.evidence_id):
            raise RunRecordError("evidence_id must look like ev_<32 hex>")
        if not isinstance(self.run_id, str) or not _RUN_ID.match(self.run_id):
            raise RunRecordError("evidence run_id must look like run_<32 hex>")
        if self.source != "tool_call":
            raise RunRecordError(f"unsupported evidence source {self.source!r}")
        try:
            provenance = Provenance(self.provenance)
            outcome = EvidenceOutcome(self.outcome)
        except ValueError as e:
            raise RunRecordError("evidence provenance or outcome is unknown") from e
        if provenance is Provenance.RUNTIME_OBSERVATION:
            if not isinstance(self.journal_call_id, str) or not self.journal_call_id:
                raise RunRecordError("a runtime observation must reference its journal record")
            if outcome is EvidenceOutcome.CLAIM:
                raise RunRecordError("a runtime observation cannot be a model claim")
        else:
            if self.journal_call_id is not None:
                raise RunRecordError("a model claim cannot reference a journal record")
            if outcome is not EvidenceOutcome.CLAIM:
                raise RunRecordError("a model claim must have the claim outcome")
        _text(self.summary, "evidence summary", MAX_EVIDENCE_SUMMARY_CHARS)
        _timestamp(self.recorded_at, "recorded_at")
        _text(self.criterion_id, "criterion_id", 64, allow_none=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": EVIDENCE_SCHEMA,
            "evidence_id": self.evidence_id,
            "run_id": self.run_id,
            "criterion_id": self.criterion_id,
            "source": self.source,
            "provenance": self.provenance,
            "outcome": self.outcome,
            "journal_call_id": self.journal_call_id,
            "summary": self.summary,
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "EvidenceRecord":
        required = {"schema", "evidence_id", "run_id", "criterion_id", "source", "provenance", "outcome",
                    "journal_call_id", "summary", "recorded_at"}
        _check_keys(data, required, set(), "evidence record")
        if data["schema"] != EVIDENCE_SCHEMA:
            raise RunRecordError(f"unsupported evidence schema {data['schema']!r}")
        return cls(
            evidence_id=data["evidence_id"], run_id=data["run_id"], criterion_id=data["criterion_id"],
            source=data["source"], provenance=data["provenance"], outcome=data["outcome"],
            journal_call_id=data["journal_call_id"], summary=data["summary"], recorded_at=data["recorded_at"],
        )


def validate_run_history(runs: Any, evidence: Any) -> None:
    """Cross-record rules: unique identifiers, and each evidence record owned by exactly one run."""
    if not isinstance(runs, list) or not isinstance(evidence, list):
        raise RunRecordError("runs and evidence must be lists")
    if len(runs) > MAX_RUNS_PER_SESSION:
        raise RunRecordError(f"a session may hold at most {MAX_RUNS_PER_SESSION} runs; it holds {len(runs)}")
    if len(evidence) > MAX_EVIDENCE_PER_SESSION:
        raise RunRecordError(
            f"a session may hold at most {MAX_EVIDENCE_PER_SESSION} evidence records; it holds {len(evidence)}")
    parsed_runs = [RunOutcome.from_dict(r) for r in runs]
    parsed_evidence = [EvidenceRecord.from_dict(e) for e in evidence]
    run_ids = [r.run_id for r in parsed_runs]
    if len(set(run_ids)) != len(run_ids):
        raise RunRecordError("run identifiers must be unique")
    evidence_ids = [e.evidence_id for e in parsed_evidence]
    if len(set(evidence_ids)) != len(evidence_ids):
        raise RunRecordError("evidence identifiers must be unique")
    by_id = {e.evidence_id: e for e in parsed_evidence}
    owner_of = {r.run_id: r for r in parsed_runs}
    for record in parsed_evidence:
        if record.run_id not in owner_of:
            raise RunRecordError(f"evidence {record.evidence_id} belongs to an unknown run")
    listed = set()
    for run in parsed_runs:
        for evidence_id in run.evidence_ids:
            if evidence_id not in by_id:
                raise RunRecordError(f"run {run.run_id} lists unknown evidence {evidence_id}")
            if by_id[evidence_id].run_id != run.run_id:
                raise RunRecordError(f"evidence {evidence_id} is listed by a run that does not own it")
            listed.add(evidence_id)
    if listed != set(evidence_ids):
        raise RunRecordError("every evidence record must be listed by its owning run")


def recover_interrupted(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Runs still marked running were not finished when their process ended. They are recorded as
    interrupted: that is a fact, and nothing is claimed about what they did before the interruption."""
    recovered = []
    for run in runs:
        if run.get("status") == RUNNING:
            run = dict(run, status=RunStatus.INTERRUPTED.value,
                       reason="the process ended before this run recorded an outcome; "
                              "it was not re-executed",
                       finished_at=_now())
        recovered.append(run)
    return recovered


@dataclass
class RunRecorder:
    """Accumulates one run's outcome. The first terminal status set wins; the runtime sets it explicitly
    on every exit path, so an unset status at the end means the run ended without a recorded outcome."""

    goal: str
    run_id: str = field(default_factory=new_run_id)
    started_at: str = field(default_factory=_now)
    status: Optional[str] = None
    reason: str = ""
    iteration_count: int = 0
    final_message: Optional[str] = None
    finished_at: Optional[str] = None
    evidence: List[EvidenceRecord] = field(default_factory=list)
    evidence_dropped: int = 0
    unknown_effects: int = 0

    def begin_iteration(self, number: int) -> None:
        self.iteration_count = number

    def end(self, status: RunStatus, reason: str, final_message: Optional[str] = None) -> None:
        if status in RESERVED_STATUSES:
            raise RunRecordError("completed_verified is reserved for the verification gate")
        if self.status is None:
            self.status = status.value
            self.reason = reason[:MAX_REASON_CHARS]
            if final_message is not None:
                self.final_message = final_message[:MAX_FINAL_MESSAGE_CHARS]

    def end_verified(self, reason: str, final_message: Optional[str] = None) -> None:
        """Set verified completion after an independent verification gate has passed."""
        if self.status is not None:
            raise RunRecordError("a run can only be verified before a terminal status is recorded")
        self.status = RunStatus.COMPLETED_VERIFIED.value
        self.reason = reason[:MAX_REASON_CHARS]
        self.final_message = (final_message[:MAX_FINAL_MESSAGE_CHARS]
                              if final_message is not None else None)

    def mark_persistence_failure(self) -> None:
        """Downgrade an in-memory verified result when its required save failed."""
        if self.status == RunStatus.COMPLETED_VERIFIED.value:
            self.status = RunStatus.NEEDS_ATTENTION.value
            self.reason = "verification passed, but the verified result was not saved"

    @property
    def open(self) -> bool:
        return self.status is None

    def record_call(self, call_id: str, tool_name: str, journal_status: str) -> Optional[EvidenceRecord]:
        """Runtime evidence from a journal transition. The summary names the status; it never copies output."""
        outcome = {"completed": EvidenceOutcome.SUCCEEDED, "failed": EvidenceOutcome.FAILED,
                   "unknown": EvidenceOutcome.UNKNOWN}[journal_status]
        if outcome is EvidenceOutcome.UNKNOWN:
            self.unknown_effects += 1
        if len(self.evidence) >= MAX_EVIDENCE_PER_RUN:
            self.evidence_dropped += 1
            return None
        summary = f"tool {tool_name}: {journal_status}; its output stays in the session journal"
        record = EvidenceRecord(
            evidence_id=new_evidence_id(), run_id=self.run_id, source="tool_call",
            provenance=Provenance.RUNTIME_OBSERVATION.value, outcome=outcome.value,
            journal_call_id=call_id, summary=summary[:MAX_EVIDENCE_SUMMARY_CHARS], recorded_at=_now())
        self.evidence.append(record)
        return record

    def _effective_status(self) -> str:
        if self.status is None:
            return RunStatus.UNKNOWN.value
        if self.status == RunStatus.COMPLETED_UNVERIFIED.value and self.unknown_effects:
            # A model stop cannot clear an external effect the runtime could not establish
            return RunStatus.UNKNOWN.value
        return self.status

    def _effective_reason(self) -> str:
        if self.status is None:
            return "the run ended without a recorded outcome"
        if self.status == RunStatus.COMPLETED_UNVERIFIED.value and self.unknown_effects:
            return (f"the model stopped, but {self.unknown_effects} external operation(s) had an unknown outcome; "
                    "verify their effects")
        return self.reason

    def snapshot(self) -> RunOutcome:
        """The run as it stands. Open runs are persisted as running."""
        finished = (self.finished_at or _now()) if not self.open else None
        status = self._effective_status() if not self.open else RUNNING
        return RunOutcome(
            run_id=self.run_id, goal=self.goal[:MAX_GOAL_CHARS], status=status,
            reason=self._effective_reason()[:MAX_REASON_CHARS], started_at=self.started_at,
            finished_at=finished, iteration_count=self.iteration_count, final_message=self.final_message,
            evidence_ids=tuple(e.evidence_id for e in self.evidence),
            evidence_dropped=self.evidence_dropped)

    def finish(self) -> RunOutcome:
        self.finished_at = self.finished_at or _now()
        return self.snapshot()
