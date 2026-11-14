"""Bounded verification of trusted acceptance criteria."""

import asyncio
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from .acceptance import AcceptanceCriterion, CriterionType
from .tools.files import _contained_target


# Attempt sequence contract (see validate_attempt_history):
#   * An attempt is one verification cycle. Attempt numbers are a per-run counter shared by all criteria
#     of that run, starting at 1, with no gaps.
#   * Every cycle evaluates every trusted criterion of its run, so each complete cycle has exactly one
#     result per trusted criterion. Checks skipped by the engine's limits are recorded as inconclusive.
#   * The runtime appends a cycle's results only after the whole cycle finishes, and persists them together.
#     So a cancelled cycle leaves no record. Legitimate partial history is therefore a run with fewer
#     complete cycles, and every persisted cycle is complete.
#   * The budget is the cycle maximum the run was started with. Each result records it, so the
#     validator can check attempts against the budget that was actually in force.
MAX_VERIFICATION_CYCLES = 10


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class VerificationResult:
    """One observation of one criterion in one verification attempt. Earlier attempts are kept, never rewritten."""
    criterion_id: str
    run_id: str
    attempt: int
    budget: int
    status: str
    observation: str
    checked_at: str

    def __post_init__(self) -> None:
        if not self.criterion_id or not self.run_id:
            raise ValueError("verification result requires criterion and run identifiers")
        if not isinstance(self.attempt, int) or isinstance(self.attempt, bool) or self.attempt < 1:
            raise ValueError("verification attempt must be a positive integer")
        if not isinstance(self.budget, int) or isinstance(self.budget, bool) or not 1 <= self.budget <= MAX_VERIFICATION_CYCLES:
            raise ValueError(f"verification budget must be an integer from 1 to {MAX_VERIFICATION_CYCLES}")
        if self.attempt > self.budget:
            raise ValueError("verification attempt exceeds its recorded budget")
        VerificationStatus(self.status)
        if not isinstance(self.observation, str) or len(self.observation) > 500:
            raise ValueError("verification observation must be a bounded string")
        if not isinstance(self.checked_at, str):
            raise ValueError("checked_at must be an ISO timestamp")
        datetime.fromisoformat(self.checked_at)

    def to_dict(self) -> Dict[str, Any]:
        return {"criterion_id": self.criterion_id, "run_id": self.run_id, "attempt": self.attempt,
                "budget": self.budget, "status": self.status, "observation": self.observation,
                "checked_at": self.checked_at}

    @classmethod
    def from_dict(cls, data: Any) -> "VerificationResult":
        required = {"criterion_id", "run_id", "attempt", "budget", "status", "observation", "checked_at"}
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("verification result has missing or unexpected fields")
        return cls(**{key: data[key] for key in required})


def _now() -> str:
    return datetime.now().isoformat()


def _job_state(result: Any) -> Optional[str]:
    """The state from a status_job envelope (JSON in content) or a plain mapping."""
    if isinstance(result, dict) and result.get("isError"):
        return None
    payload = result
    if isinstance(result, dict) and isinstance(result.get("content"), str):
        try:
            payload = json.loads(result["content"])
        except ValueError:
            return None
    state = payload.get("state") if isinstance(payload, dict) else None
    return state if isinstance(state, str) else None


class VerificationEngine:
    """Runs only typed checks and stops after the configured work budget."""

    def __init__(self, max_checks: int = 20, max_seconds: float = 30.0, max_file_bytes: int = 1 << 20):
        self.max_checks = max_checks
        self.max_seconds = max_seconds
        self.max_file_bytes = max_file_bytes

    async def verify(self, criteria: Iterable[AcceptanceCriterion], run_id: str, attempt: int = 1, budget: int = 1,
                     command_runner: Optional[Callable[..., Awaitable[Dict[str, Any]]]] = None,
                     job_checker: Optional[Callable[..., Awaitable[Dict[str, Any]]]] = None) -> List[VerificationResult]:
        selected = [criterion for criterion in criteria if criterion.run_id == run_id and criterion.trusted]
        deadline = time.monotonic() + self.max_seconds
        results = []
        for index, criterion in enumerate(selected):
            if index >= self.max_checks:
                # Never dropped silently: an unchecked required criterion is recorded as inconclusive, not passed
                results.append(self._result(criterion, VerificationStatus.INCONCLUSIVE,
                                            "not checked: the verification check limit was reached", attempt, budget))
                continue
            if time.monotonic() >= deadline:
                results.append(self._result(criterion, VerificationStatus.INCONCLUSIVE, "verification budget exhausted", attempt, budget))
                continue
            try:
                status, observation = await asyncio.wait_for(
                    self._check(criterion, command_runner, job_checker),
                    max(0.001, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                status, observation = VerificationStatus.INCONCLUSIVE, "verification timed out"
            except Exception as exc:
                status, observation = VerificationStatus.INCONCLUSIVE, f"verification failed: {type(exc).__name__}"
            results.append(self._result(criterion, status, observation, attempt, budget))
        return results

    async def _check(self, criterion, command_runner, job_checker):
        params = criterion.parameters
        expected = criterion.expected
        if criterion.criterion_type == CriterionType.FILE_EXISTS.value:
            target, error = _contained_target(params.get("path", ""))
            if error:
                return VerificationStatus.FAILED, "path is outside the working directory"
            exists = target.exists()
            expected_exists = bool(expected if isinstance(expected, bool) else True)
            if not exists:
                return (VerificationStatus.PASSED, "file is absent as required") if not expected_exists else (VerificationStatus.FAILED, "file is missing")
            kind = params.get("type", "file")
            if kind == "file" and not target.is_file():
                return VerificationStatus.FAILED, "path is not a file"
            if kind == "directory" and not target.is_dir():
                return VerificationStatus.FAILED, "path is not a directory"
            return (VerificationStatus.PASSED, "file exists") if expected_exists else (VerificationStatus.FAILED, "file exists unexpectedly")
        if criterion.criterion_type == CriterionType.FILE_CONTENT.value:
            target, error = _contained_target(params.get("path", ""))
            if error or not target.is_file():
                return VerificationStatus.FAILED, "file is missing or outside the working directory"
            if target.stat().st_size > self.max_file_bytes:
                return VerificationStatus.INCONCLUSIVE, "file exceeds verification read limit"
            text = target.read_bytes().decode("utf-8")
            expected_text = params.get("contains", expected)
            if not isinstance(expected_text, str):
                return VerificationStatus.INCONCLUSIVE, "content expectation is not text"
            return ((VerificationStatus.PASSED, "expected content observed") if expected_text in text
                    else (VerificationStatus.FAILED, "expected content was not observed"))
        if criterion.criterion_type == CriterionType.COMMAND_EXIT.value:
            if command_runner is None:
                return VerificationStatus.INCONCLUSIVE, "no authorized verification command runner"
            result = await command_runner(params.get("command", ""), params.get("timeout_seconds"))
            if not isinstance(result, dict):
                return VerificationStatus.INCONCLUSIVE, "verification command produced no result"
            if result.get("requiresPermission"):
                return VerificationStatus.INCONCLUSIVE, "verification command was not authorized and was not run"
            if result.get("isError"):
                return VerificationStatus.INCONCLUSIVE, "verification command did not produce a confirmed result"
            code = result.get("exitCode")
            if not isinstance(code, int) or isinstance(code, bool):
                return VerificationStatus.INCONCLUSIVE, "verification exit status was not reported"
            if result.get("processGroupEmpty") is not True:
                return VerificationStatus.INCONCLUSIVE, "could not confirm the command's process group was empty"
            if code < 0:
                return VerificationStatus.INCONCLUSIVE, "verification command was terminated by a signal"
            expected_code = expected if isinstance(expected, int) and not isinstance(expected, bool) else 0
            return ((VerificationStatus.PASSED, "verification command returned expected exit code")
                    if code == expected_code else (VerificationStatus.FAILED, f"unexpected exit code {code}"))
        if criterion.criterion_type == CriterionType.JOB_STATE.value:
            if job_checker is None:
                return VerificationStatus.INCONCLUSIVE, "no owned job checker"
            result = await job_checker(params.get("job_id"))
            state = _job_state(result)
            if state is None:
                return VerificationStatus.INCONCLUSIVE, "job state was not reported"
            if state == "unknown":
                return VerificationStatus.INCONCLUSIVE, "job state is unknown"
            return ((VerificationStatus.PASSED, "owned job reached expected state") if state == expected
                    else (VerificationStatus.FAILED, f"observed job state: {state}"))
        return VerificationStatus.INCONCLUSIVE, "unsupported criterion type"

    @staticmethod
    def _result(criterion, status, observation, attempt, budget):
        return VerificationResult(criterion.criterion_id, criterion.run_id, attempt, budget, status.value,
                                  observation, _now())

    @staticmethod
    def current(results: Iterable[VerificationResult]) -> Dict[str, VerificationResult]:
        """The latest attempt's result for each criterion. Earlier attempts remain in the history."""
        latest: Dict[str, VerificationResult] = {}
        for result in results:
            held = latest.get(result.criterion_id)
            if held is None or result.attempt > held.attempt:
                latest[result.criterion_id] = result
        return latest

    @staticmethod
    def feedback(results: Iterable[VerificationResult]) -> str:
        return "Verification failed:\n" + "\n".join(
            f"- {result.criterion_id}: {result.status} — {result.observation}"
            for result in results if result.status != VerificationStatus.PASSED.value)

    @staticmethod
    def all_passed(results: Iterable[VerificationResult]) -> bool:
        values = list(results)
        return bool(values) and all(result.status == VerificationStatus.PASSED.value for result in values)


def validate_results(results: Any, criteria: Iterable[AcceptanceCriterion]) -> List[VerificationResult]:
    if not isinstance(results, list):
        raise ValueError("verification_results must be a list")
    known = {criterion.criterion_id: criterion for criterion in criteria}
    parsed = [VerificationResult.from_dict(item) for item in results]
    if len({(item.criterion_id, item.attempt) for item in parsed}) != len(parsed):
        raise ValueError("verification results must be unique per criterion and attempt")

    for result in parsed:
        criterion = known.get(result.criterion_id)
        if criterion is None or criterion.run_id != result.run_id:
            raise ValueError("verification result is not owned by its criterion")
    return parsed


def validate_attempt_history(results: List[VerificationResult], criteria: Iterable[AcceptanceCriterion],
                             run_statuses: Dict[str, str]) -> None:
    """Semantic rules for persisted verification history. Raises ValueError for anything inconsistent.

    run_statuses maps each recorded run id to its persisted status. Runs with no results are not constrained here.
    """
    trusted_by_run: Dict[str, set] = {}
    for criterion in criteria:
        if criterion.trusted:
            trusted_by_run.setdefault(criterion.run_id, set()).add(criterion.criterion_id)
    by_run: Dict[str, List[VerificationResult]] = {}
    for item in results:
        if item.run_id not in run_statuses:
            raise ValueError("verification result belongs to a run that is not recorded")
        by_run.setdefault(item.run_id, []).append(item)
    for run_id, items in by_run.items():
        budgets = {item.budget for item in items}
        if len(budgets) != 1:
            raise ValueError(f"run {run_id} mixes verification budgets")
        budget = budgets.pop()
        attempts = sorted({item.attempt for item in items})
        if attempts != list(range(1, len(attempts) + 1)):
            raise ValueError(f"run {run_id} has a gap in its verification attempts")
        if attempts[-1] > budget:
            raise ValueError(f"run {run_id} exceeds its verification budget of {budget}")
        required = trusted_by_run.get(run_id, set())
        for attempt in attempts:
            covered = {item.criterion_id for item in items if item.attempt == attempt}
            if covered != required:
                raise ValueError(f"verification cycle {attempt} of run {run_id} does not cover every trusted criterion")
        last = {item.criterion_id for item in items if item.attempt == attempts[-1]}
        if last != required:
            raise ValueError(f"run {run_id} has an incomplete final verification cycle")
