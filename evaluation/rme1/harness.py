"""RME-1 host orchestration: trials, classification, bounded records. Runs on the host, never in the agent image."""

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scenarios import (MAX_TURNS, REPO_ROOT, RESULT_BYTES_LIMIT, SCENARIOS, TRIAL_TOOL_ALLOWLIST,  # noqa: E402
                       VERIFICATION_CYCLES, WALL_CLOCK_SECONDS, SHELL_TIMEOUT_SECONDS,
                       build_criteria, make_workspace, protected_unchanged)
from evaluator import HIDDEN_CHECKS, run_hidden_check, write_hidden_check  # noqa: E402

"""RME-1 evaluation harness: scenario fixtures, trusted criteria, independent evaluation, classification, and
bounded result records. Uses only the public Agent Loop runtime API and the standard library.

A real-model trial runs only when run_trial(..., execute=True) is called explicitly and the configured endpoint
answers. Without that, the result status is "not_executed" and no model is contacted.
"""

import asyncio
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# The only commands the model may run in a trial. Explicit and narrow; nothing is granted by ALLOW_ALL.

VERIFIED_SUCCESS = "VERIFIED_SUCCESS"
FALSE_POSITIVE_VERIFICATION = "FALSE_POSITIVE_VERIFICATION"
UNVERIFIED_COMPLETION = "UNVERIFIED_COMPLETION"
TASK_FAILURE = "TASK_FAILURE"
PROVIDER_FAILURE = "PROVIDER_FAILURE"
RUNTIME_FAILURE = "RUNTIME_FAILURE"
INTERRUPTED = "INTERRUPTED"
NOT_EXECUTED = "NOT_EXECUTED"
EVALUATION_ENVIRONMENT_FAILURE = "EVALUATION_ENVIRONMENT_FAILURE"  # result frame missing or corrupt; never success
INVALID_MODEL_MISMATCH = "INVALID_MODEL_MISMATCH"                  # requests did not use the intended model
INVALID_POLICY_MISMATCH = "INVALID_POLICY_MISMATCH"
INVALID_RESULT_SESSION = "INVALID_RESULT_SESSION"
INVALID_RESULT = "INVALID_RESULT"
DEADLINE_OVERRUN = "DEADLINE_OVERRUN"  # runtime reported completion after the deadline was reached; success not established  # RESULT frame corrupt, malformed, or with an unknown run status  # RESULT and SESSION frames contradict
LIMIT_REACHED = "LIMIT_REACHED"                         # runtime: iteration limit, independent evaluator FAIL or absent
LIMIT_REACHED_EVALUATOR_PASS = "LIMIT_REACHED_EVALUATOR_PASS"  # runtime: iteration limit; the work product passed the
                                                               # hidden checks, but the run never completed or verified
DEADLINE_INTERRUPTED = "DEADLINE_INTERRUPTED"           # supervisor deadline; the runner saved an INTERRUPTED RunOutcome
SUPERVISOR_TIMEOUT_NO_OUTCOME = "SUPERVISOR_TIMEOUT_NO_OUTCOME"  # forced termination; no RunOutcome was published
BLOCKED = "BLOCKED"  # the model requested an operation that policy denied; the runtime's own terminal status


def classify(run_status: Optional[str], reason: str, evaluator_passed: Optional[bool]) -> str:
    """Map a RunOutcome status and the independent evaluator result to the RME classification."""
    if run_status == "completed_verified":
        if evaluator_passed is True:
            return VERIFIED_SUCCESS
        return FALSE_POSITIVE_VERIFICATION
    if run_status == "completed_unverified":
        return UNVERIFIED_COMPLETION
    if run_status == "provider_failed":
        return PROVIDER_FAILURE
    if run_status == "interrupted":
        return INTERRUPTED
    if run_status == "blocked":
        # Valid benchmark outcome. Never a success, never a retry trigger, and never recast as UNKNOWN.
        return BLOCKED
    if run_status == "limit_reached":
        return LIMIT_REACHED_EVALUATOR_PASS if evaluator_passed is True else LIMIT_REACHED
    if run_status == "needs_attention":
        return TASK_FAILURE if "verification" in reason or "acceptance" in reason else RUNTIME_FAILURE
    return RUNTIME_FAILURE


def environment() -> Dict[str, str]:
    return {"python": platform.python_version(), "os": f"{platform.system()} {platform.release()}",
            "endpoint": "http://localhost:1234/v1/chat/completions (configured; not contacted unless executed)"}


def bounded(record: Dict[str, Any]) -> str:
    text = json.dumps(record, indent=2, sort_keys=True)
    if len(text.encode("utf-8")) > RESULT_BYTES_LIMIT:
        raise ValueError("result record exceeds the size bound")
    return text


def _tool_counts(execution: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in execution:
        counts[record["tool_name"]] = counts.get(record["tool_name"], 0) + 1
    return counts


def run_trial(scenario_id: str, root: Path, trial_id: str, evaluator_dir: Path, execute: bool = False,
              endpoint_ok: bool = False) -> Dict[str, Any]:
    """One trial. Without execute=True (or without a reachable endpoint) no model is contacted."""
    scenario = SCENARIOS[scenario_id]
    fixture = make_workspace(scenario, root, trial_id)
    record: Dict[str, Any] = {"evaluation_id": f"{scenario_id}:{trial_id}", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                              "environment": environment(), "budgets": {"max_turns": MAX_TURNS,
                              "verification_cycles": VERIFICATION_CYCLES, "wall_clock_seconds": WALL_CLOCK_SECONDS,
                              "shell_timeout_seconds": SHELL_TIMEOUT_SECONDS},
                              "tool_allowlist": sorted(TRIAL_TOOL_ALLOWLIST), "workspace": fixture["workspace"]}
    if not execute:
        record.update({"classification": NOT_EXECUTED, "reason": "trial not executed: model not contacted"})
        return record
    if not endpoint_ok:
        record.update({"classification": PROVIDER_FAILURE, "reason": "the LM Studio endpoint is unavailable"})
        return record
    return _execute(scenario, fixture, record, evaluator_dir)


def _execute(scenario, fixture, record, evaluator_dir):
    """Real execution path: the public runtime API with the real provider. Requires an explicit caller decision."""
    from src import config, loop, runtime
    from src.tools.mcp_client import MCPClientManager
    workspace = Path(fixture["workspace"])
    previous_cwd = os.getcwd()
    previous_allow = set(config.ALLOWED_TOOLS)
    previous_iterations = config.MAX_ITERATIONS
    started = time.monotonic()
    rt = runtime.AgentRuntime()
    previous_sessions_dir = runtime.session.SESSIONS_DIR
    session_dir = workspace.parent / "sessions"  # trial sessions are kept with the trial, never in the home directory
    try:
        runtime.session.SESSIONS_DIR = session_dir
        os.chdir(workspace)
        config.ALLOWED_TOOLS = set(TRIAL_TOOL_ALLOWLIST)
        config.MAX_ITERATIONS = MAX_TURNS
        session = rt.open_session(f"rme-{scenario.scenario_id}-{workspace.name}")
        outcome = rt.event_loop.run_until_complete(session.run_goal(
            scenario.goal, criteria=build_criteria(scenario), verification_cycles=VERIFICATION_CYCLES))
        session.close()
        status, reason = outcome.run_status.value, outcome.reason
    except Exception as exc:  # recorded, not hidden
        status, reason = "runtime_error", f"{type(exc).__name__}: {exc}"[:300]
    finally:
        config.ALLOWED_TOOLS = previous_allow
        config.MAX_ITERATIONS = previous_iterations
        runtime.session.SESSIONS_DIR = previous_sessions_dir
        os.chdir(previous_cwd)
        rt.shutdown()
    elapsed = round(time.monotonic() - started, 1)
    previous_dir = runtime.session.SESSIONS_DIR
    runtime.session.SESSIONS_DIR = session_dir
    try:
        saved = runtime.session.load_session(f"rme-{scenario.scenario_id}-{workspace.name}") or {}
    finally:
        runtime.session.SESSIONS_DIR = previous_dir
    evaluator = run_hidden_check(scenario, workspace, evaluator_dir)
    unchanged = protected_unchanged(scenario, workspace, fixture["initial_hashes"])
    record.update({
        "final_run_status": status, "run_reason": reason[:300], "wall_clock_seconds": elapsed,
        "tool_counts": _tool_counts(saved.get("execution", [])),
        "verification_attempts": max([r.get("attempt", 0) for r in saved.get("verification_results", [])] or [0]),
        "runtime_verification": [{"criterion_id": r["criterion_id"], "attempt": r["attempt"], "status": r["status"]}
                                 for r in saved.get("verification_results", [])][-12:],
        "independent_evaluator": evaluator, "protected_files_unchanged": unchanged,
        "human_interventions": 0,
    })
    record["classification"] = classify(status, reason, evaluator["passed"] and unchanged)
    return record


def write_record(record: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{record['evaluation_id'].replace(':', '_')}.json"
    path.write_text(bounded(record))
    return path


def cleanup_workspace(workspace: str) -> None:
    """Remove only a directory that the harness created under a temporary root."""
    path = Path(workspace).resolve()
    if path.exists() and Path(tempfile.gettempdir()).resolve() in path.parents:
        shutil.rmtree(path)
