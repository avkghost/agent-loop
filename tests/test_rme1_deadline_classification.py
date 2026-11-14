"""Deadline classification matrix. Every row calls supervisor.classify_trial, the same function main() uses.
Recorded real sessions and scripted sessions only: no LM Studio, no Docker, no network."""

import copy
import sys
from pathlib import Path

import pytest

from src.session import SessionValidationError, validate_session_data
from tests.test_rme1_terminal_integrity import (  # noqa: F401 (fixture and helpers reused on purpose)
    MODEL, _classify, _load, _summary, completed_session)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evaluation" / "rme1"))
import harness  # noqa: E402  (loaded by the same module as the tests above)
import supervisor  # noqa: E402

DRILL = "session_interrupted_drill.json"
BLOCKED = "session_blocked_glm16.json"
LIMIT = "session_limit_reached_glm15.json"
NO_SESSION = (None, "MISSING")
CORRUPT_SESSION = (None, "CORRUPT")


def _deadline(run_status, session, session_status, *, signalled=True, reason="external deadline", graceful=True):
    summary = _summary(run_status, reason, deadline_signalled=signalled) if run_status is not None else None
    if isinstance(summary, dict) and run_status is None:
        summary = _summary("x", reason)
    return _classify(summary, session, session_status=session_status, timed_out=True, graceful=graceful)


# 1-9 and boundary rows. Expected classification and the outcome label that must be recorded.
TIMEOUT_MATRIX = [
    pytest.param("1 interrupted + valid matching SESSION",
                 lambda: _deadline("interrupted", _load(DRILL), "OK"),
                 harness.DEADLINE_INTERRUPTED, "DEADLINE_GRACEFUL", id="t1-interrupted-valid"),
    pytest.param("2 interrupted + missing SESSION",
                 lambda: _deadline("interrupted", None, "MISSING"),
                 harness.INVALID_RESULT_SESSION, "DEADLINE_CHECK_FAILED", id="t2-interrupted-missing"),
    pytest.param("3 interrupted + corrupt SESSION",
                 lambda: _deadline("interrupted", None, "CORRUPT"),
                 harness.INVALID_RESULT_SESSION, "DEADLINE_CHECK_FAILED", id="t3-interrupted-corrupt"),
    pytest.param("4 runtime_error + missing SESSION",
                 lambda: _deadline("runtime_error", None, "MISSING", signalled=False, reason="RuntimeError: boom"),
                 harness.RUNTIME_FAILURE, "DEADLINE_RUNTIME_ERROR", id="t4-runtime-error-missing"),
    pytest.param("6 null run_status",
                 lambda: _classify({"run_status": None, "reason": "r", "policy": {}}, _load(DRILL), timed_out=True,
                                   graceful=True),
                 harness.INVALID_RESULT, "RESULT_MALFORMED", id="t6-null-status"),
    pytest.param("7 unknown run_status",
                 lambda: _classify({"run_status": "maybe_done", "reason": "r", "policy": {}}, _load(DRILL),
                                   timed_out=True, graceful=True),
                 harness.INVALID_RESULT, "RESULT_MALFORMED", id="t7-unknown-status"),
    pytest.param("8 interrupted RESULT + mismatched SESSION (blocked)",
                 lambda: _deadline("interrupted", _load(BLOCKED), "OK"),
                 harness.INVALID_RESULT_SESSION, "DEADLINE_CHECK_FAILED", id="t8-mismatched-session"),
    pytest.param("9 no RESULT",
                 lambda: _classify(None, None, status="MISSING", session_status="MISSING", timed_out=True,
                                   graceful=False),
                 harness.SUPERVISOR_TIMEOUT_NO_OUTCOME, "DEADLINE_FORCED_NO_OUTCOME", id="t9-no-result"),
    pytest.param("boundary: interrupted without a deadline signal",
                 lambda: _deadline("interrupted", _load(DRILL), "OK", signalled=False),
                 harness.INVALID_RESULT_SESSION, "DEADLINE_CHECK_FAILED", id="b1-no-signal"),
    pytest.param("boundary: corrupt RESULT frame",
                 lambda: supervisor.classify_trial(
                     status="CORRUPT", detail="checksum", summary=None, timed_out=True, graceful=False,
                     session_status="MISSING", session_doc=None, scenario="RME-01", model=MODEL,
                     policy_id="RME-01-POLICY-V2", evaluator_passed=False),
                 harness.INVALID_RESULT, "RESULT_CORRUPT", id="b2-corrupt-result"),
    pytest.param("boundary: blocked after a deadline (valid session) is not DEADLINE_INTERRUPTED",
                 lambda: _deadline("blocked", _load(BLOCKED), "OK", signalled=False),
                 harness.BLOCKED, "DEADLINE_RESULT_NOT_INTERRUPTED", id="b3-blocked-after-deadline"),
]


@pytest.mark.parametrize("label, run, expected, outcome", TIMEOUT_MATRIX)
def test_timeout_matrix_row(label, run, expected, outcome):
    out = run()
    assert out["classification"] == expected, f"{label}: got {out['classification']} ({out['reason']})"
    assert out["outcome"] == outcome, label
    if expected != harness.DEADLINE_INTERRUPTED:
        assert out["classification"] != harness.DEADLINE_INTERRUPTED, label


def test_row5_runtime_error_with_a_valid_matching_session_is_not_representable():
    """The session validator's run statuses exclude runtime_error (the runner's exception path records no run), so no
    valid SESSION can match a runtime_error RESULT. The classification therefore rests on the RESULT alone."""
    doc = copy.deepcopy(_load(DRILL))
    doc["runs"][0]["status"] = "runtime_error"
    with pytest.raises(SessionValidationError):
        validate_session_data(doc)
    out = _deadline("runtime_error", None, "MISSING", signalled=False, reason="RuntimeError: boom")
    assert out["classification"] == harness.RUNTIME_FAILURE and out["outcome"] == "DEADLINE_RUNTIME_ERROR"


def test_row10_no_timeout_valid_completed_verified_is_verified_success(completed_session):
    out = _classify(_summary("completed_verified"), completed_session, scenario="RME-02", evaluator=True)
    assert out["classification"] == harness.VERIFIED_SUCCESS and out["outcome"] == "COMPLETED"


def test_row11_no_timeout_blocked_is_blocked():
    out = _classify(_summary("blocked", "the user denied permission for tool run_command; the run stopped"),
                    _load(BLOCKED), evaluator=True)
    assert out["classification"] == harness.BLOCKED and out["outcome"] == "COMPLETED"


def test_row12_no_timeout_limit_reached_is_limit_reached():
    out = _classify(_summary("limit_reached", "the iteration limit was reached"), _load(LIMIT))
    assert out["classification"] == harness.LIMIT_REACHED and out["outcome"] == "COMPLETED"
