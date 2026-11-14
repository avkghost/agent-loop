"""Deadline race classification. Every assertion calls supervisor.classify_trial or the drill's wrapper around it.
Recorded sessions, a scripted completed session, and offline stand-ins only: no LM Studio, no Docker, no network."""

import copy
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

from src.session import validate_session_data
from tests.test_rme1_terminal_integrity import (  # noqa: F401 (fixture and helpers reused on purpose)
    MODEL, _classify, _load, _summary, completed_session)

REPO = Path(__file__).resolve().parents[1]
RME = REPO / "evaluation" / "rme1"
sys.path.insert(0, str(RME))
import harness  # noqa: E402
import supervisor  # noqa: E402

spec = importlib.util.spec_from_file_location("rme1_drill_race", RME / "drill.py")
drill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drill)

DRILL, BLOCKED, LIMIT = "session_interrupted_drill.json", "session_blocked_glm16.json", "session_limit_reached_glm15.json"


def _timed(run_status, session, session_status="OK", *, signalled=True, graceful=True, reason="external deadline",
           evaluator=True):
    summary = _summary(run_status, reason, deadline_signalled=signalled)
    return _classify(summary, session, session_status=session_status, timed_out=True, graceful=graceful,
                     evaluator=evaluator)


def _completed_unverified_session():
    """A validator-accepted completed_unverified session. It is derived: only the recorded run's status is changed."""
    doc = copy.deepcopy(_load(BLOCKED))
    doc["runs"][0]["status"] = "completed_unverified"
    validate_session_data(doc)
    return doc


# 1. timed-out completed_verified with a valid SESSION and evaluator PASS -> DEADLINE_OVERRUN
def test_timeout_completed_verified_with_valid_session_and_evaluator_pass_is_deadline_overrun(completed_session):
    out = _classify(_summary("completed_verified", "ok", scenario="RME-02", deadline_signalled=True),
                    completed_session, scenario="RME-02", timed_out=True, graceful=True, evaluator=True)
    assert out["classification"] == harness.DEADLINE_OVERRUN
    assert out["classification"] not in (harness.VERIFIED_SUCCESS, harness.FALSE_POSITIVE_VERIFICATION)
    assert out["run_status"] == "completed_verified", "the runtime status is preserved"
    assert out["checks"]["evaluator_passed"] is True, "the evaluator result is kept separately"


# 2. timed-out completed_verified with a missing SESSION -> INVALID_RESULT_SESSION
def test_timeout_completed_verified_with_missing_session_is_invalid_result_session():
    out = _timed("completed_verified", None, session_status="MISSING")
    assert out["classification"] == harness.INVALID_RESULT_SESSION


# 3. timed-out completed_verified with a contradictory SESSION -> INVALID_RESULT_SESSION
def test_timeout_completed_verified_with_contradictory_session_is_invalid_result_session():
    out = _timed("completed_verified", _load(BLOCKED))
    assert out["classification"] == harness.INVALID_RESULT_SESSION
    assert "contradicts" in out["reason"] or "no verification evidence" in out["reason"]


# 4. timed-out completed_unverified with a valid SESSION -> DEADLINE_OVERRUN (non-success)
def test_timeout_completed_unverified_with_valid_session_is_deadline_overrun():
    out = _timed("completed_unverified", _completed_unverified_session())
    assert out["classification"] == harness.DEADLINE_OVERRUN
    assert out["classification"] != harness.UNVERIFIED_COMPLETION


# 5. no timeout, completed_verified, evaluator PASS -> VERIFIED_SUCCESS (non-timeout behaviour unchanged)
def test_no_timeout_completed_verified_with_evaluator_pass_is_verified_success(completed_session):
    out = _classify(_summary("completed_verified", "ok", scenario="RME-02"), completed_session, scenario="RME-02",
                    evaluator=True)
    assert out["classification"] == harness.VERIFIED_SUCCESS and out["outcome"] == "COMPLETED"


# 6. timeout, interrupted, valid SESSION and deadline signal -> DEADLINE_INTERRUPTED
def test_timeout_interrupted_with_valid_session_and_signal_is_deadline_interrupted():
    out = _timed("interrupted", _load(DRILL), signalled=True,
                 reason="external deadline: the goal was cancelled and its state saved")
    assert out["classification"] == harness.DEADLINE_INTERRUPTED and out["outcome"] == "DEADLINE_GRACEFUL"


# 7. timeout, interrupted, missing SESSION -> INVALID_RESULT_SESSION
def test_timeout_interrupted_with_missing_session_is_invalid_result_session():
    assert _timed("interrupted", None, session_status="MISSING")["classification"] == harness.INVALID_RESULT_SESSION


# 8. timeout, no RESULT -> SUPERVISOR_TIMEOUT_NO_OUTCOME
def test_timeout_without_result_is_supervisor_timeout_no_outcome():
    out = supervisor.classify_trial(status="MISSING", detail="", summary=None, timed_out=True, graceful=False,
                                    session_status="MISSING", session_doc=None, scenario="RME-01", model=MODEL,
                                    policy_id="RME-01-POLICY-V2", evaluator_passed=True)
    assert out["classification"] == harness.SUPERVISOR_TIMEOUT_NO_OUTCOME and out["run_status"] is None


# 9. timeout, BLOCKED, valid SESSION -> BLOCKED, not VERIFIED_SUCCESS, with deadline metadata
def test_timeout_blocked_with_valid_session_stays_blocked_with_deadline_metadata():
    out = _timed("blocked", _load(BLOCKED), signalled=True,
                 reason="the user denied permission for tool run_command; the run stopped")
    assert out["classification"] == harness.BLOCKED
    assert out["checks"]["deadline"] == {"timed_out": True, "graceful": True}


# 10. timeout, LIMIT_REACHED, valid SESSION -> LIMIT_REACHED with deadline metadata
def test_timeout_limit_reached_with_valid_session_stays_limit_reached_with_deadline_metadata():
    out = _timed("limit_reached", _load(LIMIT), signalled=True, reason="the iteration limit was reached",
                 evaluator=False)
    assert out["classification"] == harness.LIMIT_REACHED
    assert out["checks"]["deadline"]["timed_out"] is True
    passed = _timed("limit_reached", _load(LIMIT), signalled=True, reason="the iteration limit was reached",
                    evaluator=True)
    assert passed["classification"] == harness.LIMIT_REACHED_EVALUATOR_PASS, "a distinct class, never a success"


# 11. timeout, runtime_error -> RUNTIME_FAILURE
def test_timeout_runtime_error_is_runtime_failure():
    out = _timed("runtime_error", None, session_status="MISSING", signalled=False, reason="RuntimeError: boom")
    assert out["classification"] == harness.RUNTIME_FAILURE and out["outcome"] == "DEADLINE_RUNTIME_ERROR"


# C. timed-out completion is never success, whatever the deadline metadata says and whichever way the run ended
@pytest.mark.parametrize("signalled, graceful", [(True, True), (False, False), (True, False), (False, True)])
def test_timed_out_completion_is_never_success_regardless_of_deadline_metadata(completed_session, signalled, graceful):
    out = _classify(_summary("completed_verified", "ok", scenario="RME-02", deadline_signalled=signalled),
                    completed_session, scenario="RME-02", timed_out=True, graceful=graceful, evaluator=True)
    assert out["classification"] == harness.DEADLINE_OVERRUN


def test_no_field_orders_completion_against_the_host_deadline():
    """Evidence for the conservative rule: the RESULT and SESSION carry no host-comparable ordering. The session's
    saved_at is a naive timestamp from the container clock; the supervisor's deadline is a host monotonic reading.
    Neither is converted into the other, so completion before the deadline cannot be established."""
    src = inspect.getsource(supervisor.classify_trial)
    assert "saved_at" not in src and "monotonic" not in src and "started" not in src


# 12. drill and supervisor agree for equivalent evidence, and the legacy shortcut is gone
MATRIX = [
    ("completed_verified", True, True, lambda: _load(DRILL)),
    ("interrupted", True, True, lambda: _load(DRILL)),
    ("blocked", True, True, lambda: _load(BLOCKED)),
    ("limit_reached", True, True, lambda: _load(LIMIT)),
]


@pytest.mark.parametrize("run_status, timed_out, graceful, session_loader", MATRIX)
def test_drill_and_supervisor_agree_for_equivalent_evidence(run_status, timed_out, graceful, session_loader):
    doc = session_loader()
    summary = _summary(run_status, "recorded", deadline_signalled=True)
    via_supervisor = supervisor.classify_trial(
        status="OK", detail="", summary=summary, timed_out=timed_out, graceful=graceful, session_status="OK",
        session_doc=doc, scenario="RME-01", model=MODEL, policy_id="RME-01-POLICY-V2", evaluator_passed=True)
    via_drill = drill.drill_classification("OK", "", summary, timed_out, graceful, "OK", doc, evaluator_passed=True)
    assert via_drill == via_supervisor["classification"]


def test_the_drill_has_no_duplicate_classification_shortcut():
    text = (RME / "drill.py").read_text()
    for legacy in ("run.timed_out and status", "harness.DEADLINE_INTERRUPTED", "UNEXPECTED_NORMAL_EXIT"):
        assert legacy not in text, f"drill still contains the legacy shortcut {legacy!r}"
    assert "supervisor.classify_trial" in text
    assert '"RME_POLICY": policy.POLICY_V2' in text, "drill trials must carry the explicit policy"
