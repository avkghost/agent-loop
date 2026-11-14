"""Terminal RESULT/SESSION integrity. Exercises the production validator (src.session.validate_session_data) and the
supervisor's pure classification path (supervisor.classify_trial), which main() also uses. Scripted providers and
recorded artifacts only: no LM Studio, no Docker, no network."""

import copy
import glob
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

from src import session
from src.session import SessionValidationError, validate_session_data
from tests import scripted_provider
from tests.conftest import make_streamed_response

REPO = Path(__file__).resolve().parents[1]
RME = REPO / "evaluation" / "rme1"
FIX = REPO / "tests" / "fixtures" / "rme1"
sys.path.insert(0, str(RME))
import policy  # noqa: E402
import scenarios  # noqa: E402
import supervisor  # noqa: E402
from deadline import SnapshotPoller  # noqa: E402

spec = importlib.util.spec_from_file_location("rme1_harness_integrity", RME / "harness.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)

MODEL = "zai-org/glm-4.7-flash"
V2 = policy.POLICY_V2


def _load(name):
    return json.loads((FIX / name).read_text())


def _summary(run_status, reason="recorded", scenario="RME-01", deadline_signalled=None):
    meta = policy.expected_metadata(V2, scenarios.SCENARIOS[scenario].tools)
    doc = {"scenario": scenario, "intended_model": MODEL, "actual_request_models": [MODEL], "request_count": 1,
           "run_status": run_status, "reason": reason,
           "policy": dict(meta, tool_schema_sha256=[meta["expected_tool_schema_sha256"]])}
    if deadline_signalled is not None:
        doc["deadline_signalled"] = deadline_signalled
    return doc


def _classify(summary, session_doc, *, session_status="OK", scenario="RME-01", status="OK", timed_out=False,
              graceful=None, evaluator=False):
    if isinstance(summary, dict):
        summary = dict(summary, scenario=scenario)  # a RESULT always names the scenario it was produced for
    return supervisor.classify_trial(status=status, detail="", summary=summary, timed_out=timed_out,
                                     graceful=graceful, session_status=session_status, session_doc=session_doc,
                                     scenario=scenario, model=MODEL, policy_id=V2, evaluator_passed=evaluator)


@pytest.fixture
def completed_session(tmp_path, monkeypatch):
    """A real validator-accepted completed_verified session, produced by the scripted harness path (no model)."""
    monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path / "unused")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    turn = lambda name, args, cid: make_streamed_response(finish_reason="tool_calls", tool_calls=[
        {"id": cid, "name": name, "arguments": json.dumps(args)}])
    queue = [turn("run_command", {"command": "python3 -m unittest -q tests.test_stats"}, "c1"),
             turn("write_file", {"path": "calc/stats.py", "content":
                  "def mean(values):\n    if not values:\n        raise ValueError('empty')\n"
                  "    return sum(values) / len(values)\n"}, "c2"),
             make_streamed_response(finish_reason="stop", content="repaired")]
    monkeypatch.setattr(scripted_provider, "POST", lambda *a, **k: queue.pop(0))
    record = harness.run_trial("RME-02", tmp_path, "integrity", tmp_path / "eval", execute=True, endpoint_ok=True)
    assert record["final_run_status"] == "completed_verified", record
    path = glob.glob(str(tmp_path / "**" / "sessions" / "rme-RME-02-*.json"), recursive=True)[0]
    doc = json.loads(Path(path).read_text())
    doc["session_id"] = "rme1-RME-02"  # the runner's identity for this scenario; no other field is touched
    validate_session_data(doc)  # the fixture itself must be accepted by the production validator
    return doc


# --- validator-level: the real session artifacts -------------------------------------------------------------

def test_recorded_sessions_are_accepted_by_the_production_validator():
    for name in ("session_blocked_glm16.json", "session_limit_reached_glm15.json", "session_interrupted_drill.json"):
        assert validate_session_data(_load(name)) == "structured", name


def test_a_corrupted_recorded_session_is_rejected_by_the_production_validator():
    doc = _load("session_blocked_glm16.json")
    doc["runs"][0].pop("status")
    with pytest.raises(SessionValidationError):
        validate_session_data(doc)


# 1. COMPLETED_VERIFIED with a valid SESSION is accepted
def test_completed_verified_with_valid_session_is_accepted(completed_session):
    out = _classify(_summary("completed_verified"), completed_session, scenario="RME-02", evaluator=True)
    assert out["classification"] == harness.VERIFIED_SUCCESS, out["reason"]


# 2-4. COMPLETED_VERIFIED with a missing, corrupt or evidence-free SESSION is rejected
def test_completed_verified_with_missing_session_is_rejected(completed_session):
    out = _classify(_summary("completed_verified"), None, session_status="MISSING", scenario="RME-02", evaluator=True)
    assert out["classification"] == harness.INVALID_RESULT_SESSION
    assert out["classification"] != harness.VERIFIED_SUCCESS


def test_completed_verified_with_corrupt_session_is_rejected(completed_session):
    # A frame with a bad checksum is CORRUPT in parse_frame and never reaches the validator.
    status, doc, _ = supervisor.parse_frame(
        "\nRME_FRAME SESSION BEGIN n\n{}\nRME_FRAME SESSION END n " + "0" * 64 + "\n", "SESSION", "n")
    assert status == "CORRUPT"
    out = _classify(_summary("completed_verified"), doc, session_status=status, scenario="RME-02", evaluator=True)
    assert out["classification"] == harness.INVALID_RESULT_SESSION


def test_completed_verified_with_an_invalid_session_document_is_rejected(completed_session):
    doc = copy.deepcopy(completed_session)
    doc["format"] = 99
    out = _classify(_summary("completed_verified"), doc, scenario="RME-02", evaluator=True)
    assert out["classification"] == harness.INVALID_RESULT_SESSION
    assert "existing validator" in out["reason"]


def test_completed_verified_with_missing_verification_evidence_is_rejected(completed_session):
    doc = copy.deepcopy(completed_session)
    doc["verification_results"] = []
    out = _classify(_summary("completed_verified"), doc, scenario="RME-02", evaluator=True)
    assert out["classification"] == harness.INVALID_RESULT_SESSION
    assert out["classification"] != harness.VERIFIED_SUCCESS


# 5. BLOCKED: valid matching SESSION accepted; missing SESSION rejected
def test_blocked_with_valid_matching_session_is_accepted():
    out = _classify(_summary("blocked", "the user denied permission for tool run_command; the run stopped"),
                    _load("session_blocked_glm16.json"), evaluator=True)
    assert out["classification"] == harness.BLOCKED, out["reason"]


def test_blocked_with_missing_session_is_rejected():
    out = _classify(_summary("blocked"), None, session_status="MISSING")
    assert out["classification"] == harness.INVALID_RESULT_SESSION


# 6. LIMIT_REACHED with a valid SESSION is accepted
def test_limit_reached_with_valid_session_is_accepted():
    out = _classify(_summary("limit_reached", "the iteration limit was reached"),
                    _load("session_limit_reached_glm15.json"))
    assert out["classification"] == harness.LIMIT_REACHED, out["reason"]


# 7-8. DEADLINE_INTERRUPTED: matching SESSION accepted; without SESSION rejected
def test_deadline_interrupted_with_matching_session_is_accepted():
    out = _classify(_summary("interrupted", "external deadline: the goal was cancelled and its state saved",
                             deadline_signalled=True),
                    _load("session_interrupted_drill.json"), timed_out=True, graceful=True)
    assert out["classification"] == harness.DEADLINE_INTERRUPTED, out["reason"]
    assert out["outcome"] == "DEADLINE_GRACEFUL"


def test_deadline_interrupted_without_session_is_rejected():
    out = _classify(_summary("interrupted"), None, session_status="MISSING", timed_out=True, graceful=True)
    assert out["classification"] == harness.INVALID_RESULT_SESSION
    assert out["classification"] != harness.DEADLINE_INTERRUPTED


# 9. SUPERVISOR_TIMEOUT_NO_OUTCOME without SESSION: preserved as a timeout, never a success
def test_forced_timeout_without_outcome_is_preserved_as_timeout_not_success():
    out = _classify(None, None, status="MISSING", session_status="MISSING", timed_out=True, graceful=False,
                    evaluator=True)
    assert out["classification"] == harness.SUPERVISOR_TIMEOUT_NO_OUTCOME
    assert out["run_status"] is None and out["outcome"] == "DEADLINE_FORCED_NO_OUTCOME"
    assert out["classification"] not in (harness.VERIFIED_SUCCESS, harness.UNVERIFIED_COMPLETION)


def test_forced_timeout_never_becomes_an_interrupted_outcome():
    out = _classify(None, _load("session_interrupted_drill.json"), status="MISSING", session_status="OK",
                    timed_out=True, graceful=False)
    assert out["classification"] == harness.SUPERVISOR_TIMEOUT_NO_OUTCOME
    assert out["classification"] != harness.INTERRUPTED and out["run_status"] is None


# 10. contradictory RESULT and SESSION are rejected
def test_result_claiming_interrupted_against_a_blocked_session_is_rejected():
    out = _classify(_summary("interrupted"), _load("session_blocked_glm16.json"))
    assert out["classification"] == harness.INVALID_RESULT_SESSION
    assert "contradicts" in out["reason"]


def test_result_claiming_completed_verified_against_a_blocked_session_is_rejected():
    out = _classify(_summary("completed_verified"), _load("session_blocked_glm16.json"), evaluator=True)
    assert out["classification"] == harness.INVALID_RESULT_SESSION
    assert out["classification"] != harness.VERIFIED_SUCCESS


def test_session_with_a_foreign_identity_is_rejected():
    doc = _load("session_blocked_glm16.json")
    out = _classify(_summary("blocked"), doc, scenario="RME-02")
    assert out["classification"] == harness.INVALID_RESULT_SESSION and "is not rme1-RME-02" in out["reason"]


# 11. a last-known snapshot does not replace a missing terminal SESSION
def test_last_known_snapshot_does_not_replace_a_missing_terminal_session(tmp_path):
    dest = tmp_path / "session_snapshot.json"
    poller = SnapshotPoller(dest, copy_out=lambda p: (p.write_text(json.dumps(_load("session_blocked_glm16.json"))) or True),
                            validate=validate_session_data)
    assert poller.poll_once() is True and dest.exists(), "a valid snapshot exists on the host"
    out = _classify(_summary("interrupted"), None, session_status="MISSING", timed_out=True, graceful=True)
    assert out["classification"] == harness.INVALID_RESULT_SESSION
    assert "snapshot" not in inspect.signature(supervisor.classify_trial).parameters, \
        "the classifier has no input through which a snapshot could stand in for a SESSION"


def test_absent_session_for_a_non_terminal_or_absent_outcome_is_not_invented(tmp_path):
    assert supervisor.terminal_session_problems({"run_status": None}, "MISSING", None, "RME-01") == []
    assert supervisor.terminal_session_problems({"run_status": "runtime_error"}, "MISSING", None, "RME-01") == []


# 12. evaluator PASS is never converted into a runtime success
def test_evaluator_pass_never_becomes_completed_verified_without_runtime_verification():
    out = _classify(_summary("blocked"), _load("session_blocked_glm16.json"), evaluator=True)
    assert out["classification"] == harness.BLOCKED
    out = _classify(_summary("interrupted", deadline_signalled=True), _load("session_interrupted_drill.json"), evaluator=True,
                    timed_out=True, graceful=True)
    assert out["classification"] == harness.DEADLINE_INTERRUPTED
