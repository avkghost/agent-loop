"""RME deadline, snapshot, and classification tests. Deterministic: child processes stand in for the container.
No LLM, no Docker, no network."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RME = REPO / "evaluation" / "rme1"
sys.path.insert(0, str(RME))
import deadline  # noqa: E402
import harness  # noqa: E402
import supervisor  # noqa: E402
from src import session  # noqa: E402

GRACEFUL_SHIM = r'''
import signal, sys, time
def on_term(signum, frame):
    print("RME_FRAME RESULT BEGIN n1", flush=True)
    print('{"run_status": "interrupted"}', flush=True)
    print("RME_FRAME RESULT END n1 " + __import__("hashlib").sha256(b'{"run_status": "interrupted"}').hexdigest(), flush=True)
    sys.exit(1)
signal.signal(signal.SIGTERM, on_term)
for i in range(10000):
    print("provider streaming chunk", i, flush=True)   # simulated provider streaming
    time.sleep(0.05)
'''

IGNORE_SHIM = r'''
import os, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
open(sys.argv[1], "w").write(str(child.pid))
time.sleep(120)
'''

SILENT_SHIM = r'''
import signal, sys, time
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(2))   # exits on TERM without publishing a frame
time.sleep(120)
'''


def _shim(tmp_path, body):
    path = tmp_path / "shim.py"
    path.write_text(body)
    return [sys.executable, str(path)]


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    except OSError:
        return True
    return bool(state) and not state.startswith("Z")


def test_timeout_during_streaming_is_graceful_when_the_runner_publishes_its_outcome(tmp_path):
    result = deadline.run_with_deadline(_shim(tmp_path, GRACEFUL_SHIM), deadline=1.0, grace=10.0)
    assert result.timed_out is True and result.graceful is True
    assert "RME_FRAME RESULT BEGIN n1" in result.stdout
    assert "provider streaming chunk" in result.stdout, "the streamed output before the deadline is preserved"


def test_forced_termination_kills_the_process_group_and_reports_not_graceful(tmp_path):
    pid_file = tmp_path / "child.pid"
    started = time.monotonic()
    result = deadline.run_with_deadline(_shim(tmp_path, IGNORE_SHIM) + [str(pid_file)], deadline=1.0, grace=1.0)
    assert result.timed_out is True and result.graceful is False
    assert time.monotonic() - started < 30, "forced termination must be bounded"
    child_pid = int(pid_file.read_text())
    time.sleep(0.2)
    assert not _pid_alive(child_pid), "no orphaned child process may survive forced termination"


def test_forced_termination_calls_the_container_remover(tmp_path):
    removed = []
    result = deadline.run_with_deadline(_shim(tmp_path, IGNORE_SHIM) + [str(tmp_path / "c.pid")],
                                        deadline=0.5, grace=0.5, force_remove=lambda: removed.append(True))
    assert result.graceful is False and removed == [True]


def test_silent_exit_after_sigterm_is_graceful_but_publishes_no_result(tmp_path):
    result = deadline.run_with_deadline(_shim(tmp_path, SILENT_SHIM), deadline=0.5, grace=10.0)
    assert result.timed_out and result.graceful is True
    assert "RME_FRAME RESULT" not in result.stdout, "no frame may be fabricated"
    assert supervisor.parse_frame(result.stdout, "RESULT", "n1")[0] == "MISSING"


def test_snapshot_poller_keeps_the_last_valid_snapshot_and_rejects_invalid_copies(tmp_path):
    dest = tmp_path / "session_snapshot.json"
    good = session.build_candidate("snap", "goal", [{"role": "system", "content": "sys"},
                                                    {"role": "user", "content": "Objective: goal"}])
    state = {"mode": "good"}

    def copy_out(path):
        if state["mode"] == "missing":
            return False
        if state["mode"] == "good":
            path.write_text(json.dumps(good))
        else:
            path.write_text('{"format": 6, "messages": [')  # a partially written or corrupt copy
        return True

    poller = deadline.SnapshotPoller(dest, copy_out=copy_out, validate=session.validate_session_data)
    assert poller.poll_once() is True and json.loads(dest.read_text())["session_id"] == "snap"
    state["mode"] = "corrupt"
    assert poller.poll_once() is False
    assert json.loads(dest.read_text())["session_id"] == "snap", "the last valid snapshot must survive a bad copy"
    assert poller.status["rejected"] == 1
    state["mode"] = "missing"
    assert poller.poll_once() is False and poller.status["last_error"] == "no session file yet"
    assert not list(tmp_path.glob("snapshot-*")), "temporary copies are removed"


def test_trial_command_mounts_only_the_workspace_and_no_session_or_result_storage(tmp_path):
    argv = supervisor.trial_argv("n", "img", ["-v", f"{tmp_path}:/work:rw"], ["cmd"], "m", {"RME_NONCE": "x"})
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert mounts == [f"{tmp_path}:/work:rw"]
    assert not any("tmp/rme1" in a and "workspace" not in a for a in argv)
    assert "--rm" in argv and "--init" in argv


def test_limit_reached_with_evaluator_pass_is_distinct_from_task_failure():
    assert harness.classify("limit_reached", "the iteration limit", True) == harness.LIMIT_REACHED_EVALUATOR_PASS
    assert harness.classify("limit_reached", "the iteration limit", False) == harness.LIMIT_REACHED
    assert harness.LIMIT_REACHED_EVALUATOR_PASS != harness.TASK_FAILURE


def test_other_terminal_outcomes_keep_their_own_classifications():
    assert harness.classify("completed_verified", "ok", True) == harness.VERIFIED_SUCCESS
    assert harness.classify("completed_verified", "ok", False) == harness.FALSE_POSITIVE_VERIFICATION
    assert harness.classify("provider_failed", "x", None) == harness.PROVIDER_FAILURE
    assert harness.classify("blocked", "denied", True) == harness.BLOCKED
    assert harness.classify("runtime_error", "x", None) == harness.RUNTIME_FAILURE


def test_deadline_outcomes_are_never_success_and_never_invent_a_status():
    assert harness.DEADLINE_INTERRUPTED not in (harness.VERIFIED_SUCCESS, harness.UNVERIFIED_COMPLETION)
    assert harness.SUPERVISOR_TIMEOUT_NO_OUTCOME not in (harness.VERIFIED_SUCCESS,)
    assert harness.classify("interrupted", "deadline", True) == harness.INTERRUPTED, \
        "the runtime status keeps its own meaning; the supervisor's label is separate"


def test_missing_or_corrupt_result_frames_are_never_success(tmp_path):
    assert supervisor.parse_frame("", "RESULT", "n")[0] == "MISSING"
    assert supervisor.parse_frame("\nRME_FRAME RESULT BEGIN n\n{}\nRME_FRAME RESULT END n " + "0" * 64 + "\n",
                                  "RESULT", "n")[0] == "CORRUPT"


def _good_doc(session_id="snap"):
    return session.build_candidate(session_id, "goal", [{"role": "system", "content": "sys"},
                                                        {"role": "user", "content": "Objective: goal"}])


def test_mid_save_failure_keeps_the_previous_snapshot_and_leaves_no_temp_file(tmp_path, monkeypatch):
    """Deterministic fault injection: the atomic replace fails after a valid copy was made. The previous valid
    snapshot must remain byte-identical and no temporary copy may be left behind."""
    dest = tmp_path / "session_snapshot.json"
    doc = _good_doc()
    poller = deadline.SnapshotPoller(dest, copy_out=lambda p: (p.write_text(json.dumps(doc)) or True),
                                     validate=session.validate_session_data)
    assert poller.poll_once() is True
    before = dest.read_bytes()
    real_replace = deadline.os.replace
    calls = {"n": 0}

    def failing_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("injected failure during save")
        return real_replace(src, dst)

    monkeypatch.setattr(deadline.os, "replace", failing_replace)
    assert poller.poll_once() is False
    assert dest.read_bytes() == before, "a failed save must not change the last valid snapshot"
    assert poller.status["rejected"] == 1 and "injected failure" in poller.status["last_rejection"]
    assert not list(tmp_path.glob("snapshot-*")), "no temporary copy may survive a failed save"


def test_vanished_container_is_reported_separately_and_keeps_the_last_valid_snapshot(tmp_path):
    dest = tmp_path / "session_snapshot.json"
    doc = _good_doc()
    state = {"gone": False}

    def copy_out(path):
        if state["gone"]:
            raise deadline.SnapshotSourceGone("No such container: x")
        path.write_text(json.dumps(doc))
        return True

    poller = deadline.SnapshotPoller(dest, copy_out=copy_out, validate=session.validate_session_data)
    assert poller.poll_once() is True
    state["gone"] = True
    assert poller.poll_once() is False
    assert poller.status["source_gone"] == 1 and poller.status["rejected"] == 0
    assert json.loads(dest.read_text())["session_id"] == "snap"


def test_snapshot_lag_interprets_the_naive_container_timestamp_as_utc(tmp_path):
    """The session writes naive datetime.now() inside the UTC container. The lag must not depend on the host zone."""
    from datetime import datetime, timezone
    doc = _good_doc()
    doc["saved_at"] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    poller = deadline.SnapshotPoller(tmp_path / "s.json", copy_out=lambda p: (p.write_text(json.dumps(doc)) or True),
                                     validate=session.validate_session_data)
    assert poller.poll_once() is True
    assert abs(poller.status["last_lag_seconds"]) < 5, poller.status["last_lag_seconds"]


def test_refused_group_kill_is_recorded_not_raised(tmp_path, monkeypatch):
    def refuse(pid, sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(deadline.os, "killpg", refuse)
    result = deadline.run_with_deadline(_shim(tmp_path, IGNORE_SHIM) + [str(tmp_path / "r.pid")],
                                        deadline=0.5, grace=0.5, force_remove=lambda: None)
    assert result.timed_out is True and result.graceful is False
    assert result.kill_error and "refused" in result.kill_error


def test_docker_cp_error_mapping_separates_missing_file_vanished_container_and_other_failures(tmp_path, monkeypatch):
    dest = tmp_path / "copy.json"

    def fake(stderr, returncode=1):
        def run(*a, **k):
            return subprocess.CompletedProcess(a, returncode, b"", stderr.encode())
        return run

    monkeypatch.setattr(supervisor.subprocess, "run", fake("Error: No such file or directory"))
    assert supervisor.docker_cp_session("n", "RME-01", dest) is False
    monkeypatch.setattr(supervisor.subprocess, "run", fake("Error response from daemon: No such container: n"))
    with pytest.raises(deadline.SnapshotSourceGone):
        supervisor.docker_cp_session("n", "RME-01", dest)
    monkeypatch.setattr(supervisor.subprocess, "run", fake("permission denied"))
    with pytest.raises(RuntimeError, match="session read failed"):
        supervisor.docker_cp_session("n", "RME-01", dest)
