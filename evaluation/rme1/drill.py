"""RME container deadline drill. HOST ONLY (never copied into the image). No LLM request is made.

Graceful drill: the real runner starts a provider request against a silent loopback server inside the container
(127.0.0.1:9999 is allowed by the egress policy; LM Studio is not contacted). The deadline sends SIGTERM through
Docker to the runner, which cancels the in-flight request. The runtime records INTERRUPTED and saves it, and the
runner publishes that outcome.

Forced drill: a stand-in saves a valid session, then ignores SIGTERM. The deadline expires, the container is removed
by force, and no RunOutcome may appear.

Usage: python drill.py [graceful|forced|both] --image <ref>
"""

import argparse
import json
import secrets
import shlex
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import harness  # noqa: E402
import policy  # noqa: E402
import supervisor  # noqa: E402
from deadline import SnapshotPoller, run_with_deadline  # noqa: E402
from src.session import validate_session_data  # noqa: E402

SINK = """import socket, time
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 9999))
s.listen(8)          # accepts connections into the backlog and never answers
time.sleep(3600)
"""

FORCED = """import pathlib
from src import session
session.SESSIONS_DIR = pathlib.Path("/tmp/sessions")
session.save_session("rme1-RME-01", "drill", [{"role": "system", "content": "drill"},
                                              {"role": "user", "content": "Objective: drill"}])
"""

GRACEFUL_CMD = ("python3 /work/drill_sink.py & export RME_LLM_URL=http://127.0.0.1:9999/v1/chat/completions; "
                "exec /usr/local/bin/python /opt/agent-loop/evaluation/rme1/runner.py RME-01 /work /tmp/sessions")
FORCED_CMD = "trap '' TERM; /usr/local/bin/python /work/drill_forced.py; sleep 3600"


def drill_classification(status, detail, summary, timed_out, graceful, session_status, session_doc,
                         evaluator_passed=False):
    """The drills' classification: supervisor.classify_trial with the drill's fixed scenario, model and policy."""
    return supervisor.classify_trial(
        status=status, detail=detail, summary=summary if status == "OK" else None, timed_out=timed_out,
        graceful=graceful, session_status=session_status, session_doc=session_doc, scenario="RME-01",
        model="zai-org/glm-4.7-flash", policy_id=policy.POLICY_V2, evaluator_passed=evaluator_passed)["classification"]


def _drill(kind, image, deadline, grace):
    stamp = time.strftime("%H%M%S")
    root = REPO / "tmp" / "rme1" / f"drill-{kind}-{stamp}-{secrets.token_hex(2)}"
    workspace, results = root / "workspace", root / "results"
    workspace.mkdir(parents=True)
    results.mkdir()
    (workspace / "drill_sink.py").write_text(SINK)
    (workspace / "drill_forced.py").write_text(FORCED)
    name = f"rme15-drill-{kind}-{stamp}"[:60]
    nonce = secrets.token_hex(16)
    command = ["sh", "-c", GRACEFUL_CMD if kind == "graceful" else FORCED_CMD]
    argv = supervisor.trial_argv(name, image, ["-v", f"{workspace}:/work:rw"], command, "zai-org/glm-4.7-flash",
                                 {"RME_NONCE": nonce, "RME_POLICY": policy.POLICY_V2})
    poller = SnapshotPoller(results / "session_snapshot.json",
                            copy_out=lambda dest: supervisor.docker_cp_session(name, "RME-01", dest),
                            validate=validate_session_data, interval=1.0).start()
    started = time.time()
    run = run_with_deadline(argv, deadline=deadline, grace=grace,
                            force_remove=lambda: supervisor._run(["docker", "rm", "-f", name], timeout=60))
    poller.stop()
    (results / "container.log").write_text(run.stdout + run.stderr)
    status, summary, detail = supervisor.parse_frame(run.stdout, "RESULT", nonce)
    snapshot = results / "session_snapshot.json"
    snapshot_doc = json.loads(snapshot.read_text()) if snapshot.exists() else None
    leftovers = supervisor.container_leftovers(name)
    record = {
        "drill": kind, "deadline_s": deadline, "grace_s": grace, "elapsed_s": run.elapsed, "kill_error": run.kill_error,
        "started_at": started, "timed_out": run.timed_out, "graceful": run.graceful,
        "container_returncode": run.returncode, "container_leftover": leftovers,
        "result_frame": {"status": status, "detail": detail,
                         "run_status": summary.get("run_status") if status == "OK" else None,
                         "reason": summary.get("reason") if status == "OK" else None,
                         "deadline_signalled": summary.get("deadline_signalled") if status == "OK" else None},
        "snapshot": {"file_present": snapshot_doc is not None, "status": poller.status,
                     "session_id": snapshot_doc.get("session_id") if snapshot_doc else None,
                     "runs_in_snapshot": [r.get("status") for r in (snapshot_doc or {}).get("runs", [])]},
    }
    # The same classifier as the production supervisor. No drill-specific classification exists.
    session_status, session_doc, _ = supervisor.parse_frame(run.stdout, "SESSION", nonce)
    record["classification"] = drill_classification(status, detail, summary, run.timed_out, run.graceful,
                                                    session_status, session_doc)
    (results / "drill.json").write_text(harness.bounded(record))
    return record


# Enforcement probe: runs inside the image with no model and no network. It installs Policy v2, asks the production
# dispatch path for a managed job, and reports what was advertised. The job's command would create a marker file.
PROBE = """import asyncio, json, os, sys
sys.path.insert(0, "/opt/agent-loop/evaluation/rme1")
import policy
import scenarios
from src import dispatch, loop
meta = policy.install("RME-01-POLICY-V2", scenarios.SCENARIOS["RME-01"].tools)
advertised = [t["function"]["name"] for t in loop.get_all_tools(None)]
rejected = asyncio.run(dispatch.dispatch_tool_async("start_job", {"command": "touch /work/started.txt"}, None, None))
listed = asyncio.run(dispatch.dispatch_tool_async("list_directory", {"path": "."}, None, None))
print("POLICY_PROBE " + json.dumps({"advertised": advertised, "start_job": rejected,
      "started_file_exists": os.path.exists("/work/started.txt"), "list_directory_isError": listed.get("isError"),
      "install_metadata": meta}, sort_keys=True))
"""


def policy_drill(image, expected_revision):
    """Non-LLM drill of the runner's real policy path and the supervisor's real validation path."""
    stamp = time.strftime("%H%M%S")
    root = REPO / "tmp" / "rme1" / f"drill-policy-{stamp}-{secrets.token_hex(2)}"
    workspace, results = root / "workspace", root / "results"
    workspace.mkdir(parents=True)
    results.mkdir()
    (workspace / "drill_sink.py").write_text(SINK)
    image_identity = supervisor.verify_image(image, expected_revision)  # the real image check
    expected = policy.expected_metadata(policy.POLICY_V2, harness.SCENARIOS["RME-01"].tools)
    out = {"image": image_identity, "expected_policy": expected}

    # A: the runner, with Policy v2 applied, publishes RESULT and SESSION through a silent sink (no LLM).
    name_a = f"rme15-policy-run-{stamp}"[:60]
    nonce = secrets.token_hex(16)
    argv = supervisor.trial_argv(name_a, image, ["-v", f"{workspace}:/work:rw"], ["sh", "-c", GRACEFUL_CMD],
                                 "zai-org/glm-4.7-flash", {"RME_NONCE": nonce, "RME_POLICY": policy.POLICY_V2})
    run = run_with_deadline(argv, deadline=12.0, grace=20.0,
                            force_remove=lambda: supervisor._run(["docker", "rm", "-f", name_a], timeout=60))
    status, summary, detail = supervisor.parse_frame(run.stdout, "RESULT", nonce)
    session_status, session_doc, session_detail = supervisor.parse_frame(run.stdout, "SESSION", nonce)
    reported = summary.get("policy") if status == "OK" else None
    out["runner"] = {
        "result_frame": status, "result_detail": detail, "session_frame": session_status,
        "session_detail": session_detail, "run_status": summary.get("run_status") if status == "OK" else None,
        "policy_problems": policy.validate_policy_metadata(
            reported, policy.POLICY_V2, expected,
            reported.get("tool_schema_sha256") if isinstance(reported, dict) else None) if status == "OK"
            else ["no RESULT frame"],
        "result_session_problems": supervisor.terminal_session_problems(
            summary, session_status, session_doc, "RME-01") if status == "OK" else [],
        "container_leftover": supervisor.container_leftovers(name_a),
    }
    out["runner"]["classification"] = supervisor.classify_trial(
        status=status, detail=detail, summary=summary if status == "OK" else None, timed_out=run.timed_out,
        graceful=run.graceful, session_status=session_status, session_doc=session_doc, scenario="RME-01",
        model="zai-org/glm-4.7-flash", policy_id=policy.POLICY_V2, evaluator_passed=False)["classification"]
    (results / "runner.log").write_text(run.stdout + run.stderr)

    # B: the enforcement probe, no model, no network use beyond the container's default egress policy.
    name_b = f"rme15-policy-probe-{stamp}"[:60]
    argv_b = supervisor.trial_argv(name_b, image, ["-v", f"{workspace}:/work:rw"],
                                   ["sh", "-c", "/usr/local/bin/python -B -c " + shlex.quote(PROBE)],
                                   "zai-org/glm-4.7-flash", {"RME_NONCE": secrets.token_hex(8), "RME_POLICY": policy.POLICY_V2})
    probe = run_with_deadline(argv_b, deadline=60.0, grace=10.0,
                              force_remove=lambda: supervisor._run(["docker", "rm", "-f", name_b], timeout=60))
    line = next((ln for ln in probe.stdout.splitlines() if ln.startswith("POLICY_PROBE ")), None)
    probe_doc = json.loads(line.split(" ", 1)[1]) if line else None
    out["probe"] = {"returncode": probe.returncode, "doc": probe_doc, "stderr_tail": probe.stderr[-500:],
                    "container_leftover": supervisor.container_leftovers(name_b)}
    (results / "probe.log").write_text(probe.stdout + probe.stderr)
    (results / "policy_drill.json").write_text(harness.bounded(out))
    return out


# Stand-in containers for the terminal-integrity drill. They print frames in the runner's format (checksummed, nonce
# bound) and call no model. The host parses them with the supervisor's own parse_frame and classify_trial.
STANDIN = r"""import hashlib, json, os, sys, time
nonce = os.environ["RME_NONCE"]
case = os.environ["RME_CASE"]
model = os.environ["RME_MODEL"]
sys.path.insert(0, "/opt/agent-loop/evaluation/rme1")
import policy
def frame(kind, obj, digest=None):
    text = json.dumps(obj, sort_keys=True)
    digest = digest or hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"\nRME_FRAME {kind} BEGIN {nonce}\n{text}\nRME_FRAME {kind} END {nonce} {digest}\n"
meta = policy.expected_metadata("RME-01-POLICY-V2", harness.SCENARIOS["RME-01"].tools)
summary = {"scenario": "RME-01", "intended_model": model, "actual_request_models": [model], "request_count": 1,
           "run_status": "interrupted", "reason": "external deadline: the goal was cancelled and its state saved",
           "policy": dict(meta, tool_schema_sha256=[meta["expected_tool_schema_sha256"]]), "deadline_signalled": True}
if case == "no_outcome_timeout":
    time.sleep(3600)
sys.stdout.write(frame("RESULT", summary)); sys.stdout.flush()
if case == "result_only_timeout":
    time.sleep(3600)
if case == "result_no_session":
    sys.exit(0)
if case == "result_corrupt_session":
    sys.stdout.write(frame("SESSION", {"session_id": "rme1-RME-01", "format": 6}, digest="0" * 64))
if case == "result_invalid_session":
    sys.stdout.write(frame("SESSION", {"session_id": "rme1-RME-01", "format": 99}))
sys.stdout.flush()
"""

SESSION_CASES = {
    # case: (expected classification, description)
    "result_only_timeout": (harness.INVALID_RESULT_SESSION, "RESULT then deadline, no SESSION"),
    "result_no_session": (harness.INVALID_RESULT_SESSION, "RESULT, clean exit, no SESSION"),
    "result_corrupt_session": (harness.INVALID_RESULT_SESSION, "RESULT, SESSION with a bad checksum"),
    "result_invalid_session": (harness.INVALID_RESULT_SESSION, "RESULT, SESSION failing the production validator"),
    "no_outcome_timeout": (harness.SUPERVISOR_TIMEOUT_NO_OUTCOME, "deadline with no frames at all"),
}


def session_drill(image, expected_revision):
    """Non-LLM container drill of terminal-result integrity through the supervisor's real parse and classify path."""
    stamp = time.strftime("%H%M%S")
    root = REPO / "tmp" / "rme1" / f"drill-session-{stamp}-{secrets.token_hex(2)}"
    (root / "workspace").mkdir(parents=True)
    (root / "results").mkdir()
    image_identity = supervisor.verify_image(image, expected_revision)
    out = {"image": image_identity, "cases": {}}
    for case, (expected, description) in SESSION_CASES.items():
        name = f"rme15-sess-{case[:20]}-{stamp}"[:60]
        nonce = secrets.token_hex(16)
        argv = supervisor.trial_argv(name, image, ["-v", f"{root / 'workspace'}:/work:rw"],
                                     ["/usr/local/bin/python", "-B", "-c", STANDIN], "zai-org/glm-4.7-flash",
                                     {"RME_NONCE": nonce, "RME_POLICY": policy.POLICY_V2, "RME_CASE": case})
        run = run_with_deadline(argv, deadline=8.0, grace=4.0,
                                force_remove=lambda: supervisor._run(["docker", "rm", "-f", name], timeout=60))
        status, summary, detail = supervisor.parse_frame(run.stdout, "RESULT", nonce)
        session_status, session_doc, session_detail = supervisor.parse_frame(run.stdout, "SESSION", nonce)
        decided = supervisor.classify_trial(
            status=status, detail=detail, summary=summary if status == "OK" else None, timed_out=run.timed_out,
            graceful=run.graceful, session_status=session_status, session_doc=session_doc, scenario="RME-01",
            model="zai-org/glm-4.7-flash", policy_id=policy.POLICY_V2, evaluator_passed=False)
        out["cases"][case] = {"description": description, "expected": expected,
                              "classification": decided["classification"], "as_expected": decided["classification"] == expected,
                              "result_frame": status, "session_frame": session_status, "session_detail": session_detail,
                              "timed_out": run.timed_out, "reason": decided["reason"][:200],
                              "stderr_tail": run.stderr[-300:],
                              "container_leftover": supervisor.container_leftovers(name)}
    (root / "results" / "session_drill.json").write_text(harness.bounded(out))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("which", nargs="?", default="both", choices=["graceful", "forced", "both", "policy", "session"])
    parser.add_argument("--image", required=True)
    parser.add_argument("--expected-revision", default=None)
    args = parser.parse_args()
    if args.which == "session":
        if not args.expected_revision:
            raise SystemExit("session drill requires --expected-revision")
        print(json.dumps(session_drill(args.image, args.expected_revision), sort_keys=True)[:6000])
        return
    if args.which == "policy":
        if not args.expected_revision:
            raise SystemExit("policy drill requires --expected-revision")
        print(json.dumps(policy_drill(args.image, args.expected_revision), sort_keys=True)[:4000])
        return
    kinds = ["graceful", "forced"] if args.which == "both" else [args.which]
    for kind in kinds:
        if kind == "graceful":
            record = _drill("graceful", args.image, deadline=12.0, grace=20.0)
        else:
            record = _drill("forced", args.image, deadline=8.0, grace=4.0)
        print(json.dumps({k: record[k] for k in ("drill", "timed_out", "graceful", "classification",
                                                  "container_leftover")}, sort_keys=True))


if __name__ == "__main__":
    main()
