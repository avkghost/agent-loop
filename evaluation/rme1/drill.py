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
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import harness  # noqa: E402
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
                                 {"RME_NONCE": nonce})
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
    if run.timed_out and status == "OK":
        record["classification"] = harness.DEADLINE_INTERRUPTED
    elif run.timed_out:
        record["classification"] = harness.SUPERVISOR_TIMEOUT_NO_OUTCOME
    else:
        record["classification"] = "UNEXPECTED_NORMAL_EXIT"
    (results / "drill.json").write_text(harness.bounded(record))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("which", nargs="?", default="both", choices=["graceful", "forced", "both"])
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
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
