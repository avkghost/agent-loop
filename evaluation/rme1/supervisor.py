"""RME host supervisor. HOST ONLY (never copied into the image).

Creates a trial under tmp/rme1/<trial>/ (workspace/ is the only mount the agent sees), verifies the image and the
model, runs the isolation probe, then runs the agent container under an external wall-clock timeout. Results are
read from validated, checksummed frames. The hidden evaluator runs on the host afterwards.

Usage: python supervisor.py <trial-id> --image <ref> --model <id> [--scenario RME-01] [--probe-only]
"""

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
from scenarios import SCENARIOS, make_workspace, protected_unchanged  # noqa: E402
from evaluator import run_hidden_check, write_hidden_check, tree_digest  # noqa: E402
import deadline  # noqa: E402
from deadline import GRACE_SECONDS, SnapshotPoller, run_with_deadline  # noqa: E402
sys.path.insert(0, str(REPO))
from src.session import validate_session_data  # noqa: E402

WALL_CLOCK = 300
ARTIFACT_LIMIT = 256 * 1024
REVISION_LABEL = "org.agent-loop.revision"
# NET_ADMIN and SET* are used only by egress-init.sh, which drops them before the agent starts. KILL lets the
# container's init (tini, root) forward SIGTERM to the agent; the agent itself runs with no capabilities.
CAPS = ["--cap-drop", "ALL", "--cap-add", "NET_ADMIN", "--cap-add", "SETUID", "--cap-add", "SETGID",
        "--cap-add", "SETPCAP", "--cap-add", "KILL"]
HARDENING = ["--init", "--network", "bridge", "--read-only", "--tmpfs", "/tmp:size=64m,mode=1777",
             "--user", "0:0", "--security-opt", "no-new-privileges", "--pids-limit", "128",
             "--memory", "1g", "--cpus", "2"]
EXPECTED_PROBES = {
    "uid": 10001, "cap_eff": "0000000000000000", "no_new_privs": "1",
    "net_lm_studio_allowed": "CONNECTED",
    "net_same_host_other_port_8080": "DENIED_TIMEOUT",
    "net_public_internet_1.1.1.1_443": "DENIED_TIMEOUT",
    "net_lan_192.168.1.1_80": "DENIED_TIMEOUT",
    "net_lan_10.0.0.1_80": "DENIED_TIMEOUT",
    "net_ipv6_public_443": "DENIED_ENETUNREACH",
    "firewall_flush_after_drop": "DENIED", "firewall_policy_change_after_drop": "DENIED",
    "fs_repo_source_write": "DENIED_EROFS", "fs_root_write": "DENIED_EROFS",
    "fs_workspace_write": "WRITABLE", "fs_host_home_visible": "ABSENT",
    "fs_docker_socket": "ABSENT", "fs_hidden_evaluator": "ABSENT",
}


class ImageRejected(Exception):
    """The image is missing, stale, or lacks the expected correction. No trial may start."""


def _run(argv, timeout=120):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def git_state() -> dict:
    head = _run(["git", "-C", str(REPO), "rev-parse", "HEAD"]).stdout.strip()
    dirty = [line[3:] for line in _run(["git", "-C", str(REPO), "status", "--porcelain"]).stdout.splitlines()]
    return {"head": head, "dirty_paths": dirty}


def verify_image(image: str) -> dict:
    """Fail closed. The image must exist, carry a revision label, and contain the RME-1.3 history correction
    and the RME model override. Returns the identity to record."""
    inspect = _run(["docker", "image", "inspect", "--format", "{{.Id}}|{{json .Config}}", image])
    if inspect.returncode != 0 or "|" not in inspect.stdout:
        raise ImageRejected(f"image not found: {image}")
    image_id, config_json = inspect.stdout.strip().split("|", 1)
    labels = (json.loads(config_json).get("Labels") or {}) if config_json not in ("", "null") else {}
    revision = labels.get(REVISION_LABEL, "")
    if not revision:
        raise ImageRejected(f"image {image} has no {REVISION_LABEL} label; it cannot be verified")
    checks = {
        "history_correction": "/opt/agent-loop/src/loop.py", "model_override": "/opt/agent-loop/evaluation/rme1/runner.py",
    }
    markers = {"history_correction": "_provider_arguments", "model_override": "RME_MODEL"}
    for key, path in checks.items():
        proc = _run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/grep", image,
                     "-c", markers[key], path])
        count = proc.stdout.strip() if proc.returncode == 0 else "0"
        if not count.isdigit() or int(count) < 1:
            raise ImageRejected(f"image {image} lacks the {key} ({markers[key]} not found in {path})")
    return {"image_ref": image, "image_id": image_id, "revision_label": revision}


def _run(argv, timeout=120):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def check_model_identity(summary: dict, intended: str):
    """None when every request used the intended model; otherwise a description of the mismatch."""
    actual = summary.get("actual_request_models", [])
    if not actual:
        return "no request model was recorded"
    if set(actual) != {intended}:
        return f"requests used {sorted(set(actual))}, intended {intended}"
    return None


def bounded_text(text: str) -> str:
    raw = text.encode("utf-8", "replace")
    if len(raw) <= ARTIFACT_LIMIT:
        return text
    return raw[:ARTIFACT_LIMIT // 2].decode("utf-8", "replace") + "\n...[truncated]...\n" + \
        raw[-ARTIFACT_LIMIT // 2:].decode("utf-8", "replace")


def parse_frame(text: str, kind: str, nonce: str):
    """Validated frame extraction. Returns (status, object, detail). Status is OK, MISSING, or CORRUPT. The begin
    and end markers must carry the same kind and nonce, and the checksum must match the enclosed bytes."""
    pattern = re.compile(r"RME_FRAME %s BEGIN %s\n(.*?)\nRME_FRAME %s END %s ([0-9a-f]{64})\n" % (
        re.escape(kind), re.escape(nonce), re.escape(kind), re.escape(nonce)), re.S)
    matches = list(pattern.finditer(text))
    if not matches:
        return "MISSING", None, f"no {kind} frame"
    if len(matches) > 1:
        return "CORRUPT", None, f"{len(matches)} {kind} frames; ambiguous"
    body, digest = matches[0].group(1), matches[0].group(2)
    import hashlib
    if hashlib.sha256(body.encode("utf-8")).hexdigest() != digest:
        return "CORRUPT", None, f"{kind} checksum mismatch"
    try:
        return "OK", json.loads(body), ""
    except ValueError as exc:
        return "CORRUPT", None, f"{kind} is not JSON: {exc}"


def trial_argv(name, image, mounts, command, model, env_extra):
    env = ["-e", f"RME_MODEL={model}"]
    for key, value in env_extra.items():
        env += ["-e", f"{key}={value}"]
    return ["docker", "run", "--rm", "--name", name] + env + HARDENING + CAPS + mounts + [image] + command


def docker_cp_session(name: str, scenario: str, dest: Path) -> bool:
    """Read the in-container session into the host. Uses `docker exec cat` rather than `docker cp`: docker cp does
    not reliably see files on a tmpfs mount (the session directory is one), while exec runs inside the container's
    mount namespace. Returns False only when no session file exists yet; any other failure raises.
    The host only writes the destination file; the model has no path into it."""
    proc = subprocess.run(["docker", "exec", name, "cat", f"/tmp/sessions/rme1-{scenario}.json"],
                          capture_output=True, timeout=60)
    if proc.returncode == 0:
        dest.write_bytes(proc.stdout)
        return True
    err = proc.stderr.decode("utf-8", "replace")
    if "No such file" in err:
        return False
    if "No such container" in err:
        raise deadline.SnapshotSourceGone(err.strip()[:160])
    raise RuntimeError(f"session read failed: {err.strip()[:200]}")


def docker_run(name, image, mounts, command, timeout, model, env_extra):
    argv = trial_argv(name, image, mounts, command, model, env_extra)
    started = time.monotonic()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"argv": argv, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
                "timed_out": False, "elapsed": round(time.monotonic() - started, 1)}
    except subprocess.TimeoutExpired as exc:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True)
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {"argv": argv, "returncode": None, "stdout": out, "stderr": err, "timed_out": True,
                "elapsed": round(time.monotonic() - started, 1)}


def container_leftovers(name):
    out = _run(["docker", "ps", "-a", "--filter", f"name={name}", "-q"])
    return len([line for line in out.stdout.split() if line])


def isolation_failures(probe):
    if not probe:
        return ["no probe output"]
    failures = [f"{key}: expected {want!r}, got {probe.get(key)!r}"
                for key, want in EXPECTED_PROBES.items() if probe.get(key) != want]
    inference = probe.get("inference", {})
    if inference.get("http") != 200 or not inference.get("content_chars"):
        failures.append(f"inference not ready: {inference}")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trial")
    parser.add_argument("--image", required=True, help="explicit image reference; no default")
    parser.add_argument("--model", required=True, help="explicit model id; no default")
    parser.add_argument("--scenario", default="RME-01")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    trial_root = REPO / "tmp" / "rme1" / args.trial
    if trial_root.exists():
        raise SystemExit(f"trial directory already exists (never reused): {trial_root}")
    results = trial_root / "results"
    results.mkdir(parents=True)
    record = {"trial": args.trial, "scenario": args.scenario, "intended_model": args.model,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "wall_clock_limit_seconds": WALL_CLOCK,
              "git": git_state()}
    try:
        record["image"] = verify_image(args.image)
    except ImageRejected as exc:
        record.update({"outcome": "IMAGE_REJECTED", "reason": str(exc)})
        (results / "record.json").write_text(harness.bounded(record))
        raise SystemExit(f"IMAGE_REJECTED: {exc}")

    scenario = SCENARIOS[args.scenario]
    fixture = make_workspace(scenario, trial_root, "workspace")
    workspace = Path(fixture["workspace"])
    evaluator_dir = trial_root / "evaluator"  # host-only; outside the workspace and never mounted
    write_hidden_check(scenario, evaluator_dir)
    record.update({"mounts": {"/work": f"{workspace} (rw)"}, "initial_hashes": fixture["initial_hashes"]})
    mounts = ["-v", f"{workspace}:/work:rw"]

    probe = docker_run(f"rme11-probe-{args.trial}"[:60], args.image, mounts,
                       ["/usr/local/bin/python", "/opt/agent-loop/evaluation/rme1/probe.py"], 120, args.model, {})
    (results / "probe.log").write_text(bounded_text(probe["stdout"] + probe["stderr"]))
    probe_json = next((json.loads(line.split(" ", 1)[1]) for line in probe["stdout"].splitlines()
                       if line.startswith("PROBE_JSON ")), None)
    record["probe"] = probe_json
    failures = isolation_failures(probe_json) if probe["returncode"] == 0 else ["probe container failed"]
    if probe_json and probe_json.get("inference", {}).get("model") not in (None, args.model):
        failures.append(f"probe model {probe_json['inference'].get('model')} != intended {args.model}")
    record["isolation_failures"] = failures
    if args.probe_only or failures:
        record["outcome"] = "PROBE_ONLY" if args.probe_only and not failures else "ISOLATION_FAILED"
        (results / "record.json").write_text(harness.bounded(record))
        print(json.dumps({"outcome": record["outcome"], "results": str(results)}))
        return

    nonce = secrets.token_hex(16)
    name = f"rme11-trial-{args.trial}"[:60]
    argv = trial_argv(name, args.image, mounts,
                      ["/usr/local/bin/python", "/opt/agent-loop/evaluation/rme1/runner.py", args.scenario, "/work",
                       "/tmp/sessions"], args.model, {"RME_NONCE": nonce})
    poller = SnapshotPoller(results / "session_snapshot.json",
                            copy_out=lambda dest: docker_cp_session(name, args.scenario, dest),
                            validate=validate_session_data, interval=5.0).start()
    run = run_with_deadline(argv, deadline=WALL_CLOCK, grace=GRACE_SECONDS,
                            force_remove=lambda: _run(["docker", "rm", "-f", name], timeout=60))
    poller.stop()
    run = {"stdout": run.stdout, "stderr": run.stderr, "returncode": run.returncode,
           "timed_out": run.timed_out, "graceful": run.graceful, "elapsed": run.elapsed, "argv": argv}
    (results / "container.log").write_text(bounded_text(run["stdout"] + run["stderr"]))
    record.update({"docker_argv_redacted": [a for a in run["argv"] if not a.startswith("RME_NONCE")],
                   "container_timed_out": run["timed_out"], "container_returncode": run["returncode"],
                   "deadline_graceful": run["graceful"], "wall_clock_seconds": run["elapsed"],
                   "snapshot_status": poller.status,
                   "container_leftover": container_leftovers(name)})

    status, summary, detail = parse_frame(run["stdout"], "RESULT", nonce)
    record["result_capture"] = {"status": status, "detail": detail}
    session_status, session_doc, session_detail = parse_frame(run["stdout"], "SESSION", nonce)
    record["session_capture"] = {"status": session_status, "detail": session_detail}
    if session_status == "OK":
        (results / "session.json").write_text(json.dumps(session_doc, sort_keys=True))
    if status == "OK":
        (results / "result.json").write_text(json.dumps(summary, sort_keys=True))
    record["agent_summary"] = summary if status == "OK" else None

    before = tree_digest(workspace)
    evaluator = run_hidden_check(scenario, workspace, evaluator_dir)
    after = tree_digest(workspace)
    unchanged = protected_unchanged(scenario, workspace, fixture["initial_hashes"])
    record.update({"independent_evaluator": evaluator, "protected_files_unchanged": unchanged,
                   "evaluator_modified_workspace": before != after})

    if run["timed_out"] and status == "OK":
        # The runner published its own RunOutcome after the deadline signal. Keep it as the runtime's record.
        run_status, reason = summary["run_status"], summary["reason"]
        classification = harness.DEADLINE_INTERRUPTED
        record["outcome"] = "DEADLINE_GRACEFUL" if run["graceful"] else "DEADLINE_FORCED_AFTER_RESULT"
    elif run["timed_out"]:
        # Forced termination with no RunOutcome: no status is invented. The last valid snapshot, if any, is
        # recorded as a snapshot, not as a RunOutcome.
        run_status, reason = None, "deadline reached; no RunOutcome was published"
        classification = harness.SUPERVISOR_TIMEOUT_NO_OUTCOME
        record["outcome"] = "DEADLINE_FORCED_NO_OUTCOME"
    elif status != "OK":
        classification, run_status, reason = harness.EVALUATION_ENVIRONMENT_FAILURE, "unavailable", \
            f"result {status}: {detail}"
        record["outcome"] = "RESULT_UNAVAILABLE"
    else:
        run_status, reason = summary["run_status"], summary["reason"]
        mismatch = check_model_identity(summary, args.model)
        if mismatch:
            classification = harness.INVALID_MODEL_MISMATCH
            reason = mismatch
        else:
            classification = harness.classify(run_status, reason, evaluator["passed"] and unchanged)
        record["outcome"] = "COMPLETED"
    record.update({"run_status": run_status, "run_reason": reason, "classification": classification})
    (results / "record.json").write_text(harness.bounded(record))
    print(json.dumps({"classification": classification, "run_status": run_status, "results": str(results)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
