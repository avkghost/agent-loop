"""RME-1.1 host supervisor. HOST ONLY (never copied into the image).

Creates a trial under tmp/rme1/<trial>/ (workspace/ is the only mount the agent sees), runs the agent container under an
external wall-clock timeout, captures bounded artifacts, then runs the hidden evaluator on the host.

Usage: python supervisor.py <trial-id> [--scenario RME-01] [--probe-only]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
from scenarios import SCENARIOS, make_workspace, protected_unchanged, build_criteria  # noqa: E402,F401
from evaluator import run_hidden_check, write_hidden_check  # noqa: E402

IMAGE = "agent-loop-rme1:1.1b"
WALL_CLOCK = 300
ARTIFACT_LIMIT = 256 * 1024
SESSION_LIMIT = 2 * 1024 * 1024
CAPS = ["--cap-drop", "ALL", "--cap-add", "NET_ADMIN", "--cap-add", "SETUID", "--cap-add", "SETGID",
        "--cap-add", "SETPCAP"]  # NET_ADMIN and SET* only for firewall setup; egress-init.sh drops them
HARDENING = ["--init", "--network", "bridge", "--read-only", "--tmpfs", "/tmp:size=64m,mode=1777",
             "--user", "0:0", "--security-opt", "no-new-privileges", "--pids-limit", "128",
             "--memory", "1g", "--cpus", "2"]


def bounded_text(text: str) -> str:
    raw = text.encode("utf-8", "replace")
    if len(raw) <= ARTIFACT_LIMIT:
        return text
    return raw[:ARTIFACT_LIMIT // 2].decode("utf-8", "replace") + "\n...[truncated]...\n" + \
        raw[-ARTIFACT_LIMIT // 2:].decode("utf-8", "replace")


def docker_run(name, mounts, command, timeout):
    argv = ["docker", "run", "--rm", "--name", name] + HARDENING + CAPS + mounts + [IMAGE] + command
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


def isolation_failures(probe):
    """Every expectation must hold. Returns the list of mismatches (empty means isolation verified)."""
    if not probe:
        return ["no probe output"]
    failures = [f"{key}: expected {want!r}, got {probe.get(key)!r}"
                for key, want in EXPECTED_PROBES.items() if probe.get(key) != want]
    inference = probe.get("inference", {})
    if inference.get("http") != 200 or not inference.get("content_chars"):
        failures.append(f"inference not ready: {inference}")
    return failures


def container_leftovers(name):
    out = subprocess.run(["docker", "ps", "-a", "--filter", f"name={name}", "-q"], capture_output=True, text=True)
    return len([line for line in out.stdout.split() if line])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trial")
    parser.add_argument("--scenario", default="RME-01")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    trial_root = REPO / "tmp" / "rme1" / args.trial
    if trial_root.exists():
        raise SystemExit(f"trial directory already exists: {trial_root}")
    results = trial_root / "results"
    results.mkdir(parents=True)
    scenario = SCENARIOS[args.scenario]
    fixture = make_workspace(scenario, trial_root, "workspace")
    workspace = Path(fixture["workspace"])
    evaluator_dir = trial_root / "evaluator"  # host-only; outside the workspace and never mounted
    write_hidden_check(scenario, evaluator_dir)

    mounts = ["-v", f"{workspace}:/work:rw"]
    record = {"trial": args.trial, "scenario": args.scenario, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "image": IMAGE, "mounts": {"/work": f"{workspace} (rw)"},
              "wall_clock_limit_seconds": WALL_CLOCK, "initial_hashes": fixture["initial_hashes"]}

    probe = docker_run(f"rme11-probe-{args.trial}"[:60], mounts,
                       ["/usr/local/bin/python", "/opt/agent-loop/evaluation/rme1/probe.py"], timeout=120)
    (results / "probe.log").write_text(bounded_text(probe["stdout"] + probe["stderr"]))
    probe_json = next((json.loads(line.split(" ", 1)[1]) for line in probe["stdout"].splitlines()
                       if line.startswith("PROBE_JSON ")), None)
    record["probe"] = probe_json
    record["probe_returncode"] = probe["returncode"]
    failures = isolation_failures(probe_json) if probe["returncode"] == 0 else ["probe container failed"]
    record["isolation_failures"] = failures
    if args.probe_only or failures:
        record["outcome"] = "PROBE_ONLY" if args.probe_only and not failures else "ISOLATION_FAILED"
        (results / "record.json").write_text(harness.bounded(record))
        print(json.dumps({"outcome": record["outcome"], "results": str(results)}))
        return

    run = docker_run(f"rme11-trial-{args.trial}"[:60], mounts,
                     ["/usr/local/bin/python", "/opt/agent-loop/evaluation/rme1/runner.py", args.scenario, "/work",
                      "/tmp/sessions"], timeout=WALL_CLOCK)
    (results / "container.log").write_text(bounded_text(run["stdout"] + run["stderr"]))
    record["docker_argv"] = run["argv"]
    record["container_timed_out"] = run["timed_out"]
    record["container_returncode"] = run["returncode"]
    record["wall_clock_seconds"] = run["elapsed"]
    record["container_leftover"] = container_leftovers(f"rme11-trial-{args.trial}"[:60])
    summary = next((json.loads(line.split(" ", 1)[1]) for line in run["stdout"].splitlines()
                    if line.startswith("RESULT_JSON ")), None)
    session_line = next((line.split(" ", 1)[1] for line in run["stdout"].splitlines()
                         if line.startswith("SESSION_JSON ")), None)
    if session_line and len(session_line) <= SESSION_LIMIT:
        (results / "session.json").write_text(session_line)
    record["agent_summary"] = summary

    evaluator = run_hidden_check(scenario, workspace, evaluator_dir)
    unchanged = protected_unchanged(scenario, workspace, fixture["initial_hashes"])
    run_status = summary["run_status"] if summary else "runtime_error"
    reason = summary["reason"] if summary else ("container timed out" if run["timed_out"] else "no summary emitted")
    if run["timed_out"]:
        run_status, reason = "interrupted", "external wall-clock limit reached"
    record["independent_evaluator"] = evaluator
    record["protected_files_unchanged"] = unchanged
    record["classification"] = harness.classify(run_status, reason, evaluator["passed"] and unchanged)
    record["run_status"] = run_status
    record["run_reason"] = reason
    record["outcome"] = "COMPLETED" if summary else "AGENT_ERROR"
    (results / "record.json").write_text(harness.bounded(record))
    print(json.dumps({"classification": record["classification"], "run_status": run_status,
                      "results": str(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
