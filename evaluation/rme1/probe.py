"""RME-1.1 isolation probes. Runs inside the final agent environment, after the firewall and privilege drop.
Prints one JSON document. Standard library only. Makes no model call except one bounded inference check."""

import errno
import json
import os
import socket
import subprocess
import sys
import http.client

RESULTS = {}
LLM_IP = os.environ.get("RME_LLM_URL", "").split("//")[-1].split(":")[0]


def tcp(host, port, family=socket.AF_INET, timeout=3):
    s = socket.socket(family, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return "CONNECTED"
    except socket.timeout:
        return "DENIED_TIMEOUT"
    except OSError as exc:
        if exc.errno == errno.ECONNREFUSED:
            return "REFUSED_REACHED_NETWORK"
        return f"DENIED_{errno.errorcode.get(exc.errno, type(exc).__name__)}"
    finally:
        s.close()


def sh(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    return proc.returncode, (proc.stdout + proc.stderr).strip()[:160]


def status_line(path, key):
    with open(path) as fh:
        for line in fh:
            if line.startswith(key):
                return line.split(":", 1)[1].strip()
    return "?"


RESULTS["uid"] = os.getuid()
RESULTS["cap_eff"] = status_line("/proc/self/status", "CapEff")
RESULTS["no_new_privs"] = status_line("/proc/self/status", "NoNewPrivs")

RESULTS["net_lm_studio_allowed"] = tcp(LLM_IP, 1234)
RESULTS["net_same_host_other_port_8080"] = tcp(LLM_IP, 8080)
RESULTS["net_public_internet_1.1.1.1_443"] = tcp("1.1.1.1", 443)
RESULTS["net_lan_192.168.1.1_80"] = tcp("192.168.1.1", 80)
RESULTS["net_lan_10.0.0.1_80"] = tcp("10.0.0.1", 80)
RESULTS["net_ipv6_public_443"] = tcp("2606:4700:4700::1111", 443, family=socket.AF_INET6)
RESULTS["net_loopback_8080_reached_network_stack"] = tcp("127.0.0.1", 8080)

rc, out = sh(["iptables", "-F", "OUTPUT"])
RESULTS["firewall_flush_after_drop"] = "DENIED" if rc != 0 else "ALLOWED_UNEXPECTED"
rc, out = sh(["iptables", "-P", "OUTPUT", "ACCEPT"])
RESULTS["firewall_policy_change_after_drop"] = "DENIED" if rc != 0 else "ALLOWED_UNEXPECTED"

for label, path in [("fs_repo_source_write", "/opt/agent-loop/probe-write"), ("fs_root_write", "/probe-write"),
                    ("fs_egress_script_write", "/opt/agent-loop/evaluation/rme1/egress-init.sh")]:
    try:
        with open(path, "w") as fh:
            fh.write("x")
        RESULTS[label] = "WRITABLE_UNEXPECTED"
        os.remove(path)
    except OSError as exc:
        RESULTS[label] = f"DENIED_{errno.errorcode.get(exc.errno, '?')}"
try:
    with open("/work/probe-write", "w") as fh:
        fh.write("ok")
    os.remove("/work/probe-write")
    RESULTS["fs_workspace_write"] = "WRITABLE"
except OSError as exc:
    RESULTS["fs_workspace_write"] = f"NOT_WRITABLE_{exc.errno}"
RESULTS["fs_host_home_visible"] = "VISIBLE_UNEXPECTED" if os.path.exists("/Users") or os.path.exists(os.path.expanduser("~/.ssh")) else "ABSENT"
RESULTS["fs_docker_socket"] = "PRESENT_UNEXPECTED" if os.path.exists("/var/run/docker.sock") else "ABSENT"
hidden = subprocess.run(["sh", "-c", "find / -xdev -name 'hidden_rme*' 2>/dev/null | head -1"],
                        capture_output=True, text=True).stdout.strip()
RESULTS["fs_hidden_evaluator"] = "PRESENT_UNEXPECTED" if hidden else "ABSENT"

# Inference readiness: a generous budget so visible output is produced; reasoning tokens are reported separately.
MODEL = os.environ.get("RME_MODEL", "zai-org/glm-4.7-flash")
payload = ('{"model": "%s", "messages": [{"role": "user", "content": "Reply with the single word OK."}],'
           ' "max_tokens": 1024, "temperature": 0.0, "stream": false}' % MODEL)
try:
    host, port, path = LLM_IP, 1234, "/v1/chat/completions"
    conn = http.client.HTTPConnection(host, port, timeout=120)
    conn.request("POST", path, body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = json.loads(resp.read(200000).decode("utf-8", "replace"))
    choice = body["choices"][0]
    RESULTS["inference"] = {"http": resp.status, "model": body.get("model"),
                            "finish_reason": choice.get("finish_reason"),
                            "content_chars": len((choice.get("message", {}).get("content") or "")),
                            "content_preview": (choice.get("message", {}).get("content") or "")[:60],
                            "usage": body.get("usage")}
except Exception as exc:
    RESULTS["inference"] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}

print("PROBE_JSON " + json.dumps(RESULTS, sort_keys=True))
