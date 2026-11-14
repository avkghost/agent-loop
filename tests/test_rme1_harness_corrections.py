"""RME-1.4 harness corrections. Deterministic: scripted provider, no LLM, no network, no real container required
(Docker-dependent checks skip where the Docker CLI or daemon is unavailable, including inside the Linux suite)."""

import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src import loop, runs, session
from src.authorization import SessionAuthorization
from src.runtime import AgentRuntime
from src.tools import shell
from src.tools.mcp_client import MCPClientManager
from tests import scripted_provider
from tests.conftest import make_streamed_response

REPO = Path(__file__).resolve().parents[1]
RME = REPO / "evaluation" / "rme1"
sys.path.insert(0, str(RME))
import evaluator  # noqa: E402
import scenarios  # noqa: E402
import supervisor  # noqa: E402


def _docker_available():
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0


needs_docker = pytest.mark.skipif(not _docker_available(), reason="Docker CLI/daemon not available here")


def _subprocess_python(code, env_extra, drop=()):
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(env_extra)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO), env=env,
                          timeout=120)


# 1. explicit image; no hardcoded default
def test_supervisor_requires_an_explicit_image_and_model():
    proc = subprocess.run([sys.executable, str(RME / "supervisor.py"), "T", "--model", "m", "--probe-only"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0 and "--image" in proc.stderr
    proc = subprocess.run([sys.executable, str(RME / "supervisor.py"), "T", "--image", "x", "--probe-only"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0 and "--model" in proc.stderr
    assert "1.1b" not in (RME / "supervisor.py").read_text(), "no hardcoded stale image reference"


# 2. stale and missing images are rejected before any container runs
@needs_docker
def test_stale_image_without_the_revision_label_is_rejected():
    with pytest.raises(supervisor.ImageRejected, match="revision label"):
        supervisor.verify_image("agent-loop-rme1:1.1b")


@needs_docker
def test_missing_image_is_rejected():
    with pytest.raises(supervisor.ImageRejected, match="not found"):
        supervisor.verify_image("agent-loop-rme1:does-not-exist-rme-test")


# 3-4. the selected model reaches the provider, and the actual request model is recorded separately
def test_model_override_reaches_the_provider_configuration():
    code = ("import sys; sys.path.insert(0, 'evaluation/rme1'); import runner; from src import provider; "
            "print(provider.GenerationOptions().model)")
    proc = _subprocess_python(code, {"RME_MODEL": "qwen/qwen3.8-27b", "RME_LLM_URL": "http://h:1/v1/chat/completions"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "qwen/qwen3.8-27b"


def test_intended_and_actual_request_models_are_recorded_and_compared():
    code = ("import sys, json; sys.path.insert(0, 'evaluation/rme1'); import runner; from src import provider; "
            "runner.install_request_recorder(); "
            "provider.build_payload([{'role': 'user', 'content': 'x'}], None, provider.GenerationOptions()); "
            "print(json.dumps(runner.REQUEST_MODELS))")
    proc = _subprocess_python(code, {"RME_MODEL": "zai-org/glm-4.7-flash", "RME_LLM_URL": "http://h:1/v1/chat/completions"})
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == ["zai-org/glm-4.7-flash"]


# 5. no default model: a missing override stops before any request
def test_missing_model_override_does_not_select_another_model():
    proc = _subprocess_python("import sys; sys.path.insert(0, 'evaluation/rme1'); import runner",
                              {"RME_LLM_URL": "http://h:1/v1/chat/completions"}, drop=("RME_MODEL",))
    assert proc.returncode != 0 and "RME_MODEL is required" in proc.stderr


# 6-8. non-interactive denial: no stdin read, a denial result, no side effect, no fabricated UNKNOWN
def _scripted_run(monkeypatch, tmp_path, replies):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(loop, "load_project_guidelines", lambda: None)
    scripted = list(replies)
    monkeypatch.setattr(scripted_provider, "POST", lambda *a, **k: scripted.pop(0))


def test_non_interactive_denial_never_reads_stdin_and_has_no_side_effect(monkeypatch, tmp_path):
    class NoStdin:
        def read(self, *a, **k):
            raise AssertionError("stdin must never be read in a non-interactive trial")

        def readline(self, *a, **k):
            raise AssertionError("stdin must never be read in a non-interactive trial")

    monkeypatch.setattr(sys, "stdin", NoStdin())
    monkeypatch.setattr(loop, "_get_single_key_input", lambda prompt, valid_keys: "3")  # the runner's denial
    spawned = []

    async def spy(*args, **kwargs):
        spawned.append(args)
        return {}

    monkeypatch.setattr(shell, "_run", spy)
    _scripted_run(monkeypatch, tmp_path, [
        make_streamed_response(finish_reason="tool_calls", tool_calls=[
            {"id": "d1", "name": "run_command", "arguments": json.dumps({"command": "uname -s"})}]),
        make_streamed_response(finish_reason="stop", content="stopping"),
    ])

    async def go():
        return await loop.agent_loop_async("try a denied command", MCPClientManager(),
                                           asyncio.get_running_loop(), session_id="denial",
                                           authorization=SessionAuthorization())
    outcome = asyncio.run(go())
    assert spawned == [], "a denied command must have no side effect"
    assert outcome.run_status is runs.RunStatus.BLOCKED
    saved = session.load_session("denial")
    session.validate_session_data(saved)
    record = next(r for r in saved["execution"] if r["tool_call_id"] == "d1")
    assert record["status"] == "failed" and record["error_category"] == "permission"
    assert not any(r["status"] == "unknown" for r in saved["execution"]), "a denial must not become UNKNOWN"
    result = next(m for m in saved["messages"] if m.get("role") == "tool" and m.get("tool_call_id") == "d1")
    assert "denied by user" in result["content"]


# 9-10. result capture is independent of terminal formatting, and missing or corrupt results are explicit
def _frame(kind, nonce, obj):
    text = json.dumps(obj, sort_keys=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"\nRME_FRAME {kind} BEGIN {nonce}\n{text}\nRME_FRAME {kind} END {nonce} {digest}\n"


def test_result_frame_is_found_despite_prompt_text_without_a_newline():
    text = "Allow this? (1. Yes, only | 2. Yes, this session | 3. No): " + _frame("RESULT", "n1", {"ok": 1})
    status, obj, _ = supervisor.parse_frame(text, "RESULT", "n1")
    assert status == "OK" and obj == {"ok": 1}


def test_missing_and_corrupt_result_frames_are_reported_explicitly():
    assert supervisor.parse_frame("no frame here", "RESULT", "n1")[0] == "MISSING"
    corrupt = _frame("RESULT", "n1", {"ok": 1}).replace('"ok": 1', '"ok": 2')
    assert supervisor.parse_frame(corrupt, "RESULT", "n1")[0] == "CORRUPT"
    doubled = _frame("RESULT", "n1", {"a": 1}) + _frame("RESULT", "n1", {"a": 1})
    assert supervisor.parse_frame(doubled, "RESULT", "n1")[0] == "CORRUPT"
    assert supervisor.parse_frame(_frame("RESULT", "n1", {"a": 1}), "RESULT", "other-nonce")[0] == "MISSING"


# 11. the evaluator writes no bytecode into the evaluated tree and leaves it unchanged
def test_evaluator_writes_no_bytecode_into_the_workspace(tmp_path):
    scenario = scenarios.SCENARIOS["RME-02"]
    created = scenarios.make_workspace(scenario, tmp_path, "evaluated")
    workspace = Path(created["workspace"])
    (workspace / "calc" / "stats.py").write_text(
        "def mean(values):\n    if not values:\n        raise ValueError('empty')\n    return sum(values) / len(values)\n")
    before = evaluator.tree_digest(workspace)
    result = evaluator.run_hidden_check(scenario, workspace, tmp_path / "evaluator")
    assert result["passed"] is True
    assert not list(workspace.rglob("__pycache__")), "the evaluator must not create bytecode in the workspace"
    assert evaluator.tree_digest(workspace) == before


# 12. isolation expectations and hardening settings are unchanged
def test_isolation_expectations_and_hardening_are_unchanged():
    expected = supervisor.EXPECTED_PROBES
    for key in ("firewall_flush_after_drop", "firewall_policy_change_after_drop", "fs_repo_source_write",
                "fs_docker_socket", "fs_hidden_evaluator", "net_public_internet_1.1.1.1_443", "uid", "cap_eff"):
        assert key in expected
    assert "--read-only" in supervisor.HARDENING and "no-new-privileges" in supervisor.HARDENING
    assert "--cap-drop" in supervisor.CAPS and "ALL" in supervisor.CAPS
    script = (RME / "egress-init.sh").read_text()
    assert "iptables -P OUTPUT DROP" in script and "ip6tables -P OUTPUT DROP" in script and "setpriv" in script
    dockerfile = (RME / "Dockerfile").read_text()
    assert "LABEL org.agent-loop.revision" in dockerfile
    copies = [line for line in dockerfile.splitlines() if line.startswith("COPY")]
    assert copies and not any("evaluator" in line or "harness" in line for line in copies), \
        "the hidden evaluator and host orchestration must not be copied into the agent image"
    assert scenarios.TRIAL_TOOL_ALLOWLIST == {"python3", "ls", "cat"}


# BLOCKED and identity checks (RME-1.4 authorization contract)
def test_blocked_is_a_valid_terminal_classification_never_a_success():
    assert evaluator is not None
    import harness as harness_module
    assert harness_module.classify("blocked", "the user denied permission", False) == harness_module.BLOCKED
    # Even if an evaluator reports a pass, a blocked run is not promoted to success
    assert harness_module.classify("blocked", "denied", True) == harness_module.BLOCKED


def test_blocked_is_not_recast_as_unknown_or_runtime_failure():
    import harness as harness_module
    assert harness_module.classify("blocked", "x", None) not in (
        harness_module.RUNTIME_FAILURE, "UNKNOWN", harness_module.EVALUATION_ENVIRONMENT_FAILURE)


def test_model_identity_check_rejects_missing_and_mismatched_models():
    assert supervisor.check_model_identity({"actual_request_models": ["a"]}, "a") is None
    assert "intended a" in supervisor.check_model_identity({"actual_request_models": ["b"]}, "a")
    assert supervisor.check_model_identity({"actual_request_models": []}, "a") is not None
    assert supervisor.check_model_identity({}, "a") is not None
