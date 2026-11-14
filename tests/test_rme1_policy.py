"""RME-01 policy v1/v2 regression tests. Scripted providers only: no LM Studio, no Docker, no network.

Guards the explicit authorization statement, the unchanged shell allowlist, the unchanged goal and criteria, the
unchanged production prompt, and the non-interactive BLOCKED path."""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from src import loop, prompts, session
from tests import scripted_provider
from tests.conftest import make_streamed_response

REPO = Path(__file__).resolve().parents[1]
RME = REPO / "evaluation" / "rme1"
sys.path.insert(0, str(RME))
import policy  # noqa: E402
import scenarios  # noqa: E402

DECLARED = scenarios.SCENARIOS["RME-01"].tools

HARNESS_PATH = RME / "harness.py"
spec = importlib.util.spec_from_file_location("rme1_harness_policy", HARNESS_PATH)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)

# Fingerprints recorded before Policy v2 (the goal also matches the stored RME-01 GLM-15 and GLM-16 sessions).
PRODUCTION_PROMPT_SHA = "0f12ffab50e903665207e37800c487c654b7b1a1ca107977ab8959b8ec12f60a"
V2_PROMPT_SHA = "54f34dddbf7d58aace7c29500f3bc2f44b16c6f619ce14d288c6db45764fa19d"  # re-pinned: authorization wording
GOAL_SHA = "27c2d4e4ca1a549f34775b00c62d642fee7a49e27ff41fd2bf9a1e311d744c32"
CRITERIA_SHA = "7d40111f6feb46377f07ed4620e4ea034ea753420cf507d6ca40fe299093aed1"


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _turn(name, arguments, call_id):
    return make_streamed_response(finish_reason="tool_calls", tool_calls=[
        {"id": call_id, "name": name, "arguments": json.dumps(arguments)}])


@pytest.fixture
def restore_prompt(monkeypatch):
    """A policy install changes the prompt constant, the tool list and dispatch; all are restored after each test."""
    monkeypatch.setattr(prompts, "SYSTEM_PROMPT", prompts.SYSTEM_PROMPT)
    yield monkeypatch
    policy.uninstall()


# --- the policy text matches the actual allowlist -------------------------------------------------------------

def test_production_prompt_and_scenario_are_unchanged_by_policy_v2():
    assert _sha(prompts.SYSTEM_PROMPT) == PRODUCTION_PROMPT_SHA
    scenario = scenarios.SCENARIOS["RME-01"]
    assert _sha(scenario.goal) == GOAL_SHA
    assert _sha(json.dumps(scenario.criteria, sort_keys=True)) == CRITERIA_SHA


def test_v2_authorized_basenames_listed_in_the_prompt_equal_the_actual_allowlist():
    text = policy.effective_prompt(policy.POLICY_V2)
    block = text.split("authorized without approval are:")[1].split("\n\n")[0]
    listed = set(re.findall(r"^- (\S+)$", block, re.M))
    assert listed == scenarios.TRIAL_TOOL_ALLOWLIST == {"python3", "ls", "cat"}


def test_v2_directs_discovery_to_list_directory_and_read_file():
    text = policy.effective_prompt(policy.POLICY_V2)
    assert "prefer `list_directory` and `read_file`" in text
    assert "Use `list_directory` and `read_file` to understand" in text


def test_find_is_never_described_as_an_allowed_shell_command():
    text = policy.effective_prompt(policy.POLICY_V2)
    negative = re.compile(r"not authorized|not approved|requires approval|BLOCKED|prefer|never|Every other|denied")
    for line in text.splitlines():
        if re.search(r"(?<![A-Za-z_])find(?![A-Za-z_])", line):
            assert negative.search(line), f"find appears without a prohibition: {line[:120]!r}"
    assert "`find . -name" not in text, "no runnable find example may remain"
    assert "| head" not in text, "no pipeline example may remain"


def test_v2_removes_the_production_examples_of_unauthorized_commands():
    text = policy.effective_prompt(policy.POLICY_V2)
    for banned in ['grep -r "pattern"', "make build", "`run_command` with `find", "Common investigation commands:\n- `find"]:
        assert banned not in text, banned


def test_unknown_policy_is_refused():
    with pytest.raises(policy.PolicyError, match="unknown policy"):
        policy.effective_prompt("RME-01-POLICY-V9")


def test_changed_production_prompt_is_rejected_before_any_replacement(monkeypatch):
    changed = policy.production_prompt().replace("make build", "build")  # the replacement target is gone
    monkeypatch.setattr(policy, "production_prompt", lambda: changed)
    with pytest.raises(policy.PolicyError, match="production system prompt changed"):
        policy.effective_prompt(policy.POLICY_V2)


def test_unrelated_prompt_change_is_rejected_even_when_every_replacement_still_matches(monkeypatch):
    changed = policy.production_prompt() + "\nAn unrelated production sentence."
    monkeypatch.setattr(policy, "production_prompt", lambda: changed)
    with pytest.raises(policy.PolicyError, match="production system prompt changed"):
        policy.effective_prompt(policy.POLICY_V2)


# --- Policy v1 remains reproducible; v2 is recorded explicitly ------------------------------------------------

def test_policy_v1_is_the_unmodified_production_prompt():
    assert policy.effective_prompt(policy.POLICY_V1) == prompts.SYSTEM_PROMPT
    meta = policy.install(policy.POLICY_V1, DECLARED)
    assert meta["effective_prompt_sha256"] == meta["production_prompt_sha256"] == PRODUCTION_PROMPT_SHA
    assert meta["replacements_applied"] == 0 and meta["shell_allowlist"] == ["cat", "ls", "python3"]


def test_policy_v2_install_records_identity_and_allowlist(restore_prompt):
    meta = policy.install(policy.POLICY_V2, DECLARED)
    assert meta["policy_id"] == "RME-01-POLICY-V2"
    assert meta["effective_prompt_sha256"] == V2_PROMPT_SHA
    assert meta["production_prompt_sha256"] == PRODUCTION_PROMPT_SHA
    assert meta["replacements_applied"] == len(policy.V2_REPLACEMENTS) == 9
    assert meta["shell_allowlist"] == sorted(scenarios.TRIAL_TOOL_ALLOWLIST)


# --- end-to-end with the scripted provider -----------------------------------------------------------------

def _capture_run(tmp_path, monkeypatch, policy_id, scripted, deny_callback=None):
    monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path / "unused-home")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    policy.install(policy_id, DECLARED)  # the trial process installs its policy before any request
    seen = {"payloads": [], "prompts": []}
    queue = list(scripted)

    def post(url, json=None, timeout=None, stream=False):
        seen["payloads"].append(json)
        seen["prompts"].append(json["messages"][0]["content"])
        if not queue:
            raise AssertionError("scripted responses exhausted")
        return queue.pop(0)

    monkeypatch.setattr(scripted_provider, "POST", post)
    approvals = []

    def approve(prompt, valid_keys):
        approvals.append(prompt)
        return "3"  # the runner's non-interactive denial

    monkeypatch.setattr(loop, "_get_single_key_input", deny_callback or approve)
    record = harness.run_trial("RME-01", tmp_path, f"policy-{policy_id}", tmp_path / "eval", execute=True,
                               endpoint_ok=True)
    return record, seen, approvals


def test_v2_model_is_sent_the_effective_authorization_instructions(tmp_path, restore_prompt):
    scripted = [_turn("run_command", {"command": "ls"}, "c1"), make_streamed_response(finish_reason="stop", content="x")]
    _, seen, _ = _capture_run(tmp_path, restore_prompt, policy.POLICY_V2, scripted)
    assert seen["prompts"][0] == policy.effective_prompt(policy.POLICY_V2)
    assert "Trial Shell Authorization (RME-01 Policy v2)" in seen["prompts"][0]


def test_v1_model_is_sent_the_unmodified_production_prompt(tmp_path, restore_prompt):
    scripted = [_turn("run_command", {"command": "ls"}, "c1"), make_streamed_response(finish_reason="stop", content="x")]
    _, seen, _ = _capture_run(tmp_path, restore_prompt, policy.POLICY_V1, scripted)
    assert seen["prompts"][0] == prompts.SYSTEM_PROMPT and _sha(seen["prompts"][0]) == PRODUCTION_PROMPT_SHA


def test_unauthorized_find_still_requires_approval_and_non_interactive_denial_ends_blocked(tmp_path, restore_prompt):
    """The exact failure of RME01-GLM-16: a find call. The production authorization still asks; the denial stops the run."""
    scripted = [_turn("run_command", {"command": 'find . -name "logsum.py"'}, "c1")]
    record, seen, approvals = _capture_run(tmp_path, restore_prompt, policy.POLICY_V2, scripted)
    assert approvals, "an unauthorized command must reach the approval prompt"
    assert record["final_run_status"] == "blocked"
    assert record["classification"] == harness.BLOCKED
    assert len(seen["payloads"]) == 1, "the denied run must not request another model turn"


def test_authorized_python3_still_runs_without_approval(tmp_path, restore_prompt):
    scripted = [_turn("run_command", {"command": "python3 -c 'print(1)'"}, "c1"),
                make_streamed_response(finish_reason="stop", content="done")]
    record, seen, approvals = _capture_run(tmp_path, restore_prompt, policy.POLICY_V2, scripted)
    assert approvals == [], "an authorized basename must not prompt"
    assert record["tool_counts"].get("run_command") == 1


def test_tool_schema_identity_is_recorded_and_stable(tmp_path, restore_prompt):
    scripted = [_turn("run_command", {"command": "ls"}, "c1"), make_streamed_response(finish_reason="stop", content="x")]
    _, seen, _ = _capture_run(tmp_path, restore_prompt, policy.POLICY_V2, scripted)
    hashes = {policy.tool_schema_sha256(p.get("tools")) for p in seen["payloads"]}
    assert len(hashes) == 1, "every request in a trial must carry the same tool schema"


# --- the runner refuses to run without an explicit policy ---------------------------------------------------

def test_runner_requires_an_explicit_policy():
    import os
    env = {k: v for k, v in os.environ.items() if k not in ("RME_POLICY",)}
    env.update({"RME_MODEL": "zai-org/glm-4.7-flash", "RME_LLM_URL": "http://h:1/v1/chat/completions"})
    proc = subprocess.run([sys.executable, "-B", "-c", "import sys; sys.path.insert(0, 'evaluation/rme1'); import runner"],
                          cwd=str(REPO), env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0 and "RME_POLICY is required" in proc.stderr
