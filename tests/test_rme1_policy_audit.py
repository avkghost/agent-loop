"""RME-01 Policy v2 audit regression tests (12 cases). Scripted providers and offline doubles only: no LM Studio,
no Docker, no network. The image-revision tests replace docker with a stub."""

import copy
import glob
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src import dispatch, loop, prompts, session
from tests import scripted_provider
from tests.conftest import make_streamed_response, protocol_violations
from tests.test_rme1_policy import _capture_run, _turn, restore_prompt  # noqa: F401 (fixture reuse)

REPO = Path(__file__).resolve().parents[1]
RME = REPO / "evaluation" / "rme1"
sys.path.insert(0, str(RME))
import policy  # noqa: E402
import scenarios  # noqa: E402
import supervisor  # noqa: E402

spec = importlib.util.spec_from_file_location("rme1_harness_audit", RME / "harness.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)

JOB_NAMES = ("start_job", "status_job", "collect_job", "terminate_job")
DECLARED = scenarios.SCENARIOS["RME-01"].tools
NOT_DECLARED = ("start_job", "status_job", "collect_job", "terminate_job", "shell_exec", "grep_files",
                "edit_file", "git_status", "git_diff")


def _run_v2_script(tmp_path, monkeypatch, restore, scripted):
    return _capture_run(tmp_path, restore, policy.POLICY_V2, scripted)


def _saved_session(tmp_path):
    files = glob.glob(str(tmp_path / "**" / "sessions" / "rme-RME-01-*.json"), recursive=True)
    assert files, "the trial session must be saved"
    return json.loads(Path(files[0]).read_text())


# 1. start_job cannot execute under Policy v2 (dispatch level, and through a model request in a trial)
def test_start_job_cannot_execute_under_policy_v2(tmp_path, restore_prompt):
    marker = tmp_path / "started.txt"
    policy.install(policy.POLICY_V2, DECLARED)
    import asyncio
    result = asyncio.run(dispatch.dispatch_tool_async("start_job", {"command": f"touch {marker}"}, None, None))
    assert result.get("isError") is True and result.get("errorCategory") == "permission"
    assert "not authorized for this scenario" in result["content"]
    assert not marker.exists()


def test_a_model_requested_start_job_is_rejected_in_a_trial_and_never_runs(tmp_path, restore_prompt):
    marker = tmp_path / "started-by-model.txt"
    scripted = [_turn("start_job", {"command": f"touch {marker}"}, "job1"),
                make_streamed_response(finish_reason="stop", content="done")]
    record, seen, approvals = _run_v2_script(tmp_path, restore_prompt, restore_prompt, scripted)
    assert not marker.exists(), "a rejected managed job must not run"
    assert approvals == [], "the rejection is explicit, not an approval request"
    execs = [e for e in _saved_session(tmp_path)["execution"] if e["tool_name"] == "start_job"]
    assert execs and execs[0]["status"] == "failed", execs


# 2. unexpected prohibited tool calls cannot bypass the restriction
@pytest.mark.parametrize("name", list(NOT_DECLARED))
def test_unexpected_prohibited_or_unknown_tools_are_rejected_directly(name, restore_prompt):
    policy.install(policy.POLICY_V2, DECLARED)
    import asyncio
    result = asyncio.run(dispatch.dispatch_tool_async(name, {"command": "true", "job_id": "x"}, None, None))
    assert result.get("isError") is True and result.get("errorCategory") == "permission", name
    sync = dispatch.dispatch_tool(name, {"command": "true"}, None, None)
    assert sync.get("isError") is True, name


def test_run_command_with_find_still_needs_approval_under_v2(tmp_path, restore_prompt):
    scripted = [_turn("run_command", {"command": "find . -name x"}, "c1")]
    record, seen, approvals = _run_v2_script(tmp_path, restore_prompt, restore_prompt, scripted)
    assert approvals and record["classification"] == harness.BLOCKED


# 3. policy v1 advertises exactly the scenario's declared tools; the production callables are restored afterwards
def test_policy_v1_advertises_only_the_declared_tools_and_restores_on_uninstall(restore_prompt):
    original_get, original_dispatch = loop.get_all_tools, dispatch.dispatch_tool_async
    policy.install(policy.POLICY_V1, DECLARED)
    names = [t["function"]["name"] for t in loop.get_all_tools(None)]
    assert names == list(DECLARED)
    assert not set(JOB_NAMES) & set(names), "managed-job tools are not advertised to any scenario"
    policy.uninstall()
    assert loop.get_all_tools is original_get and dispatch.dispatch_tool_async is original_dispatch


def test_policy_v1_request_carries_exactly_the_scenario_tool_schemas(tmp_path, restore_prompt):
    scripted = [make_streamed_response(finish_reason="stop", content="x")]
    record, seen, _ = _capture_run(tmp_path / "v1", restore_prompt, policy.POLICY_V1, scripted)
    assert seen["payloads"][0]["tools"] == policy.advertised_tools(policy.POLICY_V1, DECLARED)


# 4. policy v2 descriptions contain no contradictory executable examples
def test_policy_v2_advertised_descriptions_contain_no_executable_prohibited_examples():
    tools = policy.advertised_tools(policy.POLICY_V2, DECLARED)
    assert [t["function"]["name"] for t in tools] == list(DECLARED)
    banned = ['run_command with "grep', 'run_command with "find', "use start_job", "approves them",
              "for a long-running process use start_job", "run_command with grep, rg, or find"]
    for tool in tools:
        text = tool["function"]["description"]
        for phrase in banned:
            assert phrase not in text, (tool["function"]["name"], phrase)


def test_policy_v2_keeps_names_and_argument_schemas_of_advertised_tools():
    production = {t["function"]["name"]: t for t in policy.production_tools()}
    for tool in policy.advertised_tools(policy.POLICY_V2, DECLARED):
        name = tool["function"]["name"]
        assert tool["function"]["parameters"] == production[name]["function"]["parameters"], name
        assert tool["type"] == production[name]["type"]


# 5. a changed production prompt is rejected (see test_rme1_policy); a changed source tool description is rejected
def test_changed_source_tool_description_is_rejected(monkeypatch):
    changed = policy.production_tools()
    changed[4]["function"]["description"] += " An unrelated change."
    monkeypatch.setattr(policy, "production_tools", lambda: changed)
    with pytest.raises(policy.PolicyError, match="schema of .* changed"):
        policy.expected_metadata(policy.POLICY_V2, DECLARED)


def test_changed_source_tool_is_rejected_at_the_advertisement_boundary(monkeypatch):
    changed = policy.production_tools()
    read = next(s for s in changed if s["function"]["name"] == "read_file")  # a declared tool
    read["function"]["parameters"]["properties"]["extra"] = {"type": "string"}
    monkeypatch.setattr(policy, "production_tools", lambda: changed)
    with pytest.raises(policy.PolicyError, match="schema of .* changed"):
        policy.advertised_tools(policy.POLICY_V2, DECLARED)


# 7-8. correct, missing and mismatched metadata
def _expected(pid):
    return policy.expected_metadata(pid, DECLARED)


def _reported(pid):
    return copy.deepcopy(_expected(pid))


def test_correct_policy_metadata_is_accepted_for_both_policies():
    for pid in policy.POLICY_IDS:
        exp = _expected(pid)
        rep = _reported(pid)
        assert policy.validate_policy_metadata(rep, pid, exp, [exp["expected_tool_schema_sha256"]] * 3) == []


@pytest.mark.parametrize("mutate, fragment", [
    (lambda r: r.pop("effective_prompt_sha256"), "lacks"),
    (lambda r: r.__setitem__("policy_id", "RME-01-POLICY-V1"), "applied policy"),
    (lambda r: r.__setitem__("production_prompt_sha256", "0" * 64), "production_prompt_sha256"),
    (lambda r: r.__setitem__("effective_prompt_sha256", "1" * 64), "effective_prompt_sha256"),
    (lambda r: r.__setitem__("replacements_applied", 8), "replacements_applied"),
    (lambda r: r.__setitem__("shell_allowlist", ["python3", "ls", "cat", "find"]), "shell_allowlist"),
    (lambda r: r.__setitem__("dispatch_guard", "none"), "dispatch_guard"),
    (lambda r: r.__setitem__("advertised_tools", list(JOB_NAMES)), "advertised_tools"),
    (lambda r: r.__setitem__("advertised_tools", list(reversed(DECLARED))), "advertised_tools"),
    (lambda r: r.__setitem__("advertised_tools", list(DECLARED) + ["edit_file"]), "advertised_tools"),
    (lambda r: r.__setitem__("contract_revision", "RME-01-CONTRACT-R1"), "contract_revision"),
    (lambda r: r.__setitem__("source_revision", "0" * 64), "source_revision"),
])
def test_mismatched_policy_metadata_is_rejected(mutate, fragment):
    exp = _expected(policy.POLICY_V2)
    rep = _reported(policy.POLICY_V2)
    mutate(rep)
    problems = policy.validate_policy_metadata(rep, policy.POLICY_V2, exp, [exp["expected_tool_schema_sha256"]])
    assert problems and any(fragment in p for p in problems), problems


def test_missing_or_malformed_tool_hashes_are_rejected():
    exp = _expected(policy.POLICY_V2)
    rep = _reported(policy.POLICY_V2)
    assert policy.validate_policy_metadata(rep, policy.POLICY_V2, exp, None)
    assert policy.validate_policy_metadata(rep, policy.POLICY_V2, exp, [])
    assert policy.validate_policy_metadata(rep, policy.POLICY_V2, exp, ["not-a-hash"])
    assert policy.validate_policy_metadata(rep, policy.POLICY_V2, exp, ["0" * 64])  # wrong schema identity
    assert policy.validate_policy_metadata(None, policy.POLICY_V2, exp, [exp["expected_tool_schema_sha256"]])


def test_result_metadata_cannot_vouch_for_itself():
    """The expected values come from the host computation, so a frame that copies its own claims still fails
    when those claims differ from the host's view."""
    exp = _expected(policy.POLICY_V2)
    rep = _reported(policy.POLICY_V1)  # a self-consistent but different policy
    assert policy.validate_policy_metadata(rep, policy.POLICY_V1, exp, [exp["expected_tool_schema_sha256"]])


# 9. contradictory RESULT and SESSION states are rejected; evaluator PASS is never a runtime success
def _session(status, runs=True, verification=None, sid="rme1-RME-01"):
    doc = {"session_id": sid, "runs": [{"status": status}] if runs else [],
           "verification_results": verification if verification is not None else []}
    return doc


def test_consistent_result_and_session_are_accepted():
    assert supervisor.validate_result_session({"run_status": "interrupted", "scenario": "RME-01"}, _session("interrupted"), "RME-01") == []
    assert supervisor.validate_result_session({"run_status": "blocked", "scenario": "RME-01"}, _session("blocked"), "RME-01") == []
    assert supervisor.terminal_session_problems({"run_status": "blocked"}, "MISSING", None, "RME-01"), \
        "an absent SESSION for a terminal outcome is a problem, never a fabricated pass"


def test_contradictory_run_status_is_rejected():
    problems = supervisor.validate_result_session({"run_status": "completed_verified"}, _session("blocked"), "RME-01")
    assert any("contradicts" in p for p in problems)


def test_terminal_status_without_any_session_run_is_rejected():
    problems = supervisor.validate_result_session({"run_status": "blocked"}, _session("x", runs=False), "RME-01")
    assert any("records no run" in p for p in problems)


def test_success_claim_without_verification_results_is_rejected():
    problems = supervisor.validate_result_session({"run_status": "completed_verified"},
                                                  _session("completed_verified", verification=[]), "RME-01")
    assert any("no verification evidence" in p for p in problems)


def test_wrong_session_identity_is_rejected():
    problems = supervisor.validate_result_session({"run_status": "blocked"}, _session("blocked", sid="rme1-RME-02"), "RME-01")
    assert any("is not rme1-RME-01" in p for p in problems)


def test_evaluator_pass_never_becomes_a_runtime_success():
    assert harness.classify("interrupted", "deadline", True) == harness.INTERRUPTED
    assert harness.classify("blocked", "denied", True) == harness.BLOCKED
    assert harness.classify("needs_attention", "acceptance", True) != harness.VERIFIED_SUCCESS
    assert harness.classify("completed_verified", "ok", True) == harness.VERIFIED_SUCCESS


# 10-11. image revision: wrong rejected, correct accepted (docker stubbed)
def _fake_docker(label, image_id="sha256:abc123"):
    inspect_out = f'{image_id}|{json.dumps({"Labels": {"org.agent-loop.revision": label}})}'

    def run(argv, timeout=120):
        if argv[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, inspect_out, "")
        return subprocess.CompletedProcess(argv, 0, "1\n", "")  # marker greps succeed
    return run


def test_wrong_image_revision_is_rejected(monkeypatch):
    monkeypatch.setattr(supervisor, "_run", _fake_docker("1531817-stale"))
    with pytest.raises(supervisor.ImageRejected, match="not the expected"):
        supervisor.verify_image("agent-loop-rme1:x", expected_revision="1531817+abc")


def test_correct_image_revision_is_accepted_and_the_immutable_id_is_recorded(monkeypatch):
    monkeypatch.setattr(supervisor, "_run", _fake_docker("1531817+abc", image_id="sha256:deadbeef"))
    identity = supervisor.verify_image("agent-loop-rme1:x", expected_revision="1531817+abc")
    assert identity["image_id"] == "sha256:deadbeef" and identity["revision_label"] == "1531817+abc"


def test_policy_v2_requires_an_explicit_expected_revision():
    proc = subprocess.run([sys.executable, "-B", str(RME / "supervisor.py"), "RME01-audit-noexp",
                           "--image", "agent-loop-rme1:x", "--model", "zai-org/glm-4.7-flash",
                           "--policy", "RME-01-POLICY-V2"], capture_output=True, text=True, timeout=60, cwd=str(REPO))
    assert proc.returncode != 0 and "require --expected-revision" in proc.stderr
    assert not (REPO / "tmp" / "rme1" / "RME01-audit-noexp").exists(), "no trial directory is created"


# 12. tool-call and journal pairing stays correct for a rejected call
def test_rejected_call_pairs_with_one_journal_record_and_a_protocol_valid_history(tmp_path, restore_prompt):
    marker = tmp_path / "paired.txt"
    scripted = [_turn("start_job", {"command": f"touch {marker}"}, "pair1"),
                make_streamed_response(finish_reason="stop", content="done")]
    record, seen, _ = _run_v2_script(tmp_path, restore_prompt, restore_prompt, scripted)
    saved = _saved_session(tmp_path)
    calls = [e for e in saved["execution"] if e["tool_name"] == "start_job"]
    assert len(calls) == 1, "exactly one journal record for the one model call"
    assert calls[0]["status"] == "failed" and calls[0]["error_category"] == "permission"
    assert protocol_violations(saved["messages"]) == []
    tool_msgs = [m for m in saved["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1 and "not authorized for this scenario" in tool_msgs[0]["content"]
