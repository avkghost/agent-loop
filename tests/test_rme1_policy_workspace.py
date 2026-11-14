"""RME-01 contract R2: per-scenario tool exposure, reviewed schemas for declared tools, and refusal of everything undeclared.

These tests check the contract, not the implementation: which tools a scenario may advertise and run, that a declared
tool's schema must be the reviewed one, that drift in tools no scenario declares does not block evaluation, and that
installing a policy is undone exactly by uninstall. Scripted doubles only: no model, no network.
"""

import asyncio
import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation" / "rme1"))
sys.path.insert(0, str(ROOT))

import policy  # noqa: E402
import scenarios  # noqa: E402
from src import dispatch, loop, prompts  # noqa: E402
from src.authorization import SessionAuthorization  # noqa: E402

EXISTING = ("run_command", "write_file", "read_file", "list_directory")
WORKSPACE = ("edit_file", "git_status", "git_diff")


@pytest.fixture(autouse=True)
def _restore_policy_after_each_test():
    original_prompt = prompts.SYSTEM_PROMPT
    yield
    policy.uninstall()
    prompts.SYSTEM_PROMPT = original_prompt


def _names(tools):
    return [schema["function"]["name"] for schema in tools]


def _digest(schema):
    return policy.sha256(policy._canonical(schema))


def _production(name):
    return next(s for s in policy.production_tools() if s["function"]["name"] == name)


# --- reviewed schemas ---------------------------------------------------------------------------------------------

def test_every_tool_a_scenario_declares_has_a_reviewed_schema_that_matches_production():
    declared = set().union(*(set(s.tools) for s in scenarios.SCENARIOS.values()))
    assert declared == set(policy.REVIEWED_TOOL_SHA256) == set(EXISTING)
    for name, digest in policy.REVIEWED_TOOL_SHA256.items():
        assert _digest(_production(name)) == digest, name


def test_every_scenario_declares_tools_that_have_a_reviewed_schema():
    for scenario in scenarios.SCENARIOS.values():
        assert scenario.tools and set(scenario.tools) <= set(policy.REVIEWED_TOOL_SHA256), scenario.scenario_id


# --- exposure is exactly what a scenario declares -----------------------------------------------------------------

@pytest.mark.parametrize("pid", [policy.POLICY_V1, policy.POLICY_V2])
def test_existing_scenarios_advertise_exactly_their_four_declared_tools_in_order(pid):
    for scenario in scenarios.SCENARIOS.values():
        policy.install(pid, scenario.tools)
        assert _names(loop.get_all_tools(None)) == list(scenario.tools) == list(EXISTING), scenario.scenario_id
        policy.uninstall()


def test_workspace_tools_are_not_exposed_to_existing_scenarios():
    for scenario in scenarios.SCENARIOS.values():
        assert not set(WORKSPACE) & set(scenario.tools), scenario.scenario_id


def test_a_reviewed_workspace_tool_is_advertised_and_runs_when_a_scenario_declares_it(tmp_path, monkeypatch):
    """edit_file has no reviewed entry in the shipped contract. Here its current schema is reviewed for this test only,
    which is the step a real scenario would take before declaring it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(policy.REVIEWED_TOOL_SHA256, "edit_file", _digest(_production("edit_file")))
    declared = ("read_file", "edit_file")
    for pid in policy.POLICY_IDS:
        (tmp_path / "f.txt").write_text("one\n", encoding="utf-8")
        policy.install(pid, declared)
        assert _names(loop.get_all_tools(None)) == list(declared)
        result = asyncio.run(dispatch.dispatch_tool_async(
            "edit_file", {"path": "f.txt", "old_string": "one", "new_string": "two"},
            authorization=SessionAuthorization()))
        assert result["isError"] is False, pid
        assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "two\n"
        policy.uninstall()


# --- undeclared tools are refused before anything runs --------------------------------------------------------------

@pytest.mark.parametrize("pid", [policy.POLICY_V1, policy.POLICY_V2])
@pytest.mark.parametrize("name", ["start_job", "status_job", "edit_file", "git_status", "git_diff"])
def test_an_undeclared_tool_is_refused_and_does_not_run(pid, name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("one\n", encoding="utf-8")
    policy.install(pid, EXISTING)
    result = asyncio.run(dispatch.dispatch_tool_async(
        name, {"path": "f.txt", "old_string": "one", "new_string": "two", "command": "touch ran.txt", "job_id": "x"},
        authorization=SessionAuthorization(allow_all=True)))
    assert result["isError"] is True and result["errorCategory"] == "permission", (pid, name)
    assert "not authorized for this scenario" in result["content"]
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "one\n"
    assert not (tmp_path / "ran.txt").exists()


def test_only_the_declared_tool_runs_under_a_narrow_declaration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("keep\n", encoding="utf-8")
    policy.install(policy.POLICY_V1, ("read_file",))
    refused = asyncio.run(dispatch.dispatch_tool_async("run_command", {"command": "ls"},
                                                       authorization=SessionAuthorization(allow_all=True)))
    assert refused["isError"] is True and "not authorized for this scenario" in refused["content"]
    read = asyncio.run(dispatch.dispatch_tool_async("read_file", {"path": "f.txt"},
                                                    authorization=SessionAuthorization()))
    assert read["isError"] is False and read["content"] == "keep\n"


# --- drift: declared tools are pinned; undeclared tools are not --------------------------------------------------

def _changed_registry(mutate):
    tools = copy.deepcopy(policy._PRODUCTION_TOOLS)  # the captured registry, not the function a test replaces
    mutate(tools)
    return tools


def _index(tools, name):
    return next(i for i, s in enumerate(tools) if s["function"]["name"] == name)


def _swap(tools, a, b):
    i, j = _index(tools, a), _index(tools, b)
    tools[i], tools[j] = tools[j], tools[i]


DECLARED_DRIFT = {
    "declared schema changed": lambda t: t[_index(t, "read_file")]["function"].__setitem__("description", "changed"),
    "declared tool missing": lambda t: t.pop(_index(t, "read_file")),
}

UNDECLARED_DRIFT = {
    "undeclared schema changed": lambda t: t[_index(t, "start_job")]["function"].__setitem__("description", "changed"),
    "undeclared tool removed": lambda t: t.pop(_index(t, "git_diff")),
    "undeclared tool added": lambda t: t.append(copy.deepcopy(t[0]) | {"function": dict(t[0]["function"], name="zz_new")}),
    "registry reordered around declared tools": lambda t: t.__setitem__(slice(0, 2), [t[1], t[0]]),
    "declared tools reordered in the registry": lambda t: _swap(t, "run_command", "write_file"),
}


@pytest.mark.parametrize("label", list(DECLARED_DRIFT))
def test_drift_in_a_declared_tool_fails_closed(label, monkeypatch):
    monkeypatch.setattr(policy, "production_tools", lambda: _changed_registry(DECLARED_DRIFT[label]))
    with pytest.raises(policy.PolicyError):
        policy.advertised_tools(policy.POLICY_V2, EXISTING)


@pytest.mark.parametrize("label", list(DECLARED_DRIFT))
def test_a_refused_install_leaves_the_production_callables_and_prompt_untouched(label, monkeypatch):
    original_get, original_dispatch = loop.get_all_tools, dispatch.dispatch_tool_async
    original_prompt = prompts.SYSTEM_PROMPT
    monkeypatch.setattr(policy, "production_tools", lambda: _changed_registry(DECLARED_DRIFT[label]))
    with pytest.raises(policy.PolicyError):
        policy.install(policy.POLICY_V2, EXISTING)
    assert loop.get_all_tools is original_get and dispatch.dispatch_tool_async is original_dispatch
    assert prompts.SYSTEM_PROMPT is original_prompt


@pytest.mark.parametrize("label", list(UNDECLARED_DRIFT))
def test_drift_in_undeclared_tools_does_not_block_a_scenario(label, monkeypatch):
    monkeypatch.setattr(policy, "production_tools", lambda: _changed_registry(UNDECLARED_DRIFT[label]))
    for pid in policy.POLICY_IDS:
        policy.install(pid, EXISTING)
        assert _names(loop.get_all_tools(None)) == list(EXISTING), (label, pid)
        policy.uninstall()


# --- scenario declarations are validated -------------------------------------------------------------------------

@pytest.mark.parametrize("declared, fragment", [
    ((), "at least one tool"),
    (("read_file", "read_file"), "twice"),
    (("read_file", "shell_exec"), "no reviewed schema"),
    (("read_file", "edit_file"), "no reviewed schema"),
])
def test_invalid_declarations_are_refused(declared, fragment):
    with pytest.raises(policy.PolicyError, match=fragment):
        policy.advertised_tools(policy.POLICY_V2, declared)


# --- metadata records the contract and the source revision ----------------------------------------------------------

def test_expected_metadata_records_contract_revision_source_revision_and_the_exposed_list():
    meta = policy.expected_metadata(policy.POLICY_V2, EXISTING)
    assert meta["contract_revision"] == policy.CONTRACT_REVISION == "RME-01-CONTRACT-R2"
    assert meta["source_revision"] == policy.source_fingerprint() == policy.source_revision()
    assert meta["advertised_tools"] == list(EXISTING)
    assert meta["expected_tool_schema_sha256"] == policy.tool_schema_sha256(
        policy.advertised_tools(policy.POLICY_V2, EXISTING))


def test_validation_flags_unexpected_exposure_and_a_reordered_advertisement():
    exp = policy.expected_metadata(policy.POLICY_V1, EXISTING)
    rep = copy.deepcopy(exp)
    rep["advertised_tools"] = list(reversed(EXISTING))
    problems = policy.validate_policy_metadata(rep, policy.POLICY_V1, exp, [exp["expected_tool_schema_sha256"]])
    assert any("advertised_tools" in p for p in problems)
    extra = policy.advertised_tools(policy.POLICY_V1, EXISTING) + [copy.deepcopy(policy.production_tools()[0])]
    problems = policy.validate_policy_metadata(copy.deepcopy(exp), policy.POLICY_V1, exp,
                                               [policy.tool_schema_sha256(extra)])
    assert any("tool schemas differ" in p for p in problems)


# --- installs are isolated and fully reversible ----------------------------------------------------------------------

def test_reinstalling_for_another_scenario_replaces_the_view_and_uninstall_restores_the_originals():
    original_get, original_dispatch = loop.get_all_tools, dispatch.dispatch_tool_async
    policy.install(policy.POLICY_V2, EXISTING)
    policy.install(policy.POLICY_V1, ("read_file",))
    assert _names(loop.get_all_tools(None)) == ["read_file"]
    policy.uninstall()
    assert loop.get_all_tools is original_get and dispatch.dispatch_tool_async is original_dispatch
    assert _names(loop.get_all_tools(None)) != ["read_file"]
