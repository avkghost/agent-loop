"""RME-1 harness validation. Deterministic; the scripted provider is used only to exercise the harness path,
never as real-model evidence. No LM Studio endpoint is contacted by any test here."""

import importlib.util
import json
from pathlib import Path

import pytest

from src import session
from tests import scripted_provider
from tests.conftest import make_streamed_response

HARNESS_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "rme1" / "harness.py"
spec = importlib.util.spec_from_file_location("rme1_harness", HARNESS_PATH)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def _turn(name, arguments, call_id):
    return make_streamed_response(finish_reason="tool_calls", tool_calls=[
        {"id": call_id, "name": name, "arguments": json.dumps(arguments)}])


def _stop(text):
    return make_streamed_response(finish_reason="stop", content=text)


def test_fixtures_are_reproducible_across_workspaces(tmp_path):
    for scenario_id, scenario in harness.SCENARIOS.items():
        first = harness.make_workspace(scenario, tmp_path / "a", f"t1-{scenario_id}")
        second = harness.make_workspace(scenario, tmp_path / "b", f"t1-{scenario_id}")
        assert first["initial_hashes"] == second["initial_hashes"], scenario_id
        for rel, content in scenario.fixtures.items():
            assert (Path(first["workspace"]) / rel).read_text() == content


def test_workspaces_are_isolated_and_never_inside_the_repository(tmp_path):
    scenario = harness.SCENARIOS["RME-01"]
    created = harness.make_workspace(scenario, tmp_path, "isolated")
    assert harness.REPO_ROOT not in Path(created["workspace"]).parents
    with pytest.raises(FileExistsError):
        harness.make_workspace(scenario, tmp_path, "isolated")
    with pytest.raises(ValueError, match="inside the Agent Loop repository"):
        harness.make_workspace(scenario, harness.REPO_ROOT / "evaluation", "inside")


def test_trusted_criteria_are_attached_with_user_origin_and_expected_values():
    for scenario_id, scenario in harness.SCENARIOS.items():
        criteria = harness.build_criteria(scenario)
        assert len(criteria) == len(scenario.criteria)
        assert all(item.trusted for item in criteria), scenario_id
        assert {item.origin for item in criteria} == {"user"}
        commands = [item for item in criteria if item.criterion_type == "command_exit"]
        assert commands and all(item.expected == 0 for item in commands), scenario_id


def test_hidden_checks_are_outside_every_workspace_and_distinguish_correct_from_wrong(tmp_path):
    scenario = harness.SCENARIOS["RME-02"]
    created = harness.make_workspace(scenario, tmp_path, "evaluator-check")
    workspace = Path(created["workspace"])
    evaluator_dir = tmp_path / "evaluator"
    script = harness.write_hidden_check(scenario, evaluator_dir)
    assert workspace not in script.parents
    assert not any(p.name == scenario.hidden_check for p in workspace.rglob("*"))

    wrong = harness.run_hidden_check(scenario, workspace, evaluator_dir)
    assert wrong["passed"] is False

    (workspace / "calc" / "stats.py").write_text(
        "def mean(values):\n    if not values:\n        raise ValueError('empty')\n    return sum(values) / len(values)\n")
    right = harness.run_hidden_check(scenario, workspace, evaluator_dir)
    assert right["passed"] is True


def test_protected_files_are_detected_when_changed(tmp_path):
    scenario = harness.SCENARIOS["RME-02"]
    created = harness.make_workspace(scenario, tmp_path, "protected")
    workspace = Path(created["workspace"])
    assert harness.protected_unchanged(scenario, workspace, created["initial_hashes"]) is True
    (workspace / "tests" / "test_stats.py").write_text("# edited\n")
    assert harness.protected_unchanged(scenario, workspace, created["initial_hashes"]) is False


@pytest.mark.parametrize("status, evaluator, reason, expected", [
    ("completed_verified", True, "", harness.VERIFIED_SUCCESS),
    ("completed_verified", False, "", harness.FALSE_POSITIVE_VERIFICATION),
    ("completed_unverified", True, "", harness.UNVERIFIED_COMPLETION),
    ("needs_attention", None, "required acceptance criteria did not pass within the verification budget", harness.TASK_FAILURE),
    ("needs_attention", None, "session could not be saved", harness.RUNTIME_FAILURE),
    ("provider_failed", None, "", harness.PROVIDER_FAILURE),
    ("interrupted", None, "", harness.INTERRUPTED),
    ("limit_reached", None, "", harness.LIMIT_REACHED),
])
def test_evaluator_separates_verified_from_false_positive_and_maps_every_status(status, evaluator, reason, expected):
    assert harness.classify(status, reason, evaluator) == expected


def test_missing_endpoint_is_an_explicit_unavailable_result_and_contacts_no_model(tmp_path):
    record = harness.run_trial("RME-01", tmp_path, "no-endpoint", tmp_path / "eval", execute=True, endpoint_ok=False)
    assert record["classification"] == harness.PROVIDER_FAILURE
    assert "unavailable" in record["reason"]


def test_default_trial_is_not_executed(tmp_path):
    record = harness.run_trial("RME-02", tmp_path, "dry", tmp_path / "eval")
    assert record["classification"] == harness.NOT_EXECUTED
    assert "model not contacted" in record["reason"]
    assert "final_run_status" not in record


def _scripted_trial(tmp_path, monkeypatch, responses, label):
    """Runs one harness trial against an exact, ordered script. Every model request is counted; a request beyond
    the script is recorded and reported as an explicit failure instead of surfacing as a transport error."""
    monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path / "unused-home")
    # Trial commands never write bytecode, so no stale cache can decide a verification result.
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    script = list(responses)
    seen = {"requests": 0, "exhausted": False}

    def post(*args, **kwargs):
        seen["requests"] += 1
        if not script:
            seen["exhausted"] = True
            raise AssertionError("scripted responses exhausted: the runtime requested an unexpected model turn")
        return script.pop(0)

    monkeypatch.setattr(scripted_provider, "POST", post)
    record = harness.run_trial("RME-02", tmp_path, label, tmp_path / "eval", execute=True, endpoint_ok=True)
    return record, seen, script


def test_harness_execution_path_with_scripted_provider_reaches_a_verified_outcome(tmp_path, monkeypatch):
    """Validates the harness path only. The scripted model is not real-model evidence."""
    scripted = [
        _turn("run_command", {"command": "python3 -m unittest -q tests.test_stats"}, "c1"),
        _turn("write_file", {"path": "calc/stats.py",
                             "content": "def mean(values):\n    if not values:\n        raise ValueError('empty')\n"
                                        "    return sum(values) / len(values)\n"}, "c2"),
        _stop("repaired"),
    ]
    record, seen, remaining = _scripted_trial(tmp_path, monkeypatch, scripted, "scripted-path")
    assert seen["exhausted"] is False and seen["requests"] == 3, seen
    assert remaining == []
    assert record["final_run_status"] == "completed_verified"
    assert record["classification"] == harness.VERIFIED_SUCCESS
    assert record["protected_files_unchanged"] is True
    assert record["independent_evaluator"]["passed"] is True
    assert record["tool_counts"].get("run_command") == 1 and record["tool_counts"].get("write_file") == 1
    assert record["verification_attempts"] == 1


def test_incorrect_repair_in_the_same_scripted_path_never_reaches_a_verified_outcome(tmp_path, monkeypatch):
    """Negative control: the same script with the original defect (+1 in the denominator) must not verify."""
    scripted = [
        _turn("run_command", {"command": "python3 -m unittest -q tests.test_stats"}, "c1"),
        _turn("write_file", {"path": "calc/stats.py",
                             "content": "def mean(values):\n    if not values:\n        raise ValueError('empty')\n"
                                        "    return sum(values) / (len(values) + 1)\n"}, "c2"),
        _stop("repaired"),
    ]
    record, seen, _ = _scripted_trial(tmp_path, monkeypatch, scripted, "scripted-wrong")
    assert record["final_run_status"] != "completed_verified"
    assert record["classification"] != harness.VERIFIED_SUCCESS


def test_result_records_are_valid_bounded_json(tmp_path):
    record = harness.run_trial("RME-03", tmp_path, "json", tmp_path / "eval")
    text = harness.bounded(record)
    assert json.loads(text)["evaluation_id"] == "RME-03:json"
    assert len(text.encode("utf-8")) < harness.RESULT_BYTES_LIMIT
    oversized = dict(record, reason="x" * (harness.RESULT_BYTES_LIMIT + 10))
    with pytest.raises(ValueError, match="size bound"):
        harness.bounded(oversized)


def test_budgets_and_allowlist_are_the_documented_values():
    assert harness.MAX_TURNS == 20
    assert harness.VERIFICATION_CYCLES == 2
    assert harness.WALL_CLOCK_SECONDS == 300
    assert harness.SHELL_TIMEOUT_SECONDS == 15
    assert harness.TRIAL_TOOL_ALLOWLIST == {"python3", "ls", "cat"}
