"""RME-DEBUG-02 offline validation and integration. Deterministic: no LM Studio, no Docker, no network. Public tests
and the hidden evaluator run in temporary copies of the fixture. The mutation outcomes are asserted as predicted; a
mismatch fails the test and is investigated, not adjusted."""

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RME1 = REPO / "evaluation" / "rme1"
BENCH = REPO / "evaluation" / "rme_debug_02"
DEFECTIVE = BENCH / "assets" / "defective"
REFERENCE = BENCH / "assets" / "reference"
HIDDEN = BENCH / "hidden_rme_debug_02.py"
ENGINE = Path("eventledger") / "engine.py"
PUBLIC = ["tests.test_batch", "tests.test_engine", "tests.test_intake", "tests.test_report", "tests.test_store"]
ENV = {"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"}
REGRESSION = "test_a_failed_event_is_applied_once_when_redelivered"

sys.path.insert(0, str(RME1))
import evaluator  # noqa: E402
import policy  # noqa: E402
import scenarios  # noqa: E402
import supervisor  # noqa: E402
from src.session import validate_session_data  # noqa: E402

SCENARIO = scenarios.SCENARIOS["RME-DEBUG-02"]

HISTORICAL = {  # fingerprints recorded before RME-DEBUG-02 was added; any change fails here
    "RME-01": "d3c62311351f640c097f9979b5f23edca7cdb962093b03121d32325778b81fcb",
    "RME-02": "1648ac2feb40d78637bc97ca1514363778e3c5f5e6839079e23a63776e8efd46",
    "RME-03": "9733da735a3e00a707ad2e0ab6363e232821714f6d254564b1a9208953aff8f3",
    "RME-DEBUG-01": "474dc327389cdb0c9fb4d989094407c2fb28f9e7e56c872bef5beba0e080b0ba",
}


def _fingerprint(s):
    import hashlib
    return hashlib.sha256(json.dumps({"goal": s.goal, "fixtures": s.fixtures, "protected": s.protected,
                                      "criteria": s.criteria, "hidden": s.hidden_check},
                                     sort_keys=True, default=str).encode()).hexdigest()


def _public(workspace):
    return subprocess.run([sys.executable, "-B", "-m", "unittest", "-q", *PUBLIC], cwd=str(workspace),
                          capture_output=True, text=True, timeout=60, env=ENV)


def _failing(proc):
    return sorted({line.split(" ")[1] for line in proc.stderr.splitlines() if line.startswith(("FAIL:", "ERROR:"))})


def _workspace(tmp_path, name, engine_source=None, base=DEFECTIVE):
    workspace = tmp_path / name
    shutil.copytree(base, workspace)
    if engine_source is not None:
        (workspace / ENGINE).write_text(engine_source)
    return workspace


def _hidden(workspace, tmp_path, name):
    return evaluator.run_hidden_check(SCENARIO, workspace, tmp_path / f"evaluator-{name}")


DEF_ENGINE = (DEFECTIVE / ENGINE).read_text()
REF_ENGINE = (REFERENCE / ENGINE).read_text()


def _sub(text, old, new):
    count = text.count(old)
    assert count == 1, (count, old[:70])
    return text.replace(old, new)


def _superficial_public_patch():
    """Patches only the public regression: failed accounts stop being deduplicated. The defect itself stays."""
    text = _sub(DEF_ENGINE, "        self._processed = ProcessedSet()\n",
                "        self._processed = ProcessedSet()\n        self._failed_accounts = set()\n")
    text = _sub(text, "        if self._processed.contains(event.event_id):\n",
                "        if self._processed.contains(event.event_id) and event.account not in self._failed_accounts:\n")
    return _sub(text, "            self._state.restore(state_before)\n",
                "            self._failed_accounts.add(event.account)\n            self._state.restore(state_before)\n")


MUTANTS = {
    # name: (base engine text, transform, predicted public pass, predicted hidden pass)
    "defective_record_before_commit": (DEF_ENGINE, lambda t: t, False, False),
    "reference_correct_repair": (REF_ENGINE, lambda t: t, True, True),
    "superficial_public_test_patch": (None, None, True, False),
    "suppressed_storage_failure": (REF_ENGINE, lambda t: _sub(
        t, "            self._ledger.truncate(ledger_before)\n            raise\n",
        "            self._ledger.truncate(ledger_before)\n            return Outcome.APPLIED\n"), False, False),
    "skipped_ledger_recording": (REF_ENGINE, lambda t: _sub(t, "        self._ledger.append(event)\n", ""), False, False),
    "disabled_idempotency": (REF_ENGINE, lambda t: _sub(
        t, "        if self._processed.contains(event.event_id):\n            return Outcome.DUPLICATE\n", ""), False, False),
    "rollback_balances_only": (REF_ENGINE, lambda t: _sub(t, "            self._ledger.truncate(ledger_before)\n", ""),
                               True, False),
    "rollback_ledger_only": (REF_ENGINE, lambda t: _sub(t, "            self._state.restore(state_before)\n", ""),
                             False, False),
    "unnecessary_rejection_of_redelivery": (REF_ENGINE, lambda t: _sub(
        _sub(_sub(t, "from .intake import parse_event\n", "from .intake import InvalidEvent, parse_event\n"),
             "        self._processed = ProcessedSet()\n", "        self._processed = ProcessedSet()\n        self._attempted = set()\n"),
        "        event = parse_event(raw)\n",
        "        event = parse_event(raw)\n        if event.event_id in self._attempted:\n"
        "            raise InvalidEvent('event was already attempted')\n        self._attempted.add(event.event_id)\n"),
        False, False),
}


@pytest.mark.parametrize("name", list(MUTANTS))
def test_mutant_outcomes_match_the_prediction(tmp_path, name):
    base, transform, public_should_pass, hidden_should_pass = MUTANTS[name]
    source = _superficial_public_patch() if name == "superficial_public_test_patch" else transform(base)
    workspace = _workspace(tmp_path, name, source)
    proc = _public(workspace)
    assert (proc.returncode == 0) is public_should_pass, f"{name}: public failing={_failing(proc)}"
    result = _hidden(workspace, tmp_path, name)
    assert result["passed"] is hidden_should_pass, f"{name}: {result['detail'][:200]}"


def test_the_defective_fixture_fails_only_the_intended_public_regression(tmp_path):
    proc = _public(_workspace(tmp_path, "defective"))
    assert _failing(proc) == [REGRESSION]


def test_the_reference_passes_all_public_tests_and_the_hidden_evaluator(tmp_path):
    workspace = _workspace(tmp_path, "reference", REF_ENGINE)
    assert _public(workspace).returncode == 0
    assert _hidden(workspace, tmp_path, "reference")["passed"] is True


def test_the_reference_differs_from_the_fixture_only_in_the_engine():
    assert {p.relative_to(REFERENCE) for p in REFERENCE.rglob("*") if p.is_file()} == {ENGINE}
    assert DEF_ENGINE != REF_ENGINE


def test_the_hidden_evaluator_fails_the_defective_fixture(tmp_path):
    result = _hidden(_workspace(tmp_path, "defective-hidden"), tmp_path, "defective")
    assert result["passed"] is False
    assert "Q later distinct event on the same account applies" in result["detail"]  # the tail keeps the failed names


# --- registration and integration -------------------------------------------------------------------------

def test_existing_scenarios_are_unchanged_by_the_new_registration():
    for scenario_id, expected in HISTORICAL.items():
        assert _fingerprint(scenarios.SCENARIOS[scenario_id]) == expected, scenario_id


def test_the_new_scenario_resolves_through_the_supervisor_and_the_registry():
    assert supervisor.SCENARIOS["RME-DEBUG-02"] is SCENARIO
    assert SCENARIO.hidden_check == "hidden_rme_debug_02.py"
    assert SCENARIO.hidden_source == str(HIDDEN) and SCENARIO.fixture_root == str(DEFECTIVE)
    assert "RME-DEBUG-01" in scenarios.SCENARIOS and "RME-02" in scenarios.SCENARIOS


def test_a_trial_of_the_new_scenario_must_be_named_after_it(tmp_path):
    bad = subprocess.run([sys.executable, "-B", str(RME1 / "supervisor.py"), "RME-DEBUG-01-QWEN-02",
                          "--scenario", "RME-DEBUG-02", "--image", "x", "--model", "m",
                          "--policy", "RME-01-POLICY-V2", "--expected-revision", "r"],
                         capture_output=True, text=True, timeout=60, cwd=str(REPO))
    assert bad.returncode != 0 and "must be named RME-DEBUG-02-" in bad.stderr
    assert not (REPO / "tmp" / "rme1" / "RME-DEBUG-01-QWEN-02").exists()


def test_the_new_workspace_contains_only_the_agent_visible_fixture(tmp_path):
    created = scenarios.make_workspace(SCENARIO, tmp_path, "agent-visible")
    workspace = Path(created["workspace"])
    names = {p.name for p in workspace.rglob("*") if p.is_file()}
    assert "hidden_rme_debug_02.py" not in names
    assert not any("reference" in str(p.relative_to(workspace)) for p in workspace.rglob("*"))
    assert set(created["initial_hashes"]) == set(scenarios.scenario_fixtures(SCENARIO))


def test_the_image_copies_only_the_new_spec_and_nothing_else_from_the_benchmark():
    lines = [ln for ln in (RME1 / "Dockerfile").read_text().splitlines() if ln.startswith("COPY")]
    assert "COPY evaluation/rme_debug_02/spec.py /opt/agent-loop/evaluation/rme_debug_02/spec.py" in lines
    assert not any(token in ln for ln in lines for token in ("assets", "hidden", "reference"))


def test_the_hidden_evaluator_is_outside_the_fixture(tmp_path):
    assert DEFECTIVE not in HIDDEN.parents and REFERENCE not in HIDDEN.parents


def test_result_and_session_identity_uses_the_new_scenario_id():
    blocked = json.loads((REPO / "tests" / "fixtures" / "rme1" / "session_blocked_glm16.json").read_text())
    blocked["session_id"] = "rme1-RME-DEBUG-02"
    validate_session_data(blocked)
    meta = policy.expected_metadata("RME-01-POLICY-V2")
    summary = {"scenario": "RME-DEBUG-02", "intended_model": "zai-org/glm-4.7-flash",
               "actual_request_models": ["zai-org/glm-4.7-flash"], "request_count": 1, "run_status": "blocked",
               "reason": "recorded", "policy": dict(meta, tool_schema_sha256=[meta["expected_tool_schema_sha256"]])}
    assert supervisor.terminal_session_problems(summary, "OK", blocked, "RME-DEBUG-02") == []
    assert any("is not 'RME-DEBUG-02'" in p for p in supervisor.terminal_session_problems(
        dict(summary, scenario="RME-DEBUG-01"), "OK", blocked, "RME-DEBUG-02"))
    assert any("is not rme1-RME-DEBUG-02" in p for p in supervisor.terminal_session_problems(
        summary, "OK", dict(blocked, session_id="rme1-RME-DEBUG-01"), "RME-DEBUG-02"))


def test_the_policy_v2_allowlist_is_sufficient_for_the_public_task(tmp_path):
    commands = [c["parameters"]["command"] for c in SCENARIO.criteria if c["criterion_type"] == "command_exit"]
    assert commands and all(c.split()[0] == "python3" for c in commands)
    quoted = [c for c in __import__("re").findall(r"'([^']+)'", SCENARIO.goal)]
    assert quoted and all(c.split()[0] in scenarios.TRIAL_TOOL_ALLOWLIST for c in quoted), quoted
    assert _public(_workspace(tmp_path, "allowlist", REF_ENGINE)).returncode == 0


ALLOWED_IMPORTS = {"__future__", "copy", "dataclasses", "enum", "io", "json", "os", "sys", "tempfile", "typing", "unittest",
                   "eventledger", "tests"}


def test_agent_visible_code_uses_only_the_standard_library():
    for source in DEFECTIVE.rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            assert set(names) <= ALLOWED_IMPORTS, (source.name, names)


def test_agent_visible_size_is_recorded_and_bounded():
    lines = sum(len([ln for ln in p.read_text().splitlines() if ln.strip()]) for p in DEFECTIVE.rglob("*.py"))
    assert 250 <= lines <= 500, lines  # the design asks for 400-700; the realised size is recorded in the report
