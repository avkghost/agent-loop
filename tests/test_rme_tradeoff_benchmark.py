"""RME-TRADEOFF-01 offline validation and integration. Deterministic: no LM Studio, no Docker, no network. Public tests
and the hidden evaluator run in temporary copies of the fixture. Mutation outcomes are predictions written before the run;
a mismatch fails the test and is investigated, not adjusted."""

import ast
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RME1 = REPO / "evaluation" / "rme1"
BENCH = REPO / "evaluation" / "rme_tradeoff_01"
DEFECTIVE = BENCH / "assets" / "defective"
STRATEGIES = BENCH / "strategies"
HIDDEN = BENCH / "hidden_rme_tradeoff_01.py"
DELIVERER = Path("delivery") / "deliverer.py"
PUBLIC = ["tests.test_deliverer", "tests.test_profiles"]
ENV = {"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"}
CORRECT = ["deliverer_recon.py", "deliverer_escalate.py", "deliverer_once.py"]

sys.path.insert(0, str(RME1))
import evaluator  # noqa: E402
import policy  # noqa: E402
import scenarios  # noqa: E402
import supervisor  # noqa: E402
from src.session import validate_session_data  # noqa: E402

SCENARIO = scenarios.SCENARIOS["RME-TRADEOFF-01"]

HISTORICAL = {  # fingerprints recorded before RME-TRADEOFF-01 was added; any change fails here
    "RME-01": "d3c62311351f640c097f9979b5f23edca7cdb962093b03121d32325778b81fcb",
    "RME-02": "1648ac2feb40d78637bc97ca1514363778e3c5f5e6839079e23a63776e8efd46",
    "RME-03": "9733da735a3e00a707ad2e0ab6363e232821714f6d254564b1a9208953aff8f3",
    "RME-DEBUG-01": "474dc327389cdb0c9fb4d989094407c2fb28f9e7e56c872bef5beba0e080b0ba",
    "RME-DEBUG-02": "99f9faabf4c2",
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


def _workspace(tmp_path, name, deliverer_source=None):
    workspace = tmp_path / name
    shutil.copytree(DEFECTIVE, workspace)
    if deliverer_source is not None:
        (workspace / DELIVERER).write_text(deliverer_source)
    return workspace


def _hidden(workspace, tmp_path, name):
    return evaluator.run_hidden_check(SCENARIO, workspace, tmp_path / f"evaluator-{name}")


def _metrics(tmp_path, name):
    return json.loads((tmp_path / f"evaluator-{name}" / "operational_metrics.json").read_text())


DEF = (DEFECTIVE / DELIVERER).read_text()
RECON = (STRATEGIES / "deliverer_recon.py").read_text()
ESCALATE = (STRATEGIES / "deliverer_escalate.py").read_text()


def _sub(text, old, new):
    count = text.count(old)
    assert count == 1, (count, old[:70])
    return text.replace(old, new)


def _superficial_public_patch():
    """Passes the public tests only: ids written by the public tests (prefix 'e') stop after an ambiguous outcome;
    every other id keeps the defect."""
    return _sub(DEF, "            except (DefiniteFailure, AmbiguousTimeout):\n                continue\n",
                "            except DefiniteFailure:\n                continue\n"
                "            except AmbiguousTimeout:\n"
                "                if event_id.startswith('e'):\n"
                "                    self._status[event_id] = UNRESOLVED\n"
                "                    return UNRESOLVED\n"
                "                continue\n")


MUTANTS = {
    # name: (source builder, predicted public pass, predicted hidden pass)
    "defective_blind_retry": (lambda: DEF, False, False),
    "superficial_public_test_patch": (_superficial_public_patch, True, False),
    "reconciliation_unavailable_authorises_send": (lambda: _sub(
        RECON, "                except ReconciliationUnavailable:\n                    break\n",
        "                except ReconciliationUnavailable:\n                    present = False\n"), True, False),
    "silent_loss_of_unresolved_status": (lambda: _sub(
        RECON, "        self._status[event_id] = UNRESOLVED if ambiguous else NOT_APPLIED\n        return self._status[event_id]\n",
        "        if ambiguous:\n            self._status.pop(event_id, None)\n            return UNRESOLVED\n"
        "        self._status[event_id] = NOT_APPLIED\n        return NOT_APPLIED\n"), False, False),
    "reconcile_budget_exceeded": (lambda: _sub(
        RECON, "lookups >= self._profile.reconcile_budget:", "lookups >= 10:"), True, False),
    "ambiguous_reported_applied": (lambda: _sub(
        ESCALATE, "            except AmbiguousTimeout:\n                self._status[event_id] = UNRESOLVED\n"
                  "                return UNRESOLVED\n",
        "            except AmbiguousTimeout:\n                self._status[event_id] = APPLIED\n                return APPLIED\n"),
        True, False),
    "ambiguous_reported_not_applied": (lambda: _sub(
        ESCALATE, "            except AmbiguousTimeout:\n                self._status[event_id] = UNRESOLVED\n"
                  "                return UNRESOLVED\n",
        "            except AmbiguousTimeout:\n                self._status[event_id] = NOT_APPLIED\n"
                  "                return NOT_APPLIED\n"), True, False),
    "unresolved_resent_without_authority": (lambda: _sub(
        ESCALATE, "        if self._status.get(event_id) in (APPLIED, UNRESOLVED):\n            return self._status[event_id]\n",
        "        if self._status.get(event_id) == APPLIED:\n            return APPLIED\n"), True, False),
}


@pytest.mark.parametrize("name", list(MUTANTS))
def test_mutant_outcomes_match_the_prediction(tmp_path, name):
    build, public_should_pass, hidden_should_pass = MUTANTS[name]
    workspace = _workspace(tmp_path, name, build())
    proc = _public(workspace)
    assert (proc.returncode == 0) is public_should_pass, f"{name}: public failing={_failing(proc)}"
    result = _hidden(workspace, tmp_path, name)
    assert result["passed"] is hidden_should_pass, f"{name}: {result['detail'][-200:]}"


def test_the_defective_fixture_fails_the_intended_public_tests_and_the_hidden_evaluator(tmp_path):
    workspace = _workspace(tmp_path, "defective")
    assert _failing(_public(workspace)) == ["test_an_ambiguous_timeout_with_effect_does_not_create_a_second_effect",
                                            "test_every_delivered_event_has_a_status_and_unresolved_events_are_reported"]
    assert _hidden(workspace, tmp_path, "defective")["passed"] is False


@pytest.mark.parametrize("name", CORRECT)
def test_each_correct_strategy_passes_the_public_tests_and_the_hidden_evaluator(tmp_path, name):
    workspace = _workspace(tmp_path, name.replace(".py", ""), (STRATEGIES / name).read_text())
    proc = _public(workspace)
    assert proc.returncode == 0, _failing(proc)
    result = _hidden(workspace, tmp_path, name.replace(".py", ""))
    assert result["passed"] is True, result["detail"][-200:]


def test_the_correct_strategies_are_structurally_different():
    texts = [(STRATEGIES / name).read_text() for name in CORRECT]
    assert len(set(texts)) == 3
    assert "lookup" not in texts[1]  # the escalating strategy never reconciles
    assert texts[0].count("self._sink.lookup(") == 1 and texts[2].count("self._sink.lookup(") == 1


def test_operational_metrics_distinguish_the_strategies_and_profiles(tmp_path):
    rows = {}
    for name in CORRECT:
        tag = name.replace(".py", "")
        workspace = _workspace(tmp_path, tag, (STRATEGIES / name).read_text())
        assert _hidden(workspace, tmp_path, tag)["passed"] is True
        rows[tag] = _metrics(tmp_path, tag)["operational"]
    escalate = rows["deliverer_escalate"]
    assert all(row["lookups"] == 0 for config, row in escalate.items())
    assert rows["deliverer_recon"]["present/live"]["lookups"] > 0
    assert rows["deliverer_recon"]["absent/live"]["lookups"] == 0
    assert rows["deliverer_recon"]["present/live"]["unresolved"] < escalate["present/live"]["unresolved"]
    assert rows["deliverer_recon"]["present/safe"]["sends"] != rows["deliverer_recon"]["present/live"]["sends"]
    for config in ["present/live", "present/safe", "unavailable/live"]:
        assert len({json.dumps(rows[tag][config], sort_keys=True) for tag in rows}) > 1, config


def test_operational_metrics_are_written_outside_the_workspace(tmp_path):
    workspace = _workspace(tmp_path, "placement", RECON)
    before = set(p.relative_to(workspace) for p in workspace.rglob("*"))
    _hidden(workspace, tmp_path, "placement")
    assert set(p.relative_to(workspace) for p in workspace.rglob("*")) == before
    assert (tmp_path / "evaluator-placement" / "operational_metrics.json").exists()


# --- isolation and integration ---------------------------------------------------------------------------

def test_no_hidden_or_strategy_asset_appears_in_the_agent_workspace(tmp_path):
    created = scenarios.make_workspace(SCENARIO, tmp_path, "agent-visible")
    workspace = Path(created["workspace"])
    names = {p.name for p in workspace.rglob("*") if p.is_file()}
    assert "hidden_rme_tradeoff_01.py" not in names
    assert not any(name.startswith("deliverer_") and name != "deliverer.py" for name in names)
    assert set(created["initial_hashes"]) == set(scenarios.scenario_fixtures(SCENARIO))


def test_the_image_copies_only_the_spec_for_this_benchmark():
    lines = [ln for ln in (RME1 / "Dockerfile").read_text().splitlines() if ln.startswith("COPY")]
    assert "COPY evaluation/rme_tradeoff_01/spec.py /opt/agent-loop/evaluation/rme_tradeoff_01/spec.py" in lines
    assert not any(("assets" in ln or "hidden" in ln or "strategies" in ln) and "rme_tradeoff_01" in ln for ln in lines)


def test_the_scenario_is_registered_and_resolves_through_the_supervisor():
    assert supervisor.SCENARIOS["RME-TRADEOFF-01"] is SCENARIO
    assert SCENARIO.hidden_check == "hidden_rme_tradeoff_01.py"
    assert SCENARIO.hidden_source == str(HIDDEN) and SCENARIO.fixture_root == str(DEFECTIVE)


def test_historical_fingerprints_are_unchanged():
    for scenario_id, expected in HISTORICAL.items():
        actual = _fingerprint(scenarios.SCENARIOS[scenario_id])
        assert actual.startswith(expected) or actual == expected, scenario_id


def test_a_trial_must_be_named_after_the_scenario(tmp_path):
    bad = subprocess.run([sys.executable, "-B", str(RME1 / "supervisor.py"), "RME01-TRADEOFF-x",
                          "--scenario", "RME-TRADEOFF-01", "--image", "x", "--model", "m",
                          "--policy", "RME-01-POLICY-V2", "--expected-revision", "r"],
                         capture_output=True, text=True, timeout=60, cwd=str(REPO))
    assert bad.returncode != 0 and "must be named RME-TRADEOFF-01-" in bad.stderr
    assert not (REPO / "tmp" / "rme1" / "RME01-TRADEOFF-x").exists()


def test_result_and_session_identity_uses_the_scenario_id():
    blocked = json.loads((REPO / "tests" / "fixtures" / "rme1" / "session_blocked_glm16.json").read_text())
    blocked["session_id"] = "rme1-RME-TRADEOFF-01"
    validate_session_data(blocked)
    meta = policy.expected_metadata("RME-01-POLICY-V2", scenarios.SCENARIOS["RME-TRADEOFF-01"].tools)
    summary = {"scenario": "RME-TRADEOFF-01", "intended_model": "zai-org/glm-4.7-flash",
               "actual_request_models": ["zai-org/glm-4.7-flash"], "request_count": 1, "run_status": "blocked",
               "reason": "recorded", "policy": dict(meta, tool_schema_sha256=[meta["expected_tool_schema_sha256"]])}
    assert supervisor.terminal_session_problems(summary, "OK", blocked, "RME-TRADEOFF-01") == []
    assert supervisor.terminal_session_problems(dict(summary, scenario="RME-DEBUG-02"), "OK", blocked,
                                                "RME-TRADEOFF-01")


def test_the_policy_v2_allowlist_is_sufficient(tmp_path):
    commands = [c["parameters"]["command"] for c in SCENARIO.criteria if c["criterion_type"] == "command_exit"]
    assert commands and all(c.split()[0] == "python3" for c in commands)
    quoted = [q for q in re.findall(r"'([^']+)'", SCENARIO.goal) if "unittest" in q]  # the commands the task names
    assert quoted and all(c.split()[0] in scenarios.TRIAL_TOOL_ALLOWLIST for c in quoted), quoted
    assert _public(_workspace(tmp_path, "allowlist", RECON)).returncode == 0


# --- design note: final deliverable checked through the trusted acceptance mechanism ---------------------

NOTE_SECTIONS = ["## Operational profile", "## Chosen strategy", "## Assumptions",
                 "## Expected operational consequences", "## Risks and trade-offs"]
GOOD_NOTE = "\n\n".join(f"{section}\nThe completed change, explained." for section in NOTE_SECTIONS) + "\n"


def _note_results(workspace, note_text):
    """Runs the scenario's trusted design-note criteria through the production verification engine."""
    import asyncio
    import dataclasses
    from src.verification import VerificationEngine
    if note_text is not None:
        (workspace / "design_note.md").write_text(note_text)
    # A trial assigns its run identifier to every trusted criterion; the test does the same.
    criteria = [dataclasses.replace(c, run_id="rt-run") for c in scenarios.build_criteria(SCENARIO)
                if c.criterion_type == "file_content"]
    results = asyncio.run(VerificationEngine().verify(criteria, run_id="rt-run"))
    return {r.criterion_id: r.status for r in results}


def test_the_design_note_sections_are_trusted_file_content_criteria():
    notes = [c for c in SCENARIO.criteria if c["criterion_type"] == "file_content"]
    assert [c["parameters"]["contains"] for c in notes] == NOTE_SECTIONS
    assert all(c["parameters"]["path"] == "design_note.md" for c in notes)
    assert not any("design_note" in c["criterion_id"] and c["criterion_type"] == "file_exists" for c in SCENARIO.criteria)


def test_a_complete_design_note_passes_every_section_criterion(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path, "complete-note", RECON)
    monkeypatch.chdir(workspace)
    statuses = _note_results(workspace, GOOD_NOTE)
    assert set(statuses.values()) == {"passed"}, statuses


def test_a_missing_design_note_fails_every_section_criterion(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path, "missing-note", RECON)
    monkeypatch.chdir(workspace)
    statuses = _note_results(workspace, None)
    assert set(statuses.values()) == {"failed"}, statuses


def test_an_incomplete_design_note_fails_only_the_missing_section(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path, "incomplete-note", RECON)
    monkeypatch.chdir(workspace)
    incomplete = GOOD_NOTE.replace("## Risks and trade-offs", "## Risks")
    statuses = _note_results(workspace, incomplete)
    assert list(statuses.values()).count("failed") == 1, statuses
    assert statuses["design_note_5"] == "failed"


def test_section_order_is_not_an_evaluation_requirement(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path, "reordered-note", RECON)
    monkeypatch.chdir(workspace)
    reordered = "\n\n".join(f"{section}\nText." for section in reversed(NOTE_SECTIONS)) + "\n"
    assert set(_note_results(workspace, reordered).values()) == {"passed"}


def test_the_timing_helper_is_removed_and_the_hidden_evaluator_ignores_the_note():
    assert not (BENCH / "process.py").exists()
    assert "design" not in HIDDEN.read_text().lower()


# --- agent-visible code and public tests -----------------------------------------------------------------

ALLOWED_IMPORTS = {"__future__", "copy", "dataclasses", "enum", "json", "os", "sys", "typing", "unittest", "config",
                   "delivery", "tests"}


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


def test_the_hidden_evaluator_pins_the_protected_files():
    text = HIDDEN.read_text()
    for rel in SCENARIO.protected:
        assert f'"{rel}"' in text, rel
