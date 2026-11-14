"""RME-1 scenario definitions: fixtures, budgets, and trusted criteria. Visible to the agent side.
Hidden evaluator checks are NOT here; see evaluator.py (host only)."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_BYTES_LIMIT = 64 * 1024
MAX_TURNS = 20
VERIFICATION_CYCLES = 2
WALL_CLOCK_SECONDS = 300
SHELL_TIMEOUT_SECONDS = 15
# The only commands the model may run in a trial. Explicit and narrow; nothing is granted by ALLOW_ALL.
TRIAL_TOOL_ALLOWLIST = {"python3", "ls", "cat"}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    goal: str
    fixtures: Dict[str, str]                # workspace-relative path -> initial content
    protected: List[str]                    # files the model must not change (hashed before the trial)
    criteria: List[Dict[str, Any]]          # trusted criteria, as plain data for AcceptanceCriterion
    hidden_check: str                       # evaluator script file name, kept OUTSIDE the workspace
    fixture_root: Optional[str] = None      # host-only directory the fixture is copied from (when fixtures is empty)
    hidden_source: Optional[str] = None     # host-only path of an external hidden evaluator (when not built in)


SCENARIOS: Dict[str, Scenario] = {
    "RME-01": Scenario(
        scenario_id="RME-01",
        goal=("Create logsum.py in the workspace with a standard-library function summarize(lines) that returns a dict "
              "mapping each severity (INFO, WARN, ERROR) to the number of lines carrying it. Input lines have the form "
              "'<ISO timestamp> <SEVERITY> <message>'. Lines that do not match this form are counted under the key "
              "'malformed' and never raise. Also write test_logsum.py with unittest tests, and make them pass with "
              "'python3 -m unittest -q test_logsum'."),
        fixtures={},
        protected=[],
        criteria=[
            {"criterion_id": "file_logsum", "criterion_type": "file_exists", "expected": True,
             "parameters": {"path": "logsum.py", "type": "file"}},
            {"criterion_id": "file_tests", "criterion_type": "file_exists", "expected": True,
             "parameters": {"path": "test_logsum.py", "type": "file"}},
            {"criterion_id": "tests_pass", "criterion_type": "command_exit", "expected": 0,
             "parameters": {"command": "python3 -m unittest -q test_logsum", "timeout_seconds": 15}},
        ],
        hidden_check="hidden_rme01.py",
    ),
    "RME-02": Scenario(
        scenario_id="RME-02",
        goal=("The function mean(values) in calc/stats.py returns wrong results. The expected behaviour is the "
              "arithmetic mean of a non-empty list of numbers, and ValueError for an empty list. Inspect the project, "
              "run its tests, repair calc/stats.py, and leave tests/test_stats.py unchanged. Confirm the repair with "
              "'python3 -m unittest -q tests.test_stats'."),
        fixtures={
            "calc/__init__.py": "",
            "calc/stats.py": ("def mean(values):\n"
                              "    if not values:\n"
                              "        raise ValueError('empty')\n"
                              "    return sum(values) / (len(values) + 1)\n"),
            "tests/__init__.py": "",
            "tests/test_stats.py": ("import unittest\n"
                                    "from calc.stats import mean\n\n\n"
                                    "class MeanTests(unittest.TestCase):\n"
                                    "    def test_mean(self):\n"
                                    "        self.assertEqual(mean([2, 4]), 3)\n\n"
                                    "    def test_empty(self):\n"
                                    "        with self.assertRaises(ValueError):\n"
                                    "            mean([])\n"),
        },
        protected=["tests/test_stats.py"],
        criteria=[
            {"criterion_id": "tests_pass", "criterion_type": "command_exit", "expected": 0,
             "parameters": {"command": "python3 -m unittest -q tests.test_stats", "timeout_seconds": 15}},
            {"criterion_id": "stats_defined", "criterion_type": "file_content", "expected": True,
             "parameters": {"path": "calc/stats.py", "contains": "def mean"}},
        ],
        hidden_check="hidden_rme02.py",
    ),
    "RME-03": Scenario(
        scenario_id="RME-03",
        goal=("Add truncate(text, n) to textutil.py. It returns text unchanged when len(text) <= n. Otherwise it returns "
              "the first n-3 characters followed by '...'. It raises ValueError when n < 3. Keep slugify() and its "
              "existing tests in tests/test_textutil.py unchanged. Add tests for truncate in tests/test_truncate.py and "
              "confirm with 'python3 -m unittest discover -s tests -q'."),
        fixtures={
            "textutil.py": ("def slugify(text):\n"
                            "    return '-'.join(text.lower().split())\n"),
            "tests/__init__.py": "",
            "tests/test_textutil.py": ("import unittest\n"
                                       "from textutil import slugify\n\n\n"
                                       "class SlugTests(unittest.TestCase):\n"
                                       "    def test_slug(self):\n"
                                       "        self.assertEqual(slugify('Hello World'), 'hello-world')\n"),
        },
        protected=["tests/test_textutil.py"],
        criteria=[
            {"criterion_id": "truncate_defined", "criterion_type": "file_content", "expected": True,
             "parameters": {"path": "textutil.py", "contains": "def truncate"}},
            {"criterion_id": "tests_pass", "criterion_type": "command_exit", "expected": 0,
             "parameters": {"command": "python3 -m unittest discover -s tests -q", "timeout_seconds": 15}},
        ],
        hidden_check="hidden_rme03.py",
    ),
}

# RME-DEBUG-01 is defined in evaluation/rme_debug/spec.py (pure data, importable in the agent image). Its fixture and
# hidden evaluator are host-only files, referenced by path and never read here. RME-01/02/03 above are unchanged.
_DEBUG_DIR = Path(__file__).resolve().parents[1] / "rme_debug"


def _load_debug_spec():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rme_debug_spec", _DEBUG_DIR / "spec.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_debug = _load_debug_spec()
SCENARIOS[_debug.SCENARIO_ID] = Scenario(
    fixtures={}, fixture_root=str(_DEBUG_DIR / _debug.ASSET_DIR), hidden_source=str(_DEBUG_DIR / _debug.HIDDEN_FILE),
    **_debug.KWARGS)


def scenario_fixtures(scenario: Scenario) -> Dict[str, str]:
    """The agent-visible files for a scenario: the literal fixtures, or every file under fixture_root (host only)."""
    if scenario.fixtures or not scenario.fixture_root:
        return dict(scenario.fixtures)
    root = Path(scenario.fixture_root)
    return {str(path.relative_to(root)): path.read_text()
            for path in sorted(root.rglob("*")) if path.is_file() and "__pycache__" not in path.parts}


# Hidden evaluator checks. These run from the harness, after the trial, against the workspace. The model never sees them.


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_workspace(scenario: Scenario, root: Path, trial_id: str) -> Dict[str, Any]:
    """A fresh, isolated workspace under root/trial_id. Refuses to reuse a directory or to sit inside the repo."""
    workspace = (root / trial_id).resolve()
    # Only the approved local artifact area inside the repository (tmp/rme1/) may hold trial workspaces
    artifact_root = (REPO_ROOT / "tmp" / "rme1").resolve()
    inside_repo = REPO_ROOT in workspace.parents or workspace == REPO_ROOT
    if inside_repo and artifact_root not in workspace.parents:
        raise ValueError("workspace must not be inside the Agent Loop repository (outside tmp/rme1/)")
    if workspace.exists():
        raise FileExistsError(f"workspace already exists: {workspace}")
    workspace.mkdir(parents=True)
    fixtures = scenario_fixtures(scenario)
    for rel, content in fixtures.items():
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    initial = {rel: file_hash(workspace / rel) for rel in fixtures}
    return {"workspace": str(workspace), "initial_hashes": initial}


def protected_unchanged(scenario: Scenario, workspace: Path, initial: Dict[str, str]) -> bool:
    return all(file_hash(workspace / rel) == initial.get(rel) for rel in scenario.protected)


def build_criteria(scenario: Scenario):
    """Trusted criteria as AcceptanceCriterion objects. origin 'user': supplied by the harness, not the model."""
    from src import acceptance
    return [acceptance.AcceptanceCriterion(criterion_id=item["criterion_id"], run_id="",
                                           criterion_type=item["criterion_type"], expected=item["expected"],
                                           origin="user", parameters=dict(item["parameters"]))
            for item in scenario.criteria]
