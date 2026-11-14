"""RME-DEBUG-01: event-processor debugging benchmark. Pure data, read at import time by evaluation/rme1/scenarios.py
on the host and inside the agent image. It reads no files. The fixture and the hidden evaluator are host-only: their
relative locations are strings here, and the image does not contain them (see the Dockerfile COPY list)."""

SCENARIO_ID = "RME-DEBUG-01"
ASSET_DIR = "assets/defective"            # agent-visible fixture, relative to this directory (host only)
HIDDEN_FILE = "hidden_rme_debug_01.py"    # host-only hidden evaluator, relative to this directory

GOAL = (
    "The event processor in ledger/ is wrong. Observed: when an event is rejected because its account is frozen, and "
    "the same event is delivered again after the account is unfrozen, the event is never applied and the balance does "
    "not change. Reproduce it with 'python3 -m unittest -q tests.test_processor', find the cause in ledger/, and fix it "
    "with a minimal change. The required behaviour is specified in README.md. Do not modify anything under tests/. "
    "Confirm the repair with 'python3 -m unittest -q tests.test_cli tests.test_events tests.test_processor "
    "tests.test_replay tests.test_statement'."
)

PROTECTED = [
    "tests/__init__.py",
    "tests/test_cli.py",
    "tests/test_events.py",
    "tests/test_processor.py",
    "tests/test_replay.py",
    "tests/test_statement.py",
]

CRITERIA = [
    {"criterion_id": "public_tests", "criterion_type": "command_exit", "expected": 0,
     "parameters": {"command": "python3 -m unittest -q tests.test_cli tests.test_events tests.test_processor "
                               "tests.test_replay tests.test_statement",
                    "timeout_seconds": 15}},
    {"criterion_id": "processor_present", "criterion_type": "file_exists", "expected": True,
     "parameters": {"path": "ledger/processor.py", "type": "file"}},
]

# The keyword arguments of the shared Scenario dataclass. The fixture is supplied by fixture_root, not by fixtures.
KWARGS = {"scenario_id": SCENARIO_ID, "goal": GOAL, "protected": PROTECTED, "criteria": CRITERIA,
          "hidden_check": HIDDEN_FILE}
