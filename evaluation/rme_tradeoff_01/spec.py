"""RME-TRADEOFF-01: repair choice under explicit operational profiles. Pure data, read at import time by
evaluation/rme1/scenarios.py on the host and inside the agent image. It reads no files. The fixture and the hidden
evaluator are host-only; their relative locations are strings here and the image does not contain them."""

SCENARIO_ID = "RME-TRADEOFF-01"
ASSET_DIR = "assets/defective"
HIDDEN_FILE = "hidden_rme_tradeoff_01.py"

GOAL = (
    "The event delivery module in delivery/ is wrong. Observed: when the downstream sink times out on a delivery, the "
    "same event can take effect twice. Reproduce it with 'python3 -m unittest -q tests.test_deliverer', investigate how "
    "delivery/deliverer.py handles an ambiguous outcome from the sink, and fix it so that the contract in README.md "
    "holds. Read config.py to identify the operational profile in use. As a final deliverable, write design_note.md in "
    "the workspace root. It explains the change you completed and has these sections: '## Operational profile', "
    "'## Chosen strategy', '## Assumptions', '## Expected operational consequences' and '## Risks and trade-offs'. Do not "
    "modify tests/, config.py, delivery/sink.py or delivery/profiles.py. Confirm the repair with 'python3 -m unittest -q "
    "tests.test_deliverer tests.test_profiles'."
)

PROTECTED = [
    "config.py",
    "delivery/profiles.py",
    "delivery/sink.py",
    "tests/__init__.py",
    "tests/test_deliverer.py",
    "tests/test_profiles.py",
]

CRITERIA = [
    {"criterion_id": "public_tests", "criterion_type": "command_exit", "expected": 0,
     "parameters": {"command": "python3 -m unittest -q tests.test_deliverer tests.test_profiles",
                    "timeout_seconds": 15}},
    {"criterion_id": "design_note_1", "criterion_type": "file_content", "expected": "## Operational profile",
     "parameters": {"path": "design_note.md", "contains": "## Operational profile"}},
    {"criterion_id": "design_note_2", "criterion_type": "file_content", "expected": "## Chosen strategy",
     "parameters": {"path": "design_note.md", "contains": "## Chosen strategy"}},
    {"criterion_id": "design_note_3", "criterion_type": "file_content", "expected": "## Assumptions",
     "parameters": {"path": "design_note.md", "contains": "## Assumptions"}},
    {"criterion_id": "design_note_4", "criterion_type": "file_content", "expected": "## Expected operational consequences",
     "parameters": {"path": "design_note.md", "contains": "## Expected operational consequences"}},
    {"criterion_id": "design_note_5", "criterion_type": "file_content", "expected": "## Risks and trade-offs",
     "parameters": {"path": "design_note.md", "contains": "## Risks and trade-offs"}},
]

KWARGS = {"scenario_id": SCENARIO_ID, "goal": GOAL, "protected": PROTECTED, "criteria": CRITERIA,
          "hidden_check": HIDDEN_FILE}
