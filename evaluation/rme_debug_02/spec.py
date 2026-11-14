"""RME-DEBUG-02: failure atomicity and redelivery in a ledger engine. Pure data, read at import time by
evaluation/rme1/scenarios.py on the host and inside the agent image. It reads no files. The fixture and the hidden
evaluator are host-only; their relative locations are strings here and the image does not contain them."""

SCENARIO_ID = "RME-DEBUG-02"
ASSET_DIR = "assets/defective"
HIDDEN_FILE = "hidden_rme_debug_02.py"

GOAL = (
    "The event engine in eventledger/ is wrong. Observed: when applying an event fails because the store cannot commit "
    "it, delivering that same event again is reported as a duplicate and the account balance never changes. Reproduce "
    "it with 'python3 -m unittest -q tests.test_engine', investigate how the engine, the store, the processed-event "
    "record and the ledger work together, and fix the cause with a minimal change. The required behaviour is specified "
    "in README.md. Do not modify anything under tests/. Confirm the repair with 'python3 -m unittest -q "
    "tests.test_batch tests.test_engine tests.test_intake tests.test_report tests.test_store'."
)

PROTECTED = ['tests/__init__.py', 'tests/test_batch.py', 'tests/test_engine.py', 'tests/test_intake.py', 'tests/test_report.py', 'tests/test_store.py']

CRITERIA = [
    {"criterion_id": "public_tests", "criterion_type": "command_exit", "expected": 0,
     "parameters": {"command": "python3 -m unittest -q tests.test_batch tests.test_engine tests.test_intake tests.test_report "
                               "tests.test_store",
                    "timeout_seconds": 15}},
    {"criterion_id": "engine_present", "criterion_type": "file_exists", "expected": True,
     "parameters": {"path": "eventledger/engine.py", "type": "file"}},
]

KWARGS = {"scenario_id": SCENARIO_ID, "goal": GOAL, "protected": PROTECTED, "criteria": CRITERIA,
          "hidden_check": HIDDEN_FILE}
