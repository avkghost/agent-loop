# RME-DEBUG-01: event-processor debugging (host and agent side)

Status: integrated with the RME supervisor and validated offline. No real-model trial has been run. The first planned
trial is `RME-DEBUG-01-GLM-01`, which needs an image rebuilt from the integrated tree (see "Before a trial").

The scenario id was renamed from `RME-02` because `RME-02` is the committed statistics-repair scenario in
`evaluation/rme1/scenarios.py`. That scenario is unchanged (its fingerprint is pinned by a test).

## What the agent must do

Inspect an existing Python project, reproduce a defect with its public tests, identify the cause, make a minimal
correction and verify it. The goal (in `spec.py`) describes the observed symptom and does not name the faulty line.

## Layout

- `spec.py`: pure data (scenario id, goal, protected test files, trusted criteria, and the relative locations of the
  fixture and hidden evaluator). It reads no files, so the agent image can import it.
- `assets/defective/`: the agent-visible project (`ledger` package, public tests, public `README.md` contract).
  Standard library only. Host-only: copied into each workspace; never mounted from here.
- `assets/reference/ledger/processor.py`: the clean reference. The only difference from the defective project.
- `hidden_rme_debug_01.py`: host-only hidden evaluator. Uses only the public API and pins the hashes of the protected
  public tests. Copied outside the workspace and run with `-B` by `evaluation/rme1/evaluator.py`.
- `../../tests/test_rme_debug_benchmark.py`: offline validation and integration tests (20).

## Integration (smallest change)

- `evaluation/rme1/scenarios.py` registers `RME-DEBUG-01` from `spec.py`. `RME-01`, `RME-02` and `RME-03` are unchanged.
  `Scenario` gained two optional host-only fields: `fixture_root` (the fixture is copied from there) and
  `hidden_source` (the external hidden evaluator). Both default to `None`.
- `evaluation/rme1/evaluator.py` copies an external `hidden_source` when one is set, and otherwise uses its built-in
  checks as before.
- `evaluation/rme1/supervisor.py`: the scenario must be registered; a non-RME-01 trial must be named
  `RME-DEBUG-01-<label>`, so artifact paths cannot collide; the RESULT's scenario must match the SESSION identity
  (`rme1-RME-DEBUG-01`) and the evaluator record names the scenario it judged.
- `evaluation/rme1/Dockerfile`: copies `evaluation/rme_debug/spec.py` only. The fixture, reference and hidden evaluator
  are not copied.
- Policy v2 is reused unchanged. Its allowlist (`python3`, `ls`, `cat`) is sufficient for the public task.

## Acceptance criteria (trusted, origin user)

- `public_tests`: `python3 -m unittest -q tests.test_cli tests.test_events tests.test_processor tests.test_replay
  tests.test_statement` exits 0.
- `processor_present`: `ledger/processor.py` exists.

The hidden evaluator is an independent layer, reported separately; runtime completion still needs the criteria.

## Offline validation matrix

| Variant | Public tests | Hidden evaluator |
|---|---|---|
| Defective starting point | FAIL, only `test_failed_operation_is_not_recorded_and_can_be_retried` | FAIL |
| Clean reference | PASS | PASS |
| Superficial bypass (skip the duplicate check for any account ever frozen) | PASS | FAIL (H1) |
| Incorrect dedupe key (account and amount) | FAIL | FAIL |
| Suppressed failure (frozen returns DUPLICATE) | FAIL | FAIL |
| Destructive (no idempotency) | FAIL | FAIL |

## Before a trial (not done)

1. Rebuild the evaluation image from the committed tree. The current image predates `spec.py` and lacks it; the runner
   imports the registry and would fail.
2. Run the preflight with `--scenario RME-DEBUG-01 --policy RME-01-POLICY-V2 --expected-revision <rev>`.
3. Execute one trial named `RME-DEBUG-01-GLM-01`, after the model and configuration are confirmed.

## Limitations and possible leakage

- The public README states the failed-event rule, which narrows the search to failure handling. The goal names the
  symptom and the frozen-account path.
- The public tests name the failing test.
- One defect is seeded, so the benchmark does not measure breadth of debugging.
- The agent-visible project is 287 non-blank lines: slightly under the design's "approximately 300-500". Not padded.
- Policy v2 forbids `find`, `grep` and `rg`. Discovery uses `list_directory` and `read_file`, which is workable here
  but a real restriction for larger code.
- Concurrency, distributed delivery and persistence are out of scope, and an agent must not assume them.
- The hidden evaluator's H1 includes a redelivery case that a superficial patch fails. An agent that could read it
  would know the trap; by design it cannot.
