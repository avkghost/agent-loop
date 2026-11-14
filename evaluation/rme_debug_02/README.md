# RME-DEBUG-02: failure atomicity and redelivery (host-side design record)

Status: fixture, reference, hidden evaluator, registration and offline validation are complete. No model request has
been made for this benchmark, and no trial is scheduled. The agent-visible contract is `assets/defective/README.md`.

## Baseline

HEAD `600b632` ("Add RME debugging benchmark") when this work started. RME-DEBUG-01, `src/` and the RME-01/02/03
scenarios were unchanged. Their fingerprints are pinned in `tests/test_rme_debug_02_benchmark.py`.

## Scenario and structure (agent-visible, `assets/defective/eventledger/`)

- `intake.py`: event contract and validation (`InvalidEvent`).
- `state.py`: balances and freeze flags, with snapshot and restore for a unit of work (`AccountFrozen`).
- `ledger.py`: append-only entries, with truncation for rollback.
- `dedupe.py`: the processed-event record (ids in order).
- `store.py`: a simulated persistence layer with deterministic commit failures (`StorageError`). No durability, no
  concurrency control.
- `engine.py`: the unit of work `apply()`, which orchestrates the other modules and the rollback.
- `batch.py`: applies a list of events and reports each outcome; a failed event does not stop the batch.
- `report.py`: `reconcile()`, a consistency check over the engine and its last committed image.

## Failure model

- F1: account frozen. Raised before any state change; no commit.
- F2: storage failure at commit. Everything changed by the unit of work is rolled back, nothing is persisted, the event
  is not recorded as processed, and the exception propagates.

## Defect and reference correction

The defect is one ordering error in `engine.py`: the processed-event record is updated before the commit. The rollback
restores balances and the ledger but not the record, so a redelivered event after a failed commit is reported as a
duplicate and never applied. The symptom is visible through the public API; the cause spans `engine.py` (ordering),
`dedupe.py` (the record has no rollback) and `store.py` (where the failure is injected).

The reference (`assets/reference/eventledger/engine.py`) is the only file that differs from the fixture. It builds the
committed image with the event included, and records the event only after the commit succeeds.

## Measured mutation matrix (asserted in `tests/test_rme_debug_02_benchmark.py`)

| Variant | Public | Hidden | Failing public tests (actual) |
|---|---|---|---|
| Defective (record before commit) | FAIL | FAIL | `test_a_failed_event_is_applied_once_when_redelivered` |
| Reference | PASS | PASS | none |
| Superficial public-test patch (skip dedupe for accounts that failed) | PASS | FAIL | none (hidden: redelivery after success) |
| Suppressed storage failure (reports APPLIED) | FAIL | FAIL | redelivery; storage failure propagates |
| Skipped ledger recording | FAIL | FAIL | normal sequence reconciles; successful commit records ledger |
| Disabled idempotency | FAIL | FAIL | batch; normal sequence; redelivery is duplicate |
| Rollback of balances only (ledger kept) | PASS | FAIL | none (hidden: ledger after failure, reconcile) |
| Rollback of ledger only (balances kept) | FAIL | FAIL | redelivery; storage failure leaves balance unchanged |
| Unnecessary rejection of redelivery | FAIL | FAIL | batch; redelivery; normal sequence; duplicate |

The predictions written into the test were asserted and held. The one variant the public suite cannot see is the
rollback of balances only, which is why the hidden evaluator checks the ledger after a failure.

## Hidden evaluator (`hidden_rme_debug_02.py`)

Host only; public API only; pins the hashes of the protected tests. Sections: protected files; F1 frozen account; F2
storage failure (propagation, full rollback, nothing persisted, reconciliation, redelivery, duplicate after success);
F3 repeated failures; B batch redelivery across batches; D distinct events around a failure; Q idempotency after success
on an account that has failed; V validation; C consistency of an ordinary sequence.

## Limitations and possible leakage

- The defect is one line of ordering, and the agent-visible README states the failure rules explicitly.
- The public tests name the failing test and point at the store-failure symptom.
- The hidden evaluator's check of a later redelivery after a success is a known trap for superficial patches; an agent
  that could read it would know the trap. By design it cannot.
- The store is a simulation. The benchmark does not test durability, crashes, concurrency or distributed delivery, and
  agents must not assume them.
- Realised size: 311 nonblank lines of agent-visible Python (code and tests), below the requested 400–700. Size was not padded.
- Policy v2 forbids `find`, `grep` and `rg`; discovery uses `list_directory` and `read_file`.

## Before a trial (not done)

1. Commit the RME-DEBUG-02 changes and rebuild the image from that clean HEAD. The validation image built here is from an
   uncommitted tree (`-dirty-debug02` label) and must not be used for a trial.
2. Run the preflight with `--scenario RME-DEBUG-02 --policy RME-01-POLICY-V2 --expected-revision <rev>`.
3. Name the first trial `RME-DEBUG-02-<model>-01` and run it once, after the model and configuration are confirmed.
