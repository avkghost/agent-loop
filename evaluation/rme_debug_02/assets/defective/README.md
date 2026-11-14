# eventledger: event application with a ledger, a processed-event record and a store

`eventledger` applies account events. Applying one event is a unit of work: it changes balances, appends a ledger
entry, records the event as processed, and commits the resulting image to the store.

## Contract (what must hold)

1. An event is applied at most once. A redelivery of an event that was applied returns `Outcome.DUPLICATE` and
   changes nothing, and the store is not written.
2. A successfully applied event has consistent records: its balance change, its ledger entry and its processed-id
   entry all exist, and the last committed image equals the live state.
3. A failed event leaves no partial committed state. Its balance change, ledger entry and processed-id entry are not
   left behind.
4. A failed event remains eligible for redelivery. Once the failure has passed, delivering the same event again
   applies it normally, exactly once.
5. Distinct events are processed independently, even when account and amount are equal.
6. Validation and error propagation are preserved: an invalid event raises `InvalidEvent`; a frozen account raises
   `AccountFrozen`; a failed commit raises `StorageError` (it is not swallowed).

## Failure model

- Failure point F1, account frozen: raised before any state is changed.
- Failure point F2, storage failure at commit: raised by `store.commit`. Everything the unit of work changed before the
  commit is rolled back, and the exception propagates to the caller.

A commit is a successful `store.commit(image)`. The image holds the balances, the ledger and the processed ids after
the event.

Not promised: durability across a crash, concurrent use of one engine, distributed transactions, and exactly-once
delivery across process restarts. The store is a simulation.

## Running the public tests

    python3 -m unittest -q tests.test_engine tests.test_intake tests.test_report tests.test_store

Do not change anything under `tests/`.
