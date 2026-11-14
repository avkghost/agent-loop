# ledger: event processor

`ledger` applies account events exactly once per `event_id`.

## Contract

- **Event**: a JSON object with `event_id` (non-empty string), `account` (non-empty string) and `amount` (integer,
  not a boolean). Anything else raises `InvalidEvent`. An invalid event is neither applied nor recorded.
- **`Processor.apply(event)`** returns `Outcome.APPLIED` when the event changes an account balance. It returns
  `Outcome.DUPLICATE` when an event with the same `event_id` was already processed successfully. A duplicate changes
  nothing.
- **Distinct `event_id`s** are independent, even when account and amount are equal.
- **Failures**: if applying an event fails (the account is frozen, which raises `AccountFrozen`), the exception
  propagates, the balance and the ledger are unchanged, and the event is not recorded as processed. Once the account
  accepts changes again, the same event can be delivered and is applied normally.
- **`Processor.ledger()`** lists applied `event_id`s in application order. Each appears at most once.
- **`replay(lines, processor)`** processes JSON lines and returns counts (`applied`, `duplicate`, `invalid`,
  `failed`) and `errors`. Invalid lines are reported, not applied.

## Running the public tests

    python3 -m unittest -q tests.test_processor tests.test_replay

Do not change anything under `tests/`.
