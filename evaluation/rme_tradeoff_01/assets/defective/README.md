# delivery: event delivery to a downstream sink

`Deliverer` sends each event to a `Sink` and records a status for every event it has delivered.

## Contract

- `Deliverer(sink, profile=None)`. `profile` defaults to `config.ACTIVE_PROFILE`. Profiles are in
  `delivery/profiles.py`.
- `deliver(event)` takes `{"event_id": "<non-empty string>"}` (other fields are ignored) and returns a status. An
  invalid event raises `InvalidEvent` before any sink call.
- Each send to the sink has one of three outcomes:
  - success: the effect is applied;
  - `DefiniteFailure`: nothing is applied;
  - `AmbiguousTimeout`: the effect may or may not have been applied. The caller cannot tell.
- Statuses:
  - `applied`: the effect exists. Terminal: later `deliver` calls send nothing for this event.
  - `not_applied`: the effect is established not to exist. Terminal for the current operation. A later `deliver`
    may send again.
  - `unresolved`: an ambiguous outcome has not been resolved. The effect is neither known to exist nor known to be
    absent. A later `deliver` must not send unless an authoritative absence has been established first.
- Budgets apply per `deliver` operation: at most `delivery_budget` sends and `reconcile_budget` lookups. Exceeding a
  budget is a contract violation. When the budget is spent the operation ends: `unresolved` if the last outcome was
  ambiguous, otherwise `not_applied`.
- Reconciliation: `sink.lookup(event_id)` returns `True` (the effect exists) or `False` (it does not). `sink.lookup` is
  `None` when the sink offers no reconciliation. A lookup may raise `ReconciliationUnavailable`. Unavailability never
  authorises a send. Only an authoritative `False` does.
- `report()` returns the sorted ids whose status is `unresolved`. Every delivered event has a status.
- Correctness does not depend on the profile. The profile sets the budgets, and therefore the operational
  consequences: the number of sends and lookups, and how many events remain unresolved.

## Profiles (`delivery/profiles.py`)

- `live`: 2 lookups and 4 sends per operation.
- `safe`: 1 lookup and 2 sends per operation.

## Not promised

Recovery after a crash during an operation; concurrent use of one deliverer; persistence of statuses across restarts;
exactly-once delivery across restarts.

## Compatibility

`Deliverer`, `InvalidEvent`, the status strings, `status()` and `report()` are stable. `config.py`,
`delivery/sink.py`, `delivery/profiles.py` and `tests/` are protected: do not change them.

## Design note

Write `design_note.md` in the workspace root as the final deliverable. It explains the change you completed. It needs
these sections (headings must appear exactly as written):

- `## Operational profile`: the profile in use (from `config.py`) and what it implies for the budgets;
- `## Chosen strategy`: the repair you implemented;
- `## Assumptions`: what the repair relies on;
- `## Expected operational consequences`: how the repair behaves under each sink capability and profile;
- `## Risks and trade-offs`: what the repair gives up or depends on.

The note is checked for these sections. Its content is reviewed separately. It does not record how the decision was
made.

## Running the public tests

    python3 -m unittest -q tests.test_deliverer tests.test_profiles
