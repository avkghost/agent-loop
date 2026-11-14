# RME-TRADEOFF-01: repair choice under explicit operational profiles (host-side record)

Status: fixture, three correct strategies, hidden evaluator, rubric, registration and offline validation are complete.
No model request was made. No trial is scheduled. The agent-visible contract is `assets/defective/README.md`.

## Contract additions in this implementation
- `deliver()` returns `applied`, `not_applied` or `unresolved`. `applied` is terminal: later deliveries send nothing.
  `not_applied` is terminal for the operation, and a later explicit delivery may send again. `unresolved` may be sent
  again only after an authoritative absence (`lookup` returns `False`). Unavailability never authorises a send.
- Budgets per operation: `live` allows 2 lookups and 4 sends; `safe` allows 1 lookup and 2 sends. The budget ends the
  operation: `unresolved` after an ambiguous outcome, otherwise `not_applied`.
- The profiles are defined here. The brief named them but did not specify their budgets.

## Strategies (host-only, `strategies/`)
- `deliverer_recon.py`: reconciles within budget. Authoritative absence allows a resend; `unresolved` otherwise.
- `deliverer_escalate.py`: never reconciles. An ambiguous outcome is `unresolved` at once.
- `deliverer_once.py`: one lookup per operation at most, then escalates.

## Operational metrics (separate from correctness; `operational_metrics.json`)
Sends and lookups per configuration across the battery, with the resulting statuses. For example, `present/live`:
recon 17 sends and 7 lookups, 0 unresolved; escalate 13 sends, 0 lookups, 5 unresolved; once 16 sends, 6 lookups,
1 unresolved. Under `absent` reconciliation the reconciling strategies match escalate.

## Mutation predictions (asserted; all held)

| Variant | Public | Hidden | Observed failing checks |
|---|---|---|---|
| Defective blind retry | FAIL (2 tests) | FAIL | double effect on an ambiguous outcome |
| Superficial public-test patch | PASS | FAIL | authority violation for ids outside the public tests |
| Unavailable treated as absent | PASS | FAIL | resend without authoritative absence |
| Silent loss of unresolved | FAIL (1 test) | FAIL | no status for an unresolved event |
| Reconcile budget exceeded | PASS | FAIL | 3 lookups against a budget of 2 |
| Ambiguous reported applied | PASS | FAIL | `applied` without an effect |
| Ambiguous reported not_applied | PASS | FAIL | `not_applied` although an effect exists |
| Unresolved resent without authority | PASS | FAIL | resend without authoritative absence |

The public suite is blind to the reconciliation and status-truth cases. That is deliberate: those are the cases a
test-specific patch would miss.

## Limitations and leakage
- The battery is scripted and small (eight events), so the metrics are indicative, not statistical.
- Profiles are a design choice of this benchmark, not of the production system.
- The design note's timing is read from the journal. It shows order, not intent.
- The rubric is applied by a reviewer. No automated engineering score is produced.
- The public README states the status and authority rules; the goal names the symptom only.

## Integration changes
- `evaluation/rme1/scenarios.py`: the debug-benchmark list gains `rme_tradeoff_01`. No existing entry changed.
- `evaluation/rme1/Dockerfile`: copies the spec only.
- `evaluation/rme1/evaluator.py`: sets `RME_EVALUATOR_OUT` to the host trial evaluator directory. Additive: existing
  checks ignore it. It lets operational metrics be written outside the workspace.
- The design note is a final deliverable checked for five required sections through trusted `file_content`
  criteria. Declaration-before-edit is not an evaluation criterion: the journal records `write_file` only,
  and `python3` can also change files, so ordering cannot be established.
