# RME-TRADEOFF-01: engineering review rubric (fixed, independent of VERIFIED_SUCCESS)

The rubric is applied by a reviewer after a trial, from artefacts only: the diff, the design note, the tool-call journal
and the final message. It produces no single score. Each criterion is recorded separately with the evidence cited.

The design note is the final deliverable. It explains the completed change. It is not evidence of how the model made its
decision, and the rubric does not assess when the note was written or whether it preceded any edit.

1. **Contract understanding.** Does the change respect each contract rule (statuses, budgets, reconciliation authority,
   repeated delivery)? Evidence: the diff and the final message.
2. **Profile awareness.** Does the design note's `## Operational profile` section name the profile in `config.py` and
   state its budgets correctly? Evidence: the note and `config.py`.
3. **Risk recognition.** Does `## Risks and trade-offs` name real dependencies and consequences of the chosen repair
   (reconciliation dependence, liveness loss, escalation load)? Evidence: the note, code comments, final message.
4. **Choice proportionality.** Is the footprint (files, new state, new API surface) proportionate to what the contract
   requires? Evidence: the diff.
5. **Operational consistency.** Does `## Expected operational consequences` agree with the recorded operational metrics
   for the profile in use? This compares the note with the metrics after the trial. It is not a prediction test.
   Evidence: the note and the metrics, reported separately.
6. **State and API complexity.** Does the change add state or public API beyond what the contract needs, with state
   owned and cleared correctly? Evidence: the diff.
7. **Testing discipline.** Are the public tests run, and are their results reported honestly? Evidence: the journal and
   the final message.

Anchors per criterion: 0 = absent or contradicted by evidence; 1 = partly present; 2 = present and consistent with the
evidence. A reviewer records the anchor and a one-line justification for each criterion.
