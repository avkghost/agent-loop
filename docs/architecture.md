# Architecture and lifecycle

The runtime composes instructions, requests a model turn, validates the
provider response, persists state, dispatches approved tools, persists results,
and continues until the model stops or a boundary requires attention.

## Durable execution state

```text
pending → running → completed | failed | unknown
```

The journal is independent of summaries and message prose. A persistence
failure before dispatch prevents the external call. An ambiguous outcome is
recorded as `unknown` and is not silently converted into success or retried.

## Ownership and compaction

`session_id` identifies the persisted conversation; `run_id` identifies one
execution. Execution records and evidence retain ownership and provenance.
Compaction shortens replaceable narrative context while retaining the objective,
protocol-valid history, and authoritative execution state.
