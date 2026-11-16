# Agent Loop 0.1.0

## Reliability baseline

- Durable `ExecutionJournal` states: `pending`, `running`, terminal, and
  conservative `unknown` recovery.
- Unresolved external calls are never automatically redispatched after a
  restart, compaction, timeout, disconnect, or ambiguous transport failure.
- Separate `run_id` and `session_id` ownership with provenance-aware inspect
  output and recovery diagnostics.
- Session persistence uses a same-directory temporary file, file `fsync`,
  atomic replacement, and directory `fsync`.
- Corrupt or partial session JSON fails closed; orphan temporary files do not
  replace a valid session.
- MCP timeout, disconnect, malformed-result, remote-error, and unadvertised
  tool paths preserve conservative outcome classification.
- Compaction preserves authoritative execution state independently of the
  narrative summary.

## Evidence-to-Claim validation

- Immutable Axum migration fixtures with provenance and source/chunk evidence.
- Shared fail-closed JSON schema validation for primary and follow-up review
  runners.
- Exact classification, attribution, uncertainty, and insufficiency checks.
- GLM and Qwen validation completed with reasoning disabled.

## Explicit limitations

- No exactly-once guarantee is claimed for external side effects.
- `unknown` means the external outcome must be reconciled manually.
- Power-loss durability for a particular filesystem, storage device, or
  hardware cache is not claimed.
- SIGKILL and power-loss cleanup of detached external processes is not claimed.
