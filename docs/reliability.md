# Reliability guarantees and limitations

## Session persistence

Session saves follow:

```text
validate → same-directory temp write → file fsync → atomic replace → directory fsync
```

Failure before replacement preserves the previous file. If replacement has
occurred but directory fsync fails, the new complete session may be observable
and the error is reported as ambiguous. Orphan temporary files do not become
active sessions. Partial or corrupt JSON fails closed.

## Recovery and MCP

Recovery preserves `unknown` for unresolved external operations and never
automatically redispatches them. MCP validation failures before submission are
local failures; timeout, disconnect, malformed response, and remote error after
submission remain conservative.

## Evidence-to-Claim

Review runners use immutable bundles, provenance, source/chunk attribution,
strict JSON schema validation, and fail-closed scoring.

## Explicit limitations

- No exactly-once guarantee is claimed for external side effects.
- Power-loss durability for a particular filesystem, device, or hardware cache
  is not claimed.
- Detached processes after SIGKILL, crash, or power loss are not guaranteed to
  be cleaned up.
- The runtime is not an operating-system sandbox.
