# Agent Loop

Agent Loop is a local, tool-using agent runtime with durable execution state,
conservative recovery, MCP integration, compaction, and evidence-aware review
helpers.

Current release: **0.1.0**

## Quick start

Requirements: Python 3.11+, an OpenAI-compatible inference endpoint, and
Node.js/npm only for npm-based MCP servers.

```bash
python -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.cli config
.venv/bin/python -m src.cli run "Your goal"
```

Useful commands:

```bash
.venv/bin/python -m src.cli repl
.venv/bin/python -m src.cli sessions
.venv/bin/python -m src.cli inspect <session-or-run-id>
.venv/bin/python -m src.cli resume <session-id>
.venv/bin/python -m src.cli tools
```

## Documentation

- [Installation and setup](docs/installation.md)
- [Usage and CLI](docs/usage.md)
- [Examples](docs/examples.md)
- [Extended examples](docs/examples-reference.md)
- [Configuration and MCP](docs/configuration.md)
- [Architecture and lifecycle](docs/architecture.md)
- [Reliability guarantees and limitations](docs/reliability.md)
- [Testing and development](docs/development.md)
- [Full reference](docs/reference.md)
- [Quick reference](docs/quick-reference.md)
- [Agent Loop 0.1.0 release notes](docs/release-notes.md)

## Reliability at a glance

- Tool calls use `pending → running → terminal/unknown` journal states.
- Unresolved external calls are never automatically redispatched on recovery.
- Sessions use validated temporary writes, file fsync, atomic replacement, and
  directory fsync.
- Corrupt or partial sessions fail closed.
- MCP ambiguous outcomes remain conservative.
- Compaction keeps execution state outside the prose summary.
- Evidence-to-Claim runners use provenance checks and strict JSON validation.

These are conservative runtime guarantees, not exactly-once or universal
power-loss guarantees. See [reliability.md](docs/reliability.md).

## Testing

```bash
.venv/bin/python -m pytest -p no:cacheprovider tests -q
```

## Project layout

```text
src/        runtime, CLI, persistence, execution, MCP, compaction
tests/      unit, integration, recovery, and policy regression tests
tools/      standalone Evidence-to-Claim review runners
docs/       focused user, maintainer, and full-reference documentation
```
