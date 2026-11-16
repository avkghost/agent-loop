# Usage and CLI

```bash
.venv/bin/python -m src.cli run "Your goal"
.venv/bin/python -m src.cli repl
.venv/bin/python -m src.cli sessions
.venv/bin/python -m src.cli inspect <session-id-or-run-id>
.venv/bin/python -m src.cli resume <session-id>
.venv/bin/python -m src.cli tools
```

Use `--allow` only when needed, `--config PATH` for an explicit configuration,
and `--cache` only when response caching is appropriate. A session may contain
multiple runs: `session_id` identifies the conversation and `run_id` one
execution. `inspect` resolves a run ID when possible.

Trusted verification criteria are supplied separately and are never authored
or changed by the model. A normal model stop is not proof of completion.

See [examples.md](examples.md) for complete task examples.
