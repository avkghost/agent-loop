# Testing and development

## Test suite

```bash
.venv/bin/python -m pytest -p no:cacheprovider tests -q
```

Focused suites include `test_session_integrity.py`,
`test_execution_state_regression.py`, and `test_mcp_sdk_contract.py`.
Subprocess crash tests verify observable recovery integrity; they do not prove
hardware power-loss guarantees.

The standalone Evidence-to-Claim runners under `tools/` use pinned sessions
and immutable hashes. Keep reasoning disabled for local GLM/Qwen pilots when
the model is prone to looping, and accept only provenance-checked scoring.

Use extended commit messages with `[*]`, `[+]`, `[!]`, and `Tests:` sections.
