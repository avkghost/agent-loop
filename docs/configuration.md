# Configuration and MCP

Configuration may come from `src/config.py`, `agent_loop.yaml`, or
`~/.agent_loop/config.yaml`; use `--config PATH` to select a file directly.
Important settings include `LM_STUDIO_URL`, `MODEL_NAME`, `REQUEST_MAX_TOKENS`,
`MAX_TOKENS`, `COMPACT_THRESHOLD`, `TEMPERATURE`, `MCP_TOOL_TIMEOUT`, and
`ALLOWED_TOOLS`. Set `TEMPERATURE` to `null` to use the server profile.

The default endpoint is LM Studio at `http://localhost:1234/v1`. Reasoning and
sampling behavior may be managed by the server when temperature is unset.

MCP tools with invalid schemas are not offered to the model. Unadvertised tools
are rejected before remote submission. After submission, timeout, disconnect,
malformed result, and remote error are conservative outcomes and are not
automatically retried.
