# Installation and setup

## Prerequisites

- Python 3.11 or newer;
- an OpenAI-compatible local inference endpoint;
- Node.js/npm only for npm-based MCP servers.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.cli config
```

Copy the MCP configuration template to `mcp_servers.json` and configure only
the servers required for the task. Do not commit secrets. `AGENTS.md` is the
workspace instruction entrypoint; linked instructions do not grant permissions.
