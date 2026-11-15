# Agent Loop

An autonomous agent loop framework with extensible agents, skills, and MCP (Model Context Protocol) tool integration. Run AI agents with custom personas, capabilities, and tool access for diverse tasks.

## Overview

**Agent Loop** is a Python-based agentic framework that:
- Executes autonomous agent loops through a compatible chat-completions API, primarily for locally hosted LLMs
- Supports pluggable **agents** (personalized AI roles) and **skills** (capability packages)
- Integrates **MCP tools** for external system access (APIs, files, commands)
- Provides CLI with subcommands for interactive (`repl`) and one-shot (`run`) execution
- Manages session history, token budgets, and response caching
- Loads project-specific rules from `AGENTS.md` guidelines

## Architecture

```
agent-loop/
├── src/
│   ├── main.py           # Entry point delegate
│   ├── cli.py            # CLI argument parsing & routing
│   ├── loop.py           # Main agent loop (tool calls, token tracking)
│   ├── config.py         # Global config (model, limits, allowed tools)
│   ├── agents_loader.py  # Load & manage agents
│   ├── skills_loader.py  # Load & manage skills
│   ├── commands_loader.py # Load & manage CLI commands
│   ├── session.py        # Session persistence
│   ├── tokenizer.py      # Token counting
│   ├── envelope.py       # API request/response wrapping
│   ├── render.py         # Format output
│   ├── dispatch.py       # Tool dispatch & execution
│   ├── cache.py          # Response caching
│   ├── compaction.py     # Token budget management
│   └── tools/
│       ├── shell.py      # Shell command execution
│       ├── files.py      # File operations
│       └── mcp_client.py # MCP tool integration
├── agents/               # Predefined agent personas (.md files)
├── skills/               # Capability packages (.md files)
├── commands/             # Custom CLI commands
├── requirements/         # Runtime and development dependencies
│   ├── base.txt          # Runtime dependencies
│   └── dev.txt           # Development and test dependencies
├── requirements.txt      # Backward-compatible runtime shim
├── tests/                # Test suite
├── mcp_servers.json      # MCP server configuration
└── AGENTS.md             # Project-specific rules
```

## LLM Backends and API Compatibility

Agent Loop separates the API protocol, the inference backend, and the language model:

- **API protocol:** The client sends OpenAI-compatible chat-completions requests to `LM_STUDIO_URL`. Requests include streaming, model selection, and tool definitions; streamed responses are expected to use the corresponding chat-completions response format, including `tool_calls` when tools are used. Although a local service such as LM Studio may also offer a Claude-compatible mode, a Claude-compatible endpoint is not automatically supported by this implementation unless it accepts this request and response schema.
- **Inference backend:** The service that runs the model. The primary development setup is a locally hosted LM Studio server. The repository also includes a configuration for a `llama-server` process from llama.cpp. A compatible API surface does not require Anthropic cloud services, and this client does not call Anthropic's API directly.
- **Language model:** The model named by `MODEL_NAME`. The default configuration uses `zai-org/glm-4.7-flash`, which is the model used during development. Other models may work when the selected backend supports the request format, context size, streaming behavior, and tool-calling format required by the agent loop.

API compatibility and model compatibility are separate concerns. A provider or local server that exposes a similarly named endpoint is not automatically supported: it must also return the response structure and tool-call behavior that this client parses.

### LM Studio

The default configuration connects to LM Studio's OpenAI-compatible chat-completions endpoint:

```python
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "zai-org/glm-4.7-flash"
```

Load a compatible model in LM Studio and make its model identifier available as `MODEL_NAME`. The current client does not configure an authentication token or send an API key; authentication settings offered by a backend are therefore not documented as supported configuration here. `LM_STUDIO_URL` is a historical name and is currently the generic endpoint used for every backend configuration.

### llama.cpp

The repository supports a direct llama.cpp setup when `llama-server` exposes the same OpenAI-compatible chat-completions endpoint expected by the client. The provided example starts the server with a Jinja chat template and points Agent Loop to it:

```bash
llama-server -m /path/to/GLM-4.7-Flash.gguf -c 131072 --jinja --port 8080
python3 agent_loop.py run "Fix the bug" --config agent_loop.llamacpp.yaml
```

`agent_loop.llamacpp.yaml` uses `http://localhost:8080/v1/chat/completions`. No proxy or translation layer is used by Agent Loop. Tool calls therefore depend on the selected llama.cpp server mode, model, and chat template producing the compatible streaming response format; other llama.cpp configurations are not automatically supported.

## Installation

### Prerequisites
- **Python 3.8+** — Main runtime
- **LM Studio or a compatible llama.cpp `llama-server`** — Choose an inference backend and load a compatible model
- **Node.js 14+** (with npm) — Needed only when using MCP servers distributed through npm, such as Context7 or context-mode

### System Requirements Check

```bash
# Verify Python
python3 --version  # Should be 3.8+

# Optional: verify Node.js & npm when using npm-based MCP servers
node --version     # Should be 14+
npm --version      # Should be 6+
```

### Setup

```bash
# Clone the repository
git clone https://github.com/avkghost/agent-loop.git
cd agent-loop

# Install runtime Python dependencies
pip install -r requirements/base.txt

# For development and tests, install the development requirements instead
pip install -r requirements/dev.txt

# Optional npm-based MCP server setup is described below

# Create your MCP config from the template and add your API key
cp mcp_servers.json.example mcp_servers.json

# (Optional) Create project rules
echo "# Your project guidelines" > AGENTS.md

# Verify setup
python3 -c "import httpx, mcp, tiktoken, yaml, rich; print('✅ Python packages OK')"
```

#### Optional: npm-based MCP servers

Use these commands only if the configured MCP servers use Node.js and `npx`:

```bash
# npx is included with npm 5.2+ (no additional setup needed)
# Configured MCP servers will auto-download on first use via npx

# Optional: pre-download MCP servers to avoid first-run delay
npx -y @upstash/context7-mcp --help >/dev/null
npx -y context-mode --help >/dev/null

# Optional: verify the MCP server packages
npx -y @upstash/context7-mcp --version 2>/dev/null && echo "✅ MCP servers OK"
```

### Dependencies

**Runtime Python Packages** (`requirements/base.txt`):
- `httpx` — asynchronous HTTP client for LLM API calls (streaming, timeouts, cancellation)
- `mcp` — Model Context Protocol library  
- `tiktoken` — Token counting for OpenAI models
- `pyyaml` — YAML parsing for agent/skill definitions
- `rich` — Terminal formatting

**Development Python Packages** (`requirements/dev.txt`):
- `pytest` — Test runner

The root `requirements.txt` remains as a compatibility shim that includes `requirements/base.txt`.

**Node.js Packages** (MCP servers, auto-installed):
- `@upstash/context7-mcp` — Context indexing & search
- `context-mode` — Context window optimization tools

## Usage

### Basic Commands

#### 1. **Run a single goal**
Execute an agent once for a specific objective:
```bash
python3 agent_loop.py run "Write a Python function to calculate Fibonacci numbers"
python3 agent_loop.py run "Debug the failing test in tests/"
python3 agent_loop.py run "Refactor the API endpoint for better performance"
```

Backward compatibility — no `run` keyword needed:
```bash
python3 agent_loop.py "Create a REST API using FastAPI"
```

#### 2. **Interactive REPL mode**
Start an interactive loop for multi-turn conversations:
```bash
python3 agent_loop.py repl
```

Then enter goals interactively:
```
🎯 Enter goal (or 'exit'/'quit' to exit): Analyze the codebase structure
🎯 Enter goal (or 'exit'/'quit' to exit): What are the entry points?
🎯 Enter goal (or 'exit'/'quit' to exit): exit
```

The REPL keeps **one session** for every goal you enter. Conversation history, run records, and session-local approvals carry from one goal to the next, and each goal is saved as its own run. Background jobs started by one goal stay running after that goal finishes, so a later goal can check, collect, or stop them. Ctrl-C cancels the goal that is running and records it as interrupted; the session stays open. Leaving the REPL closes the session, which stops its background jobs, closes its MCP connections, and then shuts the runtime down.

**Entering multi-line text.** A goal may span several lines. Paste it with bracketed paste and press Enter once to submit: a paste is taken whole, its newlines included, and it never submits by itself. Typed text is edited with Backspace; Enter submits; Ctrl-C cancels the prompt; Ctrl-D on an empty prompt ends the REPL.

*Requirements and limits.* Multi-line paste depends on the terminal: it must support bracketed paste mode (check your terminal's documentation; it is not universal). Terminals that do not support it deliver a paste as ordinary typing. Then each newline is an Enter: a paste that arrives in one read is still kept whole, but a paste delivered line by line submits at its first newline, and the rest is typed into the next prompt. Paste into such a terminal line by line is not recommended. Terminal.app and Midnight Commander have not been verified for this change.

**Permission prompts.** A permission prompt accepts only one deliberate key (`1`, `2`, or `3`). Text pasted or typed as a burst, and any key that is not one of those, is ignored and the prompt keeps waiting. Ctrl-C denies. On a terminal without bracketed paste, a single pasted `1`, `2`, or `3` delivered as one byte cannot be told apart from a keypress; paste into such a terminal while a permission prompt is open is not safe.

#### 3. **Run with a named agent**
Use a specific agent persona with predefined system prompt and tool access:
```bash
python3 agent_loop.py agent coder "Implement error handling in main.py"
python3 agent_loop.py agent researcher "Research best practices for Python async"
python3 agent_loop.py agent investigator "Debug the socket timeout issue"
```

Available agents: `coder`, `investigator`, `researcher`, `code-reviewer`, `local-assistant`

#### 4. **Run with a skill**
Use a skill (capability package) with restricted tool access:
```bash
python3 agent_loop.py skill data-analyst "Analyze the test results in test_report.json"
python3 agent_loop.py skill python-helper "Format and lint the Python files"
```

#### 5. **List available agents**
```bash
python3 agent_loop.py agents
# Output:
# coder (0.7°)           — Professional software engineer
# investigator          — Debugging and root-cause analysis
# researcher           — Research & knowledge synthesis
# code-reviewer        — Code review & quality assurance
# local-assistant      — Local file operations
```

#### 6. **List available skills**
```bash
python3 agent_loop.py skills
# Output:
# data-analyst         — Analyze data, extract insights
# python-helper        — Python code formatting & linting
# research-assistant   — Web research & information gathering
```

#### 7. **View configuration**
```bash
python3 agent_loop.py config
```

Show effective config with tool allowlists and limits.

#### 8. **List available tools**
```bash
python3 agent_loop.py tools --config my-config.yaml
```

#### 9. **Resume a saved session**
```bash
python3 agent_loop.py sessions          # List saved sessions
python3 agent_loop.py resume <session-id> # Resume a session
```
`resume` restores the saved conversation. A tool call that was in progress when the process stopped is answered as interrupted and is not run again. See "Tool Failures, Interruptions, and Limits" under Token Management.

#### 10. **Verify a task against trusted criteria**
```bash
python3 agent_loop.py run "Make value() return 2" --criteria criteria.json --verification-cycles 2
python3 agent_loop.py resume <session-id> --criteria criteria.json   # optional: checks the same contract
```
`criteria.json` is a JSON object whose only key is `criteria`: a non-empty list of checks in the existing criterion types. Each check has `criterion_id`, `criterion_type`, `expected` and `parameters`; `description` is optional.

- `file_exists`: `path`, optional `type` (`file` or `directory`); `expected` is `true` or `false`.
- `file_content`: `path`, `contains` (non-empty text); `expected` is `true`.
- `command_exit`: `command`, optional `timeout_seconds` (1 to 600); `expected` is the exit code.

Paths must stay inside the working directory. A command must be allowed without approval, and may not use shell operators. A document with any problem is refused before a session is created or the model is contacted; every problem is listed. `job_state` cannot be declared in a file.

The model cannot add or change these checks. When the model stops, the runtime runs the checks itself. Failed checks are returned to the model, up to `--verification-cycles` attempts. A run is reported as one of:

- `VERIFIED`: every criterion passed.
- `FAILED VERIFICATION`: a required criterion did not pass within the budget.
- `COMPLETED WITHOUT TRUSTED CRITERIA`: no criteria were given, so nothing was verified.
- `INTERRUPTED OR INCOMPLETE`, `AUTHORIZATION OR EXECUTION FAILURE`, or `NEEDS ATTENTION` (for example, an unresolved tool outcome, which withholds verified completion).

Each criterion's latest result and evidence are printed. On `resume`, the trusted criteria saved with the session are carried over to the new run. If `--criteria` is also given, it must describe the same checks; otherwise nothing runs. Criteria cannot be added to a session that started without them.

### Command-Line Options

| Option | Commands | Description |
|--------|----------|-------------|
| `--allow` | `run`, `repl`, `agent`, `skill` | Allow all tools (skip permission prompts) |
| `--config FILE` | `run`, `repl`, `agent`, `skill`, `tools`, `config` | Load custom config file |
| `--cache` | `run`, `repl`, `agent`, `skill` | Enable response caching |
| `--criteria PATH` | `run`, `resume` | Trusted acceptance criteria (see Verify a task) |
| `--verification-cycles N` | `run`, `resume` | Verification attempts, 1 to 10 (default 2) |

### Examples

#### Example 1: Code Generation
```bash
python3 agent_loop.py agent coder "Create a REST API with FastAPI that:
- Accepts POST /analyze with JSON payload
- Validates input with Pydantic
- Returns analysis results
- Includes proper error handling"
```

#### Example 2: Bug Investigation
```bash
python3 agent_loop.py agent investigator "The tests in tests/test_integration.py are failing. 
Investigate the root cause, show the error, and suggest a fix."
```

#### Example 3: Data Analysis
```bash
python3 agent_loop.py skill data-analyst "Analyze the CSV file data/metrics.csv and 
identify trends, outliers, and key statistics."
```

#### Example 4: Interactive Research
```bash
python3 agent_loop.py repl
# Then:
# 🎯 Enter goal: What are best practices for Python async?
# 🎯 Enter goal: Show me an example of asyncio.gather()
# 🎯 Enter goal: How do I test async functions?
# 🎯 Enter goal: exit
```

#### Example 5: Code Review
```bash
python3 agent_loop.py agent code-reviewer "Review the code in src/api.py. 
Check for: logic errors, edge cases, performance issues, and security vulnerabilities."
```

## Agents & Skills

### Agents (Personas)
Agents are AI personas with specific expertise and tool access. Each has a custom system prompt and allowed tools. The `temperature` and `max_tokens` settings in an agent file are parsed but not applied to requests: the primary loop uses the global generation settings described under Configuration.

**Available Agents:**

1. **coder** — Professional software engineer
   - Expertise: Implementation, SOLID principles, best practices
   - Tools: `run_command`, `read_file`, `write_file`, `list_directory`
   - Temp: 0.7

2. **investigator** — Root-cause analysis & debugging
   - Expertise: Tracing issues, reading logs, identifying patterns
   - Tools: `run_command`, `read_file`, `list_directory`, `grep`
   - Temp: 0.5

3. **researcher** — Knowledge synthesis & learning
   - Expertise: Gathering info, comparing approaches, explaining concepts
   - Tools: `run_command`, `read_file`, `search_docs`
   - Temp: 0.3

4. **code-reviewer** — Code quality & security
   - Expertise: Best practices, performance, security, maintainability
   - Tools: `read_file`, `list_directory`, `git`
   - Temp: 0.3

5. **local-assistant** — File & system operations
   - Expertise: File management, scripting, automation
   - Tools: `run_command`, `read_file`, `write_file`, `list_directory`
   - Temp: 0.5

### Skills (Capability Packages)
Skills provide restricted tool sets for specialized tasks.

**Available Skills:**

1. **data-analyst** — Data analysis & insights
   - Tools: `run_command`, `read_file`, `list_directory`

2. **python-helper** — Python code formatting
   - Tools: `run_command`, `read_file`, `write_file`
   - Scripts: `format_check.py` for linting

3. **research-assistant** — Information gathering
   - Tools: `search_docs`, `read_file`, `run_command`

## Configuration

### Config File (src/config.py)

Key settings:

```python
# LLM endpoint and model
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "zai-org/glm-4.7-flash"
LM_STUDIO_TIMEOUT = 60

# Loop Control
MAX_ITERATIONS = 100
MAX_TOKENS = 131072                    # Total conversation budget
COMPACT_THRESHOLD = 0.8                # Start compacting at 80%

# Model Context Window
MODEL_CONTEXT_WINDOW = 131072          # Model's actual context window
REQUEST_MAX_TOKENS = 8192              # Per-request output limit (YAML: REQUEST_MAX_TOKENS)
TEMPERATURE = None                     # Explicit sampling temperature (YAML: TEMPERATURE); None = LM Studio's setting

# Tool Access
ALLOWED_TOOLS = {"ls", "cat", "grep", "find", ...}
ALLOWED_MCP_TOOLS = set()              # Dynamically populated
ALLOW_ALL = False                      # CLI --allow flag
```

**Important:** Update `MODEL_CONTEXT_WINDOW` to match your loaded model's actual context size. See [Token Management](#token-management) for details.

The name `LM_STUDIO_URL` is retained for compatibility, but it is used as the generic API endpoint for the selected backend. The current implementation has no separate provider or authentication setting. YAML configuration files can override `LM_STUDIO_URL`, `MODEL_NAME`, `MAX_ITERATIONS`, `MAX_TOKENS`, `COMPACT_THRESHOLD`, `ALLOWED_TOOLS`, `REQUEST_MAX_TOKENS`, and `TEMPERATURE`. `MODEL_CONTEXT_WINDOW` is defined in `src/config.py` only.

**Generation parameters: who decides what.**
- *LM Studio's sampling settings* (for example temperature when `TEMPERATURE` is unset, `top_p`, `top_k`, penalties, seed, and reasoning budget) are the default. Agent Loop never sends those fields.
- *Explicit Agent Loop override.* `TEMPERATURE` (a number from 0 to 2, or `null` for LM Studio's setting) is sent only when it is set. Invalid values are rejected at startup, before any request is built.
- *Conversation token budget.* `MAX_TOKENS` limits the whole conversation, and it triggers compaction. It is never sent to the model.
- *Per-request output limit.* `REQUEST_MAX_TOKENS` (default 8192, a positive integer) is always sent as `max_tokens`. It cannot be disabled.
- *Compaction* uses its own fixed options (temperature 0.3, 500 output tokens), which are unaffected by these settings.
- *Agent files.* The `temperature` and `max_tokens` values in an agent file are parsed but not used by the primary loop; they have no effect on requests.

### MCP Servers (mcp_servers.json)

Define external tools and APIs:

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp", "--api-key", "..."]
    },
    "context-mode": {
      "command": "npx",
      "args": ["-y", "context-mode"]
    }
  }
}
```

### Project Rules (AGENTS.md)

Create `AGENTS.md` in the project root to inject custom rules into all agent system prompts:

```markdown
# Project Guidelines

## Code Style
- Use 2-space indents
- Follow PEP 8 for Python
- Use type hints on public APIs

## Tool Restrictions
- Never use `sudo` without explicit approval
- Always test changes locally first
- Document breaking changes
```

## Advanced Usage

### Custom Configuration File

```bash
python3 agent_loop.py run "Fix the bug" --config custom-config.yaml
```

### Using llama.cpp

`agent_loop.llamacpp.yaml` points the agent at a llama.cpp `llama-server` using the same model and a 131072-token context:

```bash
llama-server -m /path/to/GLM-4.7-Flash.gguf -c 131072 --jinja --port 8080
python3 agent_loop.py run "Fix the bug" --config agent_loop.llamacpp.yaml
```

This example requires a `llama-server` endpoint compatible with the client's OpenAI-style streaming chat-completions and tool-call format. The context size is set by the server's `-c` flag. `LM_STUDIO_URL`, `MODEL_NAME`, `MAX_TOKENS`, `MAX_ITERATIONS`, `COMPACT_THRESHOLD`, `ALLOWED_TOOLS`, `REQUEST_MAX_TOKENS`, and `TEMPERATURE` can be set from a YAML file. `MODEL_CONTEXT_WINDOW` must be changed in `src/config.py`.

### Response Caching

```bash
python3 agent_loop.py run "Analyze the logs" --cache
```

Caches responses to avoid re-calling the LLM for identical inputs.

### Bulk Tool Access

```bash
python3 agent_loop.py agent coder "Refactor the API" --allow
```

Allows all defined tools without prompting.

**Warning:** The `--allow` flag disables tool approval prompts. Use it only in trusted environments, especially when shell commands and filesystem write operations are available. Enabled MCP tools may also access external services or data.

## API Integration

### Direct Python Usage

```python
from src.loop import agent_loop_async
from src.tools.mcp_client import MCPClientManager
import asyncio

# Create MCP client
mcp_client = MCPClientManager()

# Run agent loop
asyncio.run(agent_loop_async(
    user_goal="Implement user authentication",
    mcp_client=mcp_client,
    system_prompt_override="You are a security expert...",
    tool_allowlist_override=["read_file", "write_file"]
))
```

## Token Management

### The Problem (Fixed)
The agent loop was terminating prematurely after 4-5 iterations with error: `finish_reason: length`

**Root Cause:** The API request incorrectly used the total conversation budget (131,072 tokens) as the per-request output limit. This set an excessively large output allocation relative to the model's context window and contributed to premature termination.

**The Fix:** 
- Added configurable `MODEL_CONTEXT_WINDOW` (your model's actual token limit)
- Added configurable `REQUEST_MAX_TOKENS` (reasonable per-request output limit, e.g., 8,192)
- Updated loop to use `config.REQUEST_MAX_TOKENS` instead of `config.MAX_TOKENS`

**Impact:** During development testing, the updated limits allowed loops to complete 20-50+ iterations instead of stopping after 4-5. Actual results depend on the model, context, prompts, and tool usage.

### Key Concepts
- **MODEL_CONTEXT_WINDOW:** Your model's actual token limit (e.g., 131,072 for GLM-4.7-flash)
- **REQUEST_MAX_TOKENS:** Per-request output limit (e.g., 8,192) — prevents model overload
- **MAX_TOKENS:** Total conversation budget (e.g., 131,072) — hard limit for entire session
- **COMPACT_THRESHOLD:** When to summarize messages (e.g., 0.8 = 80% of budget)

### How Token Management Works
1. **Each request:** Limited to `REQUEST_MAX_TOKENS` (8,192)
2. **After each response:** Tokens counted and tracked
3. **At 80% threshold:** Old messages summarized to preserve context
4. **At 100%:** Loop terminates gracefully

### Configuration for Different Models

| Model | Context | Request Max | Strategy |
|-------|---------|------------|----------|
| Small (4K) | 4,096 | 1,024 | Conservative; many iterations |
| Medium (8K) | 8,192 | 2,048 | Balanced; 15-20 observed in testing |
| Large (16K-32K) | 16,384+ | 4,096-8,192 | Generous; 20-30 observed in testing |
| XLarge (131K) | 131,072 | 8,192 | Full capacity; 50+ observed in testing |

Iteration counts are observations from development testing, not guarantees; actual behavior varies by model, prompts, and tool usage.

**Update for your model:**
```python
# In src/config.py
MODEL_CONTEXT_WINDOW = <your_model_context>
REQUEST_MAX_TOKENS = <safe_per_request_limit>
```

### How Token Accounting Works

**Token Counting Heuristic:**
```python
def get_token_count(text):
    return len(text) // 4  # Simple: 4 chars ≈ 1 token
```
This is approximate. For precise counting, consider using the `tiktoken` library.

**Token Consumption Flow:**
1. **Per request:** Limited to `REQUEST_MAX_TOKENS` (e.g., 8,192)
2. **After response:** Tokens counted and tracked
3. **At 80% threshold:** Old messages summarized via LLM
4. **Fallback:** Hard truncation if compaction fails

### Message Compaction Strategy

When messages exceed 80% of `MAX_TOKENS`:

**Step 1: LLM-based summarization (preferred)**
- Keeps: structured state containing the original objective (verbatim), the replaceable current summary, and the recent tail (last 4 messages, moved forward if needed so the tail never starts with a tool result)
- Summarizes: everything between the system prompt and that tail
- The provider-facing message is derived from that state; message text and summary labels are never parsed back as state metadata.
- Each successful compaction replaces the previous summary rather than appending another summary.
- Rejected, and falls back to Step 2, when the request times out or fails, the reply is malformed or empty, or the summary exceeds 20% of `MAX_TOKENS` (heuristic: 4 characters per token)

**Step 2: Hard truncation (fallback)**
- Keeps: system prompt, the original objective (verbatim), and the largest protocol-valid recent tail that fits the budget
- Discards: the replaceable summary and other older context, including whole tool-call groups that fall outside the tail
- Used if LLM summarization is rejected or fails
- If the system prompt, objective, and required protocol context cannot fit, the run stops with an explicit compaction error before another provider request.

Compaction runs at most once per loop iteration and never re-dispatches a tool.

Sessions saved by the current runtime use format 5, with `objective`, `summary`, `messages`, durable `execution` records, `runs`, `evidence`, and `legacy_history` persisted as separate fields. Format 4 sessions resume with their conversation and execution records, but without recorded run history. Format 3 sessions may resume without execution history. Clean objective-only format 2 or unversioned sessions may also resume. Legacy histories containing merged summary markers are rejected as ambiguous rather than reinterpreted. Any other format is rejected.

### Tool Failures, Interruptions, and Limits

- **Explicit tool execution failures** are recorded as a terminal result for that call, marked `unknown_outcome` and not retryable. The run continues with the next call. The tool is never retried automatically, because it may already have had an effect. Unexpected programming errors are not converted into tool results.
- **Durable tool-call state.** Each call is persisted as `pending` before dispatch, `running` before external invocation, and then as `completed`, `failed`, or `unknown`. These records are authoritative and survive compaction; they are not reconstructed from summaries. A persistence failure stops execution before another external invocation.
- **Interrupted runs.** Each assistant turn and each tool result is saved as it is recorded. If the process stops while a call is in progress, `resume` answers that call as interrupted. It does **not** re-run it. Whether its side effect happened is unknown and must be checked by hand.
- **Resume restores the conversation history and the durable call records.** It does not restore an agent's position inside a loop iteration, and it cannot tell whether an unknown external effect happened. This is not crash-safe recovery and is not exactly-once execution.
- **MCP calls** are awaited on the running agent loop. A validation error before submission (for example an invalid tool name) is a definitive failure. A transport failure, timeout, malformed response, or remote `isError` after submission is `unknown_outcome` and is never retried, because the remote side may already have acted.
- **File writes** encode the content first and write a temporary file in the same directory before replacing the target. A failure before the replacement leaves the existing file unchanged and is reported as a definitive failure. Only a replacement that reports an error after the temporary file was consumed is `unknown_outcome`. `read_file` accepts UTF-8 text only; binary content is refused as a validation error.
- **Path containment** is checked on resolved paths (symlinks followed) by path component. It is not an operating-system sandbox and does not defend against other processes changing the filesystem concurrently.
- **Session writes** are replaced atomically (temporary file, then rename), so an interrupted write keeps the previous session file. Durability on power loss is not guaranteed.
- **Model requests.** Requests go through an asynchronous provider (`src/provider.py`) on httpx with explicit connect, read, write, and pool timeouts. A turn is complete only when its stream carries a finish reason and the `[DONE]` terminator. An incomplete, malformed, length-limited, or failed turn is discarded: nothing from it is kept or dispatched, and the run reports the reason. Timeouts are retried with non-blocking backoff, but only before any tool call from that turn runs. Cancelling a run closes the HTTP response. Token usage is labelled `reported` when the server provides it, and `estimated` otherwise. Context summaries use the same provider, and a failed summary is reported before the bounded fallback runs.
- **Shell commands are foreground only.** Syntax is checked quote-aware before approval or execution. A background `&` is rejected when it is unquoted and unescaped and not part of `&&`, `&>`, or `2>&1`; so `sleep 100 &` and `sleep 1 & echo done` are rejected, while `echo "a & b"`, `echo a\&b`, `a && b`, and `cmd 2>&1` are allowed. Detaching commands (`nohup`, `setsid`, `disown`, `screen`, `tmux`, `at`, `batch`) are rejected. Unsupported grammar is rejected before any process starts: command substitution `$(...)` and backticks, subshells `( ... )`, and here-documents `<<`. Background servers and watchers are not run this way; use managed jobs (below) for them.
- **Shell limits.** Default timeout 120 s; `timeout_seconds` may be 1 to 600. Stdin is `/dev/null`, so a command cannot read the REPL's input. Each stream keeps at most 1 MiB in memory; the model and journal see only its first and last 16 KiB, with a truncation note. Exit codes are reported (`[exit code N]`) and are not treated as tool errors.
- **Timeouts and cancellation.** On timeout or cancellation the command's process group is stopped: SIGTERM, then SIGKILL after 5 s if the group is not empty. Cleanup runs to completion even if the command is cancelled again; the cancellation then propagates. A leader is reaped only after its group is confirmed empty. Group membership is read from the process table (`ps`), zombies are ignored, and a failed or unavailable membership check is never treated as an empty group. A timeout is `unknown_outcome` and is not retried. Descendants that stay in the group are stopped. Descendants that start their own session (`setsid`, daemons) leave the group and are not tracked.
- **Unconfirmed stops.** A leader whose group could not be confirmed empty is retained, still unreaped, so its process-group id stays reserved. Retained groups are retried on later commands and are SIGKILLed at normal exit. At most 8 may be retained: while the cap is reached, new commands are refused before anything starts, and no retained process is discarded. If another part of the process reaps a command's leader, the command is reported as `unknown_outcome`, its group is not signalled, and its exit status is not claimed. SIGKILL of the agent process, power loss, and a process-group id reused in the narrow window between a leader's exit and its reap are not covered.
- **Platforms.** POSIX systems only (developed and tested on macOS and Linux). Shell execution requires `ps` (procps) to verify process groups; it is refused before anything starts if `ps` is missing. Shell execution is refused on other systems rather than run with weaker guarantees.
- **Approval and allowlist.** Approval controls which commands the agent may start. It is not an operating-system sandbox. Each grant has one scope:
  - `ALLOWED_TOOLS` in configuration names base commands (for example `grep`). Any invocation of a listed command runs without a prompt. This is deliberately broad, and you narrow it by editing the list.
  - "Yes, only" applies to one invocation. It is consumed by that invocation, and it is discarded if the attempt fails validation, raises an error, or is cancelled.
  - "Yes, this session" for an unlisted command approves its exact command text, after surrounding whitespace is trimmed, for the rest of that session. `uname -s` does not approve `uname -m`, and `curl -I https://a` does not approve `curl https://b`.
  - A command that uses shell operators (pipes, redirects, `&&`, `||`) is approved by its exact text only.
  - MCP tools are approved by tool name, for the session or once. Arguments are not part of the match.
  - Approvals belong to the session in which they were given and never reach another session. They are not saved: a resumed session, or a new process, starts from the configuration.
  - `--allow` approves every action. Use it only in trusted environments.
  - The built-in file tools (`read_file`, `list_directory`, `write_file`) are not approval-gated. They are limited to the working directory (see path containment above).
  - `kill`, `jobs`, `bg`, and `fg` are not automatically allowed; using them requires approval.
- **Search state.** Whether a session has run a codebase search (`grep`, `rg`, `find`) is kept for that session only. It starts empty for each session, including a resumed one, and it is not saved. It controls only the warning that `write_file` shows when it creates a new file before any search. A refused search request does not count.

**Example (illustrative, not a measured run):**
```
Before compaction: 45 messages (116,736 tokens / 131,072 = 89%)
After compaction:  6 messages (24,576 tokens) + summary
→ Loop continues for 20+ more iterations
```

### Run Outcomes and Evidence

Every goal ends with a `RunOutcome`, and the CLI prints it as `🧾 Run <id>: <status> — <reason>`. The outcome is also returned to programmatic callers. Statuses mean exactly what the runtime can establish:

| Status | When |
|---|---|
| `completed_unverified` | The model stopped normally. Nothing was verified. |
| `needs_attention` | Durable state could not be saved, so a person must check. |
| `blocked` | The user refused a requested action. The run stops rather than route around that decision. |
| `unknown` | An external operation's outcome is unknown, or the run hit an unexpected error. A later model stop does not clear an unknown external effect. |
| `interrupted` | Cancelled or interrupted, or a run whose process ended before it recorded an outcome. |
| `provider_failed` | The model request timed out, failed, was incomplete or malformed, returned nothing, or sent an unexpected finish reason. |
| `limit_reached` | The iteration budget, the context budget, or the response length limit was reached. |
| `completed_verified` | Reserved for a future verification gate. The runtime never emits it. |

Recoverable tool errors do not end a run. The model sees the error and can continue.

**Evidence** records are runtime observations taken from the execution journal. Each names the journal call it came from, and each carries a short summary, never the tool's output. A completed call is a success observation, and an unknown call is an unknown observation, never a success. A model's own statements are not evidence, so a run is never verified because its output says so.

**Acceptance criteria.** A program that calls the runtime API (`AgentSession.run_goal(goal, criteria=...)` or `agent_loop_async(..., criteria=...)`) may attach bounded criteria to a goal. The CLI does not supply criteria. Four criterion types are supported: a contained file exists, a bounded file-content check, a command with an expected exit code, and an owned managed job in an expected state. Criteria are bound to the run that receives them, and they cannot be changed or moved to another run.

Only trusted criteria (origin `user` or `trusted`) are verified. A criterion proposed by the model (origin `model`) is never trusted: it cannot satisfy a missing requirement, and it cannot make a run verified.

When the model stops without requesting tools, the runtime checks the trusted criteria itself. A passing check set produces `completed_verified`. A failing or inconclusive set returns the failed criteria to the model. This repeats for at most `verification_cycles` attempts (default 2); then the run ends as `needs_attention`. A run with no trusted criteria ends as `completed_unverified`.

Verification commands run only through the normal authorization policy. Verification never prompts: a command that needs permission is not run, and its journal record is `failed`. A timeout, a signal, an unconfirmed exit status, a dispatch exception, or an unknown outcome is inconclusive and never counts as a pass. If this run has an unknown tool outcome, verified completion is withheld. Earlier journal records are not changed by verification.

**Session format 5** stores runs and evidence next to the conversation and the journal. Sessions in formats 2 to 4 still load, and their earlier history is recorded as unrecorded (`legacy_history`). Nothing is reconstructed for them. A run that was still open when the process ended is recovered as `interrupted`, and it is never re-executed.

Sessions that contain acceptance criteria are saved as format 6, which adds the run-owned `criteria` and `verification_results` fields. Each verification observation records its attempt number and the cycle budget it ran under. Attempt numbers are per run and shared by all of that run's criteria, start at 1, have no gaps, and never exceed the budget (1 to 10, default 2). Every persisted cycle covers every trusted criterion, and earlier attempts are kept. A run is `completed_verified` only if the latest attempt passed every trusted criterion. Format 5 and earlier sessions still load, and format 5 cannot carry `completed_verified`. A failed save of a verified result is reported as `needs_attention`, never as verified success.

### Managed Background Jobs

For a long-running local process (a server, a watcher), the agent starts a **managed job** instead of a foreground command:

| Tool | Purpose |
|---|---|
| `start_job(command, lifetime_seconds?)` | Start the command and return an opaque `job_id`. Same command rules and approvals as `run_command`. |
| `status_job(job_id)` | State, reason, exit status, group state, remaining lifetime. |
| `collect_job(job_id, cursor?)` | Output since the cursor, 16 KiB per stream per call. Pass back the returned cursor. |
| `terminate_job(job_id)` | SIGTERM to the process group, SIGKILL after a bounded grace period. |

**Lifecycle.** `starting` → `running` → one of:
- `completed`: the leader exited with status 0.
- `failed`: nonzero exit, or killed by a signal nobody requested.
- `terminated`: a stop was requested (by the model, the lifetime limit, the end of the owning session, or runtime shutdown) and the process group was confirmed empty.
- `unknown`: a stop could not be confirmed, or another part of the process reaped the leader. Processes may still be running, and the job is never reported as completed or failed. It moves to `terminated` only when a later check confirms the group is empty.

**Ownership and lifetime.**
- A job belongs to the agent session that started it. Other sessions get the same "not found" answer for any id they do not own.
- A job survives the end of each goal in its session, so a later goal in the same session can call `status_job`, `collect_job`, or `terminate_job` on it. Closing the session (the REPL on exit, or the end of a one-shot run) terminates the session's jobs within a bounded time. A stop that cannot be confirmed is reported, not dropped. Every job is terminated at runtime shutdown.
- Jobs are not saved in the session file. A resumed session cannot see or restart an earlier job, and nothing is restarted automatically.
- Each job has a lifetime watchdog (default 600 s, at most 3600 s). It runs on its own worker thread, so it fires even while a model request is blocking the agent.

**Cursors.** A cursor is opaque and authenticated. It is valid only for the job that issued it and only for the current runtime. A forged, altered, or foreign cursor gets one generic answer that reveals nothing about the other job. A cursor older than the buffer is accepted, and the dropped bytes are reported.

**Shutdown and failures.** Runtime shutdown is refused while any session is running a goal. The refusal names the busy sessions, nothing is closed, and the runtime stays usable: finish or cancel the goal, then shut down again. Once a shutdown is accepted, no session can start a goal. If a session's jobs or MCP connections cannot be confirmed closed, the runtime stays pending, keeps its job worker and event loop, and a later shutdown retries only what is still unconfirmed. A completed shutdown is idempotent, and a closed runtime is never reopened implicitly. Failures in a job's helper tasks are recorded in diagnostics and never left unretrieved. If output capture, lifetime enforcement, or leader supervision fails, the job is stopped rather than left unmanaged.

**Output.** Each stream keeps at most 1 MiB in memory. Older bytes are dropped first, and `collect_job` reports how many bytes were dropped. Output is drained continuously, so a chatty job cannot block on a full pipe.

**Limits** (in `src/config.py`): `JOB_MAX_CONCURRENT` (4), `JOB_DEFAULT_LIFETIME_SECONDS` (600), `JOB_MAX_LIFETIME_SECONDS` (3600), `JOB_OUTPUT_BUFFER_BYTES` (1 MiB), `JOB_RETENTION_SECONDS` (600), `JOB_RETENTION_MAX` (32), `JOB_MAX_RETAINED` (8), `JOB_GRACE_SECONDS` (5), `JOB_KILL_CONFIRM_SECONDS` (5), `JOB_SHUTDOWN_TIMEOUT_SECONDS` (30).

**Runtime shutdown and crashes.** On a normal exit the runtime terminates every running job, escalates where needed, reaps, and closes its pipes. Unconfirmed stops are reported at exit. This is done explicitly by the CLI and does not depend on `atexit` alone. If the process is killed (SIGKILL, power loss, a crash), no cleanup runs. Jobs run in their own sessions, so they can keep running after the agent is gone. Nothing in Agent Loop reaps them afterwards.

**Platform.** POSIX only, and `ps` (procps) must be installed, as for foreground commands.

### Troubleshooting Token Issues

**Symptom:** Loop stops after 4-5 iterations  
**Solution:** 
- Check `MODEL_CONTEXT_WINDOW` matches your loaded model
- Verify `REQUEST_MAX_TOKENS` is reasonable (not > 1/3 of context window)
- See [Version History](#version-history) for known fixes

**Symptom:** Responses are too short  
**Solution:** Increase `REQUEST_MAX_TOKENS` (up to 1/3 of context window)

**Symptom:** Out of memory errors  
**Solution:** First determine whether the issue is context-budget pressure or actual RAM/VRAM exhaustion. For context-budget pressure, decrease `REQUEST_MAX_TOKENS` or lower `COMPACT_THRESHOLD` to trigger compaction earlier. If RAM or VRAM is exhausted while loading or running the model, changing `REQUEST_MAX_TOKENS` alone may not help; use a smaller model or a backend configuration that fits available hardware.

**Symptom:** Compaction never triggers  
**Solution:** Check `COMPACT_THRESHOLD` (default 0.8 = 80%); lower it to trigger sooner

## Testing

Install development dependencies before running the test suite:

```bash
pip install -r requirements/dev.txt
```

```bash
# Run test suite
python3 -m pytest tests/

# Run specific test
python3 -m pytest tests/test_integration.py::test_agent_loop

# With verbose output
python3 -m pytest tests/ -v
```

## Version History & Current Status

### Current Configuration (2026-10-19)
**Model Loaded:** GLM-4.7-flash with 131,072 token context window

| Setting | Value | Purpose |
|---------|-------|---------|
| `MODEL_CONTEXT_WINDOW` | 131,072 | Model's actual context capacity |
| `REQUEST_MAX_TOKENS` | 8,192 | Per-request output limit |
| `MAX_TOKENS` | 131,072 | Total conversation budget |
| `MAX_ITERATIONS` | 100 | Maximum loop iterations |
| `COMPACT_THRESHOLD` | 0.8 | Trigger compaction at 80% (104,858 tokens) |

**Expected Behavior:**
- Development testing observed 50+ iterations before token exhaustion
- Compaction activates automatically at 80% threshold
- Agent can handle complex multi-step tasks

### Version History

| Date | Issue | Fix | Status |
|------|-------|-----|--------|
| 2026-10-19 | Loop ends prematurely after 4-5 iterations with `finish_reason: length` | Added configurable `REQUEST_MAX_TOKENS`; now properly respects model's context window | ✅ **Fixed** |

**Previous:** API was requesting `max_tokens = 131,072` per response → token budget exhausted in 4-5 iterations  
**Current:** API requests `max_tokens = 8,192` per response → testing observed room for 50+ iterations

---

## Troubleshooting

### Issue: "Connection refused" for the inference backend
**Solution:** Ensure the configured LM Studio or llama.cpp server is running, then update `LM_STUDIO_URL` in `src/config.py` or the selected YAML configuration.

### Issue: Tool permission denied
**Solution:** Prefer adding only the required tools to `ALLOWED_TOOLS` in `src/config.py`. As an alternative, `--allow` bypasses tool approval prompts; see [Bulk Tool Access](#bulk-tool-access) and use it only in trusted environments.

### Issue: Slow responses / timeouts
**Solution:** Check the configured inference backend's performance and increase `LM_STUDIO_TIMEOUT` if appropriate. If individual generations are too long, reduce `REQUEST_MAX_TOKENS`; `MAX_TOKENS` is the total conversation budget, not the per-response output limit.

### Issue: MCP tools not found
**Solution:** Verify `mcp_servers.json` is properly configured and MCP servers are running.

## Development

### Project Structure

- **Agents:** Create `.md` file in `agents/` with YAML frontmatter + system prompt
- **Skills:** Create `.md` file in `skills/` with allowed tools
- **Commands:** Create `.py` file in `commands/` inheriting from base command class
- **Tools:** Add Python modules in `src/tools/` exposing tool functions

### Adding a New Agent

Create `agents/my-agent.md`:

```markdown
---
temperature: 0.5
max_tokens: 2048
tools:
  - read_file
  - write_file
---

You are a specialized assistant for [task description].

## Guidelines
- [Guideline 1]
- [Guideline 2]
```

### Adding a New Tool

Create `src/tools/my_tool.py`:

```python
def tool_function(param: str) -> str:
    """Tool description for LLM."""
    return result

# Export as OpenAI-compatible function schema
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tool_function",
            "description": "...",
            "parameters": {...}
        }
    }
]
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

1. Create a feature branch
2. Add tests for new functionality
3. Ensure tests pass: `pytest tests/`
4. Submit a pull request

---

For detailed agent/skill definitions, see:
- Agent examples: `agents/*.md`
- Skill examples: `skills/*.md`
- Configuration: `src/config.py`
- Project rules: `AGENTS.md`
