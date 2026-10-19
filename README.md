# Agent Loop

An autonomous agent loop framework with extensible agents, skills, and MCP (Model Context Protocol) tool integration. Run AI agents with custom personas, capabilities, and tool access for diverse tasks.

## Overview

**Agent Loop** is a Python-based agentic framework that:
- Executes autonomous agent loops with Claude API
- Supports pluggable **agents** (personalized AI roles) and **skills** (capability packages)
- Integrates **MCP tools** for external system access (APIs, files, commands)
- Provides CLI with subcommands for interactive (`repl`) and one-shot (`run`) execution
- Manages session history, token budgets, and response caching
- Loads project-specific rules from `AGENTS.md` guidelines

## Architecture

```
agent_loop/
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
├── tests/                # Test suite
├── mcp_servers.json      # MCP server configuration
└── AGENTS.md             # Project-specific rules
```

## Installation

### Prerequisites
- **Python 3.8+** — Main runtime
- **Node.js 14+** (with npm) — Required for MCP servers (context7, context-mode)
- **LM Studio instance** with a model loaded (e.g., GLM-4.7-flash with 131K context)

### System Requirements Check

```bash
# Verify Python
python3 --version  # Should be 3.8+

# Verify Node.js & npm
node --version     # Should be 14+
npm --version      # Should be 6+
```

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd agent_loop

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies (for MCP servers)
# npx is included with npm 5.2+ (no additional setup needed)
# The MCP servers will auto-download on first use via npx

# (Optional) Pre-download MCP servers to avoid first-run delay
npx -y @upstash/context7-mcp --help >/dev/null
npx -y context-mode --help >/dev/null

# Create your MCP config from the template and add your API key
cp mcp_servers.json.example mcp_servers.json

# (Optional) Create project rules
echo "# Your project guidelines" > AGENTS.md

# Verify setup
python3 -c "import requests, mcp, tiktoken, yaml, rich; print('✅ Python packages OK')"
npx -y @upstash/context7-mcp --version 2>/dev/null && echo "✅ MCP servers OK"
```

### Dependencies

**Python Packages** (`requirements.txt`):
- `requests` — HTTP client for LLM API calls
- `mcp` — Model Context Protocol library  
- `tiktoken` — Token counting for OpenAI models
- `pyyaml` — YAML parsing for agent/skill definitions
- `rich` — Terminal formatting

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

### Command-Line Options

| Option | Commands | Description |
|--------|----------|-------------|
| `--allow` | `run`, `repl`, `agent`, `skill` | Allow all tools (skip permission prompts) |
| `--config FILE` | `run`, `repl`, `agent`, `skill`, `tools`, `config` | Load custom config file |
| `--cache` | `run`, `repl`, `agent`, `skill` | Enable response caching |

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
Agents are AI personas with specific expertise and tool access. Each has a custom system prompt, temperature, and allowed tools.

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
# LLM Model
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "zai-org/glm-4.7-flash"
LM_STUDIO_TIMEOUT = 60

# Loop Control
MAX_ITERATIONS = 100
MAX_TOKENS = 131072                    # Total conversation budget
COMPACT_THRESHOLD = 0.8                # Start compacting at 80%

# Model Context Window
MODEL_CONTEXT_WINDOW = 131072          # Model's actual context window
REQUEST_MAX_TOKENS = 8192              # Per-request output limit

# Tool Access
ALLOWED_TOOLS = {"ls", "cat", "grep", "find", ...}
ALLOWED_MCP_TOOLS = set()              # Dynamically populated
ALLOW_ALL = False                      # CLI --allow flag
```

**Important:** Update `MODEL_CONTEXT_WINDOW` to match your loaded model's actual context size. See [Token Management](#token-management) for details.

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

**Root Cause:** The API request was setting `max_tokens` to the total conversation budget (131,072) instead of a per-request limit. Each response tried to output ~128KB of text, exhausting the context in just 4-5 iterations.

**The Fix:** 
- Added configurable `MODEL_CONTEXT_WINDOW` (your model's actual token limit)
- Added configurable `REQUEST_MAX_TOKENS` (reasonable per-request output limit, e.g., 8,192)
- Updated loop to use `config.REQUEST_MAX_TOKENS` instead of `config.MAX_TOKENS`

**Impact:** Loops now run 20-50+ iterations instead of 4-5, allowing complex multi-step tasks to complete.

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
| Medium (8K) | 8,192 | 2,048 | Balanced; 15-20 iterations |
| Large (16K-32K) | 16,384+ | 4,096-8,192 | Generous; 20-30 iterations |
| XLarge (131K) | 131,072 | 8,192 | Full capacity; 50+ iterations |

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
- Keeps: system prompt + last 4 messages
- Summarizes: everything in between
- Result: Preserves key context while reducing token count

**Step 2: Hard truncation (fallback)**
- Keeps: system prompt + last 4 messages
- Discards: everything else
- Used if LLM summarization fails

**Example:**
```
Before compaction: 45 messages (116,736 tokens / 131,072 = 89%)
After compaction:  6 messages (24,576 tokens) + summary
→ Loop continues for 20+ more iterations
```

### Troubleshooting Token Issues

**Symptom:** Loop stops after 4-5 iterations  
**Solution:** 
- Check `MODEL_CONTEXT_WINDOW` matches your loaded model
- Verify `REQUEST_MAX_TOKENS` is reasonable (not > 1/3 of context window)
- See [Version History](#version-history) for known fixes

**Symptom:** Responses are too short  
**Solution:** Increase `REQUEST_MAX_TOKENS` (up to 1/3 of context window)

**Symptom:** Out of memory errors  
**Solution:** Decrease `REQUEST_MAX_TOKENS` or lower `COMPACT_THRESHOLD` to trigger compaction earlier

**Symptom:** Compaction never triggers  
**Solution:** Check `COMPACT_THRESHOLD` (default 0.8 = 80%); lower it to trigger sooner

## Testing

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
- Loops run 50+ iterations before token exhaustion
- Compaction activates automatically at 80% threshold
- Agent can handle complex multi-step tasks

### Version History

| Date | Issue | Fix | Status |
|------|-------|-----|--------|
| 2026-10-19 | Loop ends prematurely after 4-5 iterations with `finish_reason: length` | Added configurable `REQUEST_MAX_TOKENS`; now properly respects model's context window | ✅ **Fixed** |

**Previous:** API was requesting `max_tokens = 131,072` per response → token budget exhausted in 4-5 iterations  
**Current:** API requests `max_tokens = 8,192` per response → room for 50+ iterations

---

## Troubleshooting

### Issue: "Connection refused" for LM Studio
**Solution:** Ensure LM Studio is running on `http://localhost:1234` or update `LM_STUDIO_URL` in `src/config.py`.

### Issue: Tool permission denied
**Solution:** Use `--allow` flag to bypass permission prompts or add tools to `ALLOWED_TOOLS` in `src/config.py`.

### Issue: Slow responses / timeouts
**Solution:** Increase `LM_STUDIO_TIMEOUT` in `src/config.py` or reduce `MAX_TOKENS`.

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

MIT License (modify as needed)

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
