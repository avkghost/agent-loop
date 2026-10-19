# Agent Loop — Quick Reference

## CLI Commands at a Glance

```bash
# Simplest: run a goal (backward-compatible syntax)
python3 agent_loop.py "Your goal here"

# Explicit run
python3 agent_loop.py run "Your goal"

# Interactive mode
python3 agent_loop.py repl

# With specific agent
python3 agent_loop.py agent <name> "goal"

# With skill
python3 agent_loop.py skill <name> "goal"

# List agents
python3 agent_loop.py agents

# List skills
python3 agent_loop.py skills

# Session management
python3 agent_loop.py sessions              # List
python3 agent_loop.py resume <session-id>   # Resume

# Config & tools
python3 agent_loop.py config                # Show configuration
python3 agent_loop.py tools                 # List available tools
```

## Command Options

| Option | Use | Example |
|--------|-----|---------|
| `--allow` | Skip tool permission prompts | `agent coder "..." --allow` |
| `--config FILE` | Use custom config | `run "..." --config custom.yaml` |
| `--cache` | Enable response caching | `run "..." --cache` |

## Agents Quick Lookup

| Agent | Best For | Temperature | Tools |
|-------|----------|-------------|-------|
| `coder` | Implementation, refactoring | 0.7 | run_command, read_file, write_file, list_directory |
| `investigator` | Debugging, root cause analysis | 0.5 | run_command, read_file, list_directory, grep |
| `researcher` | Explaining, learning, research | 0.3 | run_command, read_file, search_docs |
| `code-reviewer` | Code review, security, performance | 0.3 | read_file, list_directory, git |
| `local-assistant` | File operations, automation | 0.5 | run_command, read_file, write_file, list_directory |

**Choose by task:**
- **"Implement this feature"** → `agent coder`
- **"Why is this failing?"** → `agent investigator`
- **"How do I...?"** → `agent researcher`
- **"Review this code"** → `agent code-reviewer`
- **"Organize files"** → `agent local-assistant`

## Skills Quick Lookup

| Skill | Best For | Allowed Tools |
|-------|----------|---------------|
| `data-analyst` | CSV, JSON, logs, metrics | run_command, read_file, list_directory |
| `python-helper` | Format, lint Python code | run_command, read_file, write_file |
| `research-assistant` | Information gathering | search_docs, read_file, run_command |

## Code Examples

### One-liner: Simple task
```bash
python3 agent_loop.py "Create a Python function to validate email addresses"
```

### Multi-step: Interactive session
```bash
python3 agent_loop.py repl
# Enter multiple goals, each building on the previous
```

### Targeted: Use agent expertise
```bash
python3 agent_loop.py agent coder "Implement user authentication with JWT"
python3 agent_loop.py agent investigator "Why are tests failing?"
python3 agent_loop.py agent code-reviewer "Security audit of auth.py"
```

### Restricted: Use skill
```bash
python3 agent_loop.py skill data-analyst "Analyze metrics.csv for trends"
python3 agent_loop.py skill python-helper "Format all Python files"
```

### With options
```bash
python3 agent_loop.py run "Fix bug" --allow --cache
python3 agent_loop.py agent coder "Refactor" --config prod-config.yaml
```

## Project Configuration

### Location: `src/config.py`

**Key settings:**
```python
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"  # LLM endpoint
MODEL_NAME = "zai-org/glm-4.7-flash"                          # Model to use
MAX_ITERATIONS = 100                                          # Max loop iterations
MAX_TOKENS = 131072                                           # Total conversation budget

# Token Management (CRITICAL)
MODEL_CONTEXT_WINDOW = 131072                                 # Model's actual context
REQUEST_MAX_TOKENS = 8192                                     # Per-request limit

ALLOWED_TOOLS = {"ls", "cat", "grep", ...}                   # Allowed shell commands
```

**Token Configuration Examples:**
- **8K model:** `MODEL_CONTEXT_WINDOW = 8192`, `REQUEST_MAX_TOKENS = 2048`
- **16K model:** `MODEL_CONTEXT_WINDOW = 16384`, `REQUEST_MAX_TOKENS = 4096`
- **131K model:** `MODEL_CONTEXT_WINDOW = 131072`, `REQUEST_MAX_TOKENS = 8192`

**Customize by:**
1. Edit `src/config.py` directly (especially token settings!)
2. Pass `--config custom.yaml` on CLI
3. Use `--allow` to bypass restrictions

### MCP Tools: `mcp_servers.json`

Define external tool providers:
```json
{
  "mcpServers": {
    "server-name": {
      "command": "...",
      "args": [...]
    }
  }
}
```

### Project Rules: `AGENTS.md`

Create in project root to inject custom system prompts:
```markdown
# Project-Specific Guidelines

## Code Style
- [Your rules]

## Tool Restrictions
- [Your restrictions]
```

## Architecture Map

```
agent_loop (entry)
  ↓
cli.py (parse args, route commands)
  ↓
loop.py (main agent loop)
  ├─ agents_loader.py (load .md agent definitions)
  ├─ skills_loader.py (load skill definitions)
  ├─ tools/ (shell, files, mcp_client)
  └─ envelope.py (API calls)

Tools flow:
agents/coder.md (definition)
  → agents_loader.py (parse YAML, load prompt)
  → loop.py (use system prompt + tools)
  → dispatch.py (execute tool calls)
  → tools/* (actual implementation)
```

## Environment Setup

```bash
# Prerequisites
python3 --version  # 3.8+
node --version     # 14+ (required for MCP servers)
npm --version      # 6+

# Install Python packages
pip install -r requirements.txt

# Pre-download MCP servers (optional, improves first-run performance)
npx -y @upstash/context7-mcp --help >/dev/null
npx -y context-mode --help >/dev/null

# Configure
cat mcp_servers.json  # Check MCP setup
cat src/config.py     # Review settings

# Verify
python3 agent_loop.py agents  # Should list agents
python3 agent_loop.py skills  # Should list skills
python3 agent_loop.py tools   # Should list available tools
```

## Debugging

### Enable verbose logging
Edit `src/config.py` and increase logging level, or use:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check what tools are available
```bash
python3 agent_loop.py tools
python3 agent_loop.py tools --config custom.yaml
```

### Trace a failing task
```bash
# Run with investigate agent (better for debugging)
python3 agent_loop.py agent investigator "Debug: [your error message]"

# Or in REPL for iterative debugging
python3 agent_loop.py repl
🎯 Show the error in [file]
🎯 What's causing line 42?
🎯 How do I fix it?
```

## Common Patterns

### Pattern 1: Code → Review → Fix
```bash
python3 agent_loop.py agent coder "Implement feature X"
python3 agent_loop.py agent code-reviewer "Review src/feature_x.py"
python3 agent_loop.py agent coder "Fix the review issues in src/feature_x.py"
```

### Pattern 2: Analyze → Explore → Learn
```bash
python3 agent_loop.py skill data-analyst "Analyze results.json"
python3 agent_loop.py repl
🎯 What does the data show?
🎯 How should I interpret these trends?
🎯 What's the root cause?
```

### Pattern 3: Debug → Fix → Verify
```bash
python3 agent_loop.py agent investigator "Debug the failing test"
python3 agent_loop.py agent coder "Implement the fix"
python3 agent_loop.py agent investigator "Verify the test passes now"
```

### Pattern 4: Interactive learning
```bash
python3 agent_loop.py repl
🎯 [Ask question 1]
🎯 [Follow-up question]
🎯 [Deep dive question]
🎯 [Example request]
🎯 exit
```

## Performance Tips

| Issue | Solution |
|-------|----------|
| Slow responses | Increase `LM_STUDIO_TIMEOUT` in config |
| Token limit exceeded | Reduce `MAX_TOKENS` or use session compaction |
| Tool permission prompts | Use `--allow` flag |
| MCP tools slow | Check MCP server health |
| Out of memory | Reduce `MAX_ITERATIONS` |

## File Structure for Reference

```
agent-loop/
├── README.md              # Full documentation ← START HERE
├── EXAMPLES.md            # Detailed usage scenarios
├── REFERENCE.md           # This file (quick reference)
├── agent_loop.py          # Entry point (delegates to src/)
├── requirements.txt       # Dependencies
├── mcp_servers.json       # MCP configuration
├── AGENTS.md             # Project rules (create if needed)
├── src/
│   ├── main.py           # Entry point
│   ├── cli.py            # CLI router
│   ├── loop.py           # Agent loop logic
│   ├── config.py         # Configuration
│   ├── agents_loader.py  # Agent loading
│   ├── skills_loader.py  # Skill loading
│   ├── tools/            # Tool implementations
│   └── [other modules]
├── agents/               # Agent definitions (.md files)
├── skills/               # Skill definitions (.md files)
├── commands/             # Custom CLI commands
└── tests/                # Test suite
```

## Common Questions

**Q: How do I add a new agent?**  
A: Create `agents/my-agent.md` with YAML frontmatter and system prompt.

**Q: How do I restrict tools for a task?**  
A: Use a skill with limited `allowed-tools`, or edit `ALLOWED_TOOLS` in config.

**Q: Can I run multiple agents in parallel?**  
A: Not directly, but use interactive mode (`repl`) to coordinate manually.

**Q: How do I cache responses?**  
A: Use `--cache` flag: `agent coder "..." --cache`

**Q: Can I integrate with my own tools?**  
A: Yes, via MCP in `mcp_servers.json` or custom tools in `src/tools/`.

**Q: How do I resume work?**  
A: Use `python3 agent_loop.py sessions` to list, then `resume <id>`.

## See Also

- **Full Docs:** `README.md`
- **Examples:** `EXAMPLES.md`
- **Agent Definitions:** `agents/*.md`
- **Skill Definitions:** `skills/*.md`
- **Configuration:** `src/config.py`
