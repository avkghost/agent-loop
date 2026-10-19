# Agent Loop — Usage Examples & Recipes

Quick reference with real-world scenarios and copy-paste examples.

## Prerequisites

Before running any examples, ensure you have:
- Python 3.8+ installed
- Node.js 14+ and npm installed (for MCP servers)
- Python packages: `pip install -r requirements.txt`
- LM Studio running with a model loaded

## Quick Start

```bash
# Single goal (simplest)
python3 agent_loop.py "Implement a login form in React"

# Interactive loop (multi-turn)
python3 agent_loop.py repl

# With a specific agent
python3 agent_loop.py agent coder "Fix the broken API endpoint"

# With a skill (restricted tools)
python3 agent_loop.py skill data-analyst "Analyze metrics.csv"
```

## Scenario 1: Code Generation

### Task: Create a REST API
```bash
python3 agent_loop.py agent coder "Create a FastAPI REST API with:
- POST /api/analyze endpoint accepting JSON
- Input validation with Pydantic models
- SQLite database integration
- Proper error handling and logging
- Include docstrings and type hints"
```

**What happens:**
1. Coder agent reads the codebase structure
2. Generates `api.py` with models, routes, and database logic
3. Creates migrations and config files
4. Writes tests in `test_api.py`

---

### Task: Convert function to async
```bash
python3 agent_loop.py agent coder "Refactor the function in src/utils.py:
1. Convert blocking I/O to async (use asyncio, aiohttp)
2. Update all call sites
3. Add async/await syntax
4. Keep the same function signature where possible
5. Update tests"
```

---

### Task: Add error handling
```bash
python3 agent_loop.py agent coder "Add comprehensive error handling to api/routes.py:
- Catch specific exceptions (DatabaseError, ValidationError, etc.)
- Return appropriate HTTP status codes
- Log errors with context
- Return user-friendly error messages
- Include retry logic for transient failures"
```

---

## Scenario 2: Debugging & Investigation

### Task: Find the root cause
```bash
python3 agent_loop.py agent investigator "The database connection test is failing:
  
Test command: pytest tests/test_db.py::test_connection -v

1. Run the test and show the error
2. Trace where the error originates
3. Check environment variables and config files
4. Identify the root cause
5. Suggest a fix"
```

**What the investigator does:**
- Runs tests to reproduce the error
- Checks configuration and environment
- Reviews related code
- Identifies the issue

---

### Task: Debug flaky tests
```bash
python3 agent_loop.py agent investigator "tests/test_integration.py is failing intermittently.

1. Run the tests 3 times
2. Capture failures and their frequency
3. Look for race conditions or timing issues
4. Check for hard-coded timeouts
5. Identify which test setup/teardown might be problematic"
```

---

### Task: Analyze logs
```bash
python3 agent_loop.py agent investigator "Analyze error.log for patterns:

1. Count errors by type
2. Find the most frequent error
3. Show the timeline of errors
4. Identify correlations (e.g., time of day, request pattern)
5. Suggest what to investigate next"
```

---

## Scenario 3: Code Review & Quality

### Task: Security audit
```bash
python3 agent_loop.py agent code-reviewer "Security review of src/auth.py:

Check for:
- SQL injection vulnerabilities
- Credential/password handling (no hardcodes, proper hashing)
- Input validation bypass
- Authentication/authorization logic
- Token expiry and refresh handling
- Rate limiting on sensitive endpoints"
```

---

### Task: Performance review
```bash
python3 agent_loop.py agent code-reviewer "Performance review of src/database.py:

Look for:
- Missing indexes on queries
- N+1 query problems
- Inefficient loops or algorithms
- Unnecessary memory allocations
- Connection pooling issues
- Caching opportunities"
```

---

### Task: Best practices check
```bash
python3 agent_loop.py agent code-reviewer "Code quality review of src/services/:

Verify:
- SOLID principles adherence
- Type hints on public methods
- Docstrings present
- Error handling completeness
- Test coverage
- Code duplication"
```

---

## Scenario 4: Data Analysis

### Task: Analyze test results
```bash
python3 agent_loop.py skill data-analyst "Analyze results.json (test report):

1. Parse the JSON file
2. Count pass/fail/skip
3. Identify slowest tests
4. Show failure patterns
5. Suggest which tests to optimize"
```

---

### Task: Process CSV metrics
```bash
python3 agent_loop.py skill data-analyst "Analyze metrics.csv:

1. Load and explore the data
2. Show summary statistics (mean, median, std dev)
3. Identify outliers (values >3 std dev)
4. Plot trends over time
5. Find correlations between columns"
```

---

### Task: Log analysis
```bash
python3 agent_loop.py skill data-analyst "Analyze access.log:

1. Count requests by status code
2. Top 10 endpoints by volume
3. Response time percentiles (p50, p95, p99)
4. Error rate by endpoint
5. Peak traffic times"
```

---

## Scenario 5: Interactive Problem Solving

### Session: API Design Workshop
```bash
python3 agent_loop.py repl

# Then input:
🎯 What are REST API best practices?
🎯 How should I version my API endpoints?
🎯 Show me an example of a well-designed error response
🎯 How do I handle rate limiting?
🎯 What authentication method should I use?
🎯 exit
```

---

### Session: Python Learning
```bash
python3 agent_loop.py repl

# Then input:
🎯 Explain async/await in Python
🎯 Show an example of asyncio.gather()
🎯 What's the difference between asyncio and threading?
🎯 How do I test async functions with pytest?
🎯 exit
```

---

### Session: Refactoring Planning
```bash
python3 agent_loop.py repl

# Then input:
🎯 I want to migrate from requests to httpx. What's the difference?
🎯 Show me a side-by-side example of requests vs httpx
🎯 What breaking changes should I watch for?
🎯 How should I test the migration?
🎯 exit
```

---

## Scenario 6: Specialized Skills

### Task: Python linting
```bash
python3 agent_loop.py skill python-helper "Format and lint all Python files in src/:

1. Check code style with PEP 8
2. Auto-format files (use black)
3. Sort imports (use isort)
4. Check type hints with mypy
5. List any remaining issues"
```

---

### Task: Documentation generation
```bash
python3 agent_loop.py skill research-assistant "Research and summarize:

1. What are the latest Python 3.12 features?
2. How do they improve code quality?
3. Which ones would benefit our project?
4. Show migration examples"
```

---

## Scenario 7: Advanced Features

### Task: With full tool access
```bash
python3 agent_loop.py agent coder "Implement user authentication" --allow
```
Skips tool permission prompts — use with trusted tasks.

---

### Task: With caching
```bash
python3 agent_loop.py run "Analyze the codebase" --cache
```
Avoids re-calling the LLM for identical prompts.

---

### Task: With custom config
```bash
python3 agent_loop.py run "Fix the bug" --config my-config.yaml
```
Load custom model, token limits, or tool allowlists.

---

### Task: Resume a session
```bash
# List all sessions
python3 agent_loop.py sessions

# Resume one
python3 agent_loop.py resume abc123def456
```

---

## Scenario 8: Batch Operations

### Multi-step refactoring
```bash
python3 agent_loop.py repl

# Step 1: Analyze
🎯 Identify code duplication in src/utils.py

# Step 2: Design
🎯 Design a refactoring plan

# Step 3: Implement
🎯 Implement the refactoring

# Step 4: Verify
🎯 Run tests and show results

# Step 5: Review
🎯 Code review the changes

🎯 exit
```

---

### Feature implementation workflow
```bash
# Step 1: Design
python3 agent_loop.py agent researcher "Research: What's the best way to implement WebSocket support in FastAPI?"

# Step 2: Implement
python3 agent_loop.py agent coder "Implement WebSocket endpoints in api/websocket.py"

# Step 3: Review
python3 agent_loop.py agent code-reviewer "Review the WebSocket implementation for security and performance"

# Step 4: Test
python3 agent_loop.py agent investigator "Write and run integration tests for WebSocket endpoints"
```

---

## Tips & Tricks

### Tip 1: Structure prompts clearly
```bash
# ✗ Vague
python3 agent_loop.py "Fix the code"

# ✓ Clear
python3 agent_loop.py "Fix the test in tests/test_auth.py:
1. Show the test failure
2. Identify the root cause
3. Implement the fix
4. Verify the test passes"
```

---

### Tip 2: Use agents for speed
```bash
# Investigator is faster at debugging
python3 agent_loop.py agent investigator "Debug the timeout issue"

# Coder is better at implementation
python3 agent_loop.py agent coder "Implement the feature"

# Code-reviewer catches quality issues
python3 agent_loop.py agent code-reviewer "Review for security"
```

---

### Tip 3: Build on previous work
```bash
python3 agent_loop.py repl

🎯 Implement basic CRUD API
# Agent creates api.py with basic endpoints

🎯 Add error handling
# Agent enhances the existing code

🎯 Add database validation
# Agent further improves it

🎯 exit
```

---

### Tip 4: Cross-check with code review
```bash
# Generate code
python3 agent_loop.py agent coder "Implement feature X"

# Review it
python3 agent_loop.py agent code-reviewer "Review the implementation in src/feature_x.py"

# Fix issues
python3 agent_loop.py agent coder "Address the review comments in src/feature_x.py"
```

---

### Tip 5: Focus on one task
```bash
# ✗ Too broad
python3 agent_loop.py "Refactor the entire codebase"

# ✓ Focused
python3 agent_loop.py "Refactor src/auth.py to use dependency injection"
```

---

## Token Management Examples

### Working with 131K Context Window

With a large model (131K tokens), you can run extended multi-step tasks:

```bash
# Complex project implementation (25+ iterations)
python3 agent_loop.py agent coder "Implement a complete REST API:
1. Design database schema
2. Create models and migrations
3. Implement CRUD endpoints
4. Add authentication
5. Add error handling
6. Write integration tests
7. Add documentation"
```

This can now run through all steps without token exhaustion.

### Checking Token Usage

Monitor token consumption in real-time. The loop prints:
```
🔄 --- ITERATION 5/100 ---
📊 Tokens: 24,576 / 131,072 (18.7%)
```

When approaching 80% threshold:
```
🧹 Compacting messages (89% of budget)...
   Before: 45 messages (116,736 tokens)
   After:  8 messages (summarized + recent, 24,576 tokens)
```

### Adjusting for Different Models

**If using a smaller model (8K context):**
```python
# In src/config.py
MODEL_CONTEXT_WINDOW = 8192
REQUEST_MAX_TOKENS = 2048  # More conservative
```

Then tasks complete in fewer iterations with tighter budgets:
```bash
python3 agent_loop.py "Simpler task suitable for 8K context"
```

---

## Troubleshooting Examples

### Issue: Tool permission denied
```bash
# Solution 1: Use --allow flag
python3 agent_loop.py run "Task" --allow

# Solution 2: Add to ALLOWED_TOOLS in src/config.py
```

---

### Issue: Slow/timeout responses
```bash
# Solution: Increase timeout in src/config.py
LM_STUDIO_TIMEOUT = 120  # was 60

# Then retry
python3 agent_loop.py "Your goal"
```

---

### Issue: MCP tools not available
```bash
# Check mcp_servers.json
cat mcp_servers.json

# Verify MCP servers are running
# Then retry with a fresh REPL session
python3 agent_loop.py repl
```

---

## Summary

| Goal | Command | Agent | Why |
|------|---------|-------|-----|
| Build code | `agent coder` | coder | Best at implementation, SOLID principles |
| Find bugs | `agent investigator` | investigator | Good at tracing, root cause analysis |
| Quality review | `agent code-reviewer` | code-reviewer | Catches edge cases, security, performance |
| Learn concepts | `repl` | researcher | Good at explanations, comparisons |
| Data analysis | `skill data-analyst` | data-analyst | Specialized for metrics, logs, CSVs |
| Interactive work | `repl` | any | Build on previous responses |

---

For full documentation, see `README.md`.
