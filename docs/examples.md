# Examples

## Code generation

```bash
.venv/bin/python -m src.cli agent coder \
  "Create a small FastAPI REST API with POST /analyze, Pydantic validation, and tests"
```

## Bug investigation

```bash
.venv/bin/python -m src.cli agent investigator \
  "Investigate the failing integration tests, identify the root cause, and suggest a fix"
```

## Data analysis

```bash
.venv/bin/python -m src.cli skill data-analyst \
  "Analyze data/metrics.csv and report trends, outliers, and key statistics"
```

## Interactive research

```bash
.venv/bin/python -m src.cli repl
```

Example goals:

```text
What are best practices for Python async?
Show me an example of asyncio.gather()
How do I test async functions?
exit
```

## Code review

```bash
.venv/bin/python -m src.cli agent code-reviewer \
  "Review src/api.py for logic errors, edge cases, performance, and security"
```

## Research with trusted verification

```bash
.venv/bin/python -m src.cli run \
  --criteria criteria/migration.json \
  "Research the migration and return an evidence-backed checklist"
```

Use trusted criteria only when the acceptance conditions are known in advance.
The model cannot modify those criteria.
