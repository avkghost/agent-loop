# context-mode 2.2.0 in Agent Loop

Prefer context-mode for large, read-only exploration and indexing when it reduces context usage. Use Agent Loop's regular tools for workspace modifications and operations requiring its authorization or verification guarantees.

This is tool-selection advice, not an enforced rule. context-mode tools are MCP tools, so Agent Loop's authorization applies to them as to any other MCP tool, and this file does not change that. Their output is an observation, not verified evidence.

When you use context-mode, call these exact names: `mcp_context-mode_ctx_execute` to run a command or analyse a file and print only the answer, `mcp_context-mode_ctx_batch` to run several commands at once, `mcp_context-mode_ctx_index` to store content for later search, and `mcp_context-mode_ctx_search` to query stored content.

- **Source labels.** A path document's source is its path relative to the project root. A URL document's source is its final URL. A `source` you choose applies only to raw content indexed with `content`. Do not pass `source` when indexing a path or a URL.
- **Source filters.** A `source` filter on `ctx_search` matches a label exactly. Use one only when you know the label; otherwise search without it.
- **No limit.** `ctx_search` has no `limit` argument. Each query returns up to 5 hits by default.
- **Continuations.** `offset`, `after` and `More:` continuations are optional. Use one only when the results you have do not answer the question. Do not page through results to be exhaustive.
- **Timeline.** `sort: "timeline"` returns chronological history across this session, earlier sessions and memory, not document content.
- **Web pages.** Index a URL with `mcp_context-mode_ctx_index` rather than fetching it with shell commands such as `curl`. Agent Loop does not run context-mode's hooks, so nothing intercepts shell commands for you.
