"""RME-01 evaluation policy versions. Evaluation infrastructure only. The production prompt, tool descriptions,
ToolRegistry and dispatch authorization are never edited: every policy change is an in-process overlay applied
by the trial runner, and each overlay fails closed if the production text it expects has changed.

- RME-01-POLICY-V1: the historical configuration. The prompt, the tool schemas, and dispatch are unmodified.
- RME-01-POLICY-V2: (a) the production prompt with exact replacements of passages that present unauthorized shell
  commands as usable, plus an authorization block; (b) a provider-visible tool overlay: managed-job tools are not
  advertised, and descriptions that advertise unauthorized commands are corrected; (c) a dispatch guard: any tool name
  other than the advertised four is rejected with the production permission envelope before it can run.

Every expected production text is pinned by hash. A drift in the production prompt or tool schemas refuses v2.
The shell allowlist is scenarios.TRIAL_TOOL_ALLOWLIST and is not changed by any policy.
"""

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src import prompts
from src import envelope
from src.tools import get_local_tools
from scenarios import TRIAL_TOOL_ALLOWLIST

POLICY_V1 = "RME-01-POLICY-V1"
POLICY_V2 = "RME-01-POLICY-V2"
POLICY_IDS = (POLICY_V1, POLICY_V2)

# Pinned production texts. Recomputed with sha256 over the exact strings (see tests).
PRODUCTION_PROMPT_SHA256 = "0f12ffab50e903665207e37800c487c654b7b1a1ca107977ab8959b8ec12f60a"
PRODUCTION_TOOLS_SHA256 = "d3097148d3a5b6a2e48fc28e4e4851324c22b8e00d5233abc93649ebd220e8f4"
PROMPT_REPLACEMENT_COUNT = 9

JOB_TOOLS = ("start_job", "status_job", "collect_job", "terminate_job")
V2_ADVERTISED = ("run_command", "write_file", "read_file", "list_directory")


class PolicyError(Exception):
    pass


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True)


# Captured at import, before any install, so the expected production text is never read from a modified constant.
_PRODUCTION_PROMPT = prompts.SYSTEM_PROMPT
_PRODUCTION_TOOLS = get_local_tools()


def production_prompt() -> str:
    return _PRODUCTION_PROMPT


def production_tools() -> List[Dict[str, Any]]:
    return copy.deepcopy(_PRODUCTION_TOOLS)


# (production text, v2 text). Each production text must occur exactly once in the production prompt.
V2_REPLACEMENTS: List[Tuple[str, str]] = [
    ('✅ WORKS: `grep -r "pattern" .`, `python3 script.py`, `find . -name "*.py" | head -5` (after approval)',
     "✅ WORKS: `python3 script.py`, `ls`, `cat FILE` (the only authorized command basenames: python3, ls, cat)"),
    ("❌ BLOCKED (need approval): Shell operators (`|`, `>`, `&&`, `||`) on first use require permission. "
     "Approve once and they work for the session.",
     "❌ BLOCKED (not authorized in this evaluation): any command whose basename is not python3, ls, or cat, and "
     "shell operators (`|`, `>`, `&&`, `||`). Such a command is denied without a prompt and the run ends as BLOCKED."),
    ("- For a process that must keep running while you work (a server, a watcher), use managed jobs instead of "
     "run_command:\n"
     "  - **start_job(command, lifetime_seconds?)** starts it and returns a job_id. Same command rules and approvals as "
     "run_command.\n"
     "  - **status_job(job_id)** reports state, reason, and exit status. **collect_job(job_id, cursor?)** reads output "
     "incrementally; pass back the cursor it returns.\n"
     "  - **terminate_job(job_id)** stops it. A job also stops at its lifetime limit, and when this run ends.\n"
     "  - Jobs belong to this run only, and they are not restarted after a resume. Treat an `unknown` state as possibly "
     "still running.",
     "- Managed jobs (start_job, status_job, collect_job, terminate_job) are not authorized in this evaluation; "
     "do not start them."),
    ("- Long compilation: `make build` with `timeout_seconds` 300",
     "- Long test run: `python3 -m unittest -q test_example` with `timeout_seconds` 300"),
    ("- Pipeline (after approval): `find . -name \"*.py\" | head -10`",
     "- Listing: `ls -la` (python3, ls, and cat are the only authorized commands)"),
    ('Some commands may require human approval and can be denied — treat a "Command denied by user." result as a '
     "signal to adapt your plan. Shell operators require one-time approval per session.",
     'Commands outside the authorized list are not approved in this evaluation. A "Command denied by user." result '
     "ends the run as BLOCKED; do not try other commands to get around it."),
    ("Before creating a new file, search the codebase first (run_command with grep, rg, or find) to check whether "
     "similar functionality already exists.",
     "Before creating a new file, look through the codebase first (with list_directory and read_file) to check whether "
     "similar functionality already exists."),
    ("1. **Explore existing code**: Use `run_command` with `find`, `grep`, `ls` to understand what already exists",
     "1. **Explore existing code**: Use `list_directory` and `read_file` to understand what already exists"),
    ('- `find . -name "*.py" -o -name "*.ts" | head -20` — explore file structure\n'
     '- `grep -r "pattern" . --include="*.py"` — search for keywords\n'
     "- `list_directory .` — understand directory layout\n"
     '- `run_command` with `find . -type f -name "*epoll*"` — check for specific topics',
     "- `list_directory .` — understand directory layout\n"
     "- `read_file <path>` — read a file to see what it does"),
]

V2_AUTHORIZATION = """

### Trial Shell Authorization (RME-01 Policy v2)

This evaluation is non-interactive. The only shell command basenames authorized without approval are:
- python3
- ls
- cat

Use them as plain commands: no pipes, redirects, `&&`, `||`, or background `&`.

For finding and reading files, prefer `list_directory` and `read_file`. Every other shell command, including `find`,
`grep`, `rg`, and `make`, requires approval. Approval will not be given here, so such a command ends the run as BLOCKED.
Managed jobs are not available in this evaluation."""

# Tool-description overlay for the advertised tools: (tool name, production text, v2 text), each count-checked.
V2_TOOL_REPLACEMENTS: List[Tuple[str, str, str]] = [
    ("run_command",
     "Commands containing shell operators (|, >, >>, <, &&, ||) run through /bin/sh only after the user approves them "
     "once per session.",
     "Commands containing shell operators (|, >, >>, <, &&, ||) are not authorized in this evaluation and are denied. "
     "Use plain commands: python3, ls, or cat."),
    ("run_command",
     '- Search codebase: run_command with "grep -r \'function_name\' ."',
     '- Read a file: run_command with "cat README.md"'),
    ("run_command",
     '- List files: run_command with "find . -name \'*.py\' -type f"',
     '- List files: run_command with "ls -la" (or list_directory)'),
    ("run_command",
     "- Some commands may require human approval (1/2/3 escalation). If denied, adapt your plan.",
     "- Commands other than python3, ls, and cat are not approved in this evaluation; a denial ends the run as BLOCKED."),
    ("run_command",
     "Background execution is not supported and is rejected; for a long-running process use start_job. Commands such "
     "as nohup, setsid, disown, screen, tmux, at, and batch are also rejected.",
     "Background execution is not supported and is rejected. Managed jobs are not authorized in this evaluation. "
     "Commands such as nohup, setsid, disown, screen, tmux, at, and batch are also rejected."),
    ("write_file",
     "search the codebase first (run_command with grep, rg, or find) to check whether similar functionality already "
     "exists.",
     "look through the codebase first (with list_directory and read_file) to check whether similar functionality "
     "already exists."),
    ("list_directory",
     "use run_command with find or ls for filtering.",
     "use run_command with ls for filtering (find is not authorized in this evaluation)."),
]


def _prompt_for(policy_id: str) -> str:
    if policy_id == POLICY_V1:
        return production_prompt()
    if policy_id != POLICY_V2:
        raise PolicyError(f"unknown policy {policy_id!r}; known: {POLICY_IDS}")
    if sha256(production_prompt()) != PRODUCTION_PROMPT_SHA256:
        raise PolicyError("the production system prompt changed; Policy v2 refuses to apply its replacements")
    text = production_prompt()
    for old, new in V2_REPLACEMENTS:
        count = text.count(old)
        if count != 1:
            raise PolicyError(f"v2 replacement target occurs {count} times in the production prompt (need exactly 1): "
                              f"{old[:60]!r}")
        text = text.replace(old, new)
    return text + V2_AUTHORIZATION


def effective_prompt(policy_id: str) -> str:
    return _prompt_for(policy_id)


def _check_source_tools(tools: List[Dict[str, Any]]) -> None:
    if sha256(_canonical(tools)) != PRODUCTION_TOOLS_SHA256:
        raise PolicyError("the production tool schemas changed; Policy v2 refuses to overlay them")


def overlay_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Policy v2 provider-visible tools. Names, parameters and the argument schemas of advertised tools are kept;
    only the descriptions listed in V2_TOOL_REPLACEMENTS change. Job tools are not advertised."""
    _check_source_tools(tools)
    out = []
    for schema in copy.deepcopy(tools):
        name = schema["function"]["name"]
        if name not in V2_ADVERTISED:
            continue
        description = schema["function"]["description"]
        for tool, old, new in V2_TOOL_REPLACEMENTS:
            if tool != name:
                continue
            count = description.count(old)
            if count != 1:
                raise PolicyError(f"v2 tool replacement for {name} occurs {count} times (need exactly 1): {old[:60]!r}")
            description = description.replace(old, new)
        schema["function"]["description"] = description
        out.append(schema)
    if [s["function"]["name"] for s in out] != [n for n in V2_ADVERTISED]:
        raise PolicyError("the advertised tool set does not match the Policy v2 list")
    return out


def _v2_tool_list() -> List[Dict[str, Any]]:
    return overlay_tools(production_tools())


def expected_metadata(policy_id: str) -> Dict[str, Any]:
    """The values a v2 (or v1) trial must report. Computed from the production texts and the overlay rules on the
    host, never read from a trial's own output. Used by the supervisor's independent validation."""
    effective = _prompt_for(policy_id)
    if policy_id == POLICY_V1:
        tools = production_tools()
        replacements, guard, advertised = 0, "none", sorted(s["function"]["name"] for s in tools)
    else:
        tools = _v2_tool_list()
        replacements, guard, advertised = len(V2_REPLACEMENTS), "enforced", list(V2_ADVERTISED)
    return {
        "policy_id": policy_id,
        "production_prompt_sha256": sha256(production_prompt()),
        "effective_prompt_sha256": sha256(effective),
        "replacements_applied": replacements,
        "shell_allowlist": sorted(TRIAL_TOOL_ALLOWLIST),
        "dispatch_guard": guard,
        "advertised_tools": sorted(advertised),
        "expected_tool_schema_sha256": sha256(_canonical(tools)),
    }


def tool_schema_sha256(tools: Any) -> str:
    """Identity of the provider-visible tool list exactly as sent in a request."""
    return sha256(_canonical(tools))


def _rejection(name: str) -> Dict[str, Any]:
    return envelope.error_permission(
        f"Tool '{name}' is not authorized under RME-01 Policy v2 (managed jobs and commands outside python3, ls, "
        "cat are not part of this evaluation). It was not run.")


_ORIGINALS: Dict[str, Any] = {}


def _original_targets() -> Dict[str, Any]:
    """The production callables, captured once. Every install starts from these, so re-installing never stacks
    overlays on top of earlier ones."""
    if not _ORIGINALS:
        from src import loop, dispatch
        _ORIGINALS.update({"loop": loop, "get_all_tools": loop.get_all_tools,
                           "dispatch_tool_async": dispatch.dispatch_tool_async,
                           "dispatch_tool": dispatch.dispatch_tool, "dispatch": dispatch})
    return _ORIGINALS


def _apply_guard(enabled: bool) -> None:
    orig = _original_targets()
    loop, dispatch = orig["loop"], orig["dispatch"]
    if not enabled:
        loop.get_all_tools = orig["get_all_tools"]
        dispatch.dispatch_tool_async = orig["dispatch_tool_async"]
        dispatch.dispatch_tool = orig["dispatch_tool"]
        return

    def guarded_get_all(*args, **kwargs):
        return overlay_tools(orig["get_all_tools"](*args, **kwargs))

    async def guarded_async(tool_name, tool_input, *args, **kwargs):
        if tool_name not in V2_ADVERTISED:
            return _rejection(str(tool_name))
        return await orig["dispatch_tool_async"](tool_name, tool_input, *args, **kwargs)

    def guarded_sync(tool_name, tool_input, *args, **kwargs):
        if tool_name not in V2_ADVERTISED:
            return _rejection(str(tool_name))
        return orig["dispatch_tool"](tool_name, tool_input, *args, **kwargs)

    loop.get_all_tools = guarded_get_all
    dispatch.dispatch_tool_async = guarded_async
    dispatch.dispatch_tool = guarded_sync


def install(policy_id: str) -> Dict[str, Any]:
    """Apply a policy to this process: prompt constant, tool overlay, dispatch guard. Idempotent: the result depends
    only on policy_id, not on earlier installs. Validates every pinned text before changing anything."""
    expected = expected_metadata(policy_id)
    effective = _prompt_for(policy_id)
    _apply_guard(policy_id == POLICY_V2)
    from src import prompts as _prompts
    _prompts.SYSTEM_PROMPT = effective
    return expected


def uninstall() -> None:
    """Restore the production prompt, tool list and dispatch (used by tests)."""
    _apply_guard(False)
    from src import prompts as _prompts
    _prompts.SYSTEM_PROMPT = production_prompt()


def validate_policy_metadata(reported: Any, requested: str, expected: Dict[str, Any],
                             actual_tool_hashes: Any) -> List[str]:
    """Independent validation. `expected` is computed on the host; `reported` is the trial's own claim. Returns the
    list of problems; an empty list means the metadata is accepted."""
    problems: List[str] = []
    if not isinstance(reported, dict):
        return ["policy metadata is missing or not an object"]
    required = ("policy_id", "production_prompt_sha256", "effective_prompt_sha256", "replacements_applied",
                "shell_allowlist", "dispatch_guard", "advertised_tools")
    missing = [k for k in required if k not in reported]
    if missing:
        return [f"policy metadata lacks {missing}"]
    if reported["policy_id"] != requested:
        problems.append(f"applied policy {reported['policy_id']!r} != requested {requested!r}")
    for key in ("production_prompt_sha256", "effective_prompt_sha256", "replacements_applied", "shell_allowlist",
                "dispatch_guard", "advertised_tools"):
        if reported[key] != expected[key]:
            problems.append(f"{key}: reported {reported[key]!r}, expected {expected[key]!r}")
    if not isinstance(actual_tool_hashes, list) or not actual_tool_hashes:
        problems.append("no provider-visible tool-schema hash was recorded")
    elif not all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in actual_tool_hashes):
        problems.append("provider-visible tool-schema hashes are malformed")
    elif set(actual_tool_hashes) != {expected["expected_tool_schema_sha256"]}:
        problems.append("provider-visible tool schemas differ from the expected overlay")
    return problems


def source_fingerprint(repo: Optional[Path] = None) -> str:
    """Traceability label for the image build: a hash over the files the image is built from. It identifies the
    inputs; it is not cryptographic build provenance."""
    root = repo or Path(__file__).resolve().parents[2]
    files = sorted(p for p in (root / "src").rglob("*.py")) + [
        root / "requirements" / "base.txt", root / "evaluation" / "rme1" / "Dockerfile",
        root / "evaluation" / "rme1" / "scenarios.py", root / "evaluation" / "rme1" / "runner.py",
        root / "evaluation" / "rme1" / "probe.py", root / "evaluation" / "rme1" / "metrics.py",
        root / "evaluation" / "rme1" / "policy.py", root / "evaluation" / "rme1" / "egress-init.sh"]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    print(source_fingerprint())
