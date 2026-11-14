"""Session-local authorization.

Grants live on the session that approved them, not in process-global state. A session's policy is built
from the CLI configuration when the session is created, and interactive approvals are added to that
session only. Nothing is persisted: a resumed session, or a new process, starts from the configuration.

Grant kinds and their scope:

  "command"       CLI policy only (``ALLOWED_TOOLS``). A base command name such as ``rm``. It is broad by
                  design, because the operator wrote it in configuration. An interactive approval can never
                  create one of these.
  "command_exact" Interactive approval of an unlisted command. The exact command text, after surrounding
                  whitespace is trimmed. Approving ``curl -I https://a`` does not approve ``curl https://b``.
  "mcp"           An MCP tool name, from ``ALLOWED_MCP_TOOLS`` or an interactive approval. The tool name is
                  the identity being approved; its arguments are not part of the match.
  "shell"         A shell command that uses operators (pipes, redirects, ``&&``, ``||``), matched by its exact
                  text. Shell operators are never approved by a base name.

Each kind can be granted for the session (persistent) or for exactly one attempt (one-time). A one-time
grant is consumed by the attempt it approves, and the loop discards any that remain when that attempt
ends, including when it fails validation or is cancelled.

The object also records whether this session has run a codebase search. That is session-local tool state,
kept here because this object is already threaded through dispatch, shell, and file tools. It is discarded
when the session closes and is never restored on resume.
"""

import threading
from typing import Iterable, Set, Tuple

from . import config

KINDS = frozenset({"command", "command_exact", "mcp", "shell"})
# Interactive approvals may only use these kinds. "command" is CLI policy only.
INTERACTIVE_KINDS = frozenset({"command_exact", "mcp", "shell"})


class SessionAuthorization:
    """Permission decisions and tool state for one live session. Thread-safe."""

    def __init__(self, allow_all: bool = False, commands: Iterable[str] = (), mcp_tools: Iterable[str] = ()):
        self._lock = threading.Lock()
        self._allow_all = bool(allow_all)
        self._persistent: dict = {
            "command": set(commands),
            "command_exact": set(),
            "mcp": set(mcp_tools),
            "shell": set(),
        }
        self._once: Set[Tuple[str, str]] = set()
        self._searched = False

    @classmethod
    def from_config(cls) -> "SessionAuthorization":
        """The initial policy derived from CLI configuration. A copy: later edits do not change it."""
        return cls(config.ALLOW_ALL, config.ALLOWED_TOOLS, config.ALLOWED_MCP_TOOLS)

    @property
    def allow_all(self) -> bool:
        return self._allow_all

    # --- checks -----------------------------------------------------------------------------

    def allows(self, kind: str, identifier: str) -> bool:
        """True if the action is allowed now. A one-time grant is consumed by this check."""
        _check_kind(kind)
        with self._lock:
            if self._allow_all:
                return True
            return self._match_locked(kind, identifier)

    def allows_command(self, base: str, command_text: str) -> bool:
        """A plain command runs if the CLI policy names its base command, or if this session approved its exact
        text. The CLI check and the interactive check are separate, so an interactive approval never widens
        the CLI policy and the CLI policy never needs the exact text."""
        with self._lock:
            if self._allow_all:
                return True
            if base in self._persistent["command"]:
                return True
            return self._match_locked("command_exact", command_text)

    def _match_locked(self, kind: str, identifier: str) -> bool:
        if identifier in self._persistent[kind]:
            return True
        if (kind, identifier) in self._once:
            self._once.discard((kind, identifier))
            return True
        return False

    # --- grants -----------------------------------------------------------------------------

    def grant(self, kind: str, identifier: str) -> None:
        """Approve for the rest of this session."""
        _check_interactive(kind)
        with self._lock:
            self._persistent[kind].add(identifier)

    def grant_once(self, kind: str, identifier: str) -> None:
        """Approve exactly one attempt of this action."""
        _check_interactive(kind)
        with self._lock:
            self._once.add((kind, identifier))

    def discard_once(self, kind: str, identifier: str) -> None:
        """Remove a one-time grant that was not consumed, for example because the attempt failed validation or
        was cancelled before it reached the check. Called by the loop when an approved attempt ends."""
        _check_kind(kind)
        with self._lock:
            self._once.discard((kind, identifier))

    def snapshot(self) -> dict:
        """Diagnostic copy of the persistent grants, and the once-grants still pending."""
        with self._lock:
            data = {kind: sorted(values) for kind, values in self._persistent.items()}
            data["once"] = sorted(f"{kind}:{identifier}" for kind, identifier in self._once)
            return data

    # --- session tool state -----------------------------------------------------------------

    @property
    def has_searched(self) -> bool:
        return self._searched

    def note_search(self) -> None:
        """Record that this session ran a codebase search (grep, rg, or find) that was allowed to start."""
        with self._lock:
            self._searched = True


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"unknown grant kind {kind!r}")


def _check_interactive(kind: str) -> None:
    _check_kind(kind)
    if kind not in INTERACTIVE_KINDS:
        raise ValueError(f"{kind!r} is CLI policy and cannot be granted interactively")
