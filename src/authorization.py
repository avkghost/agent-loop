"""Session-local authorization.

Grants live on the session that approved them, not in process-global state. The initial policy is a
snapshot of the CLI configuration taken when the session is created. Grants are never restored from
disk: a resumed session starts from the configuration again.

Grant kinds:
  "command"  a base command name such as ``curl`` (persistent grants apply to that name)
  "mcp"      an MCP tool name (persistent grants apply to that name)
  "shell"    an exact shell command string (never a prefix or a pattern)
"""

import threading
from typing import Iterable, Set, Tuple

from . import config

KINDS = frozenset({"command", "mcp", "shell"})


class SessionAuthorization:
    """Permission decisions for one live session. Thread-safe; the job worker may consult it."""

    def __init__(self, allow_all: bool = False, commands: Iterable[str] = (), mcp_tools: Iterable[str] = ()):
        self._lock = threading.Lock()
        self._allow_all = bool(allow_all)
        self._persistent: dict = {
            "command": set(commands),
            "mcp": set(mcp_tools),
            "shell": set(),
        }
        self._once: Set[Tuple[str, str]] = set()

    @classmethod
    def from_config(cls) -> "SessionAuthorization":
        """The initial policy derived from CLI configuration. A copy: later edits do not change it."""
        return cls(config.ALLOW_ALL, config.ALLOWED_TOOLS, config.ALLOWED_MCP_TOOLS)

    @property
    def allow_all(self) -> bool:
        return self._allow_all

    def allows(self, kind: str, identifier: str) -> bool:
        """True if the action is allowed now. A one-time grant is consumed by this check."""
        _check_kind(kind)
        with self._lock:
            if self._allow_all:
                return True
            if identifier in self._persistent[kind]:
                return True
            if (kind, identifier) in self._once:
                self._once.discard((kind, identifier))
                return True
            return False

    def grant(self, kind: str, identifier: str) -> None:
        """Approve for the rest of this session."""
        _check_kind(kind)
        with self._lock:
            self._persistent[kind].add(identifier)

    def grant_once(self, kind: str, identifier: str) -> None:
        """Approve exactly one attempt of this action."""
        _check_kind(kind)
        with self._lock:
            self._once.add((kind, identifier))

    def snapshot(self) -> dict:
        """Diagnostic copy of the persistent grants."""
        with self._lock:
            return {kind: sorted(values) for kind, values in self._persistent.items()}


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"unknown grant kind {kind!r}")
