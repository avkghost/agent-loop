"""Static registry of local tool descriptors: names, provider-visible schemas, and routing kinds.

The registry is routing and schema data only. It grants nothing. Permission decisions stay with
SessionAuthorization and the handlers that consult it (shell validation, the MCP gate, job start).

Kinds are declarative routing information:
  file   built-in file tools, contained to the working directory
  shell  run_command, a foreground shell command
  job    managed-job tools, owned by the calling session
  git    read-only Git inspection (fixed commands, no model-supplied Git options)
  mcp    names with the mcp_ prefix, registered by MCP servers at runtime; never registered locally

The registry is built once, finalized, and then immutable. It is not a plugin system: nothing is discovered
dynamically, and no entry points or hot reloading are involved.
"""

import copy
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

FILE = "file"
SHELL = "shell"
JOB = "job"
GIT = "git"
MCP = "mcp"
KINDS = frozenset({FILE, SHELL, JOB, GIT})
MCP_PREFIX = "mcp_"


class RegistryError(ValueError):
    """A registration was refused. Registration happens at startup, so this is a programming error."""


@dataclass(frozen=True)
class ToolDescriptor:
    name: str
    kind: str
    schema: Dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise RegistryError("tool name must be a non-empty string")
        if self.name.startswith(MCP_PREFIX):
            raise RegistryError(f"local tool {self.name!r} may not use the {MCP_PREFIX} prefix, which is reserved for MCP")
        if self.kind not in KINDS:
            raise RegistryError(f"unknown tool kind {self.kind!r}")
        declared = self.schema.get("function", {}).get("name") if isinstance(self.schema, dict) else None
        if declared != self.name:
            raise RegistryError(f"schema name {declared!r} does not match descriptor name {self.name!r}")
        # Own a private copy: a caller keeping the original dict must not change a registered schema later
        object.__setattr__(self, "schema", copy.deepcopy(self.schema))


class ToolRegistry:
    """Ordered, duplicate-free descriptors. Lookup is exact-match; unknown names resolve to None."""

    def __init__(self) -> None:
        self._items: List[ToolDescriptor] = []
        self._by_name: Dict[str, ToolDescriptor] = {}
        self._final = False

    def register(self, descriptor: ToolDescriptor) -> None:
        if self._final:
            raise RegistryError("the registry is finalized; tools cannot be registered after startup")
        if descriptor.name in self._by_name:
            raise RegistryError(f"duplicate tool registration: {descriptor.name}")
        self._items.append(descriptor)
        self._by_name[descriptor.name] = descriptor

    def finalize(self) -> "ToolRegistry":
        self._final = True
        return self

    @property
    def finalized(self) -> bool:
        return self._final

    def get(self, name: Any) -> Optional[ToolDescriptor]:
        """A descriptor whose schema is a fresh copy, so the registry's own definition cannot be altered here."""
        if not isinstance(name, str):
            return None
        descriptor = self._by_name.get(name)
        if descriptor is None:
            return None
        return dataclasses.replace(descriptor, schema=copy.deepcopy(descriptor.schema))

    def kind_of(self, name: Any) -> Optional[str]:
        """The routing kind for a name. Names with the MCP prefix route to MCP; everything else unknown is None."""
        if isinstance(name, str) and name in self._by_name:
            return self._by_name[name].kind
        if isinstance(name, str) and name.startswith(MCP_PREFIX):
            return MCP
        return None

    def names(self) -> List[str]:
        return [item.name for item in self._items]

    def schemas(self) -> List[Dict[str, Any]]:
        """Fresh copies of the provider-visible schemas, in registration order."""
        return [copy.deepcopy(item.schema) for item in self._items]
