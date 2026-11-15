"""Workspace instructions: the AGENTS.md entrypoint and the local Markdown documents it links to.

Discovery. Starting at the workspace directory, the nearest AGENTS.md is used. The search walks toward the parents and
stops at the first directory that contains a .git entry (the project root, included), or before the user's home
directory (excluded), whichever comes first. Nothing above home is ever considered.

Loading. The entrypoint is read, and every Markdown link in it whose target ends in .md is followed, depth first, in
order of appearance. Links inside code spans and fenced code blocks are not directives. A document is loaded once.
Each target must resolve inside the instruction root (the directory that holds the entrypoint), with symlinks resolved.
Remote and absolute targets are rejected. Any failure, including a missing, unreadable, or oversized document, raises
InstructionError. Nothing is truncated.

Instructions are text for the model. They grant no tools, approvals, or paths, and they do not change the workspace.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

ENTRY_NAME = "AGENTS.md"
MAX_INSTRUCTION_BYTES = 64 * 1024  # total bytes of all loaded instruction documents

_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
_LINK = re.compile(r"\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


class InstructionError(Exception):
    """Workspace instructions cannot be loaded safely. The run must not start with a partial set."""


@dataclass(frozen=True)
class Instructions:
    root: Path               # the instruction root: the directory that holds the entrypoint
    files: Tuple[str, ...]   # loaded documents, relative to root, entrypoint first
    text: str                # the composed text sent to the model
    size: int                # total bytes of the loaded documents
    cycles: Tuple[str, ...]  # back-references that closed a cycle (already loaded, not loaded again)


def find_entry(start: Path, home: Path) -> Optional[Path]:
    """The nearest AGENTS.md at or above start, bounded by the project root and by the home directory."""
    home = home.resolve()
    current = start.resolve()
    for directory in [current, *current.parents]:
        if directory == home:
            return None
        candidate = directory / ENTRY_NAME
        if candidate.is_file():
            return candidate
        if (directory / ".git").exists():
            return None
    return None


def _local_markdown_links(text: str) -> List[str]:
    """Targets of Markdown links that name a local .md document, in order. Other links are ignored."""
    stripped = _CODE.sub("", text)
    targets: List[str] = []
    for match in _LINK.finditer(stripped):
        target = match.group(1).split("#", 1)[0]
        if not target:
            continue
        if _SCHEME.match(target):
            if target.lower().endswith(".md") or ".md#" in match.group(1).lower():
                raise InstructionError(f"remote instruction sources are not allowed: {match.group(1)}")
            continue
        if target.startswith(("/", "~")):
            if target.lower().endswith(".md"):
                raise InstructionError(f"absolute paths are not allowed as instruction sources: {target}")
            continue
        if target.lower().endswith(".md") and target not in targets:
            targets.append(target)
    return targets


def _read_checked(root: Path, path: Path, label: str) -> Tuple[str, int]:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise InstructionError(f"{label} resolves outside the instruction root (possibly through a symbolic link)")
    if not resolved.is_file():
        raise InstructionError(f"required instruction document is missing: {label}")
    try:
        data = resolved.read_bytes()
    except OSError as exc:
        raise InstructionError(f"required instruction document is unreadable: {label} ({exc.strerror or exc})") from exc
    if len(data) > MAX_INSTRUCTION_BYTES:
        raise InstructionError(f"{label} is larger than the instruction limit of {MAX_INSTRUCTION_BYTES} bytes")
    try:
        return data.decode("utf-8"), len(data)
    except UnicodeDecodeError as exc:
        raise InstructionError(f"required instruction document is not UTF-8 text: {label}") from exc


def load(entry: Path) -> Instructions:
    """Load the entrypoint and every local document it links to, transitively, with duplicates and cycles handled."""
    root = entry.parent.resolve()
    documents: List[Tuple[str, str]] = []
    cycles: List[str] = []
    seen: set = set()
    total = 0

    def visit(path: Path, rel: str, chain: Tuple[str, ...]) -> None:
        nonlocal total
        if rel in seen:
            if rel in chain:
                cycles.append(" -> ".join(chain + (rel,)))
            return
        seen.add(rel)
        text, size = _read_checked(root, path, rel)
        total += size
        if total > MAX_INSTRUCTION_BYTES:
            raise InstructionError(f"the instruction documents exceed the limit of {MAX_INSTRUCTION_BYTES} bytes "
                                   f"at {rel}; nothing was loaded")
        documents.append((rel, text))
        for target in _local_markdown_links(text):
            child = path.parent / target
            resolved = child.resolve()
            if not resolved.is_relative_to(root):
                raise InstructionError(f"{rel} links to a document outside the instruction root: {target}")
            visit(child, resolved.relative_to(root).as_posix(), chain + (rel,))

    visit(entry, ENTRY_NAME, ())
    text = "\n\n".join(f"--- {rel} ---\n{body}" for rel, body in documents)
    return Instructions(root=root, files=tuple(rel for rel, _ in documents), text=text, size=total,
                        cycles=tuple(cycles))


def discover(start: Path, home: Path) -> Optional[Instructions]:
    """Find and load the workspace instructions for start, or return None when there are none."""
    entry = find_entry(start, home)
    if entry is None:
        return None
    return load(entry)
