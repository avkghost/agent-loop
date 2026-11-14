"""File operations tools (read, write, list).

Containment: every path is resolved (following symlinks) and must lie inside the working
directory, checked by path components. This is a path check, not an OS sandbox; it does not
protect against adversarial filesystem changes made concurrently by other processes.
"""

import os
import stat
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from .. import config, envelope


def _contained_target(path_str: str) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """Resolve a path inside the working directory. Returns (target, None) or (None, error envelope)."""
    cwd = Path.cwd().resolve()
    target = (cwd / path_str).resolve()
    if not target.is_relative_to(cwd):
        return None, envelope.error_permission(
            f"Error: Path {path_str} resolves outside working directory {cwd}",
            description="Paths must resolve within the working directory."
        )
    return target, None


def read_file_safe(path_str: str) -> Dict[str, Any]:
    """Read a UTF-8 text file inside the working directory. Reading has no side effects."""
    try:
        target, error = _contained_target(path_str)
        if error:
            return error

        if not target.exists():
            return envelope.error_validation(
                f"Error: File not found: {target}",
                fixable=True,
                description="The file does not exist at the specified path."
            )

        data = target.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return envelope.error_validation(
                f"Error: {target.name} is not UTF-8 text (binary or other encoding); it cannot be read as text.",
                fixable=False,
                description="The file contents could not be decoded as UTF-8."
            )
        return envelope.success(text)

    except PermissionError as e:
        return envelope.error_permission(
            f"Error: Permission denied: {str(e)}",
            description="You don't have permission to read this file."
        )
    except OSError as e:
        return envelope.error_validation(
            f"Error reading file: {str(e)}",
            fixable=False,
            description="An unexpected error occurred while reading the file."
        )


def list_directory_safe(path_str: str) -> Dict[str, Any]:
    """List directory contents with file metadata."""
    try:
        target, error = _contained_target(path_str)
        if error:
            return error

        # Check directory exists
        if not target.exists():
            return envelope.error_validation(
                f"Error: Directory not found: {target}",
                fixable=True,
                description="The directory does not exist at the specified path."
            )

        if not target.is_dir():
            return envelope.error_validation(
                f"Error: Not a directory: {target}",
                fixable=True,
                description="The path points to a file, not a directory."
            )

        # List directory contents
        entries = []
        for item in sorted(target.iterdir()):
            size = item.stat().st_size if item.is_file() else None
            entries.append({
                "name": item.name,
                "type": "file" if item.is_file() else "directory",
                "size": size,
            })

        listing = f"Directory: {target}\n\n"
        for entry in entries:
            type_indicator = "[FILE]" if entry["type"] == "file" else "[DIR]"
            size_str = f" ({entry['size']} bytes)" if entry["size"] is not None else ""
            listing += f"{type_indicator} {entry['name']}{size_str}\n"

        return envelope.success(listing)

    except PermissionError as e:
        return envelope.error_permission(
            f"Error: Permission denied: {str(e)}",
            description="You don't have permission to list this directory."
        )
    except OSError as e:
        return envelope.error_validation(
            f"Error listing directory: {str(e)}",
            fixable=False,
            description="An unexpected error occurred while listing the directory."
        )


def write_file_safe(path_str: str, content: str, searched: bool = False) -> Dict[str, Any]:
    """Write UTF-8 text inside the working directory without ever truncating the target first.

    Outcome classification:
      * encoding failure, or any failure before the temporary file is complete: definitive
        failure, and the existing target is byte-for-byte unchanged;
      * failure while replacing the target: definitive if the temporary file still exists
        (the replace did not happen), otherwise unknown_outcome;
      * success: the complete new content is in place.
    """
    try:
        target, error = _contained_target(path_str)
        if error:
            return error

        if not isinstance(content, str):
            return envelope.error_validation(
                "Error: content must be a string; nothing was written.",
                fixable=True,
            )
        try:
            data = content.encode("utf-8")
        except UnicodeEncodeError as e:
            return envelope.error_validation(
                f"Error: content is not valid UTF-8 text ({e.reason}); nothing was written.",
                fixable=True,
                description="Replace the unencodable characters and retry."
            )

        # Soft nudge: warn if creating a new file without searching first
        warning = ""
        if not target.exists() and not searched:
            warning = "⚠️ Note: No codebase search (grep/rg/find) was run this session before creating this file. Consider checking for existing implementations first.\n"

        target.parent.mkdir(parents=True, exist_ok=True)

        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as e:
            _remove_quietly(tmp)
            return envelope.error_validation(
                f"Error writing file: {e}; the existing file was not modified.",
                fixable=False,
                description="The temporary file could not be completed; nothing was replaced."
            )

        try:
            os.replace(tmp, target)
        except OSError as e:
            if tmp.exists():
                _remove_quietly(tmp)
                return envelope.error_validation(
                    f"Error replacing file: {e}; the existing file was not modified.",
                    fixable=False,
                    description="The replacement did not happen."
                )
            return envelope.error_unknown_outcome(
                f"Error replacing file: {e}; the file may have been replaced. Not retried.",
                description="The replacement reported an error after the temporary file was consumed."
            )

        return envelope.success(warning + f"File written successfully: {target}")

    except PermissionError as e:
        return envelope.error_permission(
            f"Error: Permission denied: {str(e)}",
            description="You don't have permission to write to this location."
        )
    except OSError as e:
        return envelope.error_validation(
            f"Error writing file: {str(e)}",
            fixable=False,
            description="An unexpected error occurred while writing the file."
        )


EDIT_MAX_FILE_BYTES = 1 << 20          # larger files are refused: an edit reads the whole file into memory
EDIT_MAX_STRING_CHARS = 64 * 1024      # per old_string / new_string


def _occurrences(text: str, needle: str, limit: int) -> int:
    """Count occurrences, overlapping ones included, stopping once limit is reached."""
    count, start = 0, 0
    while count < limit:
        index = text.find(needle, start)
        if index < 0:
            break
        count += 1
        start = index + 1
    return count


def _symlinked_component(path_str: str) -> Optional[Dict[str, Any]]:
    """Refuse a path that names a symbolic link anywhere below the workspace root.

    The walk uses lstat on each component, so no alias is followed before it is judged. The workspace root itself may be
    reached through links: it is the trust boundary, and an ancestor outside it is not this check's concern. Absolute
    paths are left to the containment check, which refuses them.
    """
    if Path(path_str).is_absolute():
        return None
    current = Path.cwd()
    for part in Path(path_str).parts:
        if part == ".":
            continue
        if part == "..":
            current = current.parent  # lexical; every component before it has already been checked
            continue
        current = current / part
        if os.path.islink(current):
            return envelope.error_permission(
                f"Error: {path_str} passes through a symbolic link ({part}); edit_file does not follow links. "
                "Nothing was changed.",
                description="Name the real path inside the workspace instead."
            )
        if not os.path.lexists(current):
            return None  # nothing below a missing component can be a link
    return None


def edit_file_safe(path_str: str, old_string: Any, new_string: Any) -> Dict[str, Any]:
    """Replace exactly one occurrence of old_string with new_string in an existing UTF-8 text file.

    Every refusal happens before the file is replaced, so a refused edit leaves the file byte-for-byte unchanged:
      * the path must resolve inside the working directory, and its final component must not be a symbolic link;
      * the target must be a regular file, not a hard-linked one, at most EDIT_MAX_FILE_BYTES, without NUL bytes,
        and valid UTF-8;
      * old_string must occur exactly once (overlapping occurrences count), and must differ from new_string;
      * the file must still hold the bytes that were read, immediately before the replacement (stale-edit check).
    The replacement is written to a temporary file in the same directory, flushed, given the original permission
    bits, and moved over the target with os.replace. Line endings and any byte-order mark are matched as stored.
    """
    try:
        if not isinstance(path_str, str) or not path_str:
            return envelope.error_validation("Error: edit_file requires a 'path' argument; nothing was changed.",
                                             fixable=True)
        if not isinstance(old_string, str) or not old_string:
            return envelope.error_validation("Error: old_string must be a non-empty string; nothing was changed.",
                                             fixable=True)
        if not isinstance(new_string, str):
            return envelope.error_validation("Error: new_string must be a string; nothing was changed.", fixable=True)
        if old_string == new_string:
            return envelope.error_validation("Error: old_string and new_string are identical; nothing was changed.",
                                             fixable=True)
        if len(old_string) > EDIT_MAX_STRING_CHARS or len(new_string) > EDIT_MAX_STRING_CHARS:
            return envelope.error_validation(
                f"Error: old_string and new_string are limited to {EDIT_MAX_STRING_CHARS} characters; nothing was changed.",
                fixable=True)
        try:
            new_string.encode("utf-8")  # refused before anything is read or written
        except UnicodeEncodeError as e:
            return envelope.error_validation(
                f"Error: new_string is not valid UTF-8 text ({e.reason}); nothing was changed.",
                fixable=True,
                description="Replace the unencodable characters and retry.")

        link_error = _symlinked_component(path_str)
        if link_error:
            return link_error
        target, error = _contained_target(path_str)
        if error:
            return error
        cwd = Path.cwd().resolve()
        if not target.exists():
            return envelope.error_validation(
                f"Error: File not found: {path_str}. edit_file only changes existing files; use write_file to create one.",
                fixable=True,
                description="The file does not exist at the specified path."
            )

        before = os.stat(target)
        if not stat.S_ISREG(before.st_mode):
            return envelope.error_validation(
                f"Error: {path_str} is not a regular file; nothing was changed.", fixable=False)
        if before.st_nlink > 1:
            return envelope.error_validation(
                f"Error: {path_str} has other hard links; replacing it would split them. Nothing was changed.",
                fixable=False)
        if before.st_size > EDIT_MAX_FILE_BYTES:
            return envelope.error_validation(
                f"Error: {path_str} is larger than {EDIT_MAX_FILE_BYTES} bytes; nothing was changed.", fixable=False)

        data = target.read_bytes()
        if len(data) > EDIT_MAX_FILE_BYTES:
            return envelope.error_validation(
                f"Error: {path_str} is larger than {EDIT_MAX_FILE_BYTES} bytes; nothing was changed.", fixable=False)
        if b"\x00" in data:
            return envelope.error_validation(
                f"Error: {path_str} looks binary (contains NUL bytes); nothing was changed.", fixable=False)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return envelope.error_validation(
                f"Error: {path_str} is not UTF-8 text; nothing was changed.", fixable=False)

        matches = _occurrences(text, old_string, limit=2)
        if matches == 0:
            return envelope.error_validation(
                f"Error: old_string was not found in {path_str}; nothing was changed. Read the file again and "
                "match it exactly. If the file uses CRLF line endings, old_string must include \\r\\n.",
                fixable=True)
        if matches > 1:
            return envelope.error_validation(
                f"Error: old_string matches more than one location in {path_str}; nothing was changed. "
                "Include more surrounding text so that it matches exactly once.",
                fixable=True)

        index = text.find(old_string)
        new_text = text[:index] + new_string + text[index + len(old_string):]
        new_data = new_text.encode("utf-8")  # cannot fail: new_string was validated above, and the rest is decoded UTF-8
        line = text.count("\n", 0, index) + 1

        # Stale-edit check: the file must still hold the bytes this edit was computed from
        if target.read_bytes() != data:
            return envelope.error_validation(
                f"Error: {path_str} changed while the edit was being prepared; nothing was changed. Read it again.",
                fixable=True)

        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp, "xb") as fh:
                fh.write(new_data)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, stat.S_IMODE(before.st_mode))
        except OSError as e:
            _remove_quietly(tmp)
            return envelope.error_validation(
                f"Error writing replacement: {e}; the existing file was not modified.",
                fixable=False,
                description="The temporary file could not be completed; nothing was replaced."
            )

        try:
            os.replace(tmp, target)
        except OSError as e:
            if tmp.exists():
                _remove_quietly(tmp)
                return envelope.error_validation(
                    f"Error replacing file: {e}; the existing file was not modified.",
                    fixable=False,
                    description="The replacement did not happen."
                )
            return envelope.error_unknown_outcome(
                f"Error replacing file: {e}; the file may have been replaced. Not retried.",
                description="The replacement reported an error after the temporary file was consumed."
            )

        removed, added = old_string.count("\n") + 1, new_string.count("\n") + 1
        display = target.relative_to(cwd)
        return envelope.success(
            f"Edited {display}: replaced 1 occurrence at line {line} ({removed} line(s) with {added} line(s)); "
            f"the file is now {len(new_data)} bytes."
        )

    except PermissionError as e:
        return envelope.error_permission(
            f"Error: Permission denied: {str(e)}; nothing was changed.",
            description="You don't have permission to modify this file."
        )
    except OSError as e:
        return envelope.error_validation(
            f"Error editing file: {str(e)}; nothing was changed.",
            fixable=False,
            description="An unexpected error occurred while editing the file."
        )


def _remove_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
