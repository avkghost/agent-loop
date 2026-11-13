"""File operations tools (read, write, list).

Containment: every path is resolved (following symlinks) and must lie inside the working
directory, checked by path components. This is a path check, not an OS sandbox; it does not
protect against adversarial filesystem changes made concurrently by other processes.
"""

import os
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


def write_file_safe(path_str: str, content: str) -> Dict[str, Any]:
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
        if not target.exists() and not config.HAS_SEARCHED_CODEBASE:
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


def _remove_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
