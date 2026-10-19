"""File operations tools (read, write, list)."""

from pathlib import Path
from typing import Dict, Any, List

from .. import config, envelope

def read_file_safe(path_str: str) -> Dict[str, Any]:
    """Read content from a file, restricted to the working directory."""
    try:
        # Resolve path relative to current working directory
        cwd = Path.cwd()
        target = (cwd / path_str).resolve()

        # Ensure target is within the working directory
        if not str(target).startswith(str(cwd)):
            return envelope.error_permission(
                f"Error: Path {path_str} resolves outside working directory {cwd}",
                description="Paths must resolve within the working directory."
            )

        # Check file exists
        if not target.exists():
            return envelope.error_validation(
                f"Error: File not found: {target}",
                fixable=True,
                description="The file does not exist at the specified path."
            )

        # Read the file
        content = target.read_text()
        return envelope.success(content)

    except PermissionError as e:
        return envelope.error_permission(
            f"Error: Permission denied: {str(e)}",
            description="You don't have permission to read this file."
        )
    except Exception as e:
        return envelope.error_validation(
            f"Error reading file: {str(e)}",
            fixable=False,
            description="An unexpected error occurred while reading the file."
        )


def list_directory_safe(path_str: str) -> Dict[str, Any]:
    """List directory contents with file metadata."""
    try:
        # Resolve path relative to current working directory
        cwd = Path.cwd()
        target = (cwd / path_str).resolve()

        # Ensure target is within the working directory
        if not str(target).startswith(str(cwd)):
            return envelope.error_permission(
                f"Error: Path {path_str} resolves outside working directory {cwd}",
                description="Paths must resolve within the working directory."
            )

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
    except Exception as e:
        return envelope.error_validation(
            f"Error listing directory: {str(e)}",
            fixable=False,
            description="An unexpected error occurred while listing the directory."
        )


def write_file_safe(path_str: str, content: str) -> Dict[str, Any]:
    """Write content to a file, restricted to the working directory."""
    try:
        # Resolve path relative to current working directory
        cwd = Path.cwd()
        target = (cwd / path_str).resolve()

        # Ensure target is within the working directory
        if not str(target).startswith(str(cwd)):
            return envelope.error_permission(
                f"Error: Path {path_str} resolves outside working directory {cwd}",
                description="Paths must resolve within the working directory."
            )

        # Soft nudge: warn if creating a new file without searching first
        file_existed = target.exists()
        warning = ""
        if not file_existed and not config.HAS_SEARCHED_CODEBASE:
            warning = "⚠️ Note: No codebase search (grep/rg/find) was run this session before creating this file. Consider checking for existing implementations first.\n"

        # Create parent directories if needed
        target.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        target.write_text(content)
        return envelope.success(warning + f"File written successfully: {target}")

    except PermissionError as e:
        return envelope.error_permission(
            f"Error: Permission denied: {str(e)}",
            description="You don't have permission to write to this location."
        )
    except Exception as e:
        return envelope.error_validation(
            f"Error writing file: {str(e)}",
            fixable=False,
            description="An unexpected error occurred while writing the file."
        )
