"""Load custom commands from Python modules."""

import importlib.util
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass


@dataclass
class Command:
    """Represents a custom command."""
    name: str
    description: str
    handler: Callable


def load_commands(commands_dir: str = "commands") -> Dict[str, Command]:
    """Load all commands from Python modules in the commands directory."""
    commands = {}
    commands_path = Path(commands_dir)

    if not commands_path.exists():
        return commands

    for py_file in sorted(commands_path.glob("*.py")):
        if py_file.name.startswith("_"):  # Skip __init__.py and internal files
            continue

        command_name = py_file.stem
        try:
            command = load_command_module(py_file)
            if command:
                commands[command_name] = command
        except Exception as e:
            print(f"⚠️  Warning: Failed to load command '{command_name}': {e}")

    return commands


def load_command_module(filepath: Path) -> Optional[Command]:
    """Load a command from a Python module."""
    try:
        # Dynamically import the module
        spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Look for a COMMAND dict/object with name, description, handler
        if hasattr(module, 'COMMAND'):
            cmd_data = module.COMMAND
            if isinstance(cmd_data, dict):
                name = cmd_data.get('name', filepath.stem)
                description = cmd_data.get('description', '')
                handler = cmd_data.get('handler')

                if handler and callable(handler):
                    return Command(
                        name=name,
                        description=description,
                        handler=handler
                    )

        return None
    except Exception as e:
        print(f"⚠️  Error loading command module {filepath}: {e}")
        return None
