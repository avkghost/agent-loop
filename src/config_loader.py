"""Configuration file loader for agent_loop."""

from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file, with fallback search path."""

    # Determine which file to load
    search_paths = [
        config_path,  # Explicit path if provided
        Path("./agent_loop.yaml"),  # Project local config
        Path.home() / ".agent_loop" / "config.yaml",  # User home config
    ]

    config_file = None
    for path in search_paths:
        if path and Path(path).exists():
            config_file = Path(path)
            break

    if not config_file:
        return {}  # No config file found, return empty dict

    # Try to load YAML
    try:
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f) or {}
        return config
    except ImportError:
        # Fallback: parse as simple key=value if yaml not available
        config = {}
        try:
            with open(config_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "=" in line:
                            key, value = line.split("=", 1)
                            config[key.strip()] = value.strip()
        except Exception:
            pass
        return config
    except Exception:
        return {}


def apply_config_overrides(config: Dict[str, Any]) -> None:
    """Apply loaded config to src.config module globals."""
    from . import config as config_module

    # Map of config keys to config module attributes
    mappings = {
        "LM_STUDIO_URL": "LM_STUDIO_URL",
        "MODEL_NAME": "MODEL_NAME",
        "MAX_ITERATIONS": "MAX_ITERATIONS",
        "MAX_TOKENS": "MAX_TOKENS",
        "COMPACT_THRESHOLD": "COMPACT_THRESHOLD",
        "ALLOWED_TOOLS": "ALLOWED_TOOLS",
    }

    for config_key, attr_name in mappings.items():
        if config_key in config:
            value = config[config_key]

            # Type conversions for numeric values
            if attr_name in ("MAX_ITERATIONS", "MAX_TOKENS"):
                value = int(value)
            elif attr_name == "COMPACT_THRESHOLD":
                value = float(value)
            elif attr_name == "ALLOWED_TOOLS" and isinstance(value, str):
                value = set(value.split(","))

            setattr(config_module, attr_name, value)
