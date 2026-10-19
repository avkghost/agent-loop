#!/usr/bin/env python3
"""Agent loop shim for backwards compatibility.

This script delegates to the modularized implementation in src/.
Allows the invocation `python3 agent_loop.py "goal"` to continue working.
"""

from src.main import main

if __name__ == "__main__":
    main()
