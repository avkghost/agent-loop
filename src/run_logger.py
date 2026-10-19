"""Structured run logging for debugging and auditing."""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


LOGS_DIR = Path.home() / ".agent_loop" / "logs"


class RunLogger:
    """Logs full run details as newline-delimited JSON."""

    def __init__(self, session_id: str, enabled: bool = True):
        """Initialize logger for a session."""
        self.session_id = session_id
        self.enabled = enabled
        self._ensure_dir()

    @staticmethod
    def _ensure_dir():
        """Ensure logs directory exists."""
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a structured event (one JSON object per line)."""
        if not self.enabled:
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": event_type,
            **data,
        }

        log_path = LOGS_DIR / f"{self.session_id}.jsonl"
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass

    def log_iteration(self, iteration: int, goal: str, messages_count: int) -> None:
        """Log iteration start."""
        self.log("iteration_start", {
            "iteration": iteration,
            "goal": goal,
            "messages_count": messages_count,
        })

    def log_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Log tool call."""
        self.log("tool_call", {
            "tool_name": tool_name,
            "arguments": arguments,
        })

    def log_observation(self, tool_name: str, observation: Dict[str, Any]) -> None:
        """Log tool observation."""
        self.log("observation", {
            "tool_name": tool_name,
            "observation": observation,
        })

    def log_completion(self, reason: str) -> None:
        """Log loop completion."""
        self.log("completion", {
            "reason": reason,
        })
