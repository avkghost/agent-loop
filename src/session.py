"""Session persistence for resuming interrupted runs."""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


SESSIONS_DIR = Path.home() / ".agent_loop" / "sessions"


def _ensure_dir():
    """Ensure sessions directory exists."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def save_session(session_id: str, goal: str, messages: List[Dict[str, Any]]) -> None:
    """Save session state to disk."""
    _ensure_dir()

    session_data = {
        "session_id": session_id,
        "goal": goal,
        "messages": messages,
        "saved_at": datetime.now().isoformat(),
    }

    session_path = SESSIONS_DIR / f"{session_id}.json"
    with open(session_path, "w") as f:
        json.dump(session_data, f, indent=2)


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Load session state from disk."""
    session_path = SESSIONS_DIR / f"{session_id}.json"

    if not session_path.exists():
        return None

    try:
        with open(session_path) as f:
            return json.load(f)
    except Exception:
        return None


def list_sessions() -> List[Dict[str, Any]]:
    """List all saved sessions."""
    _ensure_dir()

    sessions = []
    for session_file in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            with open(session_file) as f:
                data = json.load(f)
            sessions.append({
                "session_id": data.get("session_id", session_file.stem),
                "goal": data.get("goal", ""),
                "saved_at": data.get("saved_at", ""),
                "message_count": len(data.get("messages", [])),
            })
        except Exception:
            continue

    return sessions


def delete_session(session_id: str) -> bool:
    """Delete a saved session."""
    session_path = SESSIONS_DIR / f"{session_id}.json"

    try:
        session_path.unlink()
        return True
    except Exception:
        return False
