"""Response caching for LM Studio calls."""

import hashlib
import json
from typing import Dict, Any, Optional


class ResponseCache:
    """In-memory cache for LM Studio responses keyed by payload hash."""

    def __init__(self, enabled: bool = False):
        """Initialize cache. Disabled by default (opt-in via CLI flag)."""
        self.enabled = enabled
        self._cache: Dict[str, Any] = {}

    def _hash_payload(self, payload: Dict[str, Any]) -> str:
        """Hash a payload dict for use as cache key."""
        # Only hash deterministic parts: messages, tools, model, temperature
        hashable = {
            "messages": json.dumps(payload.get("messages", []), sort_keys=True),
            "model": payload.get("model", ""),
            "temperature": payload.get("temperature", 0.2),
            "tools_count": len(payload.get("tools", [])),  # Not tools themselves (large)
        }
        key_str = json.dumps(hashable, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response if it exists and caching is enabled."""
        if not self.enabled:
            return None

        key = self._hash_payload(payload)
        return self._cache.get(key)

    def set(self, payload: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Cache a response if caching is enabled."""
        if not self.enabled:
            return

        key = self._hash_payload(payload)
        self._cache[key] = response

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def size(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)
