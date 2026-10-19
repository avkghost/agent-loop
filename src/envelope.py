"""Structured error/result envelope following MCP conventions."""

import json
from typing import Optional, Any, Dict

# Error categories per lesson 2.2
CATEGORY_TRANSIENT = "transient"      # Timeouts, service unavailable, rate limits
CATEGORY_VALIDATION = "validation"    # Invalid input format, malformed data
CATEGORY_BUSINESS = "business"        # Business rule violation (not used in this agent)
CATEGORY_PERMISSION = "permission"    # Access denied, path outside boundary

def success(content: str) -> Dict[str, Any]:
    """Successful tool result (not an error)."""
    return {
        "content": content,
        "isError": False,
        "errorCategory": None,
        "isRetryable": None,
    }

def empty_result() -> Dict[str, Any]:
    """Valid empty result (tool ran successfully, found nothing)."""
    return success("[No results found]")

def error_transient(content: str, description: Optional[str] = None) -> Dict[str, Any]:
    """Transient error: retry-able (timeout, service unavailable, rate limit)."""
    return {
        "content": content,
        "isError": True,
        "errorCategory": CATEGORY_TRANSIENT,
        "isRetryable": True,
        "description": description,
    }

def error_validation(content: str, fixable: bool = True, description: Optional[str] = None) -> Dict[str, Any]:
    """Validation error: fixable (true) or not (false)."""
    return {
        "content": content,
        "isError": True,
        "errorCategory": CATEGORY_VALIDATION,
        "isRetryable": fixable,
        "description": description,
    }

def error_permission(content: str, description: Optional[str] = None) -> Dict[str, Any]:
    """Permission error: not retry-able."""
    return {
        "content": content,
        "isError": True,
        "errorCategory": CATEGORY_PERMISSION,
        "isRetryable": False,
        "description": description,
    }

def normalize_mcp_result(mcp_result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an MCP CallToolResult into our envelope shape."""
    # If the result already has our structure, pass through
    if "isError" in mcp_result:
        return mcp_result

    # Otherwise wrap it
    return {
        "content": str(mcp_result.get("content", "")),
        "isError": mcp_result.get("isError", False),
        "errorCategory": mcp_result.get("errorCategory"),
        "isRetryable": mcp_result.get("isRetryable"),
    }

def serialize_for_model(envelope: Dict[str, Any]) -> str:
    """Serialize the envelope dict to JSON for the model."""
    # Include only the essential fields for the model
    data = {
        "content": envelope.get("content", ""),
        "isError": envelope.get("isError", False),
    }
    if envelope.get("errorCategory"):
        data["errorCategory"] = envelope["errorCategory"]
    if envelope.get("isRetryable") is not None:
        data["isRetryable"] = envelope["isRetryable"]
    return json.dumps(data)
