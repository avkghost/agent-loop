"""Structured error/result envelope following MCP conventions."""

import json
from typing import Optional, Any, Dict

# Error categories per lesson 2.2
CATEGORY_TRANSIENT = "transient"      # Timeouts, service unavailable, rate limits
CATEGORY_VALIDATION = "validation"    # Invalid input format, malformed data
CATEGORY_BUSINESS = "business"        # Business rule violation (not used in this agent)
CATEGORY_PERMISSION = "permission"    # Access denied, path outside boundary
CATEGORY_UNKNOWN_OUTCOME = "unknown_outcome"  # Tool raised; its side effects may or may not have happened

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

class ToolExecutionError(Exception):
    """An explicit tool-boundary failure with an unknown external outcome.

    Dispatcher implementations may raise this when an operation may have started
    but did not produce a reliable result. Ordinary programming errors must use
    normal exceptions so they are not misreported as external side effects.
    """


def error_unknown_outcome(content: str, description: Optional[str] = None) -> Dict[str, Any]:
    """Tool raised an exception. The outcome is unknown, so the call is never marked retryable."""
    return {
        "content": content,
        "isError": True,
        "errorCategory": CATEGORY_UNKNOWN_OUTCOME,
        "isRetryable": False,
        "description": description,
    }


_MISSING = object()


def mcp_result_envelope(tool_name: str, result: Any) -> Dict[str, Any]:
    """Convert an MCP call result into an envelope, validating its structure first.

    A result that is missing, has an unexpected shape, or has malformed content is never
    reported as success: the call was already submitted, so its outcome is unknown.
    Valid empty content is a successful empty result.
    """
    malformed = error_unknown_outcome(
        f"MCP tool {tool_name} returned a malformed response; its outcome cannot be verified. It was not retried.",
        description="The response did not match the MCP result structure.",
    )
    if result is None:
        return malformed
    if isinstance(result, dict):
        content = result.get("content", _MISSING)
        is_error = result.get("isError", False)
    else:
        content = getattr(result, "content", _MISSING)
        is_error = getattr(result, "isError", False)
    if content is _MISSING or not isinstance(content, list) or not isinstance(is_error, bool):
        return malformed

    parts = []
    for item in content:
        if isinstance(item, dict):
            text, kind = item.get("text"), item.get("type")
        else:
            text, kind = getattr(item, "text", None), getattr(item, "type", None)
        if isinstance(text, str):
            parts.append(text)
        elif isinstance(kind, str) and kind:
            parts.append(f"[{kind} content omitted]")  # non-text MCP content is not representable here
        else:
            return malformed
    body = "\n".join(parts)
    if is_error:
        return error_unknown_outcome(
            body or f"MCP tool {tool_name} reported an error.",
            description="The remote tool reported an error after the call was submitted; "
                        "side effects may have occurred. It was not retried.",
        )
    return success(body)


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
