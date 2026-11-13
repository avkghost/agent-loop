"""Smart context compaction with LLM-based summarization."""

import asyncio
import json
import requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from . import config, envelope, prompts


SUMMARY_BUDGET_FRACTION = 0.2  # max share of MAX_TOKENS a summary may occupy
SUMMARY_MARKER = "[Context Summary]"
RECENT_USER_MARKER = "[Recent User Message]"


class CannotCompactError(RuntimeError):
    """The required context cannot fit within the configured conversation budget."""


@dataclass
class CompactionState:
    """State that is authoritative for compaction and independent of model text."""

    objective: Optional[str]
    summary: Optional[str] = None


def _recent_tail_start(messages: List[Dict[str, Any]], keep: int = 4) -> int:
    """Return the start of a recent tail that does not begin with a tool result.

    A tool result whose assistant declaration would be cut off is dropped together with
    the rest of its group, so every retained tool message keeps its declaration.
    """
    start = max(len(messages) - keep, 1)
    while start < len(messages) and messages[start].get("role") == "tool":
        start += 1
    return start


def _recent_tail(messages: List[Dict[str, Any]], keep: int = 4) -> List[Dict[str, Any]]:
    """Return the protocol-valid recent tail."""
    return messages[_recent_tail_start(messages, keep):]


def _legacy_objective(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Get an objective only from an unstructured, marker-free legacy history."""
    if any(
        marker in str(message.get("content") or "")
        for message in messages
        for marker in (SUMMARY_MARKER, RECENT_USER_MARKER)
    ):
        raise CannotCompactError(
            "Cannot compact legacy conversation: marker-bearing history has ambiguous state."
        )
    if len(messages) > 1 and messages[1].get("role") == "user":
        return str(messages[1].get("content", ""))
    return None


def _assemble(system_msg: Dict[str, Any], objective: Optional[str],
              summary_text: Optional[str], tail: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build [system, one user message (objective and/or summary), tail].

    The objective and summary share one user message, and a user message that starts the
    tail is merged in too, so the provider never receives adjacent user roles.
    """
    parts = []
    if objective is not None:
        parts.append(objective)
    if summary_text is not None:
        parts.append(f"{SUMMARY_MARKER}\n{summary_text}")
    tail = list(tail)
    if not parts:
        return [system_msg] + tail
    if tail and tail[0].get("role") == "user":
        parts.append(f"{RECENT_USER_MARKER}\n{tail[0].get('content', '')}")
        tail = tail[1:]
    return [system_msg, {"role": "user", "content": "\n\n".join(parts)}] + tail


def _estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate context size with the same four-characters-per-token heuristic."""
    return sum(len(json.dumps(message, ensure_ascii=False)) // 4 for message in messages)


def _within_budget(messages: List[Dict[str, Any]]) -> bool:
    return _estimate_tokens(messages) <= config.MAX_TOKENS


def _budgeted_assembly(
    system_msg: Dict[str, Any],
    objective: Optional[str],
    summary_text: Optional[str],
    messages: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """Retain the largest recent protocol-valid tail that fits the budget."""
    for keep in range(4, -1, -1):
        candidate = _assemble(system_msg, objective, summary_text, _recent_tail(messages, keep))
        if _within_budget(candidate):
            return candidate
    return None


def _fallback_or_raise(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    objective = _legacy_objective(messages)
    result = _budgeted_assembly(messages[0], objective, None, messages)
    if result is None:
        raise CannotCompactError(
            "Cannot compact conversation: the system prompt, original objective, "
            "and required recent protocol messages exceed MAX_TOKENS."
        )
    return result


def _fallback_with_state(messages: List[Dict[str, Any]], state: CompactionState) -> List[Dict[str, Any]]:
    """Drop the replaceable summary and retain only the structured objective."""
    result = _budgeted_assembly(messages[0], state.objective, None, messages)
    if result is None:
        raise CannotCompactError(
            "Cannot compact conversation: the system prompt, original objective, "
            "and required recent protocol messages exceed MAX_TOKENS."
        )
    state.summary = None
    return result


def truncate_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the system message, the original objective, and a protocol-valid recent tail."""
    if not messages:
        return messages
    return _assemble(messages[0], _legacy_objective(messages), None, _recent_tail(messages))


def restore_structured_state(messages: List[Dict[str, Any]], state: CompactionState) -> List[Dict[str, Any]]:
    """Render persisted structured state without parsing model-generated message text."""
    if not messages:
        return messages
    if len(messages) < 2 or messages[1].get("role") != "user":
        restored = list(messages)
        restored.insert(1, {"role": "user", "content": state.objective or ""})
        return restored
    if state.summary is None:
        restored = list(messages)
        restored[1] = {"role": "user", "content": state.objective or ""}
        return restored
    return _assemble(messages[0], state.objective, state.summary, _recent_tail(messages))


async def compact_messages(
    messages: List[Dict[str, Any]],
    event_loop: asyncio.AbstractEventLoop,
    mcp_client: Optional[Any] = None,
    state: Optional[CompactionState] = None,
) -> List[Dict[str, Any]]:
    """Compact messages by summarizing dropped middle messages via LLM.

    Strategy: Keep the system prompt plus a protocol-valid recent tail, summarizing
    everything in between. Falls back to hard truncation if the LLM call fails or
    the summary exceeds its configured budget share.
    """

    if len(messages) <= 5:
        return messages  # Not enough messages to compact

    # Try LLM-based summarization
    try:
        # Extract everything before the same adjusted tail retained below.
        system_msg = messages[0]
        tail_start = _recent_tail_start(messages)
        if state is None:
            objective, previous_summary = _legacy_objective(messages), None
        else:
            objective, previous_summary = state.objective, state.summary
        middle_start = 2 if objective is not None else 1
        middle_messages = messages[middle_start:tail_start]

        # Create a summary prompt
        previous_text = previous_summary or "(none)"
        middle_text = json.dumps(middle_messages, indent=2)
        summary_prompt = f"""Summarize this conversation history concisely, preserving key facts, decisions, and progress so far:

Previous compacted summary (carry forward useful context, but replace it with your new summary):
{previous_text}

{middle_text}

Provide a brief summary (2-3 sentences) of what has been done and decided."""

        # Call LLM to summarize
        payload = {
            "model": config.MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                {"role": "user", "content": summary_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500,
        }

        response = requests.post(config.LM_STUDIO_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        summary_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        # A summary may use at most this share of the conversation budget (heuristic: 4 chars ~= 1 token).
        # Larger output is rejected and the controlled hard-truncation fallback is used instead.
        if summary_text and len(summary_text) // 4 <= config.MAX_TOKENS * SUMMARY_BUDGET_FRACTION:
            # Objective and summary share one user message; retain the largest tail that fits.
            candidate = _budgeted_assembly(system_msg, objective, summary_text, messages)
            if candidate is not None:
                if state is not None:
                    state.summary = summary_text
                return candidate

    except Exception:
        pass  # Fall through to hard truncation

    # Fallback: bounded hard truncation, without carrying forward old summaries.
    if state is not None:
        return _fallback_with_state(messages, state)
    return _fallback_or_raise(messages)
