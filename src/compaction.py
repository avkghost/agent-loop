"""Smart context compaction with LLM-based summarization."""

import asyncio
import json
import requests
from typing import List, Dict, Any, Optional

from . import config, envelope, prompts


async def compact_messages(
    messages: List[Dict[str, Any]],
    event_loop: asyncio.AbstractEventLoop,
    mcp_client: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Compact messages by summarizing dropped middle messages via LLM.

    Strategy: Keep system prompt + last 4 messages, summarize everything in between.
    Falls back to hard truncation if LLM call fails.
    """

    if len(messages) <= 5:
        return messages  # Not enough messages to compact

    # Try LLM-based summarization
    try:
        # Extract messages to summarize (everything except system + last 4)
        system_msg = messages[0]
        middle_messages = messages[1:-4]
        last_messages = messages[-4:]

        # Create a summary prompt
        middle_text = json.dumps(middle_messages, indent=2)
        summary_prompt = f"""Summarize this conversation history concisely, preserving key facts, decisions, and progress so far:

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

        if summary_text:
            # Create summary message
            summary_msg = {
                "role": "assistant",
                "content": f"[Context Summary]\n{summary_text}"
            }

            # Return: system + summary + last 4 messages
            return [system_msg, summary_msg] + last_messages

    except Exception:
        pass  # Fall through to hard truncation

    # Fallback: hard truncation (original behavior)
    return messages[:1] + messages[-4:]
