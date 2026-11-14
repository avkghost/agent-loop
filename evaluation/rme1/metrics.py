"""RME evaluation-side observation of provider turns. Runs inside the agent image and on the host.

wrap_complete() wraps the existing provider's complete() method. It records what the provider returned and how long
the call took, and it changes nothing the runtime sees. Token counts are recorded only when the provider itself
reported usage that passed its validation and the turn succeeded. Nothing is estimated here.

Not available without a production change: time to first token (the stream callback belongs to the loop) and
reasoning tokens (the provider's usage object does not carry them).
"""

import time
from typing import Any, Dict, List

NOT_AVAILABLE = None


def wrap_complete(provider_cls: type, records: List[Dict[str, Any]]):
    original = provider_cls.complete

    async def timed(self, messages, tools, options, on_text=None):
        started = time.monotonic()
        turn = await original(self, messages, tools, options, on_text)
        duration = round(time.monotonic() - started, 3)
        usage = getattr(turn, "usage", None)
        reported = usage is not None and usage.source == "reported" and bool(getattr(turn, "succeeded", False))
        entry = {
            "index": len(records) + 1,
            "duration_s": duration,
            "outcome": getattr(turn.outcome, "value", str(turn.outcome)),
            "usage_source": usage.source if usage is not None else None,
            "usage": ({"prompt": usage.prompt_tokens, "completion": usage.completion_tokens,
                       "total": usage.total_tokens} if reported else None),
            "usage_error": getattr(turn, "usage_error", None),
            "output_tokens_per_second": (round(usage.completion_tokens / duration, 2)
                                         if reported and duration > 0 else None),
            "time_to_first_token_s": NOT_AVAILABLE,
            "reasoning_tokens": NOT_AVAILABLE,
        }
        records.append(entry)
        return turn

    provider_cls.complete = timed
    return original
