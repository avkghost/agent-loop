"""Replay a JSON-lines event file through a Processor. The summary is deterministic. Invalid lines are reported and
not applied; a frozen account is reported as failed."""

import json

from .events import InvalidEvent
from .processor import AccountFrozen


def replay(lines, processor):
    summary = {"applied": 0, "duplicate": 0, "invalid": 0, "failed": 0, "errors": []}
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            summary["invalid"] += 1
            summary["errors"].append(f"line {number}: {exc.msg}")
            continue
        try:
            outcome = processor.apply(raw)
        except InvalidEvent as exc:
            summary["invalid"] += 1
            summary["errors"].append(f"line {number}: {exc}")
        except AccountFrozen as exc:
            summary["failed"] += 1
            summary["errors"].append(f"line {number}: {exc}")
        else:
            summary[outcome.value] += 1
    return summary
