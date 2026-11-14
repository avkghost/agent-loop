"""Deliverer (strategy: escalate immediately). Host-only validation implementation of the contract in README.md."""

import config
from .profiles import get_profile
from .sink import AmbiguousTimeout, DefiniteFailure

APPLIED = "applied"
NOT_APPLIED = "not_applied"
UNRESOLVED = "unresolved"


class InvalidEvent(ValueError):
    """The event does not meet the contract. Nothing is sent for an invalid event."""


def _event_id(event):
    if not isinstance(event, dict) or not isinstance(event.get("event_id"), str) or not event["event_id"]:
        raise InvalidEvent("event must be an object with a non-empty string event_id")
    return event["event_id"]


class Deliverer:
    def __init__(self, sink, profile=None):
        self._sink = sink
        self._profile = get_profile(profile or config.ACTIVE_PROFILE)
        self._status = {}

    def status(self, event_id):
        return self._status.get(event_id)

    def report(self):
        return sorted(event_id for event_id, status in self._status.items() if status == UNRESOLVED)

    def deliver(self, event):
        event_id = _event_id(event)
        if self._status.get(event_id) in (APPLIED, UNRESOLVED):
            return self._status[event_id]
        sends = 0
        while sends < self._profile.delivery_budget:
            sends += 1
            try:
                self._sink.send(event_id)
            except DefiniteFailure:
                continue
            except AmbiguousTimeout:
                self._status[event_id] = UNRESOLVED
                return UNRESOLVED
            self._status[event_id] = APPLIED
            return APPLIED
        self._status[event_id] = NOT_APPLIED
        return NOT_APPLIED
