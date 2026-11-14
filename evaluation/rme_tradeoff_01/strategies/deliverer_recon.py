"""Deliverer (strategy: reconcile within budget). Host-only validation implementation of the contract in README.md."""

import config
from .profiles import get_profile
from .sink import AmbiguousTimeout, DefiniteFailure, ReconciliationUnavailable

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
        if self._status.get(event_id) == APPLIED:
            return APPLIED
        # An unresolved event may be sent again only after an authoritative absence has been established.
        ambiguous = self._status.get(event_id) == UNRESOLVED
        sends = lookups = 0
        while True:
            if ambiguous:
                if self._sink.lookup is None or lookups >= self._profile.reconcile_budget:
                    break
                lookups += 1
                try:
                    present = self._sink.lookup(event_id)
                except ReconciliationUnavailable:
                    break
                if present:
                    self._status[event_id] = APPLIED
                    return APPLIED
                ambiguous = False
            if sends >= self._profile.delivery_budget:
                break
            sends += 1
            try:
                self._sink.send(event_id)
            except DefiniteFailure:
                continue
            except AmbiguousTimeout:
                ambiguous = True
                continue
            self._status[event_id] = APPLIED
            return APPLIED
        self._status[event_id] = UNRESOLVED if ambiguous else NOT_APPLIED
        return self._status[event_id]
