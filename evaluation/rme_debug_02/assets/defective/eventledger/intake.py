"""Event intake. An event is a JSON object with event_id (non-empty string), account (non-empty string) and amount
(integer, not a boolean). Anything else raises InvalidEvent before any state is read or changed."""

from dataclasses import dataclass


class InvalidEvent(ValueError):
    """The event does not meet the contract. An invalid event is neither applied nor recorded."""


@dataclass(frozen=True)
class Event:
    event_id: str
    account: str
    amount: int


def parse_event(raw) -> Event:
    if not isinstance(raw, dict):
        raise InvalidEvent("event must be a JSON object")
    event_id, account, amount = raw.get("event_id"), raw.get("account"), raw.get("amount")
    if not isinstance(event_id, str) or not event_id:
        raise InvalidEvent("event_id must be a non-empty string")
    if not isinstance(account, str) or not account:
        raise InvalidEvent("account must be a non-empty string")
    if not isinstance(amount, int) or isinstance(amount, bool):
        raise InvalidEvent("amount must be an integer")
    return Event(event_id=event_id, account=account, amount=amount)
